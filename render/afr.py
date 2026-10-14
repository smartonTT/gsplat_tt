"""Task #581: alternate-frame rendering (AFR) on 2 chips, GSPLAT_TT_AFR=1 (default off).

render/run.py --back-to-back with GSPLAT_TT_AFR=1 does not open a device itself. It becomes
the coordinator: it starts one worker process per chip (run.py --afr-worker K/N), worker K
renders the views at sweep positions K, K+N, ... on its own chip, with its own DeviceState,
programs and B4a fold chain (its next view is N positions on, so the fold carries N -> N+2 on
the same chip). Every stage keeps process-wide state (device_state.cpp and about 250 statics
in render/host), so one process per chip is the cheap way to get one host thread and one
DeviceState per device; docs/afr-t581/DESIGN.md has the reasoning and the expected numbers.

Protocol (worker stdout lines starting with MSG, coordinator -> worker stdin "GO <p>"):
  worker: ready (after its warm-up, waiting for GO 0) -> pass {p, t0, t1, digests on p=0}
          for every pass -> done {rc, hero_md5}
  coordinator: starts worker 0, waits for its ready, then worker 1 (so the second never
  compiles kernels into the shared JIT cache at the same time), then sends GO p to all
  workers per pass and times a pass as max(t1) - min(t0) over the chips (CLOCK_MONOTONIC,
  which is system-wide; Python's time.monotonic is per process on macOS). Other worker output is forwarded with an "[afrK] " prefix, so no worker line
  starts with "B2B " or "TTW_TIMING".

Device selection (worker K, PCIe id D = K-th entry of GSPLAT_TT_AFR_DEVICES, default "0,1"):
  default: TT_VISIBLE_DEVICES=D and GSPLAT_TT_DEVICE_ID=0, so each process sees one chip;
  GSPLAT_TT_AFR_VISIBLE=0: TT_VISIBLE_DEVICES unset and GSPLAT_TT_DEVICE_ID=D.

Task #624, single-frame split (GSPLAT_TT_SPLIT=1 with GSPLAT_TT_AFR=1, exactly 2 chips):
both workers render every view, worker K only its tile window (GSPLAT_TT_SPLIT_WINDOW, set
here from GSPLAT_TT_SPLIT_AXIS x|y, default x, and GSPLAT_TT_SPLIT_AT, default 16 tiles;
render/host/split_window.h). A pass still times max(t1) - min(t0) over the chips. Pass 0's
window crops go through a .npy file per worker (a stand-in for a real transfer, not timed);
the coordinator joins them, times the join (split_copy_ms per view) and md5s the joined
frames, so raw_md5 compares with the one-chip run. GSPLAT_TT_SPLIT_LOCKSTEP=1 (latency arm):
the coordinator also releases every view to both chips at once ("V <p> <i>") and times each
view as max(t1) - min(t0), so the split's per-frame latency is measured, not its throughput.
"""
import hashlib
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time

MSG = "AFR_MSG "
_DEFAULT_DEVICES = "0,1"


def now():
    """Seconds on a clock all processes share (CLOCK_MONOTONIC on Linux and macOS)."""
    if hasattr(time, "clock_gettime") and hasattr(time, "CLOCK_MONOTONIC"):
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    return time.time()


def enabled(env=os.environ):
    return env.get("GSPLAT_TT_AFR", "0") not in ("", "0")


def devices(env=os.environ):
    """PCIe ids of the chips, one worker each. At least 2, no repeats."""
    raw = env.get("GSPLAT_TT_AFR_DEVICES", "") or _DEFAULT_DEVICES
    try:
        devs = [int(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"GSPLAT_TT_AFR_DEVICES={raw!r}: want comma-separated ids")
    if len(devs) < 2 or len(set(devs)) != len(devs) or min(devs) < 0:
        raise ValueError(f"GSPLAT_TT_AFR_DEVICES={raw!r}: want >= 2 distinct ids >= 0")
    return devs


def parse_worker(spec):
    """'K/N' -> (K, N)."""
    k, n = (int(x) for x in spec.split("/"))
    if not 0 <= k < n or n < 2:
        raise ValueError(f"--afr-worker {spec!r}: want K/N with 0 <= K < N, N >= 2")
    return k, n


def split(n_views, n_chips):
    """Sweep positions per chip: chip k gets k, k + n_chips, ..."""
    return [list(range(k, n_views, n_chips)) for k in range(n_chips)]


def merge(parts, n_views):
    """parts[k] = chip k's per-view values in its split() order -> one list in sweep order."""
    out = [None] * n_views
    for k, vals in enumerate(parts):
        idx = split(n_views, len(parts))[k]
        if len(vals) != len(idx):
            raise ValueError(f"chip {k}: {len(vals)} values for {len(idx)} views")
        for i, v in zip(idx, vals):
            out[i] = v
    return out


def raw_md5(digests):
    """The sweep md5 run.py prints as raw_md5 (md5 of the per-view md5s in sweep order)."""
    return hashlib.md5("".join(digests).encode()).hexdigest()[:8]


def period_ms(stamps, n_views):
    """ms/view of one pass over all chips: stamps = [(t0, t1)] in seconds, one per chip."""
    return (max(t1 for _, t1 in stamps) - min(t0 for t0, _ in stamps)) * 1000.0 / n_views


def split_mode(env=os.environ):
    """Task #624: None (off) or (axis, at) for GSPLAT_TT_SPLIT=1."""
    if env.get("GSPLAT_TT_SPLIT", "0") in ("", "0"):
        return None
    axis = env.get("GSPLAT_TT_SPLIT_AXIS", "") or "x"
    raw = env.get("GSPLAT_TT_SPLIT_AT", "") or "16"
    try:
        at = int(raw)
    except ValueError:
        raise ValueError(f"GSPLAT_TT_SPLIT_AT={raw!r}: want a tile index")
    if axis not in ("x", "y"):
        raise ValueError(f"GSPLAT_TT_SPLIT_AXIS={axis!r}: want x or y")
    return axis, at


def lockstep(env=os.environ):
    return env.get("GSPLAT_TT_SPLIT_LOCKSTEP", "0") not in ("", "0")


def split_windows(mode, tiles_x, tiles_y):
    """The two tile windows (x0, x1, y0, y1), half-open, of split_mode() `mode`."""
    axis, at = mode
    full = tiles_x if axis == "x" else tiles_y
    if not 0 < at < full:
        raise ValueError(f"GSPLAT_TT_SPLIT_AT={at}: want 0 < at < {full} tiles")
    if axis == "x":
        return [(0, at, 0, tiles_y), (at, tiles_x, 0, tiles_y)]
    return [(0, tiles_x, 0, at), (0, tiles_x, at, tiles_y)]


def worker_window(env=os.environ):
    """The worker's own window from GSPLAT_TT_SPLIT_WINDOW, or None when the split is off."""
    if env.get("GSPLAT_TT_SPLIT", "0") in ("", "0"):
        return None
    win = tuple(int(x) for x in env["GSPLAT_TT_SPLIT_WINDOW"].split(","))
    if len(win) != 4:
        raise ValueError(f"GSPLAT_TT_SPLIT_WINDOW={env['GSPLAT_TT_SPLIT_WINDOW']!r}: want x0,x1,y0,y1")
    return win


def crop(img, win, tile=32):
    x0, x1, y0, y1 = win
    return img[y0 * tile:y1 * tile, x0 * tile:x1 * tile]


def join(crops, windows, shape, tile=32):
    """One frame of `shape` from the chips' window crops."""
    import numpy as np
    out = np.empty(shape, dtype=crops[0].dtype)
    for c, (x0, _, y0, _) in zip(crops, windows):
        out[y0 * tile:y0 * tile + c.shape[0], x0 * tile:x0 * tile + c.shape[1]] = c
    return out


def join_views(crop_paths, windows, shape, tile=32):
    """Pass 0's frames from the workers' crop files -> (frames, join ms per view). Loading
    the files is not timed (it stands in for the chip-to-host transfer)."""
    import numpy as np
    stacks = [np.load(p) for p in crop_paths]
    if len({len(s) for s in stacks}) != 1:
        raise ValueError(f"crop files hold {[len(s) for s in stacks]} views")
    frames, ms = [], []
    for i in range(len(stacks[0])):
        t0 = time.perf_counter()
        frames.append(join([s[i] for s in stacks], windows, shape, tile))
        ms.append((time.perf_counter() - t0) * 1000.0)
    return frames, ms


def worker_env(base, k, dev, visible=None, window=None):
    env = dict(base)
    env["GSPLAT_TT_AFR"] = "0"
    env["GSPLAT_TT_AFR_WORKER"] = str(k)
    if visible is None:
        visible = base.get("GSPLAT_TT_AFR_VISIBLE", "1") not in ("", "0")
    if visible:
        env["TT_VISIBLE_DEVICES"] = str(dev)
        env["GSPLAT_TT_DEVICE_ID"] = "0"
    else:
        env.pop("TT_VISIBLE_DEVICES", None)
        env["GSPLAT_TT_DEVICE_ID"] = str(dev)
    if window is not None:
        env["GSPLAT_TT_SPLIT"] = "1"
        env["GSPLAT_TT_SPLIT_WINDOW"] = ",".join(str(v) for v in window)
    return env


def worker_argv(argv, k, n, windowed=False):
    """run.py arguments for worker k: --afr-worker k/n; workers past the first get --no-ref
    and their own --iter-dir <dir>/afr<k> (their hero_clean.png is their warm-up hero).
    windowed (split): --dump-views is dropped, the coordinator dumps the joined frames."""
    out, i = [], 0
    iter_dir = None
    while i < len(argv):
        a = argv[i]
        if a == "--afr-worker":
            i += 2
            continue
        if a.startswith("--afr-worker="):
            i += 1
            continue
        if a == "--iter-dir" and i + 1 < len(argv):
            iter_dir = argv[i + 1]
            i += 2
            continue
        if a.startswith("--iter-dir="):
            iter_dir = a.split("=", 1)[1]
            i += 1
            continue
        if windowed and a == "--dump-views":
            i += 2
            continue
        if windowed and a.startswith("--dump-views="):
            i += 1
            continue
        out.append(a)
        i += 1
    if k > 0:
        iter_dir = f"{iter_dir or 'render-clean'}/afr{k}"
        if "--no-ref" not in out:
            out.append("--no-ref")
    if iter_dir is not None:
        out += ["--iter-dir", iter_dir]
    return out + ["--afr-worker", f"{k}/{n}"]


class WorkerLink:
    """Worker side of the protocol (run.py _back_to_back calls it around every pass)."""

    def __init__(self, k, n, inp=None, out=None, window=None, lockstep=False):
        self.k, self.n = k, n
        self.inp = inp if inp is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self.t0 = self.t1 = self.tv0 = 0.0
        self.window, self.lockstep = window, lockstep

    def send(self, **ev):
        ev["k"] = self.k
        self.out.write(MSG + json.dumps(ev) + "\n")
        self.out.flush()

    def before_pass(self, p):
        if p == 0:
            self.send(ev="ready")
        line = self.inp.readline().strip()
        if line != f"GO {p}":
            raise SystemExit(f"[afr{self.k}] expected 'GO {p}', got {line!r}; stopping")
        self.t0 = now()

    def before_view(self, p, i):
        """Lockstep (split latency arm): wait for the coordinator's 'V <p> <i>'."""
        line = self.inp.readline().strip()
        if line != f"V {p} {i}":
            raise SystemExit(f"[afr{self.k}] expected 'V {p} {i}', got {line!r}; stopping")
        self.tv0 = now()

    def after_view(self, p, i):
        self.send(ev="view", p=p, i=i, t0=self.tv0, t1=now())

    def mark_end(self):
        self.t1 = now()

    def save_crops(self, imgs, out_dir):
        """Split: pass 0's window crops (uint8 frames) to <out_dir>/split_crops.npy."""
        import numpy as np
        path = os.path.abspath(os.path.join(str(out_dir), "split_crops.npy"))
        np.save(path, np.stack([crop(im, self.window) for im in imgs]))
        return path

    def after_pass(self, p, digests=None, crops=None):
        ev = {"crops": crops} if crops is not None else {}
        self.send(ev="pass", p=p, t0=self.t0, t1=self.t1 or now(), digests=digests, **ev)
        self.t1 = 0.0

    def done(self, rc, hero_md5=None):
        self.send(ev="done", rc=rc, hero_md5=hero_md5)


def _reader(k, stream, q, log):
    for line in stream:
        line = line.rstrip("\n")
        if line.startswith(MSG):
            try:
                q.put((k, json.loads(line[len(MSG):])))
                continue
            except ValueError:
                pass
        log(f"[afr{k}] {line}")
    q.put((k, {"ev": "eof"}))


def coordinate(cmds, n_passes, n_views, log=print, popen=subprocess.Popen, timeout_s=None,
               windowed=False, lock=False):
    """Run the workers. cmds = [(argv, env)] per chip. Returns (rc, info): rc 0 when every
    worker ended with 0, else the first non-zero one (8 for a protocol error or a worker that
    ended early, 5 when the chips' hero frames differ); info has the merged pass-0 digests,
    the per-pass periods (ms/view over all chips) and the per-chip ms per own view.
    windowed (split): every chip renders every view; info["crops"] = pass 0's crop files,
    no digest merge and no hero compare (the caller joins). lock: views are released one at
    a time and info["view_ms"][p] = each view's max(t1) - min(t0)."""
    n = len(cmds)
    q = queue.Queue()
    procs, threads = [], []
    info = {"periods": [], "chip_ms": [[] for _ in range(n)], "digests": None,
            "hero_md5": [None] * n, "worker_rc": [None] * n, "split": windowed,
            "crops": None, "view_ms": []}

    pending = [[] for _ in range(n)]  # events read but not yet taken, per worker

    def get(want, who, p=None, i=None):
        """Next event of each worker in `who`, which must be `want` (pass p); None if a
        worker ended, sent something else or timed out. Events are taken in order per
        worker: one worker's next event never stands in for another's."""
        got = {}
        while True:
            for k in who:
                if k in got or not pending[k]:
                    continue
                ev = pending[k].pop(0)
                if ev.get("ev") == "eof":
                    log(f"[afr] worker {k} ended while waiting for '{want}'")
                    return None
                if ev.get("ev") == "done":
                    info["worker_rc"][k] = ev.get("rc")
                if ev.get("ev") != want or (p is not None and ev.get("p") != p) \
                        or (i is not None and ev.get("i") != i):
                    log(f"[afr] worker {k}: got {ev.get('ev')} (rc {ev.get('rc')}, pass "
                        f"{ev.get('p')}) while waiting for '{want}'"
                        + ("" if p is None else f" pass {p}"))
                    return None
                got[k] = ev
            if len(got) == len(who):
                return got
            try:
                k, ev = q.get(timeout=timeout_s)
            except queue.Empty:
                log(f"[afr] timed out waiting for '{want}' from {sorted(set(who) - set(got))}")
                return None
            pending[k].append(ev)

    def stop_all():
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()

    def fail(rc=8):
        stop_all()
        codes = [p.returncode for p in procs]
        info["worker_rc"] = [info["worker_rc"][i] if info["worker_rc"][i] is not None
                             else (codes[i] if i < len(codes) else None) for i in range(n)]
        bad = [c for c in info["worker_rc"] if c not in (None, 0) and c > 0]
        return (bad[0] if bad else rc), info

    for k, (argv, env) in enumerate(cmds):
        log(f"[afr] starting worker {k}: TT_VISIBLE_DEVICES={env.get('TT_VISIBLE_DEVICES', '')} "
            f"GSPLAT_TT_DEVICE_ID={env.get('GSPLAT_TT_DEVICE_ID', '')}")
        p = popen(argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        procs.append(p)
        t = threading.Thread(target=_reader, args=(k, p.stdout, q, log), daemon=True)
        t.start()
        threads.append(t)
        # Stagger: the next worker starts once this one finished its warm-up (JIT compile).
        if get("ready", [k]) is None:
            return fail()
        log(f"[afr] worker {k} ready")

    def send_all(line):
        try:
            for p in procs:
                p.stdin.write(line + "\n")
                p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            log(f"[afr] worker stdin closed before {line}: {e}")
            return False
        return True

    first = None
    for p_i in range(n_passes):
        t_go = now()
        if not send_all(f"GO {p_i}"):
            return fail()
        if lock:
            info["view_ms"].append([])
            for v in range(n_views):
                if not send_all(f"V {p_i} {v}"):
                    return fail()
                gv = get("view", list(range(n)), p_i, v)
                if gv is None:
                    return fail()
                info["view_ms"][-1].append(
                    period_ms([(gv[k]["t0"], gv[k]["t1"]) for k in range(n)], 1))
        got = get("pass", list(range(n)), p_i)
        if got is None:
            return fail()
        t_back = now()
        stamps = [(got[k]["t0"], got[k]["t1"]) for k in range(n)]
        info["periods"].append(period_ms(stamps, n_views))
        for k in range(n):
            n_own = n_views if windowed else len(split(n_views, n)[k])
            info["chip_ms"][k].append((stamps[k][1] - stamps[k][0]) * 1000.0 / max(1, n_own))
        if p_i == 0:
            first = [got[k]["digests"] for k in range(n)]
            if windowed:
                info["crops"] = [got[k].get("crops") for k in range(n)]
        log(f"[run] AFR pass {p_i}: {info['periods'][-1] * n_views:.1f} ms for {n_views} views "
            f"= {info['periods'][-1]:.3f} ms/frame (coordinator {1000.0 * (t_back - t_go) / n_views:.3f}; "
            + " ".join(f"chip{k} {info['chip_ms'][k][-1]:.3f}" for k in range(n))
            + " ms per own view)")
    for p in procs:
        p.stdin.close()
    done = get("done", list(range(n)))
    for p in procs:
        p.wait()
    for t in threads:
        t.join(timeout=5)
    if done is None:
        return fail()
    for k in range(n):
        info["worker_rc"][k] = done[k].get("rc", procs[k].returncode)
        info["hero_md5"][k] = done[k].get("hero_md5")
    if first is not None and all(d is not None for d in first) and not windowed:
        info["digests"] = merge(first, n_views)
    bad = [c for c in info["worker_rc"] if c != 0]
    if bad:
        return bad[0], info
    hm = [h for h in info["hero_md5"] if h]
    if not windowed and len(hm) == n and len(set(hm)) != 1:
        log(f"[afr] chips rendered different hero frames: {info['hero_md5']}")
        return 5, info
    return 0, info


def split_fields(info, n_warm, windows, copy_ms):
    """Task #624: the split_* fields of the B2B line (latency over the timed passes)."""
    out = (f"split_windows={';'.join(','.join(str(v) for v in w) for w in windows)} "
           f"split_copy_ms={statistics.mean(copy_ms) if copy_ms else 0.0:.3f} ")
    lat = [x for vm in info["view_ms"][1 + n_warm:] or info["view_ms"] for x in vm]
    if lat:
        out += (f"split_latency_ms={statistics.mean(lat):.3f} "
                f"split_latency_ms_median={statistics.median(lat):.3f} ")
    return out


def summary_line(info, scene, n_views, n_warm, drop, gap_ms, devs, out_dir, extra=""):
    """The B2B line run.py prints, from coordinate()'s info, plus afr_* fields."""
    periods = info["periods"]
    timed = periods[1 + n_warm:] or periods
    ms_frame = statistics.mean(timed)
    ms_median = statistics.median(timed)
    md5 = raw_md5(info["digests"]) if info["digests"] else "none"
    chip = [statistics.median(c[1 + n_warm:] or c) if c else 0.0 for c in info["chip_ms"]]
    hm = info["hero_md5"]
    hero_eq = ("yes" if len(set(hm)) == 1 else "NO") if all(hm) else "unknown"
    if info.get("split"):
        hero_eq = "split"
    return (f"B2B scene={scene} n_views={n_views} passes={len(timed)} warmup={n_warm} "
            f"drop={'yes' if drop else 'no'} "
            f"ms_frame={ms_frame:.3f} fps={1000.0 / ms_frame:.2f} "
            f"ms_frame_median={ms_median:.3f} "
            f"pass_ms_frame={','.join(f'{x:.3f}' for x in timed)} "
            f"check_pass_ms_frame={periods[0]:.3f} raw_md5={md5} "
            f"gap_ms={gap_ms:.2f} render_ms_frame={ms_frame:.3f} "
            f"identical_across_passes={'unchecked' if drop else ('NO' if 5 in info['worker_rc'] else 'yes')} "
            f"afr_chips={len(devs)} afr_devices={','.join(str(d) for d in devs)} "
            f"afr_chip_ms_own_view={','.join(f'{c:.3f}' for c in chip)} "
            f"afr_hero_equal={hero_eq} {extra}out={out_dir}"), ms_frame, ms_median
