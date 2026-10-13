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


def worker_env(base, k, dev, visible=None):
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
    return env


def worker_argv(argv, k, n):
    """run.py arguments for worker k: --afr-worker k/n; workers past the first get --no-ref
    and their own --iter-dir <dir>/afr<k> (their hero_clean.png is their warm-up hero)."""
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

    def __init__(self, k, n, inp=None, out=None):
        self.k, self.n = k, n
        self.inp = inp if inp is not None else sys.stdin
        self.out = out if out is not None else sys.stdout
        self.t0 = self.t1 = 0.0

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

    def mark_end(self):
        self.t1 = now()

    def after_pass(self, p, digests=None):
        self.send(ev="pass", p=p, t0=self.t0, t1=self.t1 or now(), digests=digests)
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


def coordinate(cmds, n_passes, n_views, log=print, popen=subprocess.Popen, timeout_s=None):
    """Run the workers. cmds = [(argv, env)] per chip. Returns (rc, info): rc 0 when every
    worker ended with 0, else the first non-zero one (8 for a protocol error or a worker that
    ended early, 5 when the chips' hero frames differ); info has the merged pass-0 digests,
    the per-pass periods (ms/view over all chips) and the per-chip ms per own view."""
    n = len(cmds)
    q = queue.Queue()
    procs, threads = [], []
    info = {"periods": [], "chip_ms": [[] for _ in range(n)], "digests": None,
            "hero_md5": [None] * n, "worker_rc": [None] * n}

    pending = [[] for _ in range(n)]  # events read but not yet taken, per worker

    def get(want, who, p=None):
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
                if ev.get("ev") != want or (p is not None and ev.get("p") != p):
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

    first = None
    for p_i in range(n_passes):
        t_go = now()
        try:
            for p in procs:
                p.stdin.write(f"GO {p_i}\n")
                p.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            log(f"[afr] worker stdin closed before GO {p_i}: {e}")
            return fail()
        got = get("pass", list(range(n)), p_i)
        if got is None:
            return fail()
        t_back = now()
        stamps = [(got[k]["t0"], got[k]["t1"]) for k in range(n)]
        info["periods"].append(period_ms(stamps, n_views))
        for k in range(n):
            n_own = len(split(n_views, n)[k])
            info["chip_ms"][k].append((stamps[k][1] - stamps[k][0]) * 1000.0 / max(1, n_own))
        if p_i == 0:
            first = [got[k]["digests"] for k in range(n)]
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
    if first is not None and all(d is not None for d in first):
        info["digests"] = merge(first, n_views)
    bad = [c for c in info["worker_rc"] if c != 0]
    if bad:
        return bad[0], info
    hm = [h for h in info["hero_md5"] if h]
    if len(hm) == n and len(set(hm)) != 1:
        log(f"[afr] chips rendered different hero frames: {info['hero_md5']}")
        return 5, info
    return 0, info


def summary_line(info, scene, n_views, n_warm, drop, gap_ms, devs, out_dir):
    """The B2B line run.py prints, from coordinate()'s info, plus afr_* fields."""
    periods = info["periods"]
    timed = periods[1 + n_warm:] or periods
    ms_frame = statistics.mean(timed)
    ms_median = statistics.median(timed)
    md5 = raw_md5(info["digests"]) if info["digests"] else "none"
    chip = [statistics.median(c[1 + n_warm:] or c) if c else 0.0 for c in info["chip_ms"]]
    hm = info["hero_md5"]
    hero_eq = ("yes" if len(set(hm)) == 1 else "NO") if all(hm) else "unknown"
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
            f"afr_hero_equal={hero_eq} out={out_dir}"), ms_frame, ms_median
