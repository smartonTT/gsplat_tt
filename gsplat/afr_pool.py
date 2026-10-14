"""Live viewer alternate-frame rendering (AFR) on N chips (task #620), off by default.

GSPLAT_VIEWER_AFR_DEVICES=0,1 (PCIe ids) switches it on. The viewer process then opens no
device: render state in render/host is per process (render/afr.py, task #581), so it starts
one worker process per chip (TT_VISIBLE_DEVICES=D, GSPLAT_TT_DEVICE_ID=0 through
render/afr.py worker_env; the ETH setup of render/eth_default.py is done once in the viewer
and inherited). Workers start one after the other, each warming its renderer (JIT compile)
before the next starts, so their compiles do not collide in the shared kernel cache.

Frame N goes to chip N % len(devices). A request carries (frame id, w2c, K, W, H, slider
settings); the worker renders it with the same Pipeline(tt_clean) as opt/viewer/viewer_clean.py
and writes the 8-bit image into a multiprocessing.shared_memory slot of its own. The viewer
keeps up to GSPLAT_VIEWER_AFR_INFLIGHT frames in flight (default: one per chip) and waits
for the oldest first, so frames leave in order (gsplat/nerfview_viewer.py FastRenderer).

Same grid on every chip (task #661): the pipeline partitions work by core count, so a different
grid changes a few pixels' bits (docs/eth-dispatch-t397: 10x10 vs 11x10). A harvested board
(bh-50 board 1: 11x10) next to a full one (12x10) gave chip-1 frames 257 px off chip 0's. Once all
workers are ready, the pool restarts any worker whose stage grid is larger than the smallest
with GSPLAT_TT_GRID_X/Y capped to it, so every chip renders the same bits. The slowest chip
paces round-robin frames anyway. GSPLAT_VIEWER_AFR_SAME_GRID=0 turns this off.
GSPLAT_VIEWER_AFR_PIDFILE names a file the pool keeps the worker pids in, so opt/viewer/viewer.sh
stop can wait for workers that outlive the viewer (t615: one still held its chip 20 s later).
"""
from __future__ import annotations

import atexit
import importlib.util
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ENV_DEVICES = "GSPLAT_VIEWER_AFR_DEVICES"
ENV_INFLIGHT = "GSPLAT_VIEWER_AFR_INFLIGHT"
ENV_MAX_WH = "GSPLAT_VIEWER_AFR_MAX_WH"
ENV_SAME_GRID = "GSPLAT_VIEWER_AFR_SAME_GRID"
ENV_PIDFILE = "GSPLAT_VIEWER_AFR_PIDFILE"
# Pipeline render knobs a request carries (the viewer's sliders).
SETTINGS = ("cull_disabled", "contrib_floor", "transmittance_threshold", "min_opacity",
            "max_radius", "k_cap", "use_isoellipse")


def devices_from_env(env=os.environ) -> Optional[list[int]]:
    """PCIe ids from GSPLAT_VIEWER_AFR_DEVICES; None when unset (single-chip viewer).
    One id runs the pool path on one chip (for an A/B of its overhead)."""
    raw = env.get(ENV_DEVICES, "").strip()
    if not raw:
        return None
    try:
        devs = [int(x) for x in raw.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"{ENV_DEVICES}={raw!r}: want comma-separated PCIe ids")
    if not devs or len(set(devs)) != len(devs) or min(devs) < 0:
        raise ValueError(f"{ENV_DEVICES}={raw!r}: want distinct ids >= 0")
    return devs


def inflight_from_env(n_chips: int, env=os.environ) -> int:
    raw = env.get(ENV_INFLIGHT, "").strip()
    n = int(raw) if raw else n_chips
    if n < 1:
        raise ValueError(f"{ENV_INFLIGHT}={raw!r}: want >= 1")
    return n


def max_wh_from_env(env=os.environ) -> tuple[int, int]:
    """Largest render size a slot holds (default 2048x2048)."""
    raw = env.get(ENV_MAX_WH, "").strip() or "2048x2048"
    w, h = (int(x) for x in raw.lower().split("x"))
    return w, h


def same_grid_from_env(env=os.environ) -> bool:
    return env.get(ENV_SAME_GRID, "1").strip() != "0"


def common_grid(grids) -> Optional[tuple[int, int]]:
    """The grid every chip can run (smallest x, smallest y); None if any is unknown."""
    if not grids or any(g is None for g in grids):
        return None
    return min(int(g[0]) for g in grids), min(int(g[1]) for g in grids)


def _load_afr():
    spec = importlib.util.spec_from_file_location("render_afr", REPO / "render" / "afr.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worker_env(base, k: int, dev: int) -> dict:
    """Worker k's environment: render/afr.py's (one visible chip, GSPLAT_TT_DEVICE_ID=0),
    without the viewer's AFR switch."""
    env = _load_afr().worker_env(base, k, dev)
    env.pop(ENV_DEVICES, None)
    return env


@dataclass
class Ticket:
    fid: int
    chip: int
    slot: int
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    H: int
    W: int


def _to_u8(img) -> Optional[np.ndarray]:
    if img is None:
        return None
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img)


def _worker_main(k, dev, factory, cfg, env, req_q, resp_q, shm_names, slot_bytes, ppid):
    os.environ.clear()
    os.environ.update(env)
    shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
    try:
        render = factory(k, dev, cfg)
    except BaseException:
        resp_q.put({"kind": "error", "err": traceback.format_exc()})
        return
    grid = getattr(render, "stage_grid", None)
    resp_q.put({"kind": "ready", "pid": os.getpid(),
                "grid": tuple(int(v) for v in grid) if grid is not None else None})
    try:
        while True:
            try:
                req = req_q.get(timeout=1.0)
            except queue.Empty:
                if os.getppid() != ppid:  # the viewer died (watchdog exit): do not linger
                    break
                continue
            if req is None:
                break
            out = {"kind": "frame", "fid": req["fid"]}
            try:
                r = render(req["w2c"], req["K"], req["H"], req["W"], req["settings"])
                img = _to_u8(r.get("image"))
                if img is not None:
                    if img.nbytes > slot_bytes:
                        raise ValueError(f"{img.shape} frame is larger than a slot "
                                         f"({slot_bytes} B, {ENV_MAX_WH})")
                    np.ndarray(img.shape, np.uint8, buffer=shms[req["slot"]].buf)[...] = img
                out.update(ok=True, shape=None if img is None else img.shape,
                           timings=dict(r.get("timings", {})),
                           sub_timings=dict(r.get("sub_timings", {})),
                           num_visible=int(r.get("num_visible", 0)), meta=r.get("meta"))
            except Exception:
                out.update(ok=False, err=traceback.format_exc())
            resp_q.put(out)
    finally:
        for s in shms:
            s.close()


class AFRPool:
    """One render worker per chip; frames go round robin, results come back in order."""

    def __init__(self, devices, factory: Callable, cfg: Optional[dict] = None, *,
                 inflight: Optional[int] = None, max_wh: tuple[int, int] = (2048, 2048),
                 base_env=None, start_method: str = "spawn", ready_timeout_s: float = 1800.0,
                 log: Callable[[str], None] = lambda m: print(m, flush=True)):
        self.devices = list(devices)
        n = len(self.devices)
        self.inflight = int(inflight or n)
        # Slots per chip: one per frame a client keeps there, +1 for a second client.
        self.slots_per_chip = math.ceil(self.inflight / n) + 1
        self.slot_bytes = int(max_wh[0]) * int(max_wh[1]) * 3
        self._log = log
        ctx = mp.get_context(start_method)
        self._cv = threading.Condition()
        self._fid = 0
        self._done: dict[int, dict] = {}
        self._free = [deque(range(self.slots_per_chip)) for _ in range(n)]
        self._state = ["starting"] * n
        self._err: list[Optional[str]] = [None] * n
        self._closed = False
        self._shm = [[shared_memory.SharedMemory(create=True, size=self.slot_bytes)
                      for _ in range(self.slots_per_chip)] for _ in range(n)]
        self._req = [ctx.Queue() for _ in range(n)]
        self._resp = [ctx.Queue() for _ in range(n)]
        self._procs: list = []
        self._readers: list[threading.Thread] = []
        atexit.register(self.close)
        self._ctx, self._factory, self._cfg = ctx, factory, cfg or {}
        self._ready_timeout_s = ready_timeout_s
        self._grid: list[Optional[tuple[int, int]]] = [None] * n
        base = dict(os.environ if base_env is None else base_env)
        self._pidfile = base.get(ENV_PIDFILE, "").strip() or None
        try:
            envs = [worker_env(base, k, dev) for k, dev in enumerate(self.devices)]
            for k in range(n):
                self._start_worker(k, envs[k])
            target = common_grid(self._grid) if same_grid_from_env(base) else None
            if n > 1 and target is not None and len(set(self._grid)) > 1:
                log(f"[afr] stage grids {self._grid} differ: capping every chip to "
                    f"{target[0]}x{target[1]} so all render the same bits")
                for k in range(n):
                    if self._grid[k] != target:
                        envs[k] = dict(envs[k], GSPLAT_TT_GRID_X=str(target[0]),
                                       GSPLAT_TT_GRID_Y=str(target[1]))
                        self._stop_worker(k)
                        self._start_worker(k, envs[k])
                        if self._grid[k] != target:
                            raise RuntimeError(f"AFR worker {k} (chip {self.devices[k]}) grid "
                                               f"{self._grid[k]} after capping to {target}")
            elif n > 1 and target is None and same_grid_from_env(base):
                log("[afr] stage grid unknown on some chip: frames may differ in a few bits "
                    "between chips")
        except BaseException:
            self.close()
            raise
        log(f"[afr] {n} chip(s) {self.devices}, grids {self._grid}, "
            f"{self.inflight} frame(s) in flight")

    def _start_worker(self, k: int, env: dict) -> None:
        dev = self.devices[k]
        p = self._ctx.Process(
            target=_worker_main, name=f"gsplat-afr{k}",
            args=(k, dev, self._factory, self._cfg, env, self._req[k], self._resp[k],
                  [s.name for s in self._shm[k]], self.slot_bytes, os.getpid()))
        t0 = time.monotonic()
        with self._cv:
            self._state[k], self._err[k], self._grid[k] = "starting", None, None
        with _environ(env):  # the child's exec environment from its first line
            p.start()
        if k < len(self._procs):
            self._procs[k] = p
        else:
            self._procs.append(p)
        self._write_pidfile()
        r = threading.Thread(target=self._read, args=(k, p), name=f"gsplat-afr{k}-rx",
                             daemon=True)
        r.start()
        if k < len(self._readers):
            self._readers[k] = r
        else:
            self._readers.append(r)
        with self._cv:  # one after the other: the JIT compiles must not collide
            if not self._cv.wait_for(lambda: self._state[k] != "starting",
                                     self._ready_timeout_s):
                raise RuntimeError(f"AFR worker {k} (chip {dev}) not ready after "
                                   f"{self._ready_timeout_s:g} s")
            if self._state[k] != "ready":
                raise RuntimeError(f"AFR worker {k} (chip {dev}) failed to start:\n"
                                   f"{self._err[k]}")
        g = self._grid[k]
        self._log(f"[afr] worker {k} chip {dev} pid {p.pid} ready in "
                  f"{time.monotonic() - t0:.1f} s, stage grid "
                  f"{f'{g[0]}x{g[1]}' if g else 'unknown'}")

    def _stop_worker(self, k: int) -> None:
        """Stop worker k (before a restart) and its reader."""
        p = self._procs[k]
        self._req[k].put(None)
        p.join(60.0)
        if p.is_alive():
            p.terminate()
            p.join(10.0)
        self._readers[k].join(5.0)

    def _write_pidfile(self) -> None:
        if not self._pidfile:
            return
        try:
            tmp = f"{self._pidfile}.tmp"
            with open(tmp, "w") as f:
                f.write("".join(f"{p.pid}\n" for p in self._procs if p.pid))
            os.replace(tmp, self._pidfile)
        except OSError as e:
            self._log(f"[afr] cannot write {self._pidfile}: {e}")

    def _read(self, k: int, proc) -> None:
        while not self._closed:
            try:
                msg = self._resp[k].get(timeout=0.5)
            except queue.Empty:
                if not proc.is_alive():
                    with self._cv:
                        if self._procs[k] is not proc:  # restarted (same-grid cap)
                            return
                        self._state[k] = "dead"
                        self._err[k] = self._err[k] or f"exit code {proc.exitcode}"
                        self._cv.notify_all()
                    return
                continue
            except (EOFError, OSError):
                return
            with self._cv:
                if msg["kind"] == "ready":
                    self._state[k] = "ready"
                    self._grid[k] = msg.get("grid")
                elif msg["kind"] == "error":
                    self._state[k], self._err[k] = "dead", msg["err"]
                else:
                    self._done[msg["fid"]] = msg
                self._cv.notify_all()

    def submit(self, extrinsics, intrinsics, H: int, W: int, settings: dict,
               timeout: Optional[float] = None) -> Ticket:
        """Queue one frame on the next chip in turn and return its ticket."""
        if W * H * 3 > self.slot_bytes:
            raise ValueError(f"{W}x{H} is larger than an AFR slot; raise {ENV_MAX_WH}")
        w2c = np.array(_np(extrinsics), dtype=np.float32)
        K = np.array(_np(intrinsics), dtype=np.float32)
        with self._cv:
            chip = self._fid % len(self.devices)
            if not self._cv.wait_for(lambda: self._free[chip] or self._state[chip] != "ready",
                                     timeout):
                raise TimeoutError(f"AFR chip {chip}: no free slot")
            self._check(chip)
            t = Ticket(self._fid, chip, self._free[chip].popleft(), w2c, K, int(H), int(W))
            self._fid += 1
            self._req[chip].put({"fid": t.fid, "slot": t.slot, "w2c": w2c, "K": K,
                                 "H": t.H, "W": t.W, "settings": dict(settings)})
        return t

    def wait(self, ticket: Ticket, timeout: Optional[float] = None):
        """The ticket's RenderResult (image copied out of its slot, which is then free)."""
        from gsplat.pipeline import RenderResult
        with self._cv:
            ok = self._cv.wait_for(lambda: ticket.fid in self._done
                                   or self._state[ticket.chip] != "ready", timeout)
            msg = self._done.pop(ticket.fid, None)
            if msg is None:
                self._check(ticket.chip)
                raise TimeoutError(f"AFR frame {ticket.fid} (chip {ticket.chip}) not back "
                                   f"after {timeout} s" if not ok else "AFR frame lost")
            img = None
            if msg.get("ok") and msg["shape"] is not None:
                buf = self._shm[ticket.chip][ticket.slot].buf
                img = np.ndarray(msg["shape"], np.uint8, buffer=buf).copy()
            self._free[ticket.chip].append(ticket.slot)
            self._cv.notify_all()
        if not msg.get("ok"):
            raise RuntimeError(f"AFR chip {ticket.chip} frame {ticket.fid} failed:\n{msg['err']}")
        res = RenderResult(image=img, timings=msg["timings"], sub_timings=msg["sub_timings"],
                           num_visible=msg["num_visible"], height=ticket.H, width=ticket.W)
        res.chip, res.fid, res.meta = ticket.chip, ticket.fid, msg.get("meta")
        return res

    def _check(self, chip: int) -> None:
        if self._state[chip] != "ready":
            raise RuntimeError(f"AFR worker {chip} (chip {self.devices[chip]}) is "
                               f"{self._state[chip]}: {self._err[chip]}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for q, p in zip(self._req, self._procs):
            try:
                q.put(None)
            except (OSError, ValueError):
                pass
        for p in self._procs:
            p.join(10.0)
            if p.is_alive():
                p.terminate()
                p.join(5.0)
        if self._pidfile and not any(p.is_alive() for p in self._procs):
            try:
                os.remove(self._pidfile)
            except OSError:
                pass
        for chip in self._shm:
            for s in chip:
                try:
                    s.close()
                    s.unlink()
                except (OSError, FileNotFoundError):
                    pass
        with self._cv:
            self._cv.notify_all()


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class _environ:
    """Swap os.environ for the duration of a block (process start only)."""

    def __init__(self, env: dict):
        self.env = env

    def __enter__(self):
        self.saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.env)

    def __exit__(self, *exc):
        os.environ.clear()
        os.environ.update(self.saved)


class AFRPipeline:
    """The viewer's ``pipeline`` in AFR mode: Pipeline's render knobs, sent with every
    request, and render/submit/wait on the pool. ``backend`` holds the knobs under the
    backend's names, for the render watchdog's pose log."""

    def __init__(self, pool: AFRPool, *, cull_disabled: bool = False,
                 contrib_floor: Optional[float] = None,
                 transmittance_threshold: float = 1.0 / 255.0, min_opacity: float = 1.0 / 255.0,
                 max_radius: int = 0, k_cap: float = 3.0, use_isoellipse: bool = False):
        from types import SimpleNamespace
        from gsplat.pipeline import DEFAULT_CONTRIB_FLOOR
        self.pool = pool
        self.inflight = pool.inflight
        self.backend = SimpleNamespace(
            cull_disabled=None, transmittance_threshold=None, min_opacity=None,
            max_radius=None, contrib_floor_override=None, _mb_contrib_floor=None, k_cap=None,
            use_isoellipse=None)
        self.cull_disabled = cull_disabled
        self.contrib_floor = DEFAULT_CONTRIB_FLOOR if contrib_floor is None else contrib_floor
        self.transmittance_threshold = transmittance_threshold
        self.min_opacity = min_opacity
        self.max_radius = max_radius
        self.k_cap = k_cap
        self.use_isoellipse = use_isoellipse
        self._sync_render_settings_to_backend()

    def _sync_render_settings_to_backend(self) -> None:
        from gsplat.pipeline import Pipeline
        Pipeline._sync_render_settings_to_backend(self)

    def settings(self) -> dict:
        return {k: getattr(self, k) for k in SETTINGS}

    def submit(self, extrinsics, intrinsics, image_height: int, image_width: int) -> Ticket:
        return self.pool.submit(extrinsics, intrinsics, image_height, image_width,
                                self.settings())

    def wait(self, ticket: Ticket):
        return self.pool.wait(ticket)

    def render(self, gaussians, extrinsics, intrinsics, image_height: int, image_width: int):
        """Synchronous render on the next chip (selftest); the gaussians are the workers'."""
        return self.wait(self.submit(extrinsics, intrinsics, image_height, image_width))

    def close(self) -> None:
        self.pool.close()


def apply_settings(pipe, settings: dict) -> None:
    """Worker side: copy changed knobs into the Pipeline and push them to its backend."""
    changed = False
    for k, v in settings.items():
        if getattr(pipe, k, None) != v:
            setattr(pipe, k, v)
            changed = True
    if changed:
        pipe._sync_render_settings_to_backend()


def tt_clean_factory(k: int, dev: int, cfg: dict):
    """Worker k's renderer: Pipeline(tt_clean) as in opt/viewer/viewer_clean.py + gsplat/viewer.py,
    the scene from cfg["ply_path"], warmed on cfg["warm"] = (w2c, K, H, W)."""
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    spec = importlib.util.spec_from_file_location("render_run", REPO / "render" / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    import backends
    import torch
    from gsplat.loading_gaussians import load_ply
    from gsplat.pipeline import Pipeline
    backends.REGISTRY["tt_clean"] = run.CleanBackend
    gauss = load_ply(cfg["ply_path"])
    pipe = Pipeline(backends.get_backend("tt_clean", verbose=bool(cfg.get("verbose")), k_cap=3.0),
                    tile_size=int(cfg.get("tile_size", 32)), k_cap=3.0)
    apply_settings(pipe, cfg.get("settings", {}))

    def render(w2c, K, H, W, settings):
        apply_settings(pipe, settings)
        r = pipe.render(gauss, torch.from_numpy(np.asarray(w2c, dtype=np.float32)),
                        torch.from_numpy(np.asarray(K, dtype=np.float32)), int(H), int(W))
        return {"image": r.image, "timings": r.timings, "sub_timings": r.sub_timings,
                "num_visible": r.num_visible}

    # The capped stage grid every stage partitions by (render/host/device_state.cpp cap_grid);
    # the pool compares it across chips (task #661). Trees without the binding report none.
    clean = getattr(pipe.backend, "_clean", None)
    if clean is not None and hasattr(clean, "stage_grid"):
        render.stage_grid = tuple(int(v) for v in clean.stage_grid())
    if cfg.get("warm") is not None:
        w2c, K, H, W = cfg["warm"]
        for _ in range(int(cfg.get("warm_n", 2))):
            render(w2c, K, H, W, cfg.get("settings", {}))
    return render


def fake_factory(k: int, dev: int, cfg: dict):
    """Test stand-in for tt_clean_factory (no device): pixel (0, 0) holds the chip index, the
    device id and the w2c x translation; ``meta`` echoes the settings and environment."""
    delay = cfg.get("delay_s", [0.0])
    if cfg.get("fail_start") == k:
        raise RuntimeError(f"fake worker {k} fails to start")
    grids = cfg.get("grids")  # per-chip compute grid; capped like device_state.cpp cap_grid
    stage_grid = None
    if grids is not None:
        gx, gy = grids[k % len(grids)]
        cx, cy = (int(os.environ.get(v, "0") or 0) for v in ("GSPLAT_TT_GRID_X", "GSPLAT_TT_GRID_Y"))
        stage_grid = (cx if 0 < cx < gx else gx, cy if 0 < cy < gy else gy)

    def render(w2c, K, H, W, settings):
        time.sleep(delay[k % len(delay)])
        img = np.zeros((H, W, 3), np.uint8)
        img[0, 0] = (k, dev, int(round(float(w2c[0][3]))) % 256)
        return {"image": img, "timings": {"total": 1.0 + k}, "num_visible": 7,
                "meta": {"settings": dict(settings), "pid": os.getpid(),
                         "TT_VISIBLE_DEVICES": os.environ.get("TT_VISIBLE_DEVICES"),
                         "GSPLAT_TT_DEVICE_ID": os.environ.get("GSPLAT_TT_DEVICE_ID"),
                         ENV_DEVICES: os.environ.get(ENV_DEVICES),
                         "stage_grid": stage_grid}}
    if stage_grid is not None:
        render.stage_grid = stage_grid
    return render


def pipelined_intervals_ms(pipeline, extrinsics, intrinsics, H: int, W: int, n: int) -> list:
    """Selftest: n frames with ``pipeline.inflight`` in flight, as FastRenderer runs them;
    ms between consecutive deliveries (the viewer's frame interval)."""
    pending, out, last = deque(), [], None
    for i in range(n + pipeline.inflight - 1):
        if i < n:
            pending.append(pipeline.submit(extrinsics, intrinsics, H, W))
        if len(pending) >= pipeline.inflight or i >= n:
            pipeline.wait(pending.popleft())
            t = time.perf_counter()
            if last is not None:
                out.append((t - last) * 1000.0)
            last = t
    while pending:
        pipeline.wait(pending.popleft())
    return out
