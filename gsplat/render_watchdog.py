"""Render watchdog and per-render pose log for the live viewer (task #357).

On 2026-10-07 one render_view call on bh-30 spun for 89 minutes; every other
renderer waited behind it and no client got frames until the viewer was
restarted by hand. `RenderWatchdog.wrap(pipeline.render)` guards each render:

- Every render's camera (w2c and K exactly as passed to the backend, the
  resolution and the backend's cull settings) goes into a ring buffer. It is
  written to `pose_log` when the watchdog fires and when the viewer exits.
- A daemon thread checks the render in flight. One that runs past the limit
  (default 5 s; a resolution's first render gets `first_limit_s`, which covers
  the JIT compile) is logged to `hang_log` with its full pose (w2c, c2w, K, W,
  H, settings), a pose JSON for opt/viewer/replay_pose.py, the ring buffer and
  every thread's stack. Then the process exits with EXIT_CODE so the supervisor
  in opt/viewer/supervise.sh restarts it.
- Debug hook: with `stall_file` set, a render that finds that file deletes it
  and sleeps for the number of seconds it holds (default 3600) inside the
  guarded region, which is how the watchdog is tested on the live viewer.

render_view releases the GIL (the viewer's heartbeat kept logging during the
89-minute hang), so a Python thread can watch it.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

EXIT_CODE = 86
# Backend attributes that change what render_view does (viewer sliders).
SETTINGS = ("min_opacity", "cull_disabled", "transmittance_threshold", "max_radius",
            "k_cap", "use_isoellipse", "contrib_floor_override", "_mb_contrib_floor")


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _np(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.array(x, dtype=np.float32, copy=True)


def _plain(v: Any) -> Any:
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def pose_record(rec: dict) -> dict:
    """JSON-ready pose of one render (w2c/K as float32 values, c2w in float64)."""
    w2c = rec["w2c"]
    return {
        "seq": rec["seq"], "utc": rec["utc"], "ms": rec["ms"],
        "width": rec["W"], "height": rec["H"],
        "w2c": w2c.tolist(),
        "c2w": np.linalg.inv(w2c.astype(np.float64)).tolist(),
        "K": rec["K"].tolist(),
        "settings": rec["settings"],
    }


class RenderWatchdog:
    def __init__(self, backend: Any, hang_log: Path, pose_log: Path, limit_s: float = 5.0,
                 first_limit_s: float = 600.0, ring: int = 256, poll_s: float | None = None,
                 stall_file: str | None = None, pyspy: str | None = None,
                 replay_cmd: str = "opt/viewer/replay_pose.py",
                 exit_fn: Callable[[int], Any] = os._exit,
                 clock: Callable[[], float] = time.monotonic):
        self.backend = backend
        self.hang_log = Path(hang_log)
        self.pose_log = Path(pose_log)
        self.limit_s = float(limit_s)
        self.first_limit_s = float(first_limit_s)
        self.poll_s = poll_s if poll_s is not None else min(0.5, max(0.05, self.limit_s / 10.0))
        self.stall_file = stall_file
        self.pyspy = pyspy
        self.replay_cmd = replay_cmd
        self._exit = exit_fn
        self._clock = clock
        self._ring: deque[dict] = deque(maxlen=ring)
        self._sizes: set[tuple[int, int]] = set()
        self._active: dict | None = None
        self._lock = threading.Lock()
        self._seq = 0
        self.fired = False
        self._thread: threading.Thread | None = None

    # -- per render ---------------------------------------------------------
    def begin(self, extrinsics: Any, intrinsics: Any, H: int, W: int) -> dict:
        settings = {k: _plain(getattr(self.backend, k, None)) for k in SETTINGS}
        with self._lock:
            self._seq += 1
            first = (int(W), int(H)) not in self._sizes
            rec = {"seq": self._seq, "utc": _utc(), "t0": self._clock(), "ms": None,
                   "W": int(W), "H": int(H), "w2c": _np(extrinsics), "K": _np(intrinsics),
                   "settings": settings,
                   "limit": self.first_limit_s if first else self.limit_s}
            self._ring.append(rec)
            self._active = rec
        return rec

    def end(self, rec: dict) -> None:
        with self._lock:
            rec["ms"] = round((self._clock() - rec["t0"]) * 1000.0, 3)
            self._sizes.add((rec["W"], rec["H"]))
            if self._active is rec:
                self._active = None

    def wrap(self, render: Callable) -> Callable:
        def guarded(gaussians, extrinsics, intrinsics, image_height, image_width, *a, **kw):
            rec = self.begin(extrinsics, intrinsics, image_height, image_width)
            try:
                self._maybe_stall()
                return render(gaussians, extrinsics, intrinsics, image_height, image_width,
                              *a, **kw)
            finally:
                self.end(rec)
        guarded.__wrapped__ = render
        return guarded

    def _maybe_stall(self) -> None:
        if not self.stall_file or not os.path.exists(self.stall_file):
            return
        try:
            s = float(Path(self.stall_file).read_text().strip() or 3600)
        except (OSError, ValueError):
            s = 3600.0
        try:
            os.unlink(self.stall_file)
        except OSError:
            pass
        print(f"[watchdog {_utc()}] STALL hook: render sleeps {s:.0f} s", flush=True)
        time.sleep(s)

    # -- watching -----------------------------------------------------------
    def start(self) -> "RenderWatchdog":
        if self.limit_s > 0 and self._thread is None:
            self._thread = threading.Thread(target=self._watch, name="gsplat-render-watchdog",
                                            daemon=True)
            self._thread.start()
        return self

    def check(self) -> bool:
        """Fire if the render in flight is over its limit. True when it fired."""
        with self._lock:
            rec = self._active
            over = rec is not None and self._clock() - rec["t0"] > rec["limit"]
        if over and not self.fired:
            self.fire(rec)
            return True
        return False

    def _watch(self) -> None:
        while not self.fired:
            time.sleep(self.poll_s)
            self.check()

    def ring_records(self) -> list[dict]:
        with self._lock:
            return [pose_record(r) for r in list(self._ring)]

    def flush_ring(self, reason: str) -> Path:
        recs = self.ring_records()
        tmp = self.pose_log.with_suffix(self.pose_log.suffix + ".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps({"flush": reason, "utc": _utc(), "pid": os.getpid(),
                                "n": len(recs)}) + "\n")
            for r in recs:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, self.pose_log)
        return self.pose_log

    def fire(self, rec: dict) -> None:
        self.fired = True
        stuck = self._clock() - rec["t0"]
        pose = pose_record(rec)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pose_json = self.hang_log.parent / f"viewer_hang_pose_{stamp}.json"
        try:
            pose_json.write_text(json.dumps(pose, indent=1) + "\n")
        except OSError as e:
            print(f"[watchdog] cannot write {pose_json}: {e}", file=sys.stderr, flush=True)
        replay = (f"TTW_ALLOW_DIRECT=1 .venv/bin/python {self.replay_cmd} "
                  f"{shlex.quote(str(pose_json))} --timeout 30")
        lines = [
            f"=== WATCHDOG {_utc()} pid={os.getpid()} render seq={rec['seq']} started "
            f"{rec['utc']} stuck {stuck:.1f} s (limit {rec['limit']:.1f} s) ===",
            f"POSE {json.dumps(pose)}",
            f"POSE_JSON {pose_json}",
            f"REPLAY {replay}",
            "RING (oldest first; ms=null is the render in flight)",
        ]
        lines += [f"  {json.dumps(r)}" for r in self.ring_records()[-32:]]
        lines.append("THREADS")
        names = {t.ident: t.name for t in threading.enumerate()}
        for tid, frame in sys._current_frames().items():
            lines.append(f"--- thread {names.get(tid, '?')} ({tid:#x})")
            lines += [s.rstrip("\n") for s in traceback.format_stack(frame)]
        if self.pyspy:
            lines.append(f"NATIVE ({self.pyspy} dump --native)")
            try:
                out = subprocess.run([self.pyspy, "dump", "--native", "--pid", str(os.getpid())],
                                     capture_output=True, text=True, timeout=60)
                lines += (out.stdout + out.stderr).splitlines()
            except (OSError, subprocess.SubprocessError) as e:
                lines.append(f"py-spy failed: {e}")
        try:
            with open(self.hang_log, "a") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.flush_ring("watchdog")
        finally:
            print(f"[watchdog {_utc()}] render seq={rec['seq']} {rec['W']}x{rec['H']} stuck "
                  f"{stuck:.1f} s: pose and stacks in {self.hang_log}, pose json {pose_json}; "
                  f"exiting {EXIT_CODE}", file=sys.stderr, flush=True)
            self._exit(EXIT_CODE)
