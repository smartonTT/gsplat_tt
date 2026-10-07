"""Viewer session bookkeeping: UTC log lines, heartbeat, pose memory (task #347).

The viewer log had no timestamps and no connect/disconnect lines (viser runs
with ``verbose=False``), so a dropped browser session left no trace. These
helpers are plain Python so they can be unit tested without a device.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import NamedTuple, Optional

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def log(msg: str) -> None:
    """One viewer log line with a UTC timestamp."""
    print(f"[viewer {utc_now()}] {msg}", flush=True)


class Heartbeat:
    """Rate-limited heartbeat: every ``active_s`` while frames or clients
    change, else every ``idle_s``. The first call always prints."""

    def __init__(self, active_s: float = 60.0, idle_s: float = 900.0) -> None:
        self.active_s = active_s
        self.idle_s = idle_s
        self._last_t: Optional[float] = None
        self._last: tuple[int, int, int] = (0, 0, 0)

    def line(self, rendered: int, sent: int, clients: int,
             now: Optional[float] = None) -> Optional[str]:
        now = time.monotonic() if now is None else now
        cur = (rendered, sent, clients)
        if self._last_t is not None:
            changed = cur != self._last
            if now - self._last_t < (self.active_s if changed else self.idle_s):
                return None
        d_r = rendered - self._last[0]
        d_s = sent - self._last[1]
        self._last_t, self._last = now, cur
        return (f"heartbeat frames_rendered={rendered} (+{d_r}) frames_sent={sent} (+{d_s}) "
                f"clients={clients}")


class Pose(NamedTuple):
    position: np.ndarray
    look_at: np.ndarray
    up_direction: np.ndarray
    fov: float
    t: float


class PoseMemory:
    """Last camera pose of a client that disconnected, so a page that
    reconnects by itself gets its view back instead of the preset."""

    def __init__(self, max_age_s: float = 1800.0) -> None:
        self.max_age_s = max_age_s
        self._pose: Optional[Pose] = None

    def remember(self, camera, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        pos = np.asarray(camera.position, dtype=np.float64)
        if not np.all(np.isfinite(pos)):
            return
        self._pose = Pose(pos.copy(), np.asarray(camera.look_at, dtype=np.float64).copy(),
                          np.asarray(camera.up_direction, dtype=np.float64).copy(),
                          float(camera.fov), now)

    def recall(self, other_clients: int, now: Optional[float] = None) -> Optional[Pose]:
        """The remembered pose if it is fresh and no other client is connected
        (otherwise the new client gets the normal preset view)."""
        now = time.monotonic() if now is None else now
        p = self._pose
        if p is None or other_clients > 0 or now - p.t > self.max_age_s:
            return None
        return p
