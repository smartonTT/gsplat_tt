"""Render watchdog, pose ring buffer and viewer supervisor (task #357).

Needs only numpy (and bash for the supervisor test). Runs under pytest or
directly:  python3 tests/unit/test_render_watchdog.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gsplat.render_watchdog import EXIT_CODE, RenderWatchdog  # noqa: E402


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class _Exit(Exception):
    pass


def _exit(code):
    raise _Exit(code)


def _backend():
    return SimpleNamespace(min_opacity=0.004, cull_disabled=False, transmittance_threshold=1e-4,
                           max_radius=-1, k_cap=3.0, use_isoellipse=False,
                           contrib_floor_override=1 / 255, _mb_contrib_floor=1 / 255)


def _pose(i):
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, 3] = [0.1 * i, -0.2, 3.0]
    K = np.array([[1097.6, 0, 512], [0, 1097.6, 512], [0, 0, 1]], dtype=np.float32)
    return w2c, K


def _wd(tmp, **kw):
    clock = _Clock()
    wd = RenderWatchdog(_backend(), tmp / "viewer_hang.log", tmp / "viewer_poses.jsonl",
                        limit_s=5.0, first_limit_s=60.0, ring=4, exit_fn=_exit, clock=clock, **kw)
    return wd, clock


def test_ring_keeps_last_poses_and_flushes():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        wd, clock = _wd(tmp)
        render = wd.wrap(lambda g, e, k, h, w: ("img", h, w))
        for i in range(6):
            w2c, K = _pose(i)
            clock.t += 1.0
            assert render(None, w2c, K, 1024, 1024) == ("img", 1024, 1024)
        lines = wd.flush_ring("exit").read_text().splitlines()
        head = json.loads(lines[0])
        assert head["flush"] == "exit" and head["n"] == 4
        recs = [json.loads(x) for x in lines[1:]]
        assert [r["seq"] for r in recs] == [3, 4, 5, 6]
        assert recs[-1]["w2c"][0][3] == np.float32(0.5).item()
        assert np.allclose(np.array(recs[-1]["c2w"]) @ np.array(recs[-1]["w2c"]), np.eye(4))
        assert recs[-1]["settings"]["contrib_floor_override"] == 1 / 255
        assert all(r["ms"] is not None for r in recs)


def test_fires_on_stuck_render_with_pose_and_stacks():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        wd, clock = _wd(tmp)
        w2c, K = _pose(1)
        rec = wd.begin(w2c, K, 1024, 1024)       # first render at this size: 60 s limit
        clock.t += 30.0
        assert wd.check() is False
        wd.end(rec)
        w2c, K = _pose(7)
        wd.begin(w2c, K, 1024, 1024)              # second: 5 s limit
        clock.t += 4.9
        assert wd.check() is False
        clock.t += 0.2
        try:
            wd.check()
            raise AssertionError("watchdog did not exit")
        except _Exit as e:
            assert e.args[0] == EXIT_CODE
        log = (tmp / "viewer_hang.log").read_text()
        assert "=== WATCHDOG" in log and "seq=2" in log
        pose = json.loads(next(x for x in log.splitlines() if x.startswith("POSE "))[5:])
        assert pose["w2c"][0][3] == np.float32(0.7).item() and pose["ms"] is None
        assert pose["width"] == 1024 and pose["K"][0][0] == np.float32(1097.6).item()
        assert "THREADS" in log and "test_fires_on_stuck_render" in log
        assert "REPLAY TTW_ALLOW_DIRECT=1" in log and "replay_pose.py" in log
        pj = Path(next(x for x in log.splitlines() if x.startswith("POSE_JSON "))[10:])
        assert json.loads(pj.read_text())["w2c"] == pose["w2c"]
        head = json.loads((tmp / "viewer_poses.jsonl").read_text().splitlines()[0])
        assert head["flush"] == "watchdog" and head["n"] == 2


def test_stall_hook_consumes_file_and_watch_thread_fires():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        fired = threading.Event()
        stall = tmp / "stall"
        wd = RenderWatchdog(_backend(), tmp / "viewer_hang.log", tmp / "viewer_poses.jsonl",
                            limit_s=0.3, first_limit_s=0.3, poll_s=0.05,
                            stall_file=str(stall), exit_fn=lambda c: fired.set())
        render = wd.wrap(lambda g, e, k, h, w: "img")
        w2c, K = _pose(2)
        assert render(None, w2c, K, 64, 64) == "img"   # no stall file: no stall
        stall.write_text("1.0")
        wd.start()
        t = time.monotonic()
        render(None, w2c, K, 64, 64)
        assert not stall.exists()
        assert fired.wait(2.0) and time.monotonic() - t < 2.0
        assert "gsplat-render-watchdog" in (tmp / "viewer_hang.log").read_text()


def test_supervisor_restarts_on_failure_and_stops_on_clean_exit():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        vdir, work = tmp / "v", tmp / "tree"
        vdir.mkdir()
        work.mkdir()
        fake = tmp / "fake_python"
        # Exits 86 (watchdog) twice, then 0 (clean stop).
        fake.write_text(f"#!/bin/bash\nn=$(cat {tmp}/n 2>/dev/null || echo 0)\n"
                        f"echo $((n+1)) > {tmp}/n\n[ $n -lt 2 ] && exit 86\nexit 0\n")
        fake.chmod(0o755)
        env = dict(os.environ, VIEWER_PYTHON=str(fake), VIEWER_BACKOFF_S="0",
                   VIEWER_MAX_RESTARTS_PER_HOUR="6")
        r = subprocess.run(["bash", str(ROOT / "opt/viewer/supervise.sh"), str(vdir), "8080"],
                           cwd=work, env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (tmp / "n").read_text().strip() == "3"
        slog = (vdir / "viewer_supervisor.log").read_text()
        assert slog.count("rc=86") == 2 and "clean exit" in slog
        assert not (vdir / "viewer.sup.pid").exists()


def test_supervisor_gives_up_at_the_hourly_cap():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        vdir = tmp / "v"
        vdir.mkdir()
        fake = tmp / "fake_python"
        fake.write_text("#!/bin/bash\nexit 86\n")
        fake.chmod(0o755)
        env = dict(os.environ, VIEWER_PYTHON=str(fake), VIEWER_BACKOFF_S="0",
                   VIEWER_MAX_RESTARTS_PER_HOUR="2")
        r = subprocess.run(["bash", str(ROOT / "opt/viewer/supervise.sh"), str(vdir), "8080"],
                           cwd=tmp, env=env, capture_output=True, text=True, timeout=30)
        assert r.returncode == 1
        slog = (vdir / "viewer_supervisor.log").read_text()
        assert slog.count("starting viewer_clean.py") == 3 and "giving up" in slog


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
