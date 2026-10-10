"""Viewer connection logging, heartbeat, pose memory and reconnect shim (task #347).

The heartbeat/pose tests need only numpy. The shim and live-server tests need
the viewer venv (viser, websockets); they are skipped elsewhere. Runs under
pytest or directly:  .venv/bin/python tests/unit/test_viewer_session.py
"""
from __future__ import annotations

import asyncio
import contextlib
import gzip
import io
import socket
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gsplat.viewer_session import Heartbeat, PoseMemory  # noqa: E402

try:
    import viser  # noqa: F401
    import websockets  # noqa: F401
    HAVE_VIEWER = True
except ImportError:
    HAVE_VIEWER = False

if "pytest" in sys.modules:
    import pytest
    needs_viewer = pytest.mark.skipif(not HAVE_VIEWER, reason="needs viser + websockets")
else:
    def needs_viewer(f):
        return f


def test_heartbeat_rate_limit() -> None:
    hb = Heartbeat(active_s=60, idle_s=900)
    assert hb.line(0, 0, 0, now=0.0) is not None  # first call prints
    assert hb.line(5, 5, 1, now=30.0) is None  # changed, but < 60 s
    line = hb.line(9, 8, 1, now=61.0)
    assert line is not None and "frames_rendered=9 (+9)" in line and "clients=1" in line
    assert hb.line(9, 8, 1, now=200.0) is None  # idle: wait 900 s
    assert hb.line(9, 8, 1, now=961.0) is not None
    assert hb.line(9, 8, 0, now=1000.0) is None  # client left, < 60 s
    assert "clients=0" in hb.line(9, 8, 0, now=1022.0)


def test_pose_memory() -> None:
    cam = SimpleNamespace(position=np.array([1.0, 2.0, 3.0]), look_at=np.zeros(3),
                          up_direction=np.array([0.0, 0.0, 1.0]), fov=0.8)
    pm = PoseMemory(max_age_s=1800)
    assert pm.recall(0, now=0.0) is None
    pm.remember(cam, now=100.0)
    cam.position = np.array([9.0, 9.0, 9.0])  # stored copy must not follow
    p = pm.recall(0, now=200.0)
    assert p is not None and np.allclose(p.position, [1, 2, 3]) and p.fov == 0.8
    assert pm.recall(1, now=200.0) is None  # another client is connected
    assert pm.recall(0, now=100.0 + 1801) is None  # too old
    pm.remember(SimpleNamespace(position=np.array([np.nan, 0, 0]), look_at=np.zeros(3),
                                up_direction=np.ones(3), fov=1.0), now=300.0)
    assert np.allclose(pm.recall(0, now=301.0).position, [1, 2, 3])  # NaN pose ignored


@needs_viewer
def test_inject_reconnect_shim() -> None:
    from gsplat.viser_patches import inject_reconnect_shim

    html = b"<!doctype html>\n<html><head><title>x</title></head><body></body></html>"
    out = inject_reconnect_shim(html)
    assert out.index(b"gsplat-reconnect-shim") < out.index(b"<title>")
    assert b"document.hasFocus" in out and b'msg.type === "retry"' in out
    assert inject_reconnect_shim(out) == out  # idempotent
    assert inject_reconnect_shim(b"<html>no head</html>") == b"<html>no head</html>"
    # Task #457: the page FPS shim rides along, once, before the page's own head.
    assert out.count(b"gsplat-pagefps-shim") == 1
    assert out.index(b"gsplat-pagefps-shim") < out.index(b"<title>")


@needs_viewer
def test_page_stats_parse_and_stale() -> None:
    from gsplat.viser_patches import PageStats
    ps = PageStats()
    assert ps.latest(now=0.0) is None
    assert not ps.update("dec=1.0&n=3", now=10.0)  # no fps: rejected
    assert not ps.update("fps=abc", now=10.0)
    assert ps.latest(now=10.0) is None
    assert ps.update("fps=118&dec=2.25&n=7.0", remote="1.2.3.4", now=10.0)
    got = ps.latest(now=11.0)
    assert got == {"fps": 118.0, "dec": 2.25, "n": 7, "remote": "1.2.3.4", "t": 10.0}
    assert ps.update("fps=5", now=12.0) and ps.latest(now=12.0)["dec"] == 0.0
    assert ps.latest(now=12.0 + PageStats.STALE_S + 0.01) is None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@needs_viewer
def test_live_server_logs_session_and_serves_shim() -> None:
    """Real viser server: page carries the shim (plain and gzip); a client's
    connect and its 1001 close are logged with id, codes and length."""
    import gsplat.viser_patches  # noqa: F401 - installs the patches
    import viser
    import websockets.asyncio.client as wsc

    port = _free_port()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        server = viser.ViserServer(host="127.0.0.1", port=port, verbose=False)
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10).read()
            assert b"gsplat-reconnect-shim" in page
            req = urllib.request.Request(f"http://127.0.0.1:{port}/",
                                         headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=10) as r:
                body = r.read()
                assert int(r.headers["Content-Length"]) == len(body)
            assert b"gsplat-reconnect-shim" in gzip.decompress(body)
            assert b"gsplat-pagefps-shim" in page
            # Page FPS report (task #457): 204, numbers land in PAGE_STATS.
            from gsplat.viser_patches import PAGE_STATS
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/gsplat/pagestats"
                                        "?fps=117&dec=1.5&n=42", timeout=10) as r:
                assert r.status == 204
            got = PAGE_STATS.latest()
            assert got is not None and got["fps"] == 117.0 and got["n"] == 42

            async def session() -> None:
                async with wsc.connect(f"ws://127.0.0.1:{port}",
                                       subprotocols=[f"viser-v{viser.__version__}"]) as ws:
                    await asyncio.sleep(0.5)
                    await ws.close(1001, "going away")

            asyncio.run(session())
            deadline = time.time() + 5
            while "disconnected" not in buf.getvalue() and time.time() < deadline:
                time.sleep(0.05)
        finally:
            server.stop()
    out = buf.getvalue()
    assert "client 0 connected from" in out, out
    line = next(ln for ln in out.splitlines() if "client 0 disconnected" in ln)
    assert "rcvd=1001 'going away'" in line and "closed_by=client" in line, line
    assert line.startswith("[viewer 20") and "Z]" in line, line


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            if not HAVE_VIEWER and name in ("test_inject_reconnect_shim",
                                            "test_page_stats_parse_and_stale",
                                            "test_live_server_logs_session_and_serves_shim"):
                print(f"SKIP {name}")
                continue
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
