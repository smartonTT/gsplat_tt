"""Viewer AFR on N chips (task #620): gsplat/afr_pool.py with a fake worker, no device.

Covers round-robin chip choice, per-worker environment, settings propagation, in-order
delivery through FastRenderer with a slow chip, worker start failure, the watchdog's pool
wait guard, and the flag-off path (no GSPLAT_VIEWER_AFR_DEVICES: FastRenderer as before).
The FastRenderer tests need the viewer venv (viser, nerfview) and are skipped elsewhere.
  .venv/bin/python -m pytest -q tests/unit/test_viewer_afr.py
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gsplat import afr_pool  # noqa: E402
from gsplat.afr_pool import AFRPipeline, AFRPool, fake_factory  # noqa: E402

try:
    import nerfview  # noqa: F401
    import viser  # noqa: F401
    HAVE_VIEWER = True
except ImportError:
    HAVE_VIEWER = False
needs_viewer = pytest.mark.skipif(not HAVE_VIEWER, reason="needs viser + nerfview")

K = np.array([[300.0, 0, 160], [0, 300.0, 128], [0, 0, 1]], np.float32)


def _w2c(i):
    w2c = np.eye(4, dtype=np.float32)
    w2c[0, 3] = float(i)
    return w2c


def _pool(devices=(0, 1), delay_s=(0.0,), **kw):
    return AFRPool(list(devices), fake_factory, {"delay_s": list(delay_s)},
                   max_wh=(320, 256), log=lambda m: None, **kw)


@pytest.fixture(scope="module")
def pool2():
    pool = _pool(devices=(0, 1), delay_s=(0.03, 0.0))
    yield pool
    pool.close()


def test_devices_from_env():
    assert afr_pool.devices_from_env({}) is None
    assert afr_pool.devices_from_env({afr_pool.ENV_DEVICES: " "}) is None
    assert afr_pool.devices_from_env({afr_pool.ENV_DEVICES: "0,1"}) == [0, 1]
    assert afr_pool.devices_from_env({afr_pool.ENV_DEVICES: "2"}) == [2]
    for bad in ("0,0", "a,1", "-1"):
        with pytest.raises(ValueError):
            afr_pool.devices_from_env({afr_pool.ENV_DEVICES: bad})
    assert afr_pool.inflight_from_env(2, {}) == 2
    assert afr_pool.inflight_from_env(2, {afr_pool.ENV_INFLIGHT: "3"}) == 3
    assert afr_pool.max_wh_from_env({}) == (2048, 2048)
    assert afr_pool.max_wh_from_env({afr_pool.ENV_MAX_WH: "1280x720"}) == (1280, 720)


def test_worker_env_one_visible_chip():
    env = afr_pool.worker_env({afr_pool.ENV_DEVICES: "0,1", "TT_VISIBLE_DEVICES": "",
                               "PATH": "/bin"}, 1, 3)
    assert env["TT_VISIBLE_DEVICES"] == "3" and env["GSPLAT_TT_DEVICE_ID"] == "0"
    assert afr_pool.ENV_DEVICES not in env and env["PATH"] == "/bin"


def test_round_robin_chips_and_worker_env(pool2):
    pipe = AFRPipeline(pool2, contrib_floor=1 / 255)
    # Each chip has inflight/n + 1 = 2 slots: 4 frames fill both chips; a 5th would block.
    assert pool2.slots_per_chip == 2
    tickets = [pipe.submit(_w2c(i), K, 256, 320) for i in range(4)]
    chips = [t.chip for t in tickets]
    assert chips == [c % 2 for c in range(tickets[0].fid, tickets[0].fid + 4)]
    with pytest.raises(TimeoutError):
        pipe.pool.submit(_w2c(9), K, 256, 320, pipe.settings(), timeout=0.2)
    for i, t in enumerate(tickets):
        r = pipe.wait(t)
        assert r.image.shape == (256, 320, 3) and r.image.dtype == np.uint8
        k, dev, x = r.image[0, 0]
        assert (k, dev, x) == (t.chip, (0, 1)[t.chip], i)
        assert r.chip == t.chip and r.fid == t.fid and r.num_visible == 7
        assert r.meta["TT_VISIBLE_DEVICES"] == str((0, 1)[t.chip])
        assert r.meta["GSPLAT_TT_DEVICE_ID"] == "0"
        assert r.meta[afr_pool.ENV_DEVICES] is None
    assert len({pipe.wait(pipe.submit(_w2c(0), K, 256, 320)).meta["pid"] for _ in range(2)}) == 2


def test_settings_reach_the_worker(pool2):
    pipe = AFRPipeline(pool2, contrib_floor=1 / 255)
    s = pipe.render(None, _w2c(1), K, 256, 320).meta["settings"]
    assert s["contrib_floor"] == 1 / 255 and s["cull_disabled"] is False
    # The viewer's sliders write the knobs, then sync (gsplat/viewer.py).
    pipe.contrib_floor = 1 / 16384
    pipe.cull_disabled = True
    pipe.min_opacity = 0.0
    pipe.max_radius = -1
    pipe._sync_render_settings_to_backend()
    for _ in range(2):  # both chips
        s = pipe.render(None, _w2c(1), K, 256, 320).meta["settings"]
        assert s["contrib_floor"] == 1 / 16384 and s["cull_disabled"] is True
        assert s["min_opacity"] == 0.0 and s["max_radius"] == -1
    assert set(s) == set(afr_pool.SETTINGS)


def test_slow_chip_frames_still_wait_in_order(pool2):
    pipe = AFRPipeline(pool2)
    tickets = [pipe.submit(_w2c(i), K, 256, 320) for i in range(4)]
    # Chip 1 (fast) finishes its frames first; waiting oldest first keeps submit order.
    xs = [int(pipe.wait(t).image[0, 0, 2]) for t in tickets]
    assert xs == [0, 1, 2, 3]
    iv = afr_pool.pipelined_intervals_ms(pipe, _w2c(0), K, 256, 320, 6)
    assert len(iv) == 5 and all(v >= 0 for v in iv)


def test_worker_start_failure_raises():
    with pytest.raises(RuntimeError, match="fake worker 1 fails to start"):
        AFRPool([0, 1], fake_factory, {"fail_start": 1}, max_wh=(64, 64), log=lambda m: None)


def test_watchdog_guards_pool_waits(pool2):
    from gsplat.render_watchdog import RenderWatchdog
    pipe = AFRPipeline(pool2)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        wd = RenderWatchdog(pipe.backend, tmp / "hang.log", tmp / "poses.jsonl",
                            limit_s=5.0, first_limit_s=60.0, ring=8)
        pipe.wait = wd.wrap_wait(pipe.wait)
        for i in range(4):
            pipe.render(None, _w2c(i), K, 256, 320)
        recs = list(wd._ring)
        assert [r["chip"] for r in recs] == [t % 2 for t in
                                             range(recs[0]["chip"], recs[0]["chip"] + 4)]
        # First render per size counts once per chip: the first two get the long limit.
        assert [r["limit"] for r in recs] == [60.0, 60.0, 5.0, 5.0]
        assert all(r["ms"] is not None for r in recs)
        assert recs[-1]["settings"]["contrib_floor_override"] == pipe.contrib_floor


# -- FastRenderer --------------------------------------------------------------------------

class _Client:
    client_id = 0

    def __init__(self):
        self.sent = []
        self.scene = self

    def set_background_image(self, image, format="jpeg", jpeg_quality=None, depth=None):
        self.sent.append(int(image[0, 0, 2]))


class _AFRViewer:
    """GaussianViewer's AFR hooks over a real pool with the fake worker."""
    state = "ready"
    _ui_active_deadline = 0.0

    def __init__(self, pipe):
        from nerfview.render_panel import RenderTabState
        self.render_tab_state = RenderTabState(render_width=320, render_height=256)
        self.pipe = pipe
        self.afr_inflight = pipe.inflight
        self.submitted, self.finished = [], []
        self.render_submit_fn = self._submit
        self.render_finish_fn = self._finish

    def _submit(self, cs, tab):
        i = int(cs.c2w[0, 3])
        self.submitted.append(i)
        return (self.pipe.submit(_w2c(i), K, 256, 320), cs)

    def _finish(self, token):
        r = self.pipe.wait(token[0])
        self.finished.append((int(r.image[0, 0, 2]), r.chip))
        return r.image

    def _after_render(self):
        pass


def _cs(i):
    import nerfview
    c2w = np.eye(4)
    c2w[0, 3] = float(i)
    return nerfview.CameraState(fov=1.0, aspect=320 / 256, c2w=c2w)


@needs_viewer
def test_fast_renderer_afr_in_order_round_robin(pool2):
    from nerfview._renderer import RenderTask
    from gsplat.nerfview_viewer import FastRenderer
    viewer = _AFRViewer(AFRPipeline(pool2))
    sent = threading.Event()
    fr = FastRenderer(viewer=viewer, client=_Client(), lock=threading.Lock(),
                      on_frame_sent=lambda *a: sent.set())
    assert fr._afr and fr.afr_depth == 2
    fr._state = "high"
    for i in range(8):
        fr.render_once_afr(RenderTask("move", _cs(i)))
        # One frame stays in flight on the other chip while the next is submitted.
        assert len(fr._inflight) == 1
    while fr._inflight:
        fr._afr_finish_oldest()
    assert viewer.submitted == list(range(8))
    assert [x for x, _ in viewer.finished] == list(range(8))  # in order, chip 0 is slower
    chips = [c for _, c in viewer.finished]
    assert all(a != b for a, b in zip(chips, chips[1:]))       # alternating chips
    assert sent.wait(5.0)
    fr.running = False
    with fr._out_cv:
        fr._out_cv.notify_all()
    sent_xs = fr.client.sent
    assert sent_xs and sent_xs == sorted(sent_xs)               # sender keeps the order


@needs_viewer
def test_fast_renderer_flag_off_unchanged():
    """No render_submit_fn (GSPLAT_VIEWER_AFR_DEVICES unset): the single-chip path, with the
    next-pose hint, exactly as before."""
    from gsplat.nerfview_viewer import FastRenderer, pipeline_frames_from_env

    class _V(_AFRViewer):
        def __init__(self):
            from nerfview.render_panel import RenderTabState
            self.render_tab_state = RenderTabState(render_width=320, render_height=256)
            self.render_fn = lambda cs, tab: np.zeros((256, 320, 3), np.uint8)

    fr = FastRenderer(viewer=_V(), client=_Client(), lock=threading.Lock())
    try:
        assert fr._afr is False and not fr._inflight
        assert fr.pipeline_frames == pipeline_frames_from_env()  # next-pose hint as before
    finally:
        fr.running = False


def test_viewer_clean_flag_off_builds_no_pool(monkeypatch):
    monkeypatch.delenv(afr_pool.ENV_DEVICES, raising=False)
    assert afr_pool.devices_from_env() is None
    src = (Path(__file__).resolve().parents[2] / "opt" / "viewer" / "viewer_clean.py").read_text()
    # Flag off: no afr_pool import, GaussianViewer gets no pipeline= (its own Pipeline(tt_clean)), the
    # watchdog wraps pipeline.render as before.
    assert "afr_devs = None\n    extra = {}\n" in src and "**extra)" in src
    assert "viewer.pipeline.render = wd.wrap(viewer.pipeline.render)" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
