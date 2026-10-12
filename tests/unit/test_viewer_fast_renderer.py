"""Viewer sender path (task #271): JPEG encoder hook and no mid-render interrupt.

Needs the viewer venv (viser, nerfview); skipped elsewhere. Runs under pytest or directly:
  .venv/bin/python tests/unit/test_viewer_fast_renderer.py
"""
from __future__ import annotations

import io
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import nerfview  # noqa: F401
    import viser  # noqa: F401
    HAVE_VIEWER = True
except ImportError:
    HAVE_VIEWER = False

if "pytest" in sys.modules:
    import pytest
    pytestmark = pytest.mark.skipif(not HAVE_VIEWER, reason="needs viser + nerfview")


def _test_image() -> np.ndarray:
    y, x = np.mgrid[0:256, 0:320].astype(np.float32)
    img = np.stack([x / 320.0, y / 256.0, 0.5 + 0.5 * np.sin(x / 17.0 + y / 23.0)], axis=-1)
    return (img * 255.0).astype(np.uint8)


def _decode(data: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def test_jpeg_encoders_roundtrip():
    from gsplat.nerfview_viewer import JPEG_ENCODERS, make_jpeg_encoder, pick_jpeg_encoder
    img = _test_image()
    usable = []
    for name in JPEG_ENCODERS:
        try:
            enc = make_jpeg_encoder(name)
        except ImportError:
            continue
        usable.append(name)
        lo, hi = enc(img, 40), enc(img, 70)
        assert lo[:2] == b"\xff\xd8", name
        assert len(hi) > len(lo), name
        dec = _decode(hi)
        assert dec.shape == img.shape, name
        # Channel order must survive: a swapped R/B would score far below this.
        assert _psnr(dec, img) > 30.0, (name, _psnr(dec, img))
    assert "viser" in usable
    assert pick_jpeg_encoder("auto")[0] == usable[0]


class _Ws:
    def __init__(self):
        self.messages = []

    def queue_message(self, msg):
        self.messages.append(msg)


class _Scene:
    def __init__(self):
        self._websock_interface = _Ws()

    def set_background_image(self, image, format="jpeg", jpeg_quality=None, depth=None):
        # viser's path (used when only its own encoder is installed): encode, queue one message.
        from viser._messages import BackgroundImageMessage
        from gsplat.nerfview_viewer import make_jpeg_encoder
        data = make_jpeg_encoder("viser")(np.ascontiguousarray(image), jpeg_quality or 90)
        self._websock_interface.queue_message(
            BackgroundImageMessage(format=format, rgb_data=data, depth_data=None))


class _Client:
    client_id = 0

    def __init__(self):
        self.scene = _Scene()


class _Viewer:
    state = "ready"
    _ui_active_deadline = 0.0

    def __init__(self, render_fn):
        from nerfview.render_panel import RenderTabState
        self.render_tab_state = RenderTabState(render_width=320, render_height=256)
        self.render_fn = render_fn

    def _after_render(self):
        pass


def test_move_mid_render_does_not_interrupt_and_frame_is_sent():
    """A move submitted while a high-quality frame renders must not abort the render: the TT
    backend would leave half a frame in its daemon pipe. nerfview interrupts through a
    sys.settrace hook; FastRenderer renders without one."""
    from nerfview._renderer import RenderTask
    from viser._messages import BackgroundImageMessage
    from gsplat.nerfview_viewer import FastRenderer
    import nerfview

    img = _test_image()
    cs = nerfview.CameraState(fov=1.0, aspect=320 / 256, c2w=np.eye(4))
    holder = {}
    finished = []

    def render_fn(camera_state, tab_state):
        holder["fr"].submit(RenderTask("move", camera_state))
        total = 0
        for i in range(1000):  # many traced lines: an armed interrupt would fire here
            total += i
        finished.append(total)
        return img

    sent = threading.Event()
    fr = FastRenderer(viewer=_Viewer(render_fn), client=_Client(), lock=threading.Lock(),
                      on_frame_sent=lambda *a: sent.set())
    holder["fr"] = fr
    fr._state = "high"
    fr.render_once(RenderTask("static", cs))
    assert finished == [sum(range(1000))]
    assert sent.wait(5.0)
    fr.running = False
    msgs = fr.client.scene._websock_interface.messages
    assert len(msgs) == 1 and isinstance(msgs[0], BackgroundImageMessage)
    assert msgs[0].format == "jpeg" and msgs[0].depth_data is None
    assert _psnr(_decode(msgs[0].rgb_data), img) > 30.0


def test_no_submit_patch():
    """viser_patches no longer wraps nerfview's Renderer.submit (the wrapper was a no-op)."""
    import gsplat.viser_patches  # noqa: F401
    from nerfview._renderer import Renderer
    assert Renderer.submit.__module__ == "nerfview._renderer"


def test_sender_threads_exit_on_disconnect():
    """Repeated connect/disconnect must not leak gsplat-send threads (review #269)."""
    import time
    from gsplat.nerfview_viewer import FastRenderer

    def senders():
        return sum(t.name == "gsplat-send" and t.is_alive() for t in threading.enumerate())

    base = senders()
    for _ in range(20):
        fr = FastRenderer(viewer=_Viewer(lambda *a: _test_image()), client=_Client(),
                          lock=threading.Lock())
        fr.running = False  # what nerfview's disconnect does
        fr._sender.join(2.0)
        assert not fr._sender.is_alive()
    time.sleep(0.05)
    assert senders() == base


def test_bad_jpeg_env_falls_back_to_auto(monkeypatch=None):
    import os
    from gsplat.nerfview_viewer import JPEG_ENCODERS, jpeg_encoder_from_env, pick_jpeg_encoder
    old = os.environ.get("GSPLAT_VIEWER_JPEG")
    try:
        os.environ["GSPLAT_VIEWER_JPEG"] = "bogus"
        assert jpeg_encoder_from_env() == "auto"
        assert pick_jpeg_encoder()[0] in JPEG_ENCODERS
        os.environ["GSPLAT_VIEWER_JPEG"] = "viser"
        assert jpeg_encoder_from_env() == "viser"
    finally:
        if old is None:
            os.environ.pop("GSPLAT_VIEWER_JPEG", None)
        else:
            os.environ["GSPLAT_VIEWER_JPEG"] = old


class _PoseViewer(_Viewer):
    """_Viewer whose client camera walks through poses 1, 2, 3, ... (task #457)."""

    def __init__(self, render_fn):
        super().__init__(render_fn)
        self.next_camera_state = None
        self.frame_post_fn = None
        self.polled = 0

    def get_camera_state(self, client):
        import nerfview
        self.polled += 1
        c2w = np.eye(4)
        c2w[0, 3] = float(self.polled)
        return nerfview.CameraState(fov=1.0, aspect=320 / 256, c2w=c2w)


def _pose_id(cs) -> float:
    return float(cs.c2w[0, 3])


def test_next_pose_hint_latching():
    """While moving, frame k renders the pose hinted with frame k-1, so the device's
    prefetched front stages always match; a static frame renders its own pose and
    drops the latch (task #457)."""
    import os
    from nerfview._renderer import RenderTask
    from gsplat.nerfview_viewer import FastRenderer
    import nerfview

    calls = []
    holder = {}

    def render_fn(camera_state, tab_state):
        nxt = holder["v"].next_camera_state
        calls.append((_pose_id(camera_state), None if nxt is None else _pose_id(nxt)))
        return _test_image()

    v = _PoseViewer(render_fn)
    holder["v"] = v

    def task_cs(x):
        c2w = np.eye(4)
        c2w[0, 3] = x
        return nerfview.CameraState(fov=1.0, aspect=320 / 256, c2w=c2w)

    old = os.environ.pop("GSPLAT_VIEWER_PIPELINE", None)
    try:
        fr = FastRenderer(viewer=v, client=_Client(), lock=threading.Lock())
        assert fr.pipeline_frames
        fr.render_once(RenderTask("move", task_cs(-1.0)))    # own pose, hints pose 1
        fr.render_once(RenderTask("move", task_cs(-2.0)))    # pose 1 (latched), hints 2
        fr.render_once(RenderTask("rerender", task_cs(-3.0)))  # pose 2, hints 3
        fr.render_once(RenderTask("static", task_cs(-4.0)))  # own pose, no hint, latch reset
        fr.render_once(RenderTask("move", task_cs(-5.0)))    # own pose again, hints 4
        assert calls == [(-1.0, 1.0), (1.0, 2.0), (2.0, 3.0), (-4.0, None), (-5.0, 4.0)], calls
        assert v.next_camera_state is None  # cleared after every render
        fr.running = False

        os.environ["GSPLAT_VIEWER_PIPELINE"] = "0"
        calls.clear()
        fr = FastRenderer(viewer=v, client=_Client(), lock=threading.Lock())
        assert not fr.pipeline_frames
        fr.render_once(RenderTask("move", task_cs(-6.0)))
        fr.render_once(RenderTask("move", task_cs(-7.0)))
        assert calls == [(-6.0, None), (-7.0, None)], calls
        fr.running = False
    finally:
        os.environ.pop("GSPLAT_VIEWER_PIPELINE", None)
        if old is not None:
            os.environ["GSPLAT_VIEWER_PIPELINE"] = old


def test_frame_post_fn_runs_on_sender_thread():
    """The letterbox (viewer.frame_post_fn) runs on the sender thread with the
    rendered frame's camera, and the sent JPEG is its output (task #457)."""
    from nerfview._renderer import RenderTask
    from gsplat.nerfview_viewer import FastRenderer
    import nerfview

    img = _test_image()
    seen = []

    def post(frame, cs):
        seen.append((threading.current_thread().name, _pose_id(cs), frame.shape))
        return np.pad(frame, ((32, 32), (0, 0), (0, 0)))  # letterbox bars

    v = _PoseViewer(lambda cs, ts: img)
    v.frame_post_fn = post
    sent = threading.Event()
    fr = FastRenderer(viewer=v, client=_Client(), lock=threading.Lock(),
                      on_frame_sent=lambda *a: sent.set())
    c2w = np.eye(4)
    c2w[0, 3] = 9.0
    fr.render_once(RenderTask("static", nerfview.CameraState(fov=1.0, aspect=320 / 256, c2w=c2w)))
    assert sent.wait(5.0)
    fr.running = False
    assert seen == [("gsplat-send", 9.0, img.shape)], seen
    msgs = fr.client.scene._websock_interface.messages
    assert len(msgs) == 1 and _decode(msgs[0].rgb_data).shape == (256 + 64, 320, 3)


def test_hint_next_pose_matches_bench_extrinsics():
    """GaussianViewer._hint_next_pose hands the backend the next pose's w2c built
    like the render's own (bitwise prefetch key), and no hint when K differs."""
    import types
    import torch
    import nerfview
    from gsplat.utils import c2w_to_w2c
    from gsplat.viewer import GaussianViewer

    c2w = np.eye(4)
    c2w[:3, 3] = [0.25, -1.5, 3.0]
    nxt = nerfview.CameraState(fov=0.9, aspect=320 / 256, c2w=c2w)
    backend = types.SimpleNamespace(next_extrinsics=None)
    gv = types.SimpleNamespace(viewer=types.SimpleNamespace(next_camera_state=nxt),
                               pipeline=types.SimpleNamespace(backend=backend), _hints=0)
    K = torch.tensor(nxt.get_K((320, 256)), dtype=torch.float32)
    GaussianViewer._hint_next_pose(gv, 320, 256, K)
    want = c2w_to_w2c(torch.from_numpy(np.asarray(c2w, dtype=np.float32)))
    assert gv._hints == 1 and torch.equal(backend.next_extrinsics, want)

    backend.next_extrinsics = None
    GaussianViewer._hint_next_pose(gv, 320, 256, K * 1.01)  # fov/size change: would miss
    assert backend.next_extrinsics is None and gv._hints == 1
    gv.viewer.next_camera_state = None
    GaussianViewer._hint_next_pose(gv, 320, 256, K)
    assert backend.next_extrinsics is None and gv._hints == 1


def test_camera_burst_renders_only_the_moving_client():
    """One client's camera drag bursts only that client: an idle page in another tab
    must not render back to back and take half the device (task #459). A render
    settings change (no client id) still bursts every client."""
    from gsplat.nerfview_viewer import GsplatViewer, burst_active

    v = types.SimpleNamespace(_ui_active_deadline=0.0, _client_active_deadline={})
    GsplatViewer.mark_ui_active(v, 1)
    assert burst_active(v, 1) and not burst_active(v, 0) and burst_active(v)
    assert not burst_active(v, 1, now=time.time() + 2.0)
    GsplatViewer.mark_ui_active(v)
    assert burst_active(v, 0) and burst_active(v, 1)

    tasks = {0: [], 1: []}

    class _R:
        def __init__(self, cid):
            self.cid = cid

        def submit(self, task):
            tasks[self.cid].append(task.action)

    v = types.SimpleNamespace(
        _ui_active_deadline=0.0, _client_active_deadline={}, _burst_running=True,
        _renderers={0: _R(0), 1: _R(1)},
        server=types.SimpleNamespace(get_clients=lambda: {0: "c0", 1: "c1"}),
        get_camera_state=lambda c: c)
    v.rerender = lambda _e: [r.submit(types.SimpleNamespace(action="rerender"))
                             for r in v._renderers.values()]
    GsplatViewer.mark_ui_active(v, 1)
    th = threading.Thread(target=GsplatViewer._ui_burst_loop, args=(v,), daemon=True)
    th.start()
    time.sleep(0.2)
    v._burst_running = False
    th.join(2.0)
    assert tasks[1] and set(tasks[1]) == {"rerender"}, tasks
    assert tasks[0] == [], tasks


def test_motion_quality_steps_down_on_backlog_and_recovers():
    """Moving-frame JPEG quality drops 5 per backlogged frame down to the floor and climbs
    back after ``recover`` clean frames; off, it stays at q40 (task #532)."""
    from gsplat.nerfview_viewer import MotionQuality
    mq = MotionQuality(lo=20, step=5, recover=3)
    assert mq.update(False) == 40
    assert [mq.update(True) for _ in range(6)] == [35, 30, 25, 20, 20, 20]
    assert [mq.update(False) for _ in range(6)] == [20, 20, 25, 25, 25, 30]
    off = MotionQuality(enabled=False)
    assert all(off.update(True) == 40 for _ in range(5))


class _PendingBuf:
    """viser AsyncMessageBuffer stand-in: queued messages stay unsent until popped."""

    def __init__(self):
        self.id_from_redundancy_key, self.message_from_id, self.n = {}, {}, 0

    def push(self, msg):
        old = self.id_from_redundancy_key.pop(msg.redundancy_key(), None)
        self.message_from_id.pop(old, None)
        self.id_from_redundancy_key[msg.redundancy_key()] = self.n
        self.message_from_id[self.n] = msg
        self.n += 1


class _PacedWs(_Ws):
    def __init__(self):
        super().__init__()
        self.buf = _PendingBuf()

    def get_message_buffer(self):
        return self.buf

    def queue_message(self, msg):
        super().queue_message(msg)
        self.buf.push(msg)


def test_backlog_paces_sends_lowers_motion_quality_and_keeps_still_q70():
    """A frame ready while the previous one is still unsent in viser's buffer (the link is
    the limit) waits for it, sends the newest frame instead, and steps the moving-frame
    quality down; a still frame is always q70, byte-identical to an unpaced send (#532)."""
    from nerfview._renderer import RenderTask
    from gsplat.nerfview_viewer import FastRenderer, make_jpeg_encoder
    import nerfview

    base = _test_image()
    imgs = {k: np.clip(base.astype(np.int32) + 40 * k, 0, 255).astype(np.uint8) for k in range(4)}
    cur = {"k": 0}
    cs = nerfview.CameraState(fov=1.0, aspect=320 / 256, c2w=np.eye(4))
    sent = []
    client = _Client()
    client.scene._websock_interface = _PacedWs()
    fr = FastRenderer(viewer=_Viewer(lambda c, t: imgs[cur["k"]]), client=client,
                      lock=threading.Lock(), on_frame_sent=lambda *a: sent.append(a))
    if fr.jpeg_encoder_name == "viser":
        fr.running = False
        return  # the viser encoder path queues through viser itself: no pacing hook
    ws = client.scene._websock_interface

    def frame(k, action):
        cur["k"] = k
        fr.render_once(RenderTask(action, cs))

    def wait_sent(n):
        t = time.time() + 5.0
        while len(sent) < n and time.time() < t:
            time.sleep(0.005)
        assert len(sent) >= n, (len(sent), n)

    frame(0, "move")
    wait_sent(1)
    assert fr.motion_quality.q == 40  # nothing was pending
    frame(1, "move")  # previous still pending: the sender waits for it ...
    time.sleep(0.01)
    frame(2, "move")  # ... and sends this newer frame instead of frame 1
    wait_sent(2)
    assert len(ws.messages) == 2
    assert fr.motion_quality.q == 35
    m = _decode(ws.messages[1].rgb_data)
    assert _psnr(m, imgs[2]) > _psnr(m, imgs[1]) + 3.0
    ws.buf.message_from_id.clear()  # viser took every frame: no backlog
    ws.buf.id_from_redundancy_key.clear()
    frame(3, "static")
    wait_sent(3)
    fr.running = False
    assert ws.messages[2].rgb_data == make_jpeg_encoder(fr.jpeg_encoder_name)(imgs[3], 70)


if __name__ == "__main__":
    if not HAVE_VIEWER:
        print("skip: needs viser + nerfview")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
