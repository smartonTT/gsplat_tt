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
    # <=: an earlier test's sender may still be exiting when base is taken; a leak only adds.
    assert senders() <= base


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


def test_render_fn_reuses_the_hinted_pose_mats():
    """Task #597: the frame that renders the last hinted pose reuses the hint's K and
    w2c (no host rebuild while the B4a-folded device idles), bit-equal to a fresh build;
    any other pose, size or fov builds them anew."""
    from collections import deque
    import torch
    import nerfview
    from gsplat.pipeline import RenderResult
    from gsplat.viewer import GaussianViewer

    calls = []
    backend = types.SimpleNamespace(next_extrinsics=None)

    def render(g, extr, intr, H, W):
        calls.append((extr, intr, backend.next_extrinsics))
        backend.next_extrinsics = None  # used by this render only, as CleanBackend does
        return RenderResult(image=np.zeros((H, W, 3), np.uint8), timings={"total": 1.0})

    gv = GaussianViewer.__new__(GaussianViewer)
    gv.__dict__.update(
        _frames_rendered=0, force_square=None, verbose=False, gaussians=None, _hints=0,
        _hinted=None, _k_cache={}, _prof=None, _render_ms=deque(), _host_ms=deque(),
        _frame_samples=[], pipeline=types.SimpleNamespace(render=render, backend=backend),
        viewer=types.SimpleNamespace(next_camera_state=None, frame_post_fn=lambda *a: None))
    gv._update_stats = lambda *a: None
    tab = types.SimpleNamespace(render_width=320, render_height=256)

    def cam(x, fov=0.9):
        c2w = np.eye(4)
        c2w[:3, 3] = [x, -1.5, 3.0]
        return nerfview.CameraState(fov=fov, aspect=320 / 256, c2w=c2w)

    def fresh(cs):
        return (torch.tensor(cs.get_K((320, 256)), dtype=torch.float32),
                GaussianViewer._render_fn.__globals__["c2w_to_w2c"](
                    torch.from_numpy(np.asarray(cs.c2w, dtype=np.float32))))

    a, b, c = cam(0.25), cam(0.5), cam(0.75)
    gv.viewer.next_camera_state = b
    GaussianViewer._render_fn(gv, a, tab)  # a fresh, hints b
    gv.viewer.next_camera_state = c
    GaussianViewer._render_fn(gv, b, tab)  # b from the hint, hints c
    gv.viewer.next_camera_state = None
    GaussianViewer._render_fn(gv, cam(0.75), tab)  # c's values, other object: fresh
    (ea, ka, ha), (eb, kb, hb), (ec, kc, hc) = calls
    assert eb is ha and kb is gv._k_cache[(0.9, 320, 256)] and hc is None
    for (e, k), cs in (((ea, ka), a), ((eb, kb), b), ((ec, kc), c)):
        fk, fe = fresh(cs)
        assert torch.equal(e, fe) and torch.equal(k, fk)
    assert gv._hinted is None  # consumed or replaced by every render

    gv.viewer.next_camera_state = cam(0.1, fov=1.1)  # fov change: no hint, no reuse
    GaussianViewer._render_fn(gv, a, tab)
    assert calls[-1][2] is None and gv._hinted is None


def test_camera_update_prepares_w2c_off_the_render_thread():
    """Task #597: the camera-update callback builds the pose's w2c (on viser's thread) and
    keeps that state as the client's latest; the render loop hints and renders from it, so
    no w2c or camera state is rebuilt on the render thread while the folded device idles."""
    import contextlib
    import torch
    import nerfview
    from gsplat.nerfview_viewer import GsplatViewer
    from gsplat.viewer import _cached_w2c, _prepare_w2c, _w2c

    submitted, built = [], []
    client = types.SimpleNamespace(client_id=7)

    def get_cs(c):
        c2w = np.eye(4)
        c2w[:3, 3] = [0.3 * (len(built) + 1), -1.5, 3.0]
        built.append(nerfview.CameraState(fov=0.9, aspect=1.25, c2w=c2w))
        return built[-1]

    gv = types.SimpleNamespace(
        _latest_cs={}, _cs_stamp={}, _cs_lock=threading.Lock(), camera_prepare_fn=_prepare_w2c, get_camera_state=get_cs,
        mark_ui_active=lambda cid: None,
        server=types.SimpleNamespace(atomic=contextlib.nullcontext),
        _renderers={7: types.SimpleNamespace(submit=submitted.append)})
    GsplatViewer._camera_moved(gv, client)
    cs = built[-1]
    assert submitted[-1].camera_state is cs and gv._latest_cs[7] is cs
    w2c = cs._gsplat_w2c
    assert torch.equal(w2c, _w2c(cs)) and _cached_w2c(cs) is w2c
    assert GsplatViewer.latest_camera_state(gv, client) is cs and len(built) == 1
    other = types.SimpleNamespace(client_id=8)  # no update yet: a fresh state
    assert GsplatViewer.latest_camera_state(gv, other) is built[-1] and len(built) == 2
    assert _cached_w2c(built[-1]) is not None  # not prepared: built now

    class Slotted:  # a state that takes no attributes: w2c built at render time
        __slots__ = ("c2w",)

    st = Slotted()
    st.c2w = cs.c2w
    _prepare_w2c(st)
    assert torch.equal(_cached_w2c(st), w2c)

    # The render loop takes the hint from latest_camera_state, not get_camera_state.
    import threading
    from nerfview._renderer import RenderTask
    from gsplat.nerfview_viewer import FastRenderer
    seen = []
    pv = _PoseViewer(lambda c, t: (seen.append(pv.next_camera_state), _test_image())[1])
    pv.latest_camera_state = lambda c: cs
    fr = FastRenderer(viewer=pv, client=_Client(), lock=threading.Lock())
    fr.render_once(RenderTask("move", cs))
    assert seen == [cs] and pv.polled == 0
    fr.running = False


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
    """Moving frames step one rung down the (scale, quality) ladder per backlogged frame to
    the floor and one rung back up after ``recover`` clean frames; off, they stay full
    size at q40 (task #532)."""
    from gsplat.nerfview_viewer import MotionQuality
    mq = MotionQuality(recover=3)
    assert mq.update(False) == (1.0, 40)
    assert [mq.update(True) for _ in range(6)] == [
        (1.0, 30), (0.75, 40), (0.75, 30), (0.75, 20), (0.75, 20), (0.75, 20)]
    assert [mq.update(False) for _ in range(6)] == [
        (0.75, 20), (0.75, 20), (0.75, 30), (0.75, 30), (0.75, 30), (0.75, 40)]
    off = MotionQuality(enabled=False)
    assert all(off.update(True) == (1.0, 40) for _ in range(5))


def test_downscale_sizes_and_keeps_full_frames():
    """0.75x moving frames keep the aspect; scale 1.0 returns the frame itself (#532)."""
    from gsplat.nerfview_viewer import downscale
    img = _test_image()
    assert downscale(img, 1.0) is img
    small = downscale(img, 0.75)
    assert small.shape == (round(img.shape[0] * 0.75), round(img.shape[1] * 0.75), 3)
    assert small.dtype == np.uint8


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
    assert (fr.motion_quality.scale, fr.motion_quality.q) == (1.0, 30)
    m = _decode(ws.messages[1].rgb_data)
    assert _psnr(m, imgs[2]) > _psnr(m, imgs[1]) + 3.0
    fr.motion_quality.rung = 2  # deeper backlog: the next moving frame goes out at 0.75x
    frame(0, "move")
    wait_sent(3)
    small = _decode(ws.messages[2].rgb_data)
    assert small.shape[:2] == (round(base.shape[0] * 0.75), round(base.shape[1] * 0.75))
    ws.buf.message_from_id.clear()  # viser took every frame: no backlog
    ws.buf.id_from_redundancy_key.clear()
    frame(3, "static")
    wait_sent(4)
    fr.running = False
    assert ws.messages[3].rgb_data == make_jpeg_encoder(fr.jpeg_encoder_name)(imgs[3], 70)


if __name__ == "__main__":
    if not HAVE_VIEWER:
        print("skip: needs viser + nerfview")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


def test_camera_moved_keeps_only_the_newest_pose():
    """Task #602: overlapping callbacks: an older pose stored last must not win, and a
    late callback must not add a disconnected client back."""
    import contextlib
    import threading
    import types
    from gsplat.nerfview_viewer import GsplatViewer

    submitted = []
    gv = types.SimpleNamespace(
        _latest_cs={}, _cs_stamp={}, _cs_lock=threading.Lock(), camera_prepare_fn=None,
        get_camera_state=lambda c: object(), mark_ui_active=lambda cid: None,
        server=types.SimpleNamespace(atomic=contextlib.nullcontext),
        _renderers={7: types.SimpleNamespace(submit=submitted.append)})
    new = types.SimpleNamespace(client_id=7, camera=types.SimpleNamespace(update_timestamp=2.0))
    old = types.SimpleNamespace(client_id=7, camera=types.SimpleNamespace(update_timestamp=1.0))
    GsplatViewer._camera_moved(gv, new)
    kept = gv._latest_cs[7]
    GsplatViewer._camera_moved(gv, old)
    assert gv._latest_cs[7] is kept and len(submitted) == 1
    gone = types.SimpleNamespace(client_id=9, camera=types.SimpleNamespace(update_timestamp=3.0))
    GsplatViewer._camera_moved(gv, gone)
    assert 9 not in gv._latest_cs and 9 not in gv._cs_stamp
