"""Viewer sender path (task #271): JPEG encoder hook and no mid-render interrupt.

Needs the viewer venv (viser, nerfview); skipped elsewhere. Runs under pytest or directly:
  .venv/bin/python tests/unit/test_viewer_fast_renderer.py
"""
from __future__ import annotations

import io
import sys
import threading
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


if __name__ == "__main__":
    if not HAVE_VIEWER:
        print("skip: needs viser + nerfview")
        sys.exit(0)
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
