"""Gsplat-specific nerfview integration."""
from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import nerfview
import numpy as np
import viser
from nerfview._renderer import Renderer, RenderTask
from viser._messages import BackgroundImageMessage
from nerfview.render_panel import RenderTabState, populate_general_render_tab

# Keep rendering for 1 s after any UI action so the FPS readout settles.
_UI_BURST_SEC = 1.0
_UI_BURST_POLL_SEC = 0.016
# Longest the sender waits for viser to take the previous frame before sending anyway.
_PACE_MAX_WAIT_SEC = 0.05

# JPEG encoders for the sender thread, in auto order. All but "viser" release the GIL for the
# whole encode, so the render thread keeps running. "viser" is viser's own encode (cv2 with a
# numpy channel swap, ~10 ms for 1024^2 on bh-35; task #271).
JPEG_ENCODERS = ("simplejpeg", "turbojpeg", "cv2", "viser")


class MotionQuality:
    """JPEG quality and scale for moving frames, sized from the send backlog (task #532).

    A frame that is ready while the previous one still sits unsent in viser's buffer means
    the link (the ssh tunnel to the page) is slower than the device: step one rung down
    ``LADDER`` so the frames get smaller. After ``recover`` clean frames step one rung back
    up. Quality alone saves little at 1024x1024 (hero: q40 97 KB, q30 86, q20 65), so the
    lower rungs send 0.75x frames (q40 59 KB, q30 50, q20 38) that the page stretches to
    the canvas. Still frames (static/update) are always full size at ``STILL``.
    """
    STILL = 70
    MAX = 40
    LADDER = ((1.0, 40), (1.0, 30), (0.75, 40), (0.75, 30), (0.75, 20))

    def __init__(self, enabled: bool = True, recover: int = 60):
        self.enabled, self.recover = enabled, recover
        self.rung, self._clean = 0, 0

    @property
    def scale(self) -> float:
        return self.LADDER[self.rung][0]

    @property
    def q(self) -> int:
        return self.LADDER[self.rung][1]

    def update(self, backlogged: bool) -> tuple[float, int]:
        """Return (scale, quality) for the next moving frame."""
        if not self.enabled:
            return self.LADDER[0]
        if backlogged:
            self.rung, self._clean = min(len(self.LADDER) - 1, self.rung + 1), 0
        else:
            self._clean += 1
            if self._clean >= self.recover and self.rung > 0:
                self.rung, self._clean = self.rung - 1, 0
        return self.LADDER[self.rung]


def downscale(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize an HxWx3 uint8 frame by ``scale`` (bilinear; cv2 releases the GIL)."""
    if scale >= 1.0:
        return img
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_LINEAR)


def adaptive_quality_from_env() -> bool:
    """GSPLAT_VIEWER_ADAPTIVE (default on; "0" keeps moving frames at q40, task #532)."""
    return os.environ.get("GSPLAT_VIEWER_ADAPTIVE", "1").strip() not in ("0", "off", "false")


def pipeline_frames_from_env() -> bool:
    """GSPLAT_VIEWER_PIPELINE (default on; "0" turns the next-pose hint off, task #457)."""
    return os.environ.get("GSPLAT_VIEWER_PIPELINE", "1").strip() not in ("0", "off", "false")


def timeline_path_from_env() -> Optional[str]:
    """GSPLAT_VIEWER_TIMELINE=<path> (task #587; off when unset): append one JSON line per
    rendered and per sent frame (perf_counter seconds), for the host/device timeline."""
    return os.environ.get("GSPLAT_VIEWER_TIMELINE", "").strip() or None


class FrameTimeline:
    """Per-frame records from the render and sender threads (deque appends are thread-safe),
    written out by the sender thread every ``flush_every`` records."""

    def __init__(self, path: str, flush_every: int = 1000) -> None:
        self.path, self.flush_every = path, flush_every
        self._q: collections.deque = collections.deque()

    def add(self, **rec) -> None:
        self._q.append(rec)

    def maybe_flush(self, force: bool = False) -> None:
        if not force and len(self._q) < self.flush_every:
            return
        lines = []
        while self._q:
            lines.append(json.dumps(self._q.popleft(), separators=(",", ":")))
        if lines:
            with open(self.path, "a") as f:
                f.write("\n".join(lines) + "\n")


def make_jpeg_encoder(name: str) -> Callable[[np.ndarray, int], bytes]:
    """Return ``encode(rgb_uint8_hwc, quality) -> jpeg bytes``; ImportError if unusable."""
    if name == "simplejpeg":
        import simplejpeg

        def encode(img: np.ndarray, quality: int) -> bytes:
            return simplejpeg.encode_jpeg(img, quality=quality, colorspace="RGB",
                                          colorsubsampling="420", fastdct=True)
        return encode
    if name == "turbojpeg":
        try:
            import turbojpeg
            tj = turbojpeg.TurboJPEG()  # loads libturbojpeg.so
        except (OSError, RuntimeError) as e:
            raise ImportError(f"libturbojpeg: {e}") from e

        def encode(img: np.ndarray, quality: int) -> bytes:
            return tj.encode(img, quality=quality, pixel_format=turbojpeg.TJPF_RGB,
                             jpeg_subsample=turbojpeg.TJSAMP_420)
        return encode
    if name == "cv2":
        import cv2

        def encode(img: np.ndarray, quality: int) -> bytes:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # SIMD swap, not viser's numpy gather
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError("cv2.imencode failed")
            return buf.tobytes()
        return encode
    if name == "viser":
        from viser._scene_api import _encode_image_binary

        def encode(img: np.ndarray, quality: int) -> bytes:
            return _encode_image_binary(img, "jpeg", jpeg_quality=quality)[1]
        return encode
    raise ValueError(f"unknown JPEG encoder {name!r}; choose from {JPEG_ENCODERS}")


def jpeg_encoder_from_env() -> str:
    """GSPLAT_VIEWER_JPEG, or "auto" (with a warning) if unset or unknown."""
    name = os.environ.get("GSPLAT_VIEWER_JPEG", "auto")
    if name != "auto" and name not in JPEG_ENCODERS:
        print(f"[viewer] WARNING: unknown GSPLAT_VIEWER_JPEG={name!r}; choose from "
              f"{JPEG_ENCODERS}; using auto", file=sys.stderr)
        return "auto"
    return name


def pick_jpeg_encoder(name: Optional[str] = None) -> tuple[str, Callable[[np.ndarray, int], bytes]]:
    """``name`` (default: env GSPLAT_VIEWER_JPEG, else "auto"): first usable in auto order."""
    name = name or jpeg_encoder_from_env()
    for cand in (JPEG_ENCODERS if name == "auto" else (name,)):
        try:
            return cand, make_jpeg_encoder(cand)
        except ImportError:
            if name != "auto":
                raise
    raise AssertionError("viser encoder is always importable")


def burst_client_ids(viewer, now: Optional[float] = None) -> list[int]:
    """Clients whose own camera moved within the burst window."""
    now = time.time() if now is None else now
    return [cid for cid, d in tuple(getattr(viewer, "_client_active_deadline", {}).items())
            if now < d]


def burst_active(viewer, client_id: Optional[int] = None, now: Optional[float] = None) -> bool:
    """True while ``client_id`` (None: any client) should render back to back: a
    viewer-wide burst (render settings changed) or that client's camera moved."""
    now = time.time() if now is None else now
    if now < viewer._ui_active_deadline:
        return True
    ids = burst_client_ids(viewer, now)
    return bool(ids) if client_id is None else client_id in ids


class FastRenderer(Renderer):
    """nerfview's Renderer without its two serial costs on every frame.

    * No ``sys.settrace`` line hook around the render, so nerfview's mid-render
      interrupt never fires: the TT backend must not be interrupted while it
      reads a frame from its daemon pipe (stale bytes would break every later
      frame). The hook also taxed every Python line of the render path.
    * JPEG encode + websocket send run on a sender thread, so frame N+1
      renders on the device while frame N is encoded. Latest frame wins: an
      unsent older frame is dropped. The encode is a GIL-free call
      (``pick_jpeg_encoder``) so it does not slow the render thread.
    * During the 1 s UI burst it renders back to back rather than at the
      burst thread's 16 ms tick, so the FPS readout shows the device speed.
    * Frames are pipelined across device stages like the bench (task #457):
      while the camera moves, each render is told the newest camera pose as
      ``viewer.next_camera_state`` and the render_fn hints it to the backend,
      which enqueues that pose's front stages (pfwc) behind this frame's blend
      (render/host/xview.h). The next frame renders exactly that latched pose,
      so the device prefetch always hits; the cost is one frame of pose
      latency. Off with GSPLAT_VIEWER_PIPELINE=0.
    * Sends are paced to the link (task #532): if the last frame is still unsent in
      viser's buffer when the next one is ready (the ssh tunnel is slower than the
      device), the sender waits for it, sends the newest frame, and lowers the
      moving-frame JPEG quality, then scale (``MotionQuality``), until the backlog
      clears. Still frames stay full size at q70. Off with GSPLAT_VIEWER_ADAPTIVE=0 (pacing stays on).
    * ``viewer.frame_post_fn(img, camera_state)`` (if set) runs on the sender
      thread before the encode (the letterbox), off the render thread.

    ``on_frame_sent(t_render_start, t_render_end, t_sent, encode_s)``
    (perf_counter seconds) is called after each send, for the on-screen stats.
    """

    def __init__(self, *args,
                 on_frame_sent: Optional[Callable[[float, float, float, float], None]] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.on_frame_sent = on_frame_sent
        self.jpeg_encoder_name, self._encode_jpeg = pick_jpeg_encoder()
        self.pipeline_frames = pipeline_frames_from_env()
        self.motion_quality = MotionQuality(adaptive_quality_from_env())
        self._last_msg = None  # last frame message handed to viser (pacing, task #532)
        self._latched = None  # pose hinted to the device with the last frame
        tl = timeline_path_from_env()
        self.timeline = FrameTimeline(tl) if tl else None
        self._out = None
        self._out_cv = threading.Condition()
        self._sender = threading.Thread(target=self._send_loop, name="gsplat-send", daemon=True)
        self._sender.start()

    @property
    def running(self) -> bool:
        return getattr(self, "_running", True)

    @running.setter
    def running(self, value: bool) -> None:
        # nerfview's disconnect sets running=False: wake the sender so it exits.
        self._running = value
        cv = getattr(self, "_out_cv", None)
        if cv is not None and not value:
            with cv:
                cv.notify_all()

    def run(self) -> None:
        while self.running:
            while not self.is_prepared_fn():
                time.sleep(0.1)
            if burst_active(self.viewer, self.client.client_id):
                # UI burst: render back to back at device speed instead of
                # waiting for the next 16 ms burst tick (that capped ~60 FPS).
                if not self._render_event.is_set():
                    self.submit(RenderTask("rerender", self.viewer.get_camera_state(self.client)))
            elif not self._render_event.wait(0.2):
                self.submit(RenderTask("static", self.viewer.get_camera_state(self.client)))
            self._render_event.clear()
            task = self._task
            assert task is not None
            if self._state == "high" and task.action == "static":
                continue
            self._state = self.transitions[self._state][task.action]
            assert task.camera_state is not None
            try:
                self.render_once(task)
            except Exception:
                traceback.print_exc()
                os._exit(1)

    def render_once(self, task: RenderTask) -> None:
        """Render one task under the viewer lock and queue it for sending."""
        te = time.perf_counter()
        cs, nxt = task.camera_state, None
        get_cs = getattr(self.viewer, "get_camera_state", None)
        if self.pipeline_frames and task.action in ("move", "rerender") and get_cs is not None:
            # Continuous motion: render the pose hinted last frame (its pfwc is already
            # on the device) and hint the newest pose for the next frame.
            if self._latched is not None:
                cs = self._latched
            nxt = get_cs(self.client)
        with self.lock:
            t0 = time.perf_counter()
            W, H = self._get_img_wh(cs.aspect)
            self.viewer.render_tab_state.viewer_width = W
            self.viewer.render_tab_state.viewer_height = H
            self.viewer.next_camera_state = nxt
            try:
                rendered = self.viewer.render_fn(cs, self.viewer.render_tab_state)
            finally:
                self.viewer.next_camera_state = None
            self.viewer._after_render()
            t1 = time.perf_counter()
            self.viewer.render_tab_state.num_view_rays_per_sec = (W * H) / max(t1 - t0, 1e-10)
        self._latched = nxt
        img, depth = rendered if isinstance(rendered, tuple) else (rendered, None)
        quality = MotionQuality.STILL if task.action in ("static", "update") else MotionQuality.MAX
        with self._out_cv:
            dropped = self._out is not None
            self._out = (img, depth, quality, t0, t1, cs)
            self._out_cv.notify()
        if self.timeline is not None:
            # te: render_once entered, t0/t1: render_fn call, dev: its C++ total ms,
            # drop: an unsent older frame was replaced.
            dev = getattr(getattr(self.viewer.render_fn, "__self__", None), "_render_ms", None)
            self.timeline.add(k="r", te=te, t0=t0, t1=t1, a=task.action, h=nxt is not None,
                              drop=dropped, dev=dev[-1] if dev else None)

    def _frame_pending(self) -> bool:
        """True while the last frame handed to viser still sits unsent in its buffer."""
        msg = self._last_msg
        if msg is None:
            return False
        try:
            buf = self.client.scene._websock_interface.get_message_buffer()
            mid = buf.id_from_redundancy_key.get(msg.redundancy_key())
            return mid is not None and buf.message_from_id.get(mid) is msg
        except AttributeError:
            return False

    def _send_loop(self) -> None:
        while self.running:
            with self._out_cv:
                while self._out is None and self.running:
                    self._out_cv.wait(0.5)  # recheck running: no leak on disconnect
                if self._out is None:
                    break
                img, depth, quality, t0, t1, cs = self._out
                self._out = None
            t_take = time.perf_counter()
            backlogged = self._frame_pending()
            if backlogged:
                # The link is slower than the device: wait (bounded) until viser takes the
                # last frame, then send the newest one instead of having viser drop it.
                t_end = time.perf_counter() + _PACE_MAX_WAIT_SEC
                while self._frame_pending() and self.running and time.perf_counter() < t_end:
                    time.sleep(0.001)
                with self._out_cv:
                    if self._out is not None:
                        img, depth, quality, t0, t1, cs = self._out
                        self._out = None
            scale = 1.0
            if quality != MotionQuality.STILL:
                scale, quality = self.motion_quality.update(backlogged)
            te = time.perf_counter()
            try:
                post = getattr(self.viewer, "frame_post_fn", None)
                if post is not None:
                    img = post(img, cs)
                if depth is None and img.dtype == np.uint8 and self.jpeg_encoder_name != "viser":
                    # set_background_image minus viser's encode: same message.
                    data = self._encode_jpeg(np.ascontiguousarray(downscale(img, scale)), quality)
                    msg = BackgroundImageMessage(format="jpeg", rgb_data=data, depth_data=None)
                    self.client.scene._websock_interface.queue_message(msg)
                    self._last_msg = msg
                else:
                    self.client.scene.set_background_image(
                        img, format="jpeg", jpeg_quality=quality, depth=depth)
            except Exception:
                traceback.print_exc()
                continue
            if self.on_frame_sent is not None:
                t_sent = time.perf_counter()
                self.on_frame_sent(t0, t1, t_sent, t_sent - te)
            if self.timeline is not None:
                # tt: taken by the sender, bl: last frame still unsent (paced), te/ts: encode
                # start / queued to viser, q/sc: JPEG quality and scale.
                self.timeline.add(k="s", t0=t0, tt=t_take, bl=backlogged, te=te,
                                  ts=time.perf_counter(), q=quality, sc=scale)
                self.timeline.maybe_flush()


class GsplatViewer(nerfview.Viewer):
    """Nerfview wrapper without the redundant Viewer Res slider.

    The upstream *Viewer Res* cap only affects nerfview's internal viewport
    sizing heuristic; gsplat renders at the explicit *Render Res* instead.
    """

    def __init__(
        self,
        *,
        default_render_width: int = 1024,
        default_render_height: int = 1024,
        on_frame_sent: Optional[Callable[[float, float, float, float], None]] = None,
        **kwargs,
    ) -> None:
        self._on_frame_sent = on_frame_sent
        # Set by FastRenderer around each render_fn call (task #457): the pose the next
        # frame will render, or None. The render_fn hints it to the backend.
        self.next_camera_state = None
        # Optional (img, camera_state) -> img run on the sender thread (letterbox).
        self.frame_post_fn: Optional[Callable] = None
        self._default_render_width = default_render_width
        self._default_render_height = default_render_height
        self._ui_active_deadline = 0.0  # all clients (render settings changed)
        # Per client: its own camera moved. Only that client renders back to back, so an
        # idle page in another tab does not take half the device (task #459).
        self._client_active_deadline: dict[int, float] = {}
        self._burst_running = True
        jpeg_encoder_from_env()  # warn on a bad GSPLAT_VIEWER_JPEG now, not at first connect
        super().__init__(**kwargs)
        self._burst_thread = threading.Thread(
            target=self._ui_burst_loop,
            name="gsplat-ui-burst",
            daemon=True,
        )
        self._burst_thread.start()

    def mark_ui_active(self, client_id: Optional[int] = None) -> None:
        """Extend the post-action render burst window (now + 1 s): for every client,
        or only for ``client_id`` (its camera moved)."""
        if client_id is None:
            self._ui_active_deadline = time.time() + _UI_BURST_SEC
        else:
            self._client_active_deadline[client_id] = time.time() + _UI_BURST_SEC

    def ui_active(self) -> bool:
        """True while any client is in its render burst."""
        return burst_active(self)

    # Back-compat alias.
    mark_camera_active = mark_ui_active

    def _ui_burst_loop(self) -> None:
        while self._burst_running:
            if self._renderers:
                if time.time() < self._ui_active_deadline:
                    self.rerender(None)
                else:
                    clients = self.server.get_clients()
                    for client_id in burst_client_ids(self):
                        r, c = self._renderers.get(client_id), clients.get(client_id)
                        if r is not None and c is not None:
                            r.submit(RenderTask("rerender", self.get_camera_state(c)))
            time.sleep(_UI_BURST_POLL_SEC)

    def _connect_client(self, client: viser.ClientHandle) -> None:
        client_id = client.client_id
        self._renderers[client_id] = FastRenderer(
            viewer=self, client=client, lock=self.lock, on_frame_sent=self._on_frame_sent,
        )
        self._renderers[client_id].start()

        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            self._last_move_time = time.time()
            self.mark_ui_active(client_id)
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

    def _disconnect_client(self, client: viser.ClientHandle) -> None:
        self._client_active_deadline.pop(client.client_id, None)
        super()._disconnect_client(client)

    def stop_burst(self) -> None:
        self._burst_running = False

    def _init_rendering_tab(self) -> None:
        self.render_tab_state = RenderTabState(
            render_width=self._default_render_width,
            render_height=self._default_render_height,
        )
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self) -> None:
        assert self.render_tab_state is not None
        assert self._rendering_folder is not None

        extra_handles = self._rendering_tab_handles.copy()
        if self.mode == "training":
            extra_handles.update(self._training_tab_handles)
        handles = populate_general_render_tab(
            self.server,
            output_dir=self.output_dir if self.output_dir is not None else Path("./results"),
            folder=self._rendering_folder,
            render_tab_state=self.render_tab_state,
            extra_handles=extra_handles,
        )
        self._rendering_tab_handles.update(handles)

        render_res = handles["render_res_vec2"]
        render_res.value = (
            self._default_render_width,
            self._default_render_height,
        )
        self.render_tab_state.render_width = self._default_render_width
        self.render_tab_state.render_height = self._default_render_height

        @render_res.on_update
        def _on_render_res(_event: viser.GuiEvent) -> None:
            self.render_tab_state.render_width = int(render_res.value[0])
            self.render_tab_state.render_height = int(render_res.value[1])
            self.mark_ui_active()
            self.rerender(_event)
