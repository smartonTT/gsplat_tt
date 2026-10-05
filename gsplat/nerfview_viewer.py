"""Gsplat-specific nerfview integration."""
from __future__ import annotations

import os
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import nerfview
import viser
from nerfview._renderer import Renderer, RenderTask
from nerfview.render_panel import RenderTabState, populate_general_render_tab

# Keep rendering for 1 s after any UI action so the FPS readout settles.
_UI_BURST_SEC = 1.0
_UI_BURST_POLL_SEC = 0.016


class FastRenderer(Renderer):
    """nerfview's Renderer without its two serial costs on every frame.

    * No ``sys.settrace`` line hook around the render. It only served the
      mid-render interrupt, which ``viser_patches`` disables anyway, and it
      taxed every Python line of the render path.
    * JPEG encode + websocket send run on a sender thread, so frame N+1
      renders on the device while frame N is encoded. Latest frame wins: an
      unsent older frame is dropped.
    * During the 1 s UI burst it renders back to back rather than at the
      burst thread's 16 ms tick, so the FPS readout shows the device speed.

    ``on_frame_sent(t_render_start, t_render_end, t_sent)`` (perf_counter
    seconds) is called after each send, for the viewer's on-screen stats.
    """

    def __init__(self, *args, on_frame_sent: Optional[Callable[[float, float, float], None]] = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.on_frame_sent = on_frame_sent
        self._out = None
        self._out_cv = threading.Condition()
        self._sender = threading.Thread(target=self._send_loop, name="gsplat-send", daemon=True)
        self._sender.start()

    def run(self) -> None:
        while self.running:
            while not self.is_prepared_fn():
                time.sleep(0.1)
            if time.time() < self.viewer._ui_active_deadline:
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
        with self.lock:
            t0 = time.perf_counter()
            W, H = self._get_img_wh(task.camera_state.aspect)
            self.viewer.render_tab_state.viewer_width = W
            self.viewer.render_tab_state.viewer_height = H
            rendered = self.viewer.render_fn(task.camera_state, self.viewer.render_tab_state)
            self.viewer._after_render()
            t1 = time.perf_counter()
            self.viewer.render_tab_state.num_view_rays_per_sec = (W * H) / max(t1 - t0, 1e-10)
        img, depth = rendered if isinstance(rendered, tuple) else (rendered, None)
        quality = 70 if task.action in ("static", "update") else 40
        with self._out_cv:
            self._out = (img, depth, quality, t0, t1)
            self._out_cv.notify()

    def _send_loop(self) -> None:
        while self.running:
            with self._out_cv:
                while self._out is None:
                    self._out_cv.wait()
                img, depth, quality, t0, t1 = self._out
                self._out = None
            try:
                self.client.scene.set_background_image(
                    img, format="jpeg", jpeg_quality=quality, depth=depth)
            except Exception:
                traceback.print_exc()
                continue
            if self.on_frame_sent is not None:
                self.on_frame_sent(t0, t1, time.perf_counter())


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
        on_frame_sent: Optional[Callable[[float, float, float], None]] = None,
        **kwargs,
    ) -> None:
        self._on_frame_sent = on_frame_sent
        self._default_render_width = default_render_width
        self._default_render_height = default_render_height
        self._ui_active_deadline = 0.0
        self._burst_running = True
        super().__init__(**kwargs)
        self._burst_thread = threading.Thread(
            target=self._ui_burst_loop,
            name="gsplat-ui-burst",
            daemon=True,
        )
        self._burst_thread.start()

    def mark_ui_active(self) -> None:
        """Extend the post-action render burst window (now + 1 s)."""
        self._ui_active_deadline = time.time() + _UI_BURST_SEC

    # Back-compat alias.
    mark_camera_active = mark_ui_active

    def _ui_burst_loop(self) -> None:
        while self._burst_running:
            if self._renderers and time.time() < self._ui_active_deadline:
                self.rerender(None)
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
            self.mark_ui_active()
            with self.server.atomic():
                camera_state = self.get_camera_state(client)
                self._renderers[client_id].submit(RenderTask("move", camera_state))

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
