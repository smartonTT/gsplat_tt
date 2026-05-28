"""Gsplat-specific nerfview integration."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import nerfview
import viser
from nerfview._renderer import Renderer, RenderTask
from nerfview.render_panel import RenderTabState, populate_general_render_tab

# Keep rendering for 1 s after any UI action so the FPS readout settles.
_UI_BURST_SEC = 1.0
_UI_BURST_POLL_SEC = 0.016


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
        **kwargs,
    ) -> None:
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
        self._renderers[client_id] = Renderer(
            viewer=self, client=client, lock=self.lock,
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
