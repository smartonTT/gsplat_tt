"""Gsplat-specific nerfview integration."""
from __future__ import annotations

from pathlib import Path

import nerfview
import viser
from nerfview.render_panel import RenderTabState, populate_general_render_tab


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
        super().__init__(**kwargs)

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
            self.rerender(_event)
