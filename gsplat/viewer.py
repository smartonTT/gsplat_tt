"""Browser-based interactive viewer for 3D Gaussian Splatting scenes."""
from __future__ import annotations

import os
import socket
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import viser
import nerfview

from backends import get_backend
from gsplat.camera_controls import ClientCameraController
from gsplat.data_structures import Gaussians
from gsplat.letterbox import letterbox_image
from gsplat.nerfview_viewer import GsplatViewer
from gsplat.pipeline import Pipeline, RenderResult, format_timings
from gsplat.utils import c2w_to_w2c


# Tile size matches the kernel (32x32) for both backends so CPU and
# kernel renders use the same tiling and are directly comparable.
TILE_SIZE = 32

# Pipeline stages, in execution order — used to lay out the benchmark table.
_STAGE_KEYS = ("project", "tile_assign", "sort", "blend")


class _FrameSample(NamedTuple):
    """One frame's per-stage timings plus the (W, H) it was rendered at."""
    timings: dict[str, float]
    sub_timings: dict[str, float]
    width: int
    height: int


def _median_by_key(
    rows: list[dict[str, float]], keys: list[str] | tuple[str, ...]
) -> dict[str, float]:
    """Median of `row[key]` across rows, skipping rows where the key is absent."""
    out: dict[str, float] = {}
    for key in keys:
        values = [r[key] for r in rows if key in r]
        if values:
            out[key] = statistics.median(values)
    return out


def _snap_render_dim(value: int) -> int:
    return max(TILE_SIZE, (value // TILE_SIZE) * TILE_SIZE)


class GaussianViewer:
    """Interactive viewer for 3D Gaussian Splatting scenes.

    Wraps nerfview/viser around a `gsplat.pipeline.Pipeline`. The pipeline
    owns the chosen backend and produces a `RenderResult` for every frame,
    including per-stage timings — so adding a new backend (CUDA, ...)
    requires no changes here as long as it's registered in
    `backends/REGISTRY`.
    """

    def __init__(
        self,
        gaussians: Gaussians,
        host: str = "0.0.0.0",
        port: int = 8080,
        backend: str = "cpu",
        render_width: int = 1024,
        render_height: int = 1024,
        force_square: int | None = None,
        verbose: bool = False,
        scene_path: str | None = None,
        # Back-compat alias: if passed, overrides render_width/height.
        max_resolution: int | None = None,
    ):
        self.gaussians = gaussians
        self.backend_name = backend
        self.scene_path = scene_path
        if max_resolution is not None:
            render_width = max_resolution
            render_height = max_resolution
        self.render_width = render_width
        self.render_height = render_height
        if force_square is not None:
            if force_square <= 0 or force_square % TILE_SIZE != 0:
                raise ValueError(
                    f"force_square={force_square} must be a positive multiple "
                    f"of TILE_SIZE={TILE_SIZE}"
                )
        self.force_square = force_square
        self.verbose = verbose

        self.pipeline = Pipeline(get_backend(backend, verbose=verbose),
                                 tile_size=TILE_SIZE)

        self._frame_samples: list[_FrameSample] = []
        self._session_start = datetime.now()
        self._camera_controllers: dict[int, ClientCameraController] = {}

        means = gaussians.means.numpy()
        opacities = gaussians.opacities.numpy()
        visible = means[opacities > 0.1]
        if visible.shape[0] < 100:
            visible = means
        lo = np.percentile(visible, 5, axis=0)
        hi = np.percentile(visible, 95, axis=0)
        self._scene_center = (lo + hi) * 0.5
        self._camera_distance = float(np.linalg.norm(hi - lo)) * 1.2

        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.scene.world_axes.visible = True

        self._stats_display = self.server.gui.add_markdown("**FPS:** --")
        self.server.gui.add_markdown(
            f"**{socket.gethostname()}**",
            order=-1000,
        )
        self._fps_controls = self.server.gui.add_checkbox(
            "FPS controls",
            initial_value=False,
            hint=(
                "Off: left-drag turntable orbit around the scene center. "
                "On: first-person look + WASD/QE movement."
            ),
        )

        self.viewer = GsplatViewer(
            server=self.server,
            render_fn=self._render_fn,
            mode="rendering",
            default_render_width=render_width,
            default_render_height=render_height,
        )

        @self._fps_controls.on_update
        def _on_control_mode(_event: viser.GuiEvent) -> None:
            mode = (
                ClientCameraController.FPS
                if self._fps_controls.value
                else ClientCameraController.ORBIT
            )
            for client in self.server.get_clients().values():
                controller = self._camera_controllers.get(client.client_id)
                if controller is None:
                    continue
                controller.set_mode(mode)
                if mode == ClientCameraController.FPS:
                    controller.snap_to_fps(client.camera)
                else:
                    controller.snap_to_orbit(client.camera)

        center = self._scene_center
        distance = self._camera_distance

        @self.server.on_client_connect
        def _on_client_connect(client: viser.ClientHandle) -> None:
            controller = ClientCameraController(center, distance)
            mode = (
                ClientCameraController.FPS
                if self._fps_controls.value
                else ClientCameraController.ORBIT
            )
            controller.set_mode(mode)
            self._camera_controllers[client.client_id] = controller

            client.camera.position = center + np.array([0.0, 0.0, distance])
            client.camera.look_at = center
            controller.apply(client.camera)

            @client.camera.on_update
            def _on_camera_update(camera: viser.CameraHandle) -> None:
                controller.set_mode(
                    ClientCameraController.FPS
                    if self._fps_controls.value
                    else ClientCameraController.ORBIT
                )
                controller.apply(camera)

        self._running = False
        if self.force_square is not None:
            flags = [f"backend={backend}",
                     f"force_square={self.force_square}x{self.force_square}"]
        else:
            flags = [
                f"backend={backend}",
                f"render={render_width}x{render_height}",
            ]
        if verbose:
            flags.append("verbose")
        print(
            f"Viewer running at http://localhost:{port} ({', '.join(flags)})",
            flush=True,
        )

    def _resolve_render_size(
        self, render_tab_state: nerfview.RenderTabState
    ) -> tuple[int, int, int, int]:
        """Pick render (W, H) and viewport (req_W, req_H) for this frame."""
        req_W = render_tab_state.viewer_width
        req_H = render_tab_state.viewer_height

        if self.force_square is not None:
            W = H = self.force_square
            return req_W, req_H, W, H

        W = _snap_render_dim(render_tab_state.render_width)
        H = _snap_render_dim(render_tab_state.render_height)
        return req_W, req_H, W, H

    def _render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Render callback invoked by nerfview for each frame."""
        wall_start = time.perf_counter()
        if self.verbose:
            print(
                f"[render-enter] viewer={render_tab_state.viewer_width}x"
                f"{render_tab_state.viewer_height} "
                f"render={render_tab_state.render_width}x"
                f"{render_tab_state.render_height}",
                flush=True,
            )

        req_W, req_H, W, H = self._resolve_render_size(render_tab_state)
        if W <= 0 or H <= 0:
            return np.zeros((max(req_H, 1), max(req_W, 1), 3), dtype=np.uint8)

        extrinsics = c2w_to_w2c(camera_state.c2w)
        intrinsics = torch.tensor(camera_state.get_K((W, H)), dtype=torch.float32)

        result = self.pipeline.render(self.gaussians, extrinsics, intrinsics, H, W)

        if result.image is None:
            image_np = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            image_np = (np.clip(result.image, 0.0, 1.0) * 255).astype(np.uint8)
            self._frame_samples.append(_FrameSample(
                timings=dict(result.timings),
                sub_timings=dict(result.sub_timings),
                width=W,
                height=H,
            ))

        display = letterbox_image(image_np, req_W, req_H)

        wall_elapsed = time.perf_counter() - wall_start
        if self.verbose:
            self._log_verbose(req_W, req_H, W, H, result, wall_elapsed)
        self._update_stats(wall_elapsed, W, H, result.num_visible)
        return display

    def _log_verbose(
        self,
        req_W: int,
        req_H: int,
        W: int,
        H: int,
        result: RenderResult,
        wall_elapsed: float,
    ) -> None:
        print(
            f"[render] viewport={req_W}x{req_H} render={W}x{H}  "
            f"visible={result.num_visible}  sorted={result.num_entries}  "
            f"backend={self.backend_name}",
            flush=True,
        )
        print(format_timings(result), flush=True)
        print(f"[wall]  {wall_elapsed * 1000:6.1f} ms", flush=True)

    def _update_stats(
        self, elapsed: float, width: int, height: int, num_visible: int,
    ) -> None:
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        self._stats_display.content = (
            f"**FPS:** {fps:.1f} | "
            f"**Render:** {width}x{height} | "
            f"**Visible:** {num_visible:,}"
        )

    @property
    def _scene_name(self) -> str:
        return Path(self.scene_path).stem if self.scene_path else "synthetic"

    def _aggregate_session_medians(
        self,
    ) -> tuple[dict[str, float], dict[str, float], list[str], float, tuple[int, int]]:
        stage_rows = [s.timings for s in self._frame_samples]
        sub_rows = [s.sub_timings for s in self._frame_samples]
        sub_keys = list(dict.fromkeys(k for s in sub_rows for k in s))

        stage_medians = _median_by_key(stage_rows, _STAGE_KEYS)
        sub_medians = _median_by_key(sub_rows, sub_keys)
        median_total = _median_by_key(stage_rows, ("total",)).get("total", 0.0)
        modal_resolution = statistics.mode(
            (s.width, s.height) for s in self._frame_samples
        )
        return stage_medians, sub_medians, sub_keys, median_total, modal_resolution

    def _benchmark_filename(self) -> str:
        ts = self._session_start.strftime("%Y-%m-%d_%H-%M-%S")
        if self.force_square is not None:
            tag = f"{self.force_square}x{self.force_square}"
        else:
            tag = f"{self.render_width}x{self.render_height}"
        return f"{self._scene_name}_{self.backend_name}_{tag}_{ts}.md"

    def _render_benchmark_md(
        self,
        stage_medians: dict[str, float],
        sub_medians: dict[str, float],
        sub_keys: list[str],
        median_total: float,
        modal_resolution: tuple[int, int],
    ) -> str:
        fps = 1000.0 / median_total if median_total > 0 else 0.0
        ts = self._session_start
        res_w, res_h = modal_resolution
        if self.force_square is not None:
            resolution = f"{res_w}x{res_h} (force_square={self.force_square})"
        else:
            resolution = (
                f"{res_w}x{res_h} "
                f"(default={self.render_width}x{self.render_height})"
            )

        rows = ["| Stage | ms |", "|---|---|"]
        for stage in _STAGE_KEYS:
            if stage in stage_medians:
                rows.append(f"| {stage} | {stage_medians[stage]:.2f} |")
            for sub in sub_keys:
                if not sub.startswith(f"{stage}.") or sub not in sub_medians:
                    continue
                depth = sub.count(".")
                leaf = sub.rsplit(".", 1)[-1]
                indent = "&nbsp;" * 4 * depth
                rows.append(f"| {indent}└─ {leaf} | {sub_medians[sub]:.2f} |")
        rows.append(f"| **Total** | **{median_total:.2f}** |")
        rows.append(f"| **FPS** | **{fps:.2f}** |")

        return "\n".join([
            f"# Benchmark: {self._scene_name}",
            "",
            f"- **Date:** {ts.strftime('%Y-%m-%d')}",
            f"- **Time:** {ts.strftime('%H:%M:%S')}",
            f"- **Backend:** {self.backend_name}",
            f"- **Scene:** {self.scene_path or '(synthetic)'}",
            f"- **Gaussians:** {self.gaussians.num_gaussians:,}",
            f"- **Resolution:** {resolution}",
            f"- **Frames sampled:** {len(self._frame_samples)}",
            "",
            "## Performance (median across frames)",
            "",
            *rows,
            "",
        ])

    def _write_benchmark(self) -> None:
        if not self._frame_samples:
            return
        out_dir = Path("benchmarks")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / self._benchmark_filename()
        path.write_text(self._render_benchmark_md(*self._aggregate_session_medians()))
        print(f"Benchmark written to {path}", flush=True)

    def run(self) -> None:
        """Block the main thread to keep the viewer alive."""
        self._running = True
        try:
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nViewer stopped.")
        finally:
            self._write_benchmark()
            self.pipeline.close()
            os._exit(0)

    def stop(self) -> None:
        """Signal the viewer to stop."""
        self._running = False
