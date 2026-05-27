"""Browser-based interactive viewer for 3D Gaussian Splatting scenes."""
from __future__ import annotations

import json
import os
import socket
import statistics
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import viser
import viser.transforms as vt
import nerfview

from backends import get_backend
# `viser_patches` MUST be imported before any viser/nerfview machinery is
# touched so the monkey-patches are in place when the server is created.
import gsplat.viser_patches  # noqa: F401
from gsplat.camera_controls import ClientCameraController
from gsplat.data_structures import Gaussians
from gsplat.letterbox import letterbox_for_aspect
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


def _failure_pattern(width: int, height: int) -> np.ndarray:
    """Visible debug image for when the pipeline raises.

    Black-on-magenta diagonal stripes — easy to spot in the browser, so a
    silent kernel/Python failure can't masquerade as "the camera moved out
    of frame". The exception is already printed via ``traceback.print_exc``.
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    stripe = ((xx + yy) // 24) % 2 == 0
    canvas[stripe] = (255, 0, 255)
    return canvas


# Initial reference frame for the orbit math:
#   world up        = -Y (scenes are authored with +Y down — see
#                     gsplat.camera_controls)
#   initial offset  = +Z * distance (camera sits in front of the scene
#                     center, looking back along -Z)
#   initial right   = +X (consistent with up × forward in the orbit frame)
_WORLD_UP_INITIAL = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_OFFSET_DIR_INITIAL = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_RIGHT_INITIAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix for ``angle`` radians around ``axis``."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    c, s = float(np.cos(angle)), float(np.sin(angle))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _orbit_pose(
    center: np.ndarray,
    distance: float,
    azim_deg: float,
    elev_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spherical → camera (position, look_at, up_direction).

    Pitch is applied around the camera's *current* right axis (i.e. the
    initial right axis rotated by yaw), so elevation always tilts up/down
    relative to the camera, not the world. The world up vector is rotated
    along with the camera so values past ±90° (i.e. through the pole) stay
    coherent — at elev = ±180° the camera ends up upside-down behind the
    target, which is what you want for "keep dragging past the pole".
    """
    azim_rad = float(np.radians(azim_deg))
    elev_rad = float(np.radians(elev_deg))

    yaw_R = _rotation_matrix(_WORLD_UP_INITIAL, azim_rad)
    right_after_yaw = yaw_R @ _RIGHT_INITIAL
    pitch_R = _rotation_matrix(right_after_yaw, elev_rad)
    rot = pitch_R @ yaw_R

    offset = rot @ (_OFFSET_DIR_INITIAL * float(distance))
    position = np.asarray(center, dtype=np.float64) + offset
    look_at = np.asarray(center, dtype=np.float64)
    up_direction = rot @ _WORLD_UP_INITIAL
    return position, look_at, up_direction


def _repo_cameras_json() -> Path:
    return Path(__file__).resolve().parent.parent / "benchmarks" / "cameras.json"


def _load_scene_preset(
    scene_path: str | None,
) -> tuple[np.ndarray, float] | None:
    """Return (hero c2w, fov_deg) from cameras.json for a fixed iconic view."""
    if not scene_path:
        return None
    cam_file = _repo_cameras_json()
    if not cam_file.exists():
        return None
    ply_name = Path(scene_path).name
    cams = json.loads(cam_file.read_text())
    for entry in cams.values():
        if Path(entry.get("ply", "")).name != ply_name:
            continue
        hero = (entry.get("views") or {}).get("hero") or {}
        c2w = hero.get("c2w")
        if c2w is not None:
            fov_deg = float(entry.get("fov_deg", 50.0))
            return np.asarray(c2w, dtype=np.float64), fov_deg
    return None


def _intrinsics_from_fov(W: int, H: int, fov_deg: float) -> torch.Tensor:
    """Square-pixel K; fov_deg applies to the longer image dimension."""
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([
        [f, 0.0, W * 0.5],
        [0.0, f, H * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return torch.from_numpy(K)


def _fov_deg_to_viser_vertical(fov_deg: float, aspect: float) -> float:
    """Convert cameras.json fov (longer dim) to viser's vertical fov radians."""
    if aspect >= 1.0:
        fov_h = np.deg2rad(fov_deg)
        return float(2.0 * np.arctan(np.tan(fov_h * 0.5) / aspect))
    return float(np.deg2rad(fov_deg))


def _c2w_to_viser_pose(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenCV c2w (cols: right, down, forward) → viser camera tuple."""
    eye = c2w[:3, 3].copy()
    forward = c2w[:3, 2]
    fn = float(np.linalg.norm(forward))
    if fn < 1e-9:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / fn
    look_at = eye + forward
    down = c2w[:3, 1]
    dn = float(np.linalg.norm(down))
    up = (-down / dn) if dn > 1e-9 else _WORLD_UP_INITIAL.copy()
    return eye, look_at, up


def _apply_opencv_c2w(
    client: viser.ClientHandle,
    c2w: np.ndarray,
    *,
    fov_deg: float | None = None,
) -> None:
    """Pin viser to an OpenCV c2w (cols: right, down, forward).

    viser's wxyz setter follows the same convention (see viser CameraHandle).
    Set ``wxyz`` last so look_at/position do not re-derive a different basis.
    """
    eye = c2w[:3, 3].copy()
    rot = c2w[:3, :3].copy()
    forward = rot[:, 2]
    fn = float(np.linalg.norm(forward))
    if fn > 1e-9:
        forward = forward / fn
    client.camera.position = eye
    client.camera.look_at = eye + forward
    client.camera.wxyz = vt.SO3.from_matrix(rot).wxyz.astype(np.float64)
    if fov_deg is not None and client.camera.aspect > 0:
        client.camera.fov = _fov_deg_to_viser_vertical(fov_deg, client.camera.aspect)


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

        preset = _load_scene_preset(scene_path)
        if preset is not None:
            preset_c2w, preset_fov_deg = preset
            self._preset_c2w: np.ndarray | None = preset_c2w
            self._preset_fov_deg: float | None = preset_fov_deg
            self._preset_active = True
            pos, look, _up = _c2w_to_viser_pose(preset_c2w)
            self._scene_center = look
            self._camera_distance = float(np.linalg.norm(pos - look))
        else:
            self._preset_c2w = None
            self._preset_fov_deg = None
            self._preset_active = False

        self._programmatic_camera = False
        self._ignore_camera_updates_until = 0.0

        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.scene.world_axes.visible = True

        self.server.gui.add_markdown(
            f"**{socket.gethostname()}**",
            order=-1000,
        )
        self._stats_display = self.server.gui.add_markdown("**FPS:** --")

        # Explicit camera controls (sliders + buttons). Mouse drag is still
        # active for fine motion, but its react-three drei orbit clamps
        # vertical at ±90° and gimbal-locks at the pole — so the sliders
        # are the only way to rotate continuously. Elevation accepts the
        # full ±180° on purpose; values outside [−90, +90] put the camera
        # past the pole (looking up while world up is below) and the up
        # vector is rotated with it so the scene doesn't suddenly invert.
        self._control_folder = self.server.gui.add_folder(
            "Camera",
            order=-900,
        )
        with self._control_folder:
            self._azim_slider = self.server.gui.add_slider(
                "Azimuth (°)",
                min=0.0, max=360.0, step=1.0, initial_value=220.0,
                hint="Rotate around the world up axis. 0° = front, 90° = right.",
            )
            self._elev_slider = self.server.gui.add_slider(
                "Elevation (°)",
                min=-180.0, max=180.0, step=1.0, initial_value=0.0,
                hint=(
                    "Tilt up/down. Goes past ±90° (the mouse-drag "
                    "clamp) — the camera flips through the pole."
                ),
            )
            self._dist_slider = self.server.gui.add_slider(
                "Distance",
                min=max(self._camera_distance * 0.05, 1e-3),
                max=self._camera_distance * 5.0,
                step=self._camera_distance * 0.01,
                initial_value=self._camera_distance,
                hint="Zoom in/out along the current camera-to-target ray.",
            )
            self._reset_view_button = self.server.gui.add_button(
                "Reset view",
                hint=(
                    "Snap back to the iconic hero camera."
                    if self._preset_c2w is not None
                    else "Snap azimuth/elevation/distance back to defaults."
                ),
            )

        if self._preset_c2w is not None:
            self.server.gui.add_markdown(
                "**Iconic view** — drag to explore; Reset restores hero camera.",
                order=-899,
            )

        self.viewer = GsplatViewer(
            server=self.server,
            render_fn=self._render_fn,
            mode="rendering",
            default_render_width=render_width,
            default_render_height=render_height,
        )

        center = self._scene_center
        default_distance = self._camera_distance

        # Suppress flag prevents the slider on_update -> camera write ->
        # client camera message -> our on_camera_update -> slider sync ->
        # on_update feedback loop.
        self._slider_suppress = False
        # Last (azim°, elev°, distance) the GaussianViewer applied to the
        # camera. Read by the slider on_update to compute deltas, and by
        # the camera-update callback to detect "user dragged away from
        # what the sliders say".
        self._last_orbit_state: tuple[float, float, float] = (220.0, 0.0, default_distance)

        def _apply_opencv_to_all_clients(c2w: np.ndarray) -> None:
            self._programmatic_camera = True
            self._slider_suppress = True
            self._ignore_camera_updates_until = time.perf_counter() + 0.5
            try:
                for client in self.server.get_clients().values():
                    _apply_opencv_c2w(
                        client,
                        c2w,
                        fov_deg=self._preset_fov_deg,
                    )
            finally:
                self._slider_suppress = False
                self._programmatic_camera = False

        def _apply_pose_to_all_clients(
            position: np.ndarray,
            look_at: np.ndarray,
            up_dir: np.ndarray,
        ) -> None:
            self._programmatic_camera = True
            self._slider_suppress = True
            try:
                for client in self.server.get_clients().values():
                    client.camera.position = position
                    client.camera.look_at = look_at
                    client.camera.up_direction = up_dir
            finally:
                self._slider_suppress = False
                self._programmatic_camera = False

        def _apply_orbit_to_all_clients(
            azim_deg: float, elev_deg: float, dist: float
        ) -> None:
            for client in self.server.get_clients().values():
                position, look_at, up_dir = _orbit_pose(
                    center, dist, azim_deg, elev_deg,
                )
                self._slider_suppress = True
                try:
                    client.camera.position = position
                    client.camera.look_at = look_at
                    client.camera.up_direction = up_dir
                finally:
                    self._slider_suppress = False
            self._last_orbit_state = (azim_deg, elev_deg, dist)

        def _apply_default_view() -> None:
            if self._preset_c2w is not None:
                self._preset_active = True
                _apply_opencv_to_all_clients(self._preset_c2w)
            else:
                self._preset_active = False
                _apply_orbit_to_all_clients(220.0, 0.0, default_distance)

        @self._azim_slider.on_update
        def _on_azim(_event: viser.GuiEvent) -> None:
            if self._slider_suppress:
                return
            self._preset_active = False
            _apply_orbit_to_all_clients(
                float(self._azim_slider.value),
                float(self._elev_slider.value),
                float(self._dist_slider.value),
            )

        @self._elev_slider.on_update
        def _on_elev(_event: viser.GuiEvent) -> None:
            if self._slider_suppress:
                return
            self._preset_active = False
            _apply_orbit_to_all_clients(
                float(self._azim_slider.value),
                float(self._elev_slider.value),
                float(self._dist_slider.value),
            )

        @self._dist_slider.on_update
        def _on_dist(_event: viser.GuiEvent) -> None:
            if self._slider_suppress:
                return
            self._preset_active = False
            _apply_orbit_to_all_clients(
                float(self._azim_slider.value),
                float(self._elev_slider.value),
                float(self._dist_slider.value),
            )

        @self._reset_view_button.on_click
        def _on_reset(_event: viser.GuiEvent) -> None:
            if self._preset_c2w is None:
                self._slider_suppress = True
                try:
                    self._azim_slider.value = 220.0
                    self._elev_slider.value = 0.0
                    self._dist_slider.value = default_distance
                finally:
                    self._slider_suppress = False
            _apply_default_view()

        @self.server.on_client_connect
        def _on_client_connect(client: viser.ClientHandle) -> None:
            controller = ClientCameraController(center, default_distance)
            controller.set_mode(ClientCameraController.ORBIT)
            self._camera_controllers[client.client_id] = controller

            if self._preset_c2w is not None:
                self._preset_active = True
                self._programmatic_camera = True
                self._slider_suppress = True
                self._ignore_camera_updates_until = time.perf_counter() + 0.5
                try:
                    _apply_opencv_c2w(
                        client,
                        self._preset_c2w,
                        fov_deg=self._preset_fov_deg,
                    )
                finally:
                    self._slider_suppress = False
                    self._programmatic_camera = False
            else:
                position, look_at, up_dir = _orbit_pose(
                    center,
                    float(self._dist_slider.value),
                    float(self._azim_slider.value),
                    float(self._elev_slider.value),
                )
                self._slider_suppress = True
                try:
                    client.camera.position = position
                    client.camera.look_at = look_at
                    client.camera.up_direction = up_dir
                finally:
                    self._slider_suppress = False

            @client.camera.on_update
            def _on_camera_update(_cam: viser.CameraHandle) -> None:
                if (
                    self._programmatic_camera
                    or self._slider_suppress
                    or time.perf_counter() < self._ignore_camera_updates_until
                ):
                    return
                if self._preset_c2w is not None:
                    self._preset_active = False
                controller.on_camera_update(client.camera)

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
    ) -> tuple[int, int]:
        """Pick the (W, H) the kernel should render at, in pixels.

        We ignore ``render_tab_state.viewer_width/height`` on purpose:
        nerfview shrinks those during interactive drag (the "low_move"
        state) which would force a low-resolution preview render. The user
        always wants the explicit Render Res; the browser handles scaling
        the result to the viewport. Aspect ratio in the browser is then
        preserved by letterboxing in :py:meth:`_render_fn`.
        """
        if self.force_square is not None:
            return self.force_square, self.force_square
        W = _snap_render_dim(render_tab_state.render_width)
        H = _snap_render_dim(render_tab_state.render_height)
        return W, H

    def _render_fn(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        """Render callback invoked by nerfview for each frame.

        Always renders at the user's Render Res — never the shrunk
        ``viewer_width/height`` that nerfview hands us during drag — and
        letterboxes the result into the camera aspect so the browser can
        scale to fill without distortion.
        """
        wall_start = time.perf_counter()
        W, H = self._resolve_render_size(render_tab_state)
        if self.verbose:
            print(
                f"[render-enter] viewer={render_tab_state.viewer_width}x"
                f"{render_tab_state.viewer_height} "
                f"render={render_tab_state.render_width}x"
                f"{render_tab_state.render_height}  "
                f"-> kernel {W}x{H}  aspect={camera_state.aspect:.3f}",
                flush=True,
            )

        if W <= 0 or H <= 0:
            return np.zeros((max(H, 1), max(W, 1), 3), dtype=np.uint8)

        render_c2w = camera_state.c2w
        intrinsics = torch.tensor(
            camera_state.get_K((W, H)), dtype=torch.float32,
        )
        extrinsics = c2w_to_w2c(render_c2w)

        try:
            result = self.pipeline.render(
                self.gaussians, extrinsics, intrinsics, H, W,
            )
        except Exception:
            traceback.print_exc()
            return _failure_pattern(W, H)

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

        display = letterbox_for_aspect(image_np, camera_state.aspect)

        wall_elapsed = time.perf_counter() - wall_start
        if self.verbose:
            self._log_verbose(W, H, result, wall_elapsed)
        self._update_stats(wall_elapsed, W, H, result.num_visible)
        return display

    def _log_verbose(
        self,
        W: int,
        H: int,
        result: RenderResult,
        wall_elapsed: float,
    ) -> None:
        print(
            f"[render] kernel={W}x{H}  "
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
