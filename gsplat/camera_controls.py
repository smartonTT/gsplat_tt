"""Client camera control helpers for the interactive viewer."""
from __future__ import annotations

from typing import Literal

import numpy as np
import viser

ControlMode = Literal["orbit", "fps"]

_WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
_FPS_LOOK_DISTANCE = 1.0


class ClientCameraController:
    """Normalize viser camera updates for orbit (turntable) or FPS modes."""

    ORBIT: ControlMode = "orbit"
    FPS: ControlMode = "fps"

    def __init__(self, scene_center: np.ndarray, fallback_distance: float) -> None:
        self.scene_center = np.asarray(scene_center, dtype=np.float64)
        self.fallback_distance = float(fallback_distance)
        self.mode: ControlMode = self.ORBIT
        self._suppress = False

    def set_mode(self, mode: ControlMode) -> None:
        self.mode = mode

    def snap_to_orbit(self, camera: viser.CameraHandle) -> None:
        """Re-anchor the camera on a turntable orbit around the scene center."""
        pos = np.asarray(camera.position, dtype=np.float64)
        offset = pos - self.scene_center
        dist = float(np.linalg.norm(offset))
        if dist < 1e-6:
            offset = np.array([0.0, 0.0, self.fallback_distance], dtype=np.float64)
            dist = float(np.linalg.norm(offset))
        direction = offset / dist
        self._write(camera, self.scene_center + direction * dist, self.scene_center)

    def snap_to_fps(self, camera: viser.CameraHandle) -> None:
        """Switch to first-person by placing the look target in front of the camera."""
        pos = np.asarray(camera.position, dtype=np.float64)
        look_at = np.asarray(camera.look_at, dtype=np.float64)
        forward = look_at - pos
        dist = float(np.linalg.norm(forward))
        if dist < 1e-6:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            forward = forward / dist
        self._write(camera, pos, pos + forward * _FPS_LOOK_DISTANCE)

    def apply(self, camera: viser.CameraHandle) -> None:
        if self._suppress:
            return
        self._suppress = True
        try:
            if self.mode == self.ORBIT:
                self._apply_orbit(camera)
            else:
                self._apply_fps(camera)
        finally:
            self._suppress = False

    def _apply_orbit(self, camera: viser.CameraHandle) -> None:
        pos = np.asarray(camera.position, dtype=np.float64)
        offset = pos - self.scene_center
        dist = float(np.linalg.norm(offset))
        if dist < 1e-6:
            return
        self._write(camera, pos, self.scene_center)

    def _apply_fps(self, camera: viser.CameraHandle) -> None:
        pos = np.asarray(camera.position, dtype=np.float64)
        look_at = np.asarray(camera.look_at, dtype=np.float64)
        forward = look_at - pos
        dist = float(np.linalg.norm(forward))
        if dist < 1e-6:
            forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            forward = forward / dist
        self._write(camera, pos, pos + forward * _FPS_LOOK_DISTANCE)

    def _write(
        self,
        camera: viser.CameraHandle,
        position: np.ndarray,
        look_at: np.ndarray,
    ) -> None:
        cur_pos = np.asarray(camera.position, dtype=np.float64)
        cur_look = np.asarray(camera.look_at, dtype=np.float64)
        if (
            np.allclose(cur_pos, position, rtol=0.0, atol=1e-5)
            and np.allclose(cur_look, look_at, rtol=0.0, atol=1e-5)
            and np.allclose(camera.up_direction, _WORLD_UP, rtol=0.0, atol=1e-5)
        ):
            return
        # Assign position before look_at: viser translates look_at with position.
        camera.position = position
        camera.look_at = look_at
        camera.up_direction = _WORLD_UP
