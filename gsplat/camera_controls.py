"""Client camera control helpers for the interactive viewer.

For *orbit* mode we deliberately DON'T touch the camera on every drag tick.
Viser's built-in orbit is quaternion-based and never gimbal-locks. The
previous implementation forced ``look_at = scene_center`` and
``up_direction = WORLD_UP`` on every ``camera.on_update`` callback, which:

  * gimbal-locked the camera at the pole (forward parallel to up → viser's
    ``y /= np.linalg.norm(y)`` divides by zero → NaN view matrix → black
    background image), and
  * pinned drag rotation to a half-circle because the snap kept yanking the
    camera back onto the equator.

So orbit mode now is "set the initial pose once, get out of the way". FPS
mode still needs a per-update handler so the look target tracks the camera.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import viser

ControlMode = Literal["orbit", "fps"]

# Viser orbits horizontal drag around ``camera.up_direction``, so left/right
# drag rotates about whichever axis we pin here, and up/down drag rotates
# about the screen-right axis. The user wants horizontal=world-Y / vertical=
# world-X. We use **-Y** as up because the scenes we render (stitch_doll,
# luigi, ...) were authored with +Y pointing "down" in the world frame, so
# +Y up makes the model render upside-down. Flipping to -Y leaves the drag
# axes intact (horizontal still rotates about world Y, vertical about world
# X — the SIGNS just flip, which is fine for orbit) and stands the scene
# upright. If a future scene comes in authored differently, expose a CLI
# flag rather than baking another value here.
WORLD_UP: np.ndarray = np.array([0.0, -1.0, 0.0], dtype=np.float64)
_FPS_LOOK_DISTANCE = 1.0


class ClientCameraController:
    """Track per-client orbit/FPS mode and snap on mode transitions."""

    ORBIT: ControlMode = "orbit"
    FPS: ControlMode = "fps"

    def __init__(self, scene_center: np.ndarray, fallback_distance: float) -> None:
        self.scene_center = np.asarray(scene_center, dtype=np.float64)
        self.fallback_distance = float(fallback_distance)
        self.mode: ControlMode = self.ORBIT
        # Used by FPS mode's on_update handler to swallow the recursive
        # camera.position/look_at writes it issues itself.
        self._suppress = False

    def set_mode(self, mode: ControlMode) -> None:
        self.mode = mode

    def install_initial_pose(self, camera: viser.CameraHandle) -> None:
        """Pin orbit pose + world up exactly once when a client connects."""
        offset = np.array([0.0, 0.0, self.fallback_distance], dtype=np.float64)
        camera.position = self.scene_center + offset
        camera.look_at = self.scene_center
        camera.up_direction = WORLD_UP

    def snap_to_orbit(self, camera: viser.CameraHandle) -> None:
        """Re-anchor the camera on a turntable orbit around the scene center."""
        pos = np.asarray(camera.position, dtype=np.float64)
        offset = pos - self.scene_center
        dist = float(np.linalg.norm(offset))
        if dist < 1e-6:
            offset = np.array([0.0, 0.0, self.fallback_distance], dtype=np.float64)
            dist = float(np.linalg.norm(offset))
        direction = offset / dist
        camera.position = self.scene_center + direction * dist
        camera.look_at = self.scene_center
        camera.up_direction = WORLD_UP

    def snap_to_fps(self, camera: viser.CameraHandle) -> None:
        """Switch to first-person by placing the look target in front of the camera."""
        pos = np.asarray(camera.position, dtype=np.float64)
        look_at = np.asarray(camera.look_at, dtype=np.float64)
        forward = look_at - pos
        dist = float(np.linalg.norm(forward))
        if dist < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        else:
            forward = forward / dist
        camera.position = pos
        camera.look_at = pos + forward * _FPS_LOOK_DISTANCE
        camera.up_direction = WORLD_UP

    def on_camera_update(self, camera: viser.CameraHandle) -> None:
        """Per-update handler. No-op for orbit (let viser do its thing)."""
        if self.mode != self.FPS:
            return
        if self._suppress:
            return
        self._suppress = True
        try:
            pos = np.asarray(camera.position, dtype=np.float64)
            look_at = np.asarray(camera.look_at, dtype=np.float64)
            forward = look_at - pos
            dist = float(np.linalg.norm(forward))
            if dist < 1e-6:
                return
            forward = forward / dist
            new_look = pos + forward * _FPS_LOOK_DISTANCE
            if np.allclose(np.asarray(camera.look_at), new_look, atol=1e-5):
                return
            camera.look_at = new_look
        finally:
            self._suppress = False
