"""Compatibility shims for viser + nerfview internals.

We import this for its side effects from ``gsplat.viewer``. Two upstream
quirks bite us hard during interactive use; both are patched here so the
actual viewer code stays clean:

1. **viser ``CameraHandle._update_wxyz`` divides by zero at the pole.**
   When the camera's forward direction becomes parallel to
   ``up_direction``, viser's Gram-Schmidt step yields a zero-length lateral
   vector and emits ``RuntimeWarning: invalid value encountered in
   divide``. The resulting NaN view matrix paints the browser black and
   pins drag rotation to a half-circle. We replace the routine with a
   gimbal-safe version that picks an alternate basis through the
   singularity so the orbit keeps spinning.

2. **nerfview's ``Renderer.submit`` interrupts the in-flight render.**
   When a new "move" task arrives mid-render, nerfview sets
   ``_may_interrupt_render = True`` and uses ``sys.settrace`` to raise
   ``InterruptRenderException`` from inside the render path. Our TT
   backend talks to a separate daemon over stdin/stdout; if the
   exception fires while we're partway through reading a frame's binary
   response, the daemon's pipe is left holding the rest of that response
   and the next frame's read sees stale bytes — every subsequent frame
   fails. We disable the interrupt; the only cost is a slightly later
   frame on rapid input bursts.
"""
from __future__ import annotations

import numpy as np


def _patch_viser_camera_handle_gimbal() -> None:
    import viser._viser as viser_internals  # type: ignore[attr-defined]
    import viser.transforms as vt

    def _safe_update_wxyz(self) -> None:
        z = np.asarray(self._state.look_at, dtype=np.float64) - np.asarray(
            self._state.position, dtype=np.float64
        )
        z_norm = float(np.linalg.norm(z))
        if z_norm < 1e-9:
            return
        z = z / z_norm

        def _gram_schmidt(reference: np.ndarray) -> np.ndarray:
            rotated = vt.SO3.exp(z * np.pi) @ reference
            ortho = rotated - np.dot(z, rotated) * z
            return ortho

        up = np.asarray(self._state.up_direction, dtype=np.float64)
        y = _gram_schmidt(up)
        y_norm = float(np.linalg.norm(y))
        if y_norm < 1e-6:
            # Forward is (nearly) parallel to up_direction. Pick a fallback
            # reference that is guaranteed not parallel to z, run the same
            # Gram-Schmidt projection, and continue. This lets the orbit
            # keep rotating right through the pole instead of NaN-locking.
            alt = (
                np.array([1.0, 0.0, 0.0])
                if abs(float(z[0])) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            y = _gram_schmidt(alt)
            y_norm = float(np.linalg.norm(y))
            if y_norm < 1e-6:
                return  # truly degenerate; leave wxyz untouched
        y = y / y_norm
        x = np.cross(y, z)
        self._state.wxyz = vt.SO3.from_matrix(
            np.stack([x, y, z], axis=1)
        ).wxyz.astype(np.float64)

    viser_internals.CameraHandle._update_wxyz = _safe_update_wxyz  # type: ignore[assignment]


def _patch_nerfview_no_interrupt() -> None:
    import nerfview._renderer as nv_renderer  # type: ignore[attr-defined]

    original_submit = nv_renderer.Renderer.submit

    def _no_interrupt_submit(self, task):  # type: ignore[no-untyped-def]
        self._may_interrupt_render = False
        return original_submit(self, task)

    nv_renderer.Renderer.submit = _no_interrupt_submit  # type: ignore[assignment]


def install_all() -> None:
    _patch_viser_camera_handle_gimbal()
    _patch_nerfview_no_interrupt()


install_all()
