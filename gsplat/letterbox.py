"""Letterbox a rendered frame for the browser viewport.

The viewer always renders at a fixed Render Res (e.g. 1024x1024), but the
browser viewport has the user's window aspect — typically wider. If we hand
viser the square image directly, the browser stretches it to fill (so the
scene looks horizontally smeared). Instead we paste the rendered image onto
a black canvas whose aspect matches the browser's; viser still scales to
fill, but now without distortion.

We do *not* resample the rendered image. The render is dropped onto the
canvas at native resolution and centered; padding is pure black.
"""
from __future__ import annotations

import numpy as np


def letterbox_for_aspect(image: np.ndarray, viewport_aspect: float) -> np.ndarray:
    """Pad *image* with black bars so the result has ``viewport_aspect``.

    The rendered image is preserved at its native resolution and centered;
    only the canvas grows. No resampling, no quality loss. Returns a uint8
    array of shape ``(canvas_h, canvas_w, 3)``.
    """
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            f"letterbox expects a (H, W, 3|4) image; got shape={image.shape}"
        )
    img_h, img_w = image.shape[:2]
    if not np.isfinite(viewport_aspect) or viewport_aspect <= 0:
        return image

    img_aspect = img_w / max(img_h, 1)
    if viewport_aspect >= img_aspect:
        # Browser is wider than the render → pad left/right.
        canvas_h = img_h
        canvas_w = max(img_w, int(round(img_h * viewport_aspect)))
    else:
        # Browser is taller than the render → pad top/bottom.
        canvas_w = img_w
        canvas_h = max(img_h, int(round(img_w / viewport_aspect)))

    if canvas_w == img_w and canvas_h == img_h:
        return image

    canvas = np.zeros((canvas_h, canvas_w, image.shape[2]), dtype=image.dtype)
    y0 = (canvas_h - img_h) // 2
    x0 = (canvas_w - img_w) // 2
    canvas[y0 : y0 + img_h, x0 : x0 + img_w] = image
    return canvas


# Back-compat: a few older callers pass viewport pixel dims rather than aspect.
def letterbox_image(
    image: np.ndarray,
    viewport_w: int,
    viewport_h: int,
) -> np.ndarray:
    if viewport_w <= 0 or viewport_h <= 0:
        return image
    return letterbox_for_aspect(image, viewport_w / viewport_h)
