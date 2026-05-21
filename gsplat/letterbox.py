"""Letterbox a rendered frame into a viewport without stretching."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def letterbox_image(
    image: np.ndarray,
    viewport_w: int,
    viewport_h: int,
) -> np.ndarray:
    """Scale *image* to fit inside (viewport_w, viewport_h), preserving aspect.

    Padded regions are black. Returns a uint8 array of shape
    (viewport_h, viewport_w, 3).
    """
    img_h, img_w = image.shape[:2]
    if viewport_w <= 0 or viewport_h <= 0:
        return image
    if img_w == viewport_w and img_h == viewport_h:
        return image

    scale = min(viewport_w / img_w, viewport_h / img_h)
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))

    if new_w != img_w or new_h != img_h:
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        tensor = F.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False,
        )
        resized = (
            tensor.squeeze(0).permute(1, 2, 0).clamp(0, 255).numpy().astype(np.uint8)
        )
    else:
        resized = image

    canvas = np.zeros((viewport_h, viewport_w, 3), dtype=np.uint8)
    y0 = (viewport_h - new_h) // 2
    x0 = (viewport_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas
