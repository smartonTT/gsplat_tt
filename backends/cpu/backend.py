"""CPU backend — pure PyTorch reference rasterizer.

All four pipeline stages fall back to the CPU defaults provided by
`gsplat.backend.Backend`, so this class is intentionally minimal: it
only needs to override `blend` (the abstract method) and route it to
`gsplat.rasterization.alpha_blend`.

Speed: ~1-2 s/frame at 256×256 on a 16-core x86 box. Used as the
correctness reference, not for interactive rendering.
"""
from __future__ import annotations

import numpy as np
import torch

from gsplat.backend import Backend
from gsplat import rasterization


class CpuBackend(Backend):
    def __init__(self, verbose: bool = False, tile_size: int = 32, **_ignored):
        """Accept (and silently ignore) the optional kwargs that the C++ /
        TT backends use (k_cap, contrib_floor, microblock, …). The numpy
        reference path doesn't apply those knobs — they're baked into the
        Python rasterization defaults — but we accept them so the viewer
        can hand the same kwargs to every backend uniformly."""
        self.verbose = verbose
        self.tile_size = tile_size

    def blend(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gaussian_ids: torch.Tensor,
        tile_ranges: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        # `alpha_blend` returns a torch.Tensor; convert to numpy for the
        # cross-backend contract. No internal sub-stages worth surfacing.
        image = rasterization.alpha_blend(
            means_2d, covs_2d, colors, opacities,
            sorted_gaussian_ids, tile_ranges,
            image_height, image_width, tile_size=self.tile_size,
        )
        return image.numpy(), {}


class CpuMicroblockBackend(CpuBackend):
    """Numpy backend with microblock culling between sort and blend.

    `mb_contrib_floor` is the per-microblock §2 keep threshold; the default
    (1/16384) hits ≥ 60 dB PSNR vs `alpha_blend` on all 30 benchmark views
    while still removing ~75% of (g, m) work. iter-009 may expose this as
    a viewer slider.
    """

    def __init__(
        self,
        verbose: bool = False,
        tile_size: int = 32,
        mb_contrib_floor: float = 1.0 / 16384.0,
    ):
        super().__init__(verbose=verbose, tile_size=tile_size)
        self.mb_contrib_floor = mb_contrib_floor

    def blend(
        self,
        means_2d: torch.Tensor,
        covs_2d: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        sorted_gaussian_ids: torch.Tensor,
        tile_ranges: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        tiles_x = (image_width + 31) // 32
        tiles_y = (image_height + 31) // 32
        mb_header, mb_stream, stats = rasterization.microblock_cull(
            means_2d,
            covs_2d,
            opacities,
            sorted_gaussian_ids,
            tile_ranges,
            tiles_x,
            tiles_y,
            tile_size=self.tile_size,
            mb_contrib_floor=self.mb_contrib_floor,
        )
        image = rasterization.alpha_blend_microblock(
            means_2d,
            covs_2d,
            colors,
            opacities,
            mb_header,
            mb_stream,
            image_height,
            image_width,
            tile_size=self.tile_size,
        )
        return image.numpy(), {
            "microblock_drop_pct": stats["drop_pct"],
            "microblock_work_reduction_pct": stats["work_reduction_pct"],
        }
