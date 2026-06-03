#!/usr/bin/env python3
"""Write microblock_cull hero fixtures from existing blend_inputs.npz."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gsplat.rasterization import alpha_blend_microblock, microblock_cull  # noqa: E402

MB_CONTRIB_FLOOR = 1.0 / 16384.0


def main() -> None:
    fixtures_dir = REPO / "tests" / "fixtures" / "hero"
    blend_path = fixtures_dir / "blend_inputs.npz"
    if not blend_path.exists():
        raise SystemExit(
            f"missing {blend_path}; run scripts/capture_reference.py --scene stitch first"
        )

    blend = dict(np.load(blend_path))
    tiles_x = int((blend["W"] + 31) // 32)
    tiles_y = int((blend["H"] + 31) // 32)

    mb_header, mb_stream, stats = microblock_cull(
        torch.from_numpy(blend["means_2d"]),
        torch.from_numpy(blend["covs_2d"]),
        torch.from_numpy(blend["opacities"]),
        torch.from_numpy(blend["sorted_gaussian_ids"]),
        torch.from_numpy(blend["tile_ranges"]),
        tiles_x,
        tiles_y,
        32,
        mb_contrib_floor=MB_CONTRIB_FLOOR,
    )

    blend_image = alpha_blend_microblock(
        torch.from_numpy(blend["means_2d"]),
        torch.from_numpy(blend["covs_2d"]),
        torch.from_numpy(blend["colors"]),
        torch.from_numpy(blend["opacities"]),
        mb_header.reshape(tiles_x * tiles_y, 32, 2),
        mb_stream,
        int(blend["H"]),
        int(blend["W"]),
    )

    np.savez(
        fixtures_dir / "microblock_cull_inputs.npz",
        means_2d=blend["means_2d"],
        covs_2d=blend["covs_2d"],
        colors=blend["colors"],
        opacities=blend["opacities"],
        sorted_gaussian_ids=blend["sorted_gaussian_ids"],
        tile_ranges=blend["tile_ranges"],
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        mb_contrib_floor=np.float32(MB_CONTRIB_FLOOR),
    )
    np.savez(
        fixtures_dir / "microblock_cull_outputs.npz",
        mb_header=mb_header.numpy(),
        mb_stream=mb_stream.numpy(),
        blend_image=blend_image.numpy(),
    )
    np.save(fixtures_dir / "blend_microblock_output.npy", blend_image.numpy())

    print(
        f"wrote microblock_cull fixtures to {fixtures_dir}\n"
        f"  pairs_in={stats['pairs_in']} pairs_out={stats['pairs_out']}\n"
        f"  drop_pct={stats['drop_pct']:.2f}% "
        f"work_reduction_pct={stats['work_reduction_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
