"""Equivalence test for iter 025 numpy `get_tile_assignments`.

Compares the new numpy implementation against a deliberately-preserved
torch reference. Sweeps multiple resolutions, gaussian counts, and rng
seeds. Each pair must match exactly (modulo stable order of equal
gaussians within a tile, which both impls preserve via repeat_interleave
/ np.repeat starting from arange).
"""

from __future__ import annotations

import numpy as np
import torch

from gsplat.rasterization import get_tile_assignments


def _torch_reference(means_2d, radii, image_height, image_width, tile_size=32):
    tiles_x = (image_width + tile_size - 1) // tile_size
    tiles_y = (image_height + tile_size - 1) // tile_size

    tile_min_x = torch.clamp((means_2d[:, 0] - radii) / tile_size, min=0, max=tiles_x - 1).int()
    tile_max_x = torch.clamp((means_2d[:, 0] + radii) / tile_size, min=0, max=tiles_x - 1).int()
    tile_min_y = torch.clamp((means_2d[:, 1] - radii) / tile_size, min=0, max=tiles_y - 1).int()
    tile_max_y = torch.clamp((means_2d[:, 1] + radii) / tile_size, min=0, max=tiles_y - 1).int()

    widths = tile_max_x - tile_min_x + 1
    heights = tile_max_y - tile_min_y + 1
    tiles_per_gaussian = widths * heights

    P = tiles_per_gaussian.sum().item()
    gaussian_ids = torch.repeat_interleave(torch.arange(means_2d.shape[0]), tiles_per_gaussian)
    min_x_rep = torch.repeat_interleave(tile_min_x, tiles_per_gaussian)
    min_y_rep = torch.repeat_interleave(tile_min_y, tiles_per_gaussian)
    widths_rep = torch.repeat_interleave(widths, tiles_per_gaussian)

    offsets = torch.arange(P) - torch.repeat_interleave(
        torch.cat([torch.zeros(1, dtype=torch.int32), tiles_per_gaussian.cumsum(0)[:-1]]),
        tiles_per_gaussian,
    )
    dy = offsets // widths_rep
    dx = offsets % widths_rep
    tile_ids = (min_y_rep + dy) * tiles_x + (min_x_rep + dx)
    return gaussian_ids, tile_ids, tiles_per_gaussian


def _gen_inputs(M, W, H, seed):
    rng = np.random.default_rng(seed)
    means = rng.uniform(low=-W * 0.1, high=W * 1.1, size=(M, 2)).astype(np.float32)
    means[:, 1] *= H / W
    radii = rng.uniform(low=0.5, high=20.0, size=M).astype(np.float32)
    return torch.from_numpy(means), torch.from_numpy(radii)


def main():
    cases = [
        (2_000, 256, 256, 0),
        (10_000, 480, 640, 1),
        (50_000, 1024, 1024, 2),
        (256_558, 480, 640, 3),
        (280_007, 1024, 1024, 4),
    ]
    failures = 0
    for M, W, H, seed in cases:
        means, radii = _gen_inputs(M, W, H, seed)
        ref = _torch_reference(means, radii, H, W)
        new = get_tile_assignments(means, radii, H, W)

        for name, a, b in zip(("gids", "tids", "per_g"), ref, new):
            ok = torch.equal(a.to(torch.int64), b.to(torch.int64))
            print(f"  {name}: {'OK' if ok else 'MISMATCH'}  (M={M} W={W} H={H} seed={seed})")
            if not ok:
                # Show first mismatch
                diff = (a.to(torch.int64) != b.to(torch.int64)).nonzero(as_tuple=True)[0]
                if diff.numel():
                    i = int(diff[0])
                    print(f"    first diff at {i}: ref={int(a[i])} new={int(b[i])}")
                failures += 1
    print(f"\n{'PASS' if failures == 0 else f'FAIL ({failures} mismatches)'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
