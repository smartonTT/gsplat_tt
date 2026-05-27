"""Dump per-tile Gaussian counts to understand load imbalance for LPT scheduling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


def build_intrinsics(W: int, H: int, fov_deg: float) -> torch.Tensor:
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(K)


def main():
    cam_data = json.loads(Path("benchmarks/cameras_v2.json").read_text())
    scene = cam_data["stitch"]
    fov_deg = float(scene["fov_deg"])
    ply_path = Path(scene["ply"])
    W, H = scene["image_size"]
    order = scene["order"]
    cams = scene["views"]
    gauss = load_ply(str(ply_path))

    from backends.cpu_cpp.backend import CpuCppBackend
    backend = CpuCppBackend(microblock=True, fused=False)
    pipeline = Pipeline(backend, tile_size=32)

    name = order[0]  # hero
    c2w = np.asarray(cams[name]["c2w"], dtype=np.float32)
    extr = c2w_to_w2c(torch.from_numpy(c2w))
    K = build_intrinsics(W, H, fov_deg)
    # Re-run stages to get tile_ranges.
    means_2d, covs_2d, depths, radii, valid_mask = backend.project(
        gauss.means, gauss.scales, gauss.rotations, extr, K, H, W, gauss.opacities
    )
    # project already filters visible Gaussians; keep all.
    # Build the opacity slice that matches the visible set.
    keep = valid_mask.bool()
    opac = gauss.opacities[keep]
    gids, tids, _tpg = backend.tile_assign(
        means_2d, radii, H, W, 32, covs_2d=covs_2d, opacities=opac,
    )
    _sg, tranges = backend.sort(gids, tids, depths, (W + 31) // 32, (H + 31) // 32)
    tranges_np = tranges.numpy()
    counts = tranges_np[:, 1] - tranges_np[:, 0]
    print(f"num_tiles={len(counts)} sum={counts.sum()} max={counts.max()} min={counts.min()} mean={counts.mean():.1f}")
    sorted_counts = np.sort(counts)[::-1]
    cum = sorted_counts.cumsum()
    print("Top 20 tiles:", sorted_counts[:20].tolist())
    print(f"Tiles holding 50% of work: {(cum < cum[-1] * 0.5).sum()+1} of {len(counts)}")
    print(f"Tiles holding 80% of work: {(cum < cum[-1] * 0.8).sum()+1} of {len(counts)}")
    print(f"Tiles holding 95% of work: {(cum < cum[-1] * 0.95).sum()+1} of {len(counts)}")
    nonzero = (counts > 0).sum()
    print(f"Non-empty tiles: {nonzero}/{len(counts)}")


if __name__ == "__main__":
    main()
