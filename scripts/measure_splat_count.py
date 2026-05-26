#!/usr/bin/env python3
"""Measure splat-count distribution under different bbox strategies.

For each scene/view we run project_gaussians, then count (Gaussian, tile)
pairs that would be produced by:
  - current_circle: r = ceil(3 * sqrt(lambda_max)) — circular AABB, what the
    pipeline ships today.
  - aabb_3sigma:    rx = 3 * sqrt(cov[0,0]),  ry = 3 * sqrt(cov[1,1]) — axis-
    aligned bbox from the covariance diagonal. Identical for symmetric
    Gaussians, strictly smaller for elongated ones. This is what the original
    diff-gaussian-rasterization CUDA implementation uses.
  - aabb_2_5sigma:  rx = 2.5*..., ry = 2.5*... — strictly tighter again. Drops
    Gaussians' last ~4% energy contour. Comparison point — quality risk
    depends on opacity distribution of the scene.

Prints total pairs and per-Gaussian tile-count histogram for each strategy.
No render; this is purely a host-side measurement to size the iter-022 win.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.rasterization import project_gaussians  # noqa: E402


def _build_intrinsics(W: int, H: int, fov_deg: float) -> torch.Tensor:
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov_deg))
    return torch.tensor([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=torch.float32)


def _c2w_to_w2c(c2w: torch.Tensor) -> torch.Tensor:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    w2c = torch.eye(4, dtype=c2w.dtype)
    w2c[:3, :3] = R_inv
    w2c[:3, 3] = t_inv
    return w2c


def _count_pairs(means_2d: torch.Tensor, rx: torch.Tensor, ry: torch.Tensor,
                 W: int, H: int, tile_size: int = 32) -> tuple[int, np.ndarray]:
    """Replicate get_tile_assignments' counting math with separate x/y radii."""
    tiles_x = (W + tile_size - 1) // tile_size
    tiles_y = (H + tile_size - 1) // tile_size
    tile_min_x = torch.clamp((means_2d[:, 0] - rx) / tile_size, min=0, max=tiles_x - 1).int()
    tile_max_x = torch.clamp((means_2d[:, 0] + rx) / tile_size, min=0, max=tiles_x - 1).int()
    tile_min_y = torch.clamp((means_2d[:, 1] - ry) / tile_size, min=0, max=tiles_y - 1).int()
    tile_max_y = torch.clamp((means_2d[:, 1] + ry) / tile_size, min=0, max=tiles_y - 1).int()
    widths = (tile_max_x - tile_min_x + 1).clamp(min=0)
    heights = (tile_max_y - tile_min_y + 1).clamp(min=0)
    per_g = (widths * heights).numpy()
    return int(per_g.sum()), per_g


def _summarize(name: str, total: int, per_g: np.ndarray, baseline: int | None = None):
    pct = "" if baseline is None else f"  ({100.0 * total / baseline - 100:+.1f}%)"
    p50, p90, p99, pmax = np.percentile(per_g, [50, 90, 99, 100]).astype(int)
    print(f"  {name:<16} total_pairs={total:>9,}{pct}   per-G  p50={p50}  p90={p90}  p99={p99}  max={pmax}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="benchmarks/cameras.json")
    ap.add_argument("--scene", default="stitch")
    ap.add_argument("--views", nargs="+", default=["hero", "side", "top"])
    args = ap.parse_args()

    cam_path = REPO / args.cameras
    cams = json.loads(cam_path.read_text())
    scene_meta = cams[args.scene]
    ply_path = REPO / scene_meta["ply"]
    W, H = scene_meta["image_size"]
    fov_deg = scene_meta["fov_deg"]

    print(f"scene={args.scene}  ply={ply_path.name}  size={W}x{H}  fov={fov_deg}")
    gs = load_ply(ply_path)
    print(f"  total Gaussians in scene: {gs.means.shape[0]:,}")

    K = _build_intrinsics(W, H, fov_deg)

    for view in args.views:
        c2w = torch.tensor(scene_meta["views"][view]["c2w"], dtype=torch.float32)
        w2c = _c2w_to_w2c(c2w)

        means_2d, covs_2d, depths, radii, valid = project_gaussians(
            gs.means, gs.scales, gs.rotations, w2c, K, H, W, opacities=gs.opacities
        )
        M = means_2d.shape[0]
        print(f"\n[{view}] visible after frustum+opacity cull: {M:,} / {gs.means.shape[0]:,}")

        a = covs_2d[:, 0, 0]
        c = covs_2d[:, 1, 1]
        sigma_x = torch.sqrt(torch.clamp(a, min=0.0))
        sigma_y = torch.sqrt(torch.clamp(c, min=0.0))

        # current_circle is exactly what the live pipeline ships.
        tot0, pg0 = _count_pairs(means_2d, radii, radii, W, H)
        _summarize("current_circle", tot0, pg0, baseline=tot0)

        # aabb_3sigma: per-axis 3σ box.
        rx3 = torch.ceil(3.0 * sigma_x)
        ry3 = torch.ceil(3.0 * sigma_y)
        tot1, pg1 = _count_pairs(means_2d, rx3, ry3, W, H)
        _summarize("aabb_3sigma", tot1, pg1, baseline=tot0)

        # aabb_2_5sigma: still axis-aligned, tighter cutoff.
        rx2 = torch.ceil(2.5 * sigma_x)
        ry2 = torch.ceil(2.5 * sigma_y)
        tot2, pg2 = _count_pairs(means_2d, rx2, ry2, W, H)
        _summarize("aabb_2_5sigma", tot2, pg2, baseline=tot0)


if __name__ == "__main__":
    main()
