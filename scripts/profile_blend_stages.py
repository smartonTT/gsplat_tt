"""Measure cull vs blend cost inside cull_and_blend by temporarily
forcing one to skip the other.

This is an out-of-band diagnostic — uses the unfused path with the
existing microblock_cull + blend_microblock split to estimate each.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.cpu_cpp.backend import CpuCppBackend  # noqa: E402
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

    backend = CpuCppBackend(microblock=True, fused=False)
    pipeline = Pipeline(backend, tile_size=32)

    name = order[0]
    c2w = np.asarray(cams[name]["c2w"], dtype=np.float32)
    extr = c2w_to_w2c(torch.from_numpy(c2w))
    K = build_intrinsics(W, H, fov_deg)
    pipeline.render(gauss, extr, K, H, W)

    # Pre-compute project + tile_assign + sort once.
    means_2d, covs_2d, depths, radii, valid_mask = backend.project(
        gauss.means, gauss.scales, gauss.rotations, extr, K, H, W, gauss.opacities
    )
    keep = valid_mask.bool()
    opac = gauss.opacities[keep]
    colors = gauss.colors[keep]
    gids, tids, _ = backend.tile_assign(means_2d, radii, H, W, 32, covs_2d=covs_2d, opacities=opac)
    sgids, tranges = backend.sort(gids, tids, depths, (W + 31) // 32, (H + 31) // 32)

    mod = backend._mod
    mn = np.ascontiguousarray(means_2d.numpy(), dtype=np.float32)
    cv = np.ascontiguousarray(covs_2d.numpy().reshape(-1, 4), dtype=np.float32)
    op = np.ascontiguousarray(opac.numpy(), dtype=np.float32)
    cl = np.ascontiguousarray(colors.numpy(), dtype=np.float32)
    sg = np.ascontiguousarray(sgids.numpy(), dtype=np.int64)
    tr = np.ascontiguousarray(tranges.numpy(), dtype=np.int64)
    tx, ty = (W + 31) // 32, (H + 31) // 32

    # Warmup.
    for _ in range(3):
        mod.microblock_cull(mn, cv, op, sg, tr, tx, ty, 32, 1.0 / 16384.0)

    N = 50
    t0 = time.perf_counter()
    for _ in range(N):
        h, s, _ = mod.microblock_cull(mn, cv, op, sg, tr, tx, ty, 32, 1.0 / 16384.0)
    t_cull = (time.perf_counter() - t0) * 1000.0 / N

    # Use h/s from last cull.
    for _ in range(3):
        mod.blend_microblock(mn, cv, cl, op, h, s, H, W, 32)
    t0 = time.perf_counter()
    for _ in range(N):
        mod.blend_microblock(mn, cv, cl, op, h, s, H, W, 32)
    t_blend = (time.perf_counter() - t0) * 1000.0 / N

    # Now fused.
    for _ in range(3):
        mod.cull_and_blend(mn, cv, cl, op, sg, tr, tx, ty, 32, H, W, 1.0 / 16384.0)
    t0 = time.perf_counter()
    for _ in range(N):
        mod.cull_and_blend(mn, cv, cl, op, sg, tr, tx, ty, 32, H, W, 1.0 / 16384.0)
    t_fused = (time.perf_counter() - t0) * 1000.0 / N

    print(f"microblock_cull:    {t_cull:6.3f} ms")
    print(f"blend_microblock:   {t_blend:6.3f} ms")
    print(f"sum unfused:        {t_cull + t_blend:6.3f} ms")
    print(f"cull_and_blend:     {t_fused:6.3f} ms")
    print(f"fusion saves:       {(t_cull + t_blend) - t_fused:6.3f} ms")


if __name__ == "__main__":
    main()
