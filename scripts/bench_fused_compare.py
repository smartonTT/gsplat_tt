"""Microbenchmark: fused cull_and_blend vs unfused microblock_cull+blend_microblock.

Times the *blend* stage only (sub-timings ignore project/tile_assign/sort), so
that we can isolate the fused-pass speedup. Renders 5 warmup + 30 timed frames.
"""
from __future__ import annotations

import json
import sys
import time
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


def run(label: str, fused: bool, gauss, cams, fov_deg, W, H, order):
    from backends.cpu_cpp.backend import CpuCppBackend
    backend = CpuCppBackend(microblock=True, fused=fused)
    pipeline = Pipeline(backend, tile_size=32)
    c2w = np.asarray(cams[order[0]]["c2w"], dtype=np.float32)
    extr = c2w_to_w2c(torch.from_numpy(c2w))
    K = build_intrinsics(W, H, fov_deg)
    for _ in range(3):
        pipeline.render(gauss, extr, K, H, W)
    samples = []
    t0_all = time.perf_counter()
    for name in order:
        c2w = np.asarray(cams[name]["c2w"], dtype=np.float32)
        extr = c2w_to_w2c(torch.from_numpy(c2w))
        K = build_intrinsics(W, H, fov_deg)
        t0 = time.perf_counter()
        pipeline.render(gauss, extr, K, H, W)
        samples.append((time.perf_counter() - t0) * 1000.0)
    elapsed = (time.perf_counter() - t0_all) * 1000.0
    samples.sort()
    p50 = samples[len(samples) // 2]
    print(f"[{label}] sum_ms={elapsed:7.1f}  p50={p50:6.2f}  min={samples[0]:6.2f}  max={samples[-1]:6.2f}")
    return elapsed, p50


def main():
    cam_data = json.loads(Path("benchmarks/cameras_v2.json").read_text())
    scene = cam_data["stitch"]
    fov_deg = float(scene["fov_deg"])
    ply_path = Path(scene["ply"])
    W, H = scene["image_size"]
    order = scene["order"]
    cams = scene["views"]
    gauss = load_ply(str(ply_path))

    # Three runs each, alternated.
    for trial in range(3):
        print(f"=== trial {trial} ===")
        run("unfused", False, gauss, cams, fov_deg, W, H, order)
        run("fused  ", True,  gauss, cams, fov_deg, W, H, order)


if __name__ == "__main__":
    main()
