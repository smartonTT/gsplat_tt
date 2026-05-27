"""Isolated micro-benchmark for microblock_cull + blend_microblock.

Loads the saved hero fixture (deterministic across runs), repeatedly calls
the C++ entrypoints, prints p50/p90 in ms.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends.cpu_cpp import _gsplat_cpu as mod  # type: ignore


def bench(label: str, fn, n: int = 30) -> float:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    p50 = ts[n // 2]
    p90 = ts[int(n * 0.9)]
    print(f"  {label:35s} p50={p50:7.2f}  p90={p90:7.2f}")
    return p50


def main() -> None:
    fix_root = Path("tests/fixtures/hero")
    proj = np.load(fix_root / "project_outputs.npz")
    blend_in = np.load(fix_root / "blend_inputs.npz")
    means_2d = np.ascontiguousarray(blend_in["means_2d"].astype(np.float32))
    covs_2d_raw = blend_in["covs_2d"].astype(np.float32)
    if covs_2d_raw.ndim == 3:
        covs_2d_raw = covs_2d_raw.reshape(covs_2d_raw.shape[0], -1)
    covs_2d = np.ascontiguousarray(covs_2d_raw)
    opac = np.ascontiguousarray(blend_in["opacities"].astype(np.float32))
    colors = np.ascontiguousarray(blend_in["colors"].astype(np.float32))
    sgids = np.ascontiguousarray(blend_in["sorted_gaussian_ids"].astype(np.int64))
    tranges = np.ascontiguousarray(blend_in["tile_ranges"].astype(np.int64))
    H = int(blend_in["H"][()])
    W = int(blend_in["W"][()])
    floor = 1.0 / 16384.0
    print(f"hero fixture H={H} W={W} M={means_2d.shape[0]}  P={sgids.shape[0]}")

    bench(
        "microblock_cull (BB-prefilter)",
        lambda: mod.microblock_cull(means_2d, covs_2d, opac, sgids, tranges, W // 32, H // 32, 32, floor),
    )
    mb_hdr, mb_str, _ = mod.microblock_cull(
        means_2d, covs_2d, opac, sgids, tranges, W // 32, H // 32, 32, floor
    )
    mb_hdr = np.asarray(mb_hdr)
    mb_str = np.asarray(mb_str)
    bench(
        "blend_microblock",
        lambda: mod.blend_microblock(
            means_2d, covs_2d, colors, opac, mb_hdr, mb_str, H, W, 32
        ),
    )


if __name__ == "__main__":
    main()
