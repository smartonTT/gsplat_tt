#!/usr/bin/env python3
"""Benchmark prep.gather variants in prepare_kernel_inputs.

Current: 9 fancy-index ops over `gids_np` (one per output column).
Candidate: build a single (M, 9) source array, do one 2D gather.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np


def _baseline(means_np, cov_inv_a, cov_inv_b, cov_inv_c, colors_np, opacities_np,
              gids_np, tile_origin_x, tile_origin_y):
    n = gids_np.shape[0]
    out = np.empty((n, 9), dtype=np.float32)
    out[:, 0] = means_np[gids_np, 0] - tile_origin_x
    out[:, 1] = means_np[gids_np, 1] - tile_origin_y
    out[:, 2] = cov_inv_a[gids_np]
    out[:, 3] = 2.0 * cov_inv_b[gids_np]
    out[:, 4] = cov_inv_c[gids_np]
    out[:, 5] = colors_np[gids_np, 0]
    out[:, 6] = colors_np[gids_np, 1]
    out[:, 7] = colors_np[gids_np, 2]
    out[:, 8] = opacities_np[gids_np]
    return out


def _stack_then_gather(means_np, cov_inv_a, cov_inv_b, cov_inv_c, colors_np, opacities_np,
                      gids_np, tile_origin_x, tile_origin_y):
    # Build a single (M, 9) source array once (per-frame), then one gather.
    source = np.empty((means_np.shape[0], 9), dtype=np.float32)
    source[:, 0] = means_np[:, 0]
    source[:, 1] = means_np[:, 1]
    source[:, 2] = cov_inv_a
    source[:, 3] = 2.0 * cov_inv_b
    source[:, 4] = cov_inv_c
    source[:, 5] = colors_np[:, 0]
    source[:, 6] = colors_np[:, 1]
    source[:, 7] = colors_np[:, 2]
    source[:, 8] = opacities_np
    out = source[gids_np]
    out[:, 0] -= tile_origin_x
    out[:, 1] -= tile_origin_y
    return out


def _take_2d(means_np, cov_inv_a, cov_inv_b, cov_inv_c, colors_np, opacities_np,
             gids_np, tile_origin_x, tile_origin_y):
    """Use np.take which is sometimes faster than fancy index."""
    source = np.empty((means_np.shape[0], 9), dtype=np.float32)
    source[:, 0] = means_np[:, 0]
    source[:, 1] = means_np[:, 1]
    source[:, 2] = cov_inv_a
    source[:, 3] = 2.0 * cov_inv_b
    source[:, 4] = cov_inv_c
    source[:, 5] = colors_np[:, 0]
    source[:, 6] = colors_np[:, 1]
    source[:, 7] = colors_np[:, 2]
    source[:, 8] = opacities_np
    out = np.take(source, gids_np, axis=0)
    out[:, 0] -= tile_origin_x
    out[:, 1] -= tile_origin_y
    return out


CANDIDATES = {
    "baseline_9_gathers": _baseline,
    "stack_then_gather": _stack_then_gather,
    "take_2d": _take_2d,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--m", type=int, default=341_426)
    ap.add_argument("--n", type=int, default=800_000)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    M = args.m
    N = args.n
    means_np = rng.random((M, 2), dtype=np.float32) * 1024.0
    cov_inv_a = rng.random(M, dtype=np.float32)
    cov_inv_b = rng.random(M, dtype=np.float32)
    cov_inv_c = rng.random(M, dtype=np.float32)
    colors_np = rng.random((M, 3), dtype=np.float32)
    opacities_np = rng.random(M, dtype=np.float32)
    gids_np = rng.integers(0, M, size=N).astype(np.int64)
    tile_origin_x = rng.random(N, dtype=np.float32) * 1024.0
    tile_origin_y = rng.random(N, dtype=np.float32) * 1024.0

    # Correctness
    ref = _baseline(means_np, cov_inv_a, cov_inv_b, cov_inv_c, colors_np, opacities_np,
                    gids_np, tile_origin_x, tile_origin_y)
    for name, fn in CANDIDATES.items():
        if name == "baseline_9_gathers":
            continue
        got = fn(means_np, cov_inv_a, cov_inv_b, cov_inv_c, colors_np, opacities_np,
                 gids_np, tile_origin_x, tile_origin_y)
        if not np.allclose(got, ref, atol=1e-5):
            print(f"  WARN: {name} disagrees with baseline (max diff {np.max(np.abs(got - ref))})")

    print(f"workload: M={M:,} Gaussians, N={N:,} pairs")
    print(f"\n{'candidate':<22} {'median ms':>10} {'min ms':>10} {'p90 ms':>10}")
    for name, fn in CANDIDATES.items():
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn(means_np, cov_inv_a, cov_inv_b, cov_inv_c, colors_np, opacities_np,
               gids_np, tile_origin_x, tile_origin_y)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        med = statistics.median(times)
        mn = min(times)
        p90 = times[int(0.9 * len(times))]
        print(f"{name:<22} {med:>10.2f} {mn:>10.2f} {p90:>10.2f}")


if __name__ == "__main__":
    main()
