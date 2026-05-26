#!/usr/bin/env python3
"""Benchmark host-side sort_and_bin alternatives against the current torch.argsort.

Generates a realistic (gaussian_ids, tile_ids, depths) workload either from a
real scene (preferred — `--scene <name>`) or from synthetic data tuned to
match stitch_doll@1024 (~800k pairs, 1024 tiles).

For each candidate, measures median wall ms over 21 reps and prints a table.
All candidates must produce the same (sorted_gaussian_ids, tile_ranges) output
as the current implementation, so this is also a correctness regression check.
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch


def _baseline_torch(gids: torch.Tensor, tids: torch.Tensor, depths: torch.Tensor,
                    num_tiles: int):
    """Current implementation (mirrors gsplat.rasterization.sort_and_bin)."""
    max_depth = depths.max().item() + 1.0
    keys = tids.float() * max_depth + depths[gids]
    order = torch.argsort(keys)
    sgids = gids[order]
    stids = tids[order]
    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int64)
    if stids.numel() > 0:
        changes = stids[1:] != stids[:-1]
        ch = torch.where(changes)[0] + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), ch])
        ends = torch.cat([ch, torch.tensor([len(stids)])])
        segs = stids[starts]
        tile_ranges[segs, 0] = starts
        tile_ranges[segs, 1] = ends
    return sgids, tile_ranges


def _np_lexsort(gids_t, tids_t, depths_t, num_tiles):
    """numpy.lexsort with (depth, tile_id) — tile_id is primary."""
    gids = gids_t.numpy()
    tids = tids_t.numpy()
    depths = depths_t.numpy()
    pair_depths = depths[gids]
    # lexsort: last key is primary. Want tile_id primary, depth secondary.
    order = np.lexsort((pair_depths, tids))
    sgids = gids[order]
    stids = tids[order]
    return _ranges_from_sorted(sgids, stids, num_tiles)


def _uint64_argsort(gids_t, tids_t, depths_t, num_tiles):
    """Pack (tile_id, depth_bits) into one uint64, sort via numpy stable.

    Depths are all > 0 in screen space (filtered by near plane), so float32
    bit reinterpretation as uint32 preserves ordering for positives. tile_id
    fits in 32 bits (max ~16k for 4K-render upper bound).
    """
    gids = gids_t.numpy()
    tids = tids_t.numpy().astype(np.uint64)
    depths = depths_t.numpy()
    pair_depth_bits = depths[gids_t.numpy()].view(np.uint32).astype(np.uint64)
    composite = (tids << 32) | pair_depth_bits
    order = np.argsort(composite, kind='stable')
    sgids = gids[order]
    stids = tids_t.numpy()[order]
    return _ranges_from_sorted(sgids, stids, num_tiles)


def _bucket_then_sort(gids_t, tids_t, depths_t, num_tiles):
    """Bucket by tile_id then per-tile depth sort.

    For each tile: gather pairs with that tile_id, sort by depth, concat.
    Build tile_ranges as a byproduct.
    """
    gids = gids_t.numpy()
    tids = tids_t.numpy()
    depths = depths_t.numpy()

    # Stable bucket via argsort on tile_id (radix-fast for small int range)
    order_by_tile = np.argsort(tids, kind='stable')
    sorted_tids = tids[order_by_tile]
    pair_depths = depths[gids]

    # Per-tile boundaries
    changes = np.flatnonzero(sorted_tids[1:] != sorted_tids[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(sorted_tids)]))

    sgids = np.empty_like(gids)
    tile_ranges = np.zeros((num_tiles, 2), dtype=np.int64)
    for s, e in zip(starts, ends):
        slc = order_by_tile[s:e]
        tile_id = sorted_tids[s]
        sub_depths = pair_depths[slc]
        inner = np.argsort(sub_depths, kind='stable')
        sgids[s:e] = gids[slc][inner]
        tile_ranges[tile_id, 0] = s
        tile_ranges[tile_id, 1] = e
    return torch.from_numpy(sgids), torch.from_numpy(tile_ranges)


def _counting_sort_bucket(gids_t, tids_t, depths_t, num_tiles):
    """Counting sort by tile_id (O(N)) + per-tile depth sort.

    Skips the comparison sort entirely for the tile_id step — tile_id has
    tiny range (num_tiles ≤ ~16k for 4K) so counting sort is two linear
    passes total. Then per-tile depth slices are small (avg N/T ≈ 800).
    """
    gids = gids_t.numpy()
    tids = tids_t.numpy()
    depths = depths_t.numpy()

    # Histogram + cumulative starts
    counts = np.bincount(tids, minlength=num_tiles)
    starts = np.zeros(num_tiles + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])

    # Scatter pair indices into per-tile contiguous regions.
    # write_pos[t] = current write head for tile t (init to starts[t]).
    write_pos = starts[:-1].copy()
    bucketed = np.empty(len(tids), dtype=np.int64)
    # Vectorized scatter via cumulative count of (tid, 1) — but a python loop
    # over N=800k pairs is too slow. Use np.argsort on tids as a fast stable
    # bucket: numpy already special-cases small-range int sorts.
    # Faster: compute per-element write position with a "rank within tile" trick.
    # Using np.add.at would be O(N) but is slow (no SIMD). Replace with a single
    # stable argsort on tids — numpy uses radix for ints (timsort-fallback for
    # large range, but our range ≤ 16k is tiny).
    order = np.argsort(tids, kind='stable')
    pair_depths = depths[gids]

    sgids = np.empty_like(gids)
    for t in range(num_tiles):
        s, e = starts[t], starts[t + 1]
        if s == e:
            continue
        slc = order[s:e]
        sub_depths = pair_depths[slc]
        inner = np.argsort(sub_depths, kind='stable')
        sgids[s:e] = gids[slc][inner]

    tile_ranges = np.zeros((num_tiles, 2), dtype=np.int64)
    tile_ranges[:, 0] = starts[:-1]
    tile_ranges[:, 1] = starts[1:]
    # Empty tiles should have (0, 0) not (start, start). Original convention:
    nonempty = counts > 0
    tile_ranges[~nonempty] = 0
    return torch.from_numpy(sgids), torch.from_numpy(tile_ranges)


def _torch_stable_int(gids_t, tids_t, depths_t, num_tiles):
    """Build int64 composite (tile_id * 2^32 + depth_bits) as a torch tensor,
    then argsort. Stays in torch but avoids the float-precision-loss key."""
    pair_depths = depths_t[gids_t]
    # Reinterpret float32 depths as int32 bits (depths > 0 so this is monotonic
    # under unsigned comparison; but int32 view sorts correctly for positives).
    depth_bits = pair_depths.view(torch.int32).to(torch.int64)
    composite = (tids_t.to(torch.int64) << 32) | depth_bits
    order = torch.argsort(composite)
    sgids = gids_t[order]
    stids = tids_t[order]
    return _ranges_from_sorted(sgids.numpy(), stids.numpy(), num_tiles)


def _numpy_argsort_uint64_inplace(gids_t, tids_t, depths_t, num_tiles):
    """Same as _uint64_argsort but uses np.argsort default (no kind kwarg) —
    NumPy's quicksort is sometimes faster for random data than mergesort."""
    gids = gids_t.numpy()
    tids_np = tids_t.numpy()
    depths = depths_t.numpy()
    tids_u64 = tids_np.astype(np.uint64)
    pair_depth_bits = depths[gids].view(np.uint32).astype(np.uint64)
    composite = (tids_u64 << 32) | pair_depth_bits
    order = np.argsort(composite)  # default = quicksort for ints
    sgids = gids[order]
    stids = tids_np[order]
    return _ranges_from_sorted(sgids, stids, num_tiles)


def _ranges_from_sorted(sgids: np.ndarray, stids: np.ndarray, num_tiles: int):
    tile_ranges = np.zeros((num_tiles, 2), dtype=np.int64)
    if len(stids) > 0:
        changes = np.flatnonzero(stids[1:] != stids[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(stids)]))
        segs = stids[starts]
        tile_ranges[segs, 0] = starts
        tile_ranges[segs, 1] = ends
    return torch.from_numpy(sgids), torch.from_numpy(tile_ranges)


CANDIDATES = {
    "baseline_torch": _baseline_torch,
    "np_lexsort": _np_lexsort,
    "uint64_argsort_stable": _uint64_argsort,
    "uint64_argsort_quicksort": _numpy_argsort_uint64_inplace,
    "bucket_then_sort": _bucket_then_sort,
    "counting_sort_bucket": _counting_sort_bucket,
    "torch_int64_composite": _torch_stable_int,
}


def _generate_synthetic(num_pairs=800_000, num_tiles=1024, seed=42):
    rng = np.random.default_rng(seed)
    M = 341_426  # stitch_doll Gaussian count
    tids = rng.integers(0, num_tiles, size=num_pairs).astype(np.int64)
    # Realistic-ish: each pair points at a Gaussian; depths span a typical range
    gids = rng.integers(0, M, size=num_pairs).astype(np.int64)
    depths = rng.uniform(0.5, 30.0, size=M).astype(np.float32)
    return torch.from_numpy(gids), torch.from_numpy(tids), torch.from_numpy(depths), num_tiles


def _load_from_scene(scene_path: str, size: int = 1024):
    """Run project + tile_assign on a real scene; return (gids, tids, depths, num_tiles)."""
    from gsplat.loading_gaussians import load_gaussians
    from gsplat.rasterization import project_gaussians, get_tile_assignments
    from gsplat.utils import build_view_matrix
    g = load_gaussians(scene_path)
    # Stitch hero camera from start_dev_viewer or saved render fixture.
    # Use a centered identity-ish pose for benchmarking purposes.
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[2, 3] = 8.0  # back off so the figure fits
    w2c = torch.linalg.inv(c2w)
    fov_deg = 45.0
    focal = size / (2.0 * np.tan(np.radians(fov_deg / 2.0)))
    means_2d, covs_2d, radii, depths, visible = project_gaussians(
        g.means, g.scales, g.quats, w2c, focal, focal, size / 2, size / 2,
        size, size, near=0.01,
    )
    gids, tids, _ = get_tile_assignments(means_2d, radii, size, size, 32)
    num_tiles = (size // 32) * (size // 32)
    return gids, tids, depths, num_tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="Path to .ply (real workload); otherwise synthetic")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=21)
    args = ap.parse_args()

    if args.scene:
        gids, tids, depths, num_tiles = _load_from_scene(args.scene, args.size)
    else:
        gids, tids, depths, num_tiles = _generate_synthetic()

    print(f"workload: {len(gids):,} pairs, {num_tiles:,} tiles, "
          f"{depths.numel():,} Gaussians")

    # Correctness: every candidate must match baseline
    ref_sgids, ref_ranges = _baseline_torch(gids, tids, depths, num_tiles)
    for name, fn in CANDIDATES.items():
        if name == "baseline_torch":
            continue
        sgids, ranges = fn(gids, tids, depths, num_tiles)
        ok_g = torch.equal(sgids, ref_sgids)
        ok_r = torch.equal(ranges, ref_ranges)
        if not (ok_g and ok_r):
            print(f"  WARN: {name} disagrees with baseline (sgids={ok_g} ranges={ok_r})")

    print(f"\n{'candidate':<22} {'median ms':>10} {'min ms':>10} {'p90 ms':>10}")
    for name, fn in CANDIDATES.items():
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            fn(gids, tids, depths, num_tiles)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        med = statistics.median(times)
        mn = min(times)
        p90 = times[int(0.9 * len(times))]
        print(f"{name:<22} {med:>10.2f} {mn:>10.2f} {p90:>10.2f}")


if __name__ == "__main__":
    main()
