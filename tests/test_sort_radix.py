"""Correctness tests for sort_and_bin — iter 017-C (63-bit composite quicksort).

Verifies that the 63-bit composite + quicksort implementation (iter 017-C) produces
byte-identical output to the iter-017 stable int64 path for a variety of synthetic
inputs, and adds per-tile depth-monotonicity assertions.
"""

import numpy as np
import numpy.testing as npt
import torch


# ---------------------------------------------------------------------------
# Reference implementation (old path, copied verbatim from e0a3640 baseline)
# ---------------------------------------------------------------------------

def _sort_and_bin_reference(gaussian_ids, tile_ids, depths, tiles_x, tiles_y):
    num_tiles = tiles_x * tiles_y
    max_depth = depths.max().item() + 1.0
    sort_keys = tile_ids.float() * max_depth + depths[gaussian_ids]
    sorted_indices = torch.argsort(sort_keys)
    sorted_gaussian_ids = gaussian_ids[sorted_indices]
    sorted_tile_ids = tile_ids[sorted_indices]

    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int64)
    if sorted_tile_ids.numel() > 0:
        changes = sorted_tile_ids[1:] != sorted_tile_ids[:-1]
        change_indices = torch.where(changes)[0] + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), change_indices])
        ends = torch.cat([change_indices, torch.tensor([len(sorted_tile_ids)])])
        segment_tiles = sorted_tile_ids[starts]
        tile_ranges[segment_tiles, 0] = starts
        tile_ranges[segment_tiles, 1] = ends

    return sorted_gaussian_ids, tile_ranges


# ---------------------------------------------------------------------------
# iter-017 implementation (stable int64 two-key composite — previous baseline)
# ---------------------------------------------------------------------------

def _sort_and_bin_new(gaussian_ids, tile_ids, depths, tiles_x, tiles_y):
    """iter-017 path: stable argsort on (tile_id<<32|depth_bits) int64 composite."""
    num_tiles = tiles_x * tiles_y

    depth_per_pair = depths[gaussian_ids]
    tile_ids_np = tile_ids.numpy().astype(np.int64)
    depth_bits = depth_per_pair.numpy().view(np.uint32).astype(np.int64)
    composite_keys = (tile_ids_np << np.int64(32)) | depth_bits

    sorted_indices_np = np.argsort(composite_keys, kind="stable")
    sorted_indices = torch.from_numpy(sorted_indices_np)
    sorted_gaussian_ids = gaussian_ids[sorted_indices]
    sorted_tile_ids = tile_ids[sorted_indices]

    tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int64)
    if sorted_tile_ids.numel() > 0:
        changes = sorted_tile_ids[1:] != sorted_tile_ids[:-1]
        change_indices = torch.where(changes)[0] + 1
        starts = torch.cat([torch.zeros(1, dtype=torch.int64), change_indices])
        ends = torch.cat([change_indices, torch.tensor([len(sorted_tile_ids)])])
        segment_tiles = sorted_tile_ids[starts]
        tile_ranges[segment_tiles, 0] = starts
        tile_ranges[segment_tiles, 1] = ends

    return sorted_gaussian_ids, tile_ranges


# ---------------------------------------------------------------------------
# iter-017-C implementation (63-bit composite quicksort — this patch)
# ---------------------------------------------------------------------------

def _sort_and_bin_017c(gaussian_ids, tile_ids, depths, tiles_x, tiles_y):
    """iter-017-C: single-pass 63-bit composite key with unstable quicksort.

    Bit layout of the int64 composite key:
        bits [62:53]  tile_id    (10 bits, ≤ 1023 for 32×32 tiles)
        bits [52:21]  depth_bits (32 bits, float32 reinterpreted as uint32)
        bits [20:0]   input_pos  (21 bits, unique pair index 0..N-1)

    Because every key is unique (input_pos is globally unique), an unstable sort
    produces the same output as a stable sort by (tile_id, depth_bits) with
    input-order tie-breaking.
    """
    num_tiles = tiles_x * tiles_y
    N = gaussian_ids.shape[0]

    gids_np   = gaussian_ids.numpy()
    tids_np   = tile_ids.numpy()
    depths_np = depths.numpy()
    dp_np     = depths_np[gids_np]

    counts  = np.bincount(tids_np.astype(np.int64) if tids_np.dtype != np.int64 else tids_np,
                          minlength=num_tiles)
    offsets = np.empty(num_tiles + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    if N == 0:
        tile_ranges_np = np.stack([offsets[:-1], offsets[1:]], axis=1)
        return gaussian_ids, torch.from_numpy(tile_ranges_np.copy())

    tids_i64 = tids_np.astype(np.int64)
    db_i64   = dp_np.view(np.uint32).astype(np.int64)

    if num_tiles <= (1 << 10) and N < (1 << 21):
        orig_i64  = np.arange(N, dtype=np.int64)
        composite = (tids_i64 << np.int64(53)) | (db_i64 << np.int64(21)) | orig_i64
        si = np.argsort(composite, kind="quicksort")
    else:
        composite = (tids_i64 << np.int64(32)) | db_i64
        si = np.argsort(composite, kind="stable")

    sorted_gaussian_ids = torch.from_numpy(gids_np[si])
    tile_ranges_np = np.stack([offsets[:-1], offsets[1:]], axis=1)
    return sorted_gaussian_ids, torch.from_numpy(tile_ranges_np.copy())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(rng, num_gaussians, num_pairs, tiles_x, tiles_y):
    num_tiles = tiles_x * tiles_y
    gaussian_ids = torch.from_numpy(rng.integers(0, num_gaussians, size=num_pairs).astype(np.int64))
    # Use int32 tile_ids to match the real pipeline (get_tile_assignments returns int32)
    tile_ids = torch.from_numpy(rng.integers(0, num_tiles, size=num_pairs).astype(np.int32))
    depths_np = rng.uniform(0.1, 100.0, size=num_gaussians).astype(np.float32)
    depths = torch.from_numpy(depths_np)
    return gaussian_ids, tile_ids, depths


def _assert_identical(name, ref_gids, ref_ranges, new_gids, new_ranges):
    npt.assert_array_equal(
        new_gids.numpy(), ref_gids.numpy(),
        err_msg=f"[{name}] sorted_gaussian_ids mismatch",
    )
    npt.assert_array_equal(
        new_ranges.numpy(), ref_ranges.numpy(),
        err_msg=f"[{name}] tile_ranges mismatch",
    )


def _assert_per_tile_depth_monotone(name, sorted_gids, tile_ranges, depths, tile_ids):
    """Assert that within every tile bucket of size ≥ 2, depths are non-decreasing.

    This is the primary semantic correctness guarantee of sort_and_bin.
    """
    depths_np   = depths.numpy()
    gids_np     = sorted_gids.numpy()
    ranges_np   = tile_ranges.numpy()
    num_tiles   = len(ranges_np)

    for t in range(num_tiles):
        s, e = int(ranges_np[t, 0]), int(ranges_np[t, 1])
        if e - s < 2:
            continue
        tile_depths = depths_np[gids_np[s:e]]
        deltas = np.diff(tile_depths.astype(np.float64))
        if not np.all(deltas >= -1e-9):
            bad = np.where(deltas < -1e-9)[0]
            raise AssertionError(
                f"[{name}] depth order violated in tile {t} at local pos {bad[0]}: "
                f"depth[{bad[0]}]={tile_depths[bad[0]]:.6g} > depth[{bad[0]+1}]={tile_depths[bad[0]+1]:.6g}"
            )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_small_random():
    """iter-017-C matches iter-017 on a small random workload."""
    rng = np.random.default_rng(42)
    gids, tids, depths = _make_inputs(rng, 500, 2000, 8, 8)
    r_gids, r_ranges = _sort_and_bin_new(gids, tids, depths, 8, 8)
    c_gids, c_ranges = _sort_and_bin_017c(gids, tids, depths, 8, 8)
    _assert_identical("small_random", r_gids, r_ranges, c_gids, c_ranges)
    _assert_per_tile_depth_monotone("small_random", c_gids, c_ranges, depths, tids)


def test_stitch_doll_scale():
    """Approximate scale of the real stitch_doll 1024x1024 workload.

    Verifies:
    1. iter-017-C output is byte-identical to iter-017 on full-scale inputs.
    2. Tile order in sorted output is non-decreasing.
    3. Within each tile bucket of size ≥ 2, depths are non-decreasing (per-tile monotone).
    4. tile_ranges[t] spans exactly the entries with tile == t in the sorted output.
    """
    rng = np.random.default_rng(7)
    gids, tids, depths = _make_inputs(rng, 280_000, 1_591_353, 32, 32)

    r_gids, r_ranges = _sort_and_bin_new(gids, tids, depths, 32, 32)
    c_gids, c_ranges = _sort_and_bin_017c(gids, tids, depths, 32, 32)

    _assert_identical("stitch_doll_scale", r_gids, r_ranges, c_gids, c_ranges)
    _assert_per_tile_depth_monotone("stitch_doll_scale", c_gids, c_ranges, depths, tids)

    # Tile order must be globally non-decreasing in the sorted output
    tids_np = tids.numpy().astype(np.int64)
    sorted_tiles = tids_np[r_gids.numpy()]
    assert np.all(np.diff(sorted_tiles) >= 0), "tile order violated at stitch_doll scale"

    # tile_ranges self-consistency
    ranges_np = c_ranges.numpy()
    for t in range(32 * 32):
        s, e = int(ranges_np[t, 0]), int(ranges_np[t, 1])
        if s < e:
            assert np.all(sorted_tiles[s:e] == t), f"tile_ranges mismatch at tile {t}"

    print(f"  (stitch_doll_scale: {len(gids):,} pairs, all checks passed)")


def test_empty():
    """Empty input: both implementations agree and return zero-sized tensors."""
    gids = torch.zeros(0, dtype=torch.int64)
    tids = torch.zeros(0, dtype=torch.int32)
    depths = torch.zeros(1, dtype=torch.float32)
    r_gids, r_ranges = _sort_and_bin_new(gids, tids, depths, 4, 4)
    c_gids, c_ranges = _sort_and_bin_017c(gids, tids, depths, 4, 4)
    _assert_identical("empty", r_gids, r_ranges, c_gids, c_ranges)


def test_single_tile():
    """All pairs in one tile — depth ordering within a single bucket."""
    rng = np.random.default_rng(99)
    num_pairs = 500
    gids = torch.from_numpy(rng.integers(0, 100, size=num_pairs).astype(np.int64))
    tids = torch.zeros(num_pairs, dtype=torch.int32)
    depths = torch.from_numpy(rng.uniform(0.5, 50.0, size=100).astype(np.float32))
    r_gids, r_ranges = _sort_and_bin_new(gids, tids, depths, 4, 4)
    c_gids, c_ranges = _sort_and_bin_017c(gids, tids, depths, 4, 4)
    _assert_identical("single_tile", r_gids, r_ranges, c_gids, c_ranges)
    _assert_per_tile_depth_monotone("single_tile", c_gids, c_ranges, depths, tids)


def test_single_entry_per_tile():
    """Exactly one pair per tile — no depth sorting needed, just bucket assignment."""
    num_tiles = 16
    gids   = torch.arange(num_tiles, dtype=torch.int64)
    tids   = torch.arange(num_tiles, dtype=torch.int32)
    depths = torch.from_numpy(np.random.default_rng(77).uniform(1.0, 10.0, size=num_tiles).astype(np.float32))
    r_gids, r_ranges = _sort_and_bin_new(gids, tids, depths, 4, 4)
    c_gids, c_ranges = _sort_and_bin_017c(gids, tids, depths, 4, 4)
    _assert_identical("single_entry_per_tile", r_gids, r_ranges, c_gids, c_ranges)
    # Monotone check on single-entry buckets (all trivially pass, but call it anyway)
    _assert_per_tile_depth_monotone("single_entry_per_tile", c_gids, c_ranges, depths, tids)


def test_equal_depths_stable():
    """Equal-depth entries within a tile must preserve input order (stability).

    The 63-bit key encodes input_pos in bits [20:0], which acts as a stable
    tie-breaker for bit-identical depth values.
    """
    rng = np.random.default_rng(55)
    num_pairs = 300
    gids   = torch.from_numpy(rng.integers(0, 50, size=num_pairs).astype(np.int64))
    # All pairs in tile 0, depths repeated across Gaussians → many equal depths
    tids   = torch.zeros(num_pairs, dtype=torch.int32)
    depths = torch.from_numpy((np.ones(50, dtype=np.float32) * 5.0))  # all same depth

    r_gids, r_ranges = _sort_and_bin_new(gids, tids, depths, 4, 4)
    c_gids, c_ranges = _sort_and_bin_017c(gids, tids, depths, 4, 4)
    _assert_identical("equal_depths_stable", r_gids, r_ranges, c_gids, c_ranges)
    _assert_per_tile_depth_monotone("equal_depths_stable", c_gids, c_ranges, depths, tids)


def test_near_zero_depths_semantic():
    """Very small depths: verify semantic ordering, not byte-identical to old reference.

    The old float composite key (tile_id * max_depth + depth) loses depth precision
    when depths ≈ 1e-7 because float32 mantissa can't distinguish them after the
    tile contribution.  The new int64 path handles these correctly.  We verify the
    new path's output is itself correctly ordered (tile first, then depth within tile).
    """
    rng = np.random.default_rng(13)
    gids, tids, depths = _make_inputs(rng, 200, 800, 4, 4)
    depths = depths * 1e-6  # tiny values
    c_gids, c_ranges = _sort_and_bin_017c(gids, tids, depths, 4, 4)

    tids_np     = tids.numpy().astype(np.int64)
    gids_np     = gids.numpy()
    depth_bits  = depths[gids].numpy().view(np.uint32).astype(np.int64)
    composite   = (tids_np << np.int64(32)) | depth_bits
    sorted_idx  = np.argsort(composite, kind="stable")
    sorted_tiles  = tids_np[sorted_idx]
    sorted_depths = depths.numpy()[gids_np[sorted_idx]]

    npt.assert_array_less(-1, np.diff(sorted_tiles.astype(np.int64)),
                          err_msg="tile order not non-decreasing")
    for tile in range(4 * 4):
        mask = sorted_tiles == tile
        tile_depths = sorted_depths[mask]
        if len(tile_depths) > 1:
            npt.assert_array_less(-1e-9, np.diff(tile_depths.astype(np.float64)),
                                  err_msg=f"depth order violated in tile {tile}")

    _assert_per_tile_depth_monotone("near_zero_depths", c_gids, c_ranges, depths, tids)


if __name__ == "__main__":
    print("Running sort_radix correctness tests (iter 017-C)...")
    test_empty()
    print("  PASS test_empty")
    test_single_entry_per_tile()
    print("  PASS test_single_entry_per_tile")
    test_single_tile()
    print("  PASS test_single_tile")
    test_equal_depths_stable()
    print("  PASS test_equal_depths_stable")
    test_small_random()
    print("  PASS test_small_random")
    test_near_zero_depths_semantic()
    print("  PASS test_near_zero_depths_semantic")
    test_stitch_doll_scale()
    print("  PASS test_stitch_doll_scale (1.6M pairs)")
    print("All tests passed.")
