"""Tests for the preallocated-buffer prep path in KernelBackend.

We cannot instantiate KernelBackend without a running daemon binary, so we
test the correctness of the hot-path computation by exercising the individual
pieces that were inlined into blend():

1. _get_tile_origins  — per-resolution tile-origin cache
2. The inline cov_inv computation matches prepare_kernel_inputs' output.
3. The inline gather + tile-origin fill matches prepare_kernel_inputs' output.
4. Buffer capacity does not grow between two calls at the same scene size.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gsplat.rasterization import (
    _get_px_py_grids,
    prepare_kernel_inputs,
    sort_and_bin,
    get_tile_assignments,
    project_gaussians,
)
from backends.tt.backend import _get_tile_origins


# ---------------------------------------------------------------------------
# Helpers that replicate the inline prep logic without a daemon
# ---------------------------------------------------------------------------

class _ScratchBuffers:
    """Minimal scratch-buffer container mirroring KernelBackend's fields."""

    def __init__(self):
        self._entry_cap = 0
        self._gauss_cap = 0
        self._tile_cap = 0
        self._attr_buf = None
        self._tile_ox_buf = None
        self._tile_oy_buf = None
        self._gids_u32_buf = None
        self._cov_inv_a = None
        self._cov_inv_b = None
        self._cov_inv_c = None
        self._det_buf = None
        self._tmp_m_buf = None
        self._tile_offsets_buf = None
        self._offsets_f32_buf = None

    def ensure(self, total_entries: int, num_visible: int, num_tiles: int) -> None:
        if total_entries > self._entry_cap:
            cap = total_entries + max(total_entries // 4, 64)
            self._attr_buf = np.empty((cap, 5), dtype=np.float32)
            self._tile_ox_buf = np.empty(cap, dtype=np.float32)
            self._tile_oy_buf = np.empty(cap, dtype=np.float32)
            self._gids_u32_buf = np.empty(cap, dtype=np.uint32)
            self._entry_cap = cap
        if num_visible > self._gauss_cap:
            cap = num_visible + max(num_visible // 4, 64)
            self._cov_inv_a = np.empty(cap, dtype=np.float32)
            self._cov_inv_b = np.empty(cap, dtype=np.float32)
            self._cov_inv_c = np.empty(cap, dtype=np.float32)
            self._det_buf = np.empty(cap, dtype=np.float32)
            self._tmp_m_buf = np.empty(cap, dtype=np.float32)
            self._gauss_cap = cap
        if num_tiles > self._tile_cap:
            cap = num_tiles + 8
            self._tile_offsets_buf = np.empty(cap + 1, dtype=np.uint32)
            self._offsets_f32_buf = np.empty(cap + 1, dtype=np.float32)
            self._tile_cap = cap

    @property
    def entry_cap(self):
        return self._entry_cap

    @property
    def gauss_cap(self):
        return self._gauss_cap


def _run_prep_fast(
    means_2d: torch.Tensor,
    covs_2d: torch.Tensor,
    sorted_gaussian_ids: torch.Tensor,
    tile_ranges: torch.Tensor,
    H: int,
    W: int,
    bufs: _ScratchBuffers,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the optimised inline prep and return (attr, tile_offsets)."""
    M = int(means_2d.shape[0])
    gids_np = sorted_gaussian_ids.numpy()
    total_entries = int(gids_np.shape[0])
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    num_tiles = tiles_x * tiles_y

    bufs.ensure(total_entries, M, num_tiles)

    attr = bufs._attr_buf[:total_entries]
    tile_ox = bufs._tile_ox_buf[:total_entries]
    tile_oy = bufs._tile_oy_buf[:total_entries]
    gids_u32 = bufs._gids_u32_buf[:total_entries]
    cov_inv_a = bufs._cov_inv_a[:M]
    cov_inv_b = bufs._cov_inv_b[:M]
    cov_inv_c = bufs._cov_inv_c[:M]
    det_buf = bufs._det_buf[:M]
    tmp_m = bufs._tmp_m_buf[:M]
    tile_offsets = bufs._tile_offsets_buf[:num_tiles + 1]

    np.copyto(gids_u32, gids_np, casting="unsafe")

    ranges_np = tile_ranges.numpy()
    counts_u32 = (ranges_np[:, 1] - ranges_np[:, 0]).astype(np.uint32)
    tile_offsets[0] = 0
    np.cumsum(counts_u32, out=tile_offsets[1:])

    covs_np = covs_2d.numpy()
    np.multiply(covs_np[:, 0, 0], covs_np[:, 1, 1], out=det_buf)
    np.multiply(covs_np[:, 0, 1], covs_np[:, 0, 1], out=tmp_m)
    np.subtract(det_buf, tmp_m, out=det_buf)
    np.clip(det_buf, 1e-6, None, out=det_buf)
    np.divide(covs_np[:, 1, 1], det_buf, out=cov_inv_a)
    np.divide(covs_np[:, 0, 1], det_buf, out=cov_inv_b)
    cov_inv_b *= -1.0
    np.divide(covs_np[:, 0, 0], det_buf, out=cov_inv_c)

    tile_ox_all, tile_oy_all = _get_tile_origins(H, W)
    for t in range(num_tiles):
        s = int(tile_offsets[t])
        e = int(tile_offsets[t + 1])
        if e > s:
            tile_ox[s:e] = tile_ox_all[t]
            tile_oy[s:e] = tile_oy_all[t]

    means_np = means_2d.numpy()
    attr[:, 0] = means_np[gids_np, 0] - tile_ox
    attr[:, 1] = means_np[gids_np, 1] - tile_oy
    attr[:, 2] = cov_inv_a[gids_np]
    attr[:, 3] = 2.0 * cov_inv_b[gids_np]
    attr[:, 4] = cov_inv_c[gids_np]

    return attr.copy(), tile_offsets.copy()


# ---------------------------------------------------------------------------
# Synthetic scene fixture
# ---------------------------------------------------------------------------

def _make_scene(N: int = 500, H: int = 128, W: int = 128, seed: int = 42):
    """Return minimal but realistic inputs for the prep pipeline."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    means = torch.rand(N, 3, generator=rng) * 4 - 2       # world coords ±2
    scales = torch.rand(N, 3, generator=rng) * 0.3 + 0.05
    rots = torch.randn(N, 4, generator=rng)
    rots = rots / rots.norm(dim=1, keepdim=True)
    opacities = torch.rand(N, generator=rng) * 0.8 + 0.1
    colors = torch.rand(N, 3, generator=rng)

    fx = fy = 100.0
    cx, cy = W / 2.0, H / 2.0
    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    extrinsics = torch.eye(4, dtype=torch.float32)
    extrinsics[2, 3] = 3.0  # camera 3 units back

    from gsplat.rasterization import project_gaussians, get_tile_assignments, sort_and_bin
    means_2d, covs_2d, depths, radii, valid = project_gaussians(
        means, scales, rots, extrinsics, intrinsics, H, W, opacities=opacities,
    )
    if means_2d.shape[0] == 0:
        pytest.skip("no visible Gaussians in synthetic scene")

    g_ids, t_ids, _ = get_tile_assignments(means_2d, radii, H, W, tile_size=32)
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = sort_and_bin(g_ids, t_ids, depths, tiles_x, tiles_y)
    return means_2d, covs_2d, colors[valid], opacities[valid], sorted_gids, tile_ranges, H, W


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tile_origins_shape_and_values():
    """_get_tile_origins returns correct (num_tiles,) arrays for several resolutions."""
    for H, W in [(128, 128), (256, 192), (1024, 1024)]:
        tiles_x = (W + 31) // 32
        tiles_y = (H + 31) // 32
        num_tiles = tiles_x * tiles_y
        ox, oy = _get_tile_origins(H, W)
        assert ox.shape == (num_tiles,)
        assert oy.shape == (num_tiles,)
        # Tile 0 is always at origin
        assert ox[0] == 0.0 and oy[0] == 0.0
        # Last tile in first row
        assert ox[tiles_x - 1] == (tiles_x - 1) * 32.0
        assert oy[tiles_x - 1] == 0.0
        # First tile in second row
        assert ox[tiles_x] == 0.0
        assert oy[tiles_x] == 32.0


def test_tile_origins_cached():
    """Second call returns same object (no recomputation)."""
    ox1, oy1 = _get_tile_origins(64, 64)
    ox2, oy2 = _get_tile_origins(64, 64)
    assert ox1 is ox2 and oy1 is oy2


def test_prep_matches_reference():
    """Fast inline prep output matches prepare_kernel_inputs reference."""
    means_2d, covs_2d, colors, opacities, sorted_gids, tile_ranges, H, W = _make_scene()

    # Reference output from the existing (non-optimised) function.
    ref_attr, ref_offsets, _, _, ref_gids = prepare_kernel_inputs(
        means_2d, covs_2d, colors, opacities, sorted_gids, tile_ranges, H, W,
        static_handled_externally=True,
    )

    bufs = _ScratchBuffers()
    fast_attr, fast_offsets = _run_prep_fast(
        means_2d, covs_2d, sorted_gids, tile_ranges, H, W, bufs
    )

    np.testing.assert_allclose(
        fast_attr, ref_attr, rtol=1e-5, atol=1e-5,
        err_msg="attr_packs mismatch between fast and reference prep",
    )
    np.testing.assert_array_equal(
        fast_offsets[:-1],   # fast returns uint32; ref returns uint32 too
        ref_offsets[:-1],
        err_msg="tile_offsets mismatch",
    )
    # gids should match (both uint32)
    np.testing.assert_array_equal(
        bufs._gids_u32_buf[: sorted_gids.shape[0]],
        ref_gids,
        err_msg="gids_u32 mismatch",
    )


def test_buffer_reuse_no_growth():
    """Calling prep twice at the same scene size does not grow buffers."""
    means_2d, covs_2d, colors, opacities, sorted_gids, tile_ranges, H, W = _make_scene()

    bufs = _ScratchBuffers()
    _run_prep_fast(means_2d, covs_2d, sorted_gids, tile_ranges, H, W, bufs)

    cap_entry_after_first = bufs.entry_cap
    cap_gauss_after_first = bufs.gauss_cap

    # Second call — same inputs, same sizes.
    _run_prep_fast(means_2d, covs_2d, sorted_gids, tile_ranges, H, W, bufs)

    assert bufs.entry_cap == cap_entry_after_first, "entry_cap grew on 2nd call"
    assert bufs.gauss_cap == cap_gauss_after_first, "gauss_cap grew on 2nd call"


def test_buffer_grows_on_larger_scene():
    """Buffer capacity grows when a larger scene arrives after a small one."""
    means_small, covs_small, colors_s, opac_s, gids_s, ranges_s, H, W = _make_scene(
        N=100, H=64, W=64, seed=1
    )
    means_large, covs_large, colors_l, opac_l, gids_l, ranges_l, H2, W2 = _make_scene(
        N=500, H=128, W=128, seed=2
    )

    bufs = _ScratchBuffers()
    _run_prep_fast(means_small, covs_small, gids_s, ranges_s, H, W, bufs)
    cap_after_small = bufs.entry_cap

    _run_prep_fast(means_large, covs_large, gids_l, ranges_l, H2, W2, bufs)
    assert bufs.entry_cap > cap_after_small, "capacity should have grown for larger scene"


def test_px_py_grids_unchanged():
    """_get_px_py_grids still returns the same cached grids (no regression)."""
    px, py = _get_px_py_grids(128, 128)
    assert px.shape == (16, 32, 32)
    assert py.shape == (16, 32, 32)
    # Tile-local: first element of every tile is always 0.5
    assert float(px[0, 0, 0]) == pytest.approx(0.5)
    assert float(py[0, 0, 0]) == pytest.approx(0.5)
