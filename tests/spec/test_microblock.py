"""Microblock cull / blend invariants (iter-006)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gsplat import rasterization

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hero"
CONTRIB_FLOOR = 15.0 / 255.0
MB_CONTRIB_FLOOR = 1.0 / 16384.0
_NUM_MB = 32


def _psnr(ref: np.ndarray, out: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - out.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def _run_cull_blend(inp: dict) -> tuple[np.ndarray, dict]:
    tiles_x = int((int(inp["W"]) + 31) // 32)
    tiles_y = int((int(inp["H"]) + 31) // 32)
    mb_header, mb_stream, stats = rasterization.microblock_cull(
        torch.from_numpy(inp["means_2d"]),
        torch.from_numpy(inp["covs_2d"]),
        torch.from_numpy(inp["opacities"]),
        torch.from_numpy(inp["sorted_gaussian_ids"]),
        torch.from_numpy(inp["tile_ranges"]),
        tiles_x,
        tiles_y,
        tile_size=int(inp.get("tile_size", 32)),
        mb_contrib_floor=MB_CONTRIB_FLOOR,
    )
    image = rasterization.alpha_blend_microblock(
        torch.from_numpy(inp["means_2d"]),
        torch.from_numpy(inp["covs_2d"]),
        torch.from_numpy(inp["colors"]),
        torch.from_numpy(inp["opacities"]),
        mb_header,
        mb_stream,
        int(inp["H"]),
        int(inp["W"]),
        tile_size=int(inp.get("tile_size", 32)),
    )
    return image.numpy(), stats


@pytest.fixture(scope="module")
def hero_blend_inputs() -> dict:
    return dict(np.load(FIXTURES / "blend_inputs.npz"))


def test_mb_stream_length_matches_header(hero_blend_inputs):
    inp = hero_blend_inputs
    tiles_x = int((int(inp["W"]) + 31) // 32)
    tiles_y = int((int(inp["H"]) + 31) // 32)
    mb_header, mb_stream, _ = rasterization.microblock_cull(
        torch.from_numpy(inp["means_2d"]),
        torch.from_numpy(inp["covs_2d"]),
        torch.from_numpy(inp["opacities"]),
        torch.from_numpy(inp["sorted_gaussian_ids"]),
        torch.from_numpy(inp["tile_ranges"]),
        tiles_x,
        tiles_y,
    )
    h = mb_header.numpy()
    assert int(h[:, :, 1].sum()) == int(mb_stream.numel())


def test_per_microblock_depth_monotonicity(hero_blend_inputs):
    inp = hero_blend_inputs
    tiles_x = int((int(inp["W"]) + 31) // 32)
    tiles_y = int((int(inp["H"]) + 31) // 32)
    mb_header, mb_stream, _ = rasterization.microblock_cull(
        torch.from_numpy(inp["means_2d"]),
        torch.from_numpy(inp["covs_2d"]),
        torch.from_numpy(inp["opacities"]),
        torch.from_numpy(inp["sorted_gaussian_ids"]),
        torch.from_numpy(inp["tile_ranges"]),
        tiles_x,
        tiles_y,
    )
    gids = inp["sorted_gaussian_ids"]
    ranges = inp["tile_ranges"]
    header = mb_header.numpy()
    stream = mb_stream.numpy()

    for tile_id in range(tiles_x * tiles_y):
        start, end = ranges[tile_id]
        if start == end:
            continue
        tile_g = gids[start:end]
        pos = {int(g): i for i, g in enumerate(tile_g)}
        for m in range(_NUM_MB):
            off, cnt = header[tile_id, m]
            if cnt == 0:
                continue
            slice_g = stream[off : off + cnt]
            indices = [pos[int(g)] for g in slice_g]
            assert indices == sorted(indices), f"tile {tile_id} mb {m} depth order broken"


def test_mask_completeness_random_synthetic():
    rng = np.random.default_rng(42)
    h, w = 32, 32
    tile_size = 32
    m = 32
    means = rng.uniform(0.5, w - 0.5, size=(m, 2)).astype(np.float32)
    opacities = rng.uniform(0.2, 1.0, size=(m,)).astype(np.float32)
    # Axis-aligned covs: clamp-to-mean is the Mahalanobis minimizer in each mb.
    covs = np.zeros((m, 2, 2), dtype=np.float32)
    for i in range(m):
        s1, s2 = rng.uniform(0.5, 4.0, size=2)
        covs[i, 0, 0] = s1 * s1
        covs[i, 1, 1] = s2 * s2
    colors = rng.uniform(0, 1, size=(m, 3)).astype(np.float32)

    # One tile (32×32): only gaussians that survive per-tile Mahalanobis cull
    # (invariant 1 is stated for post-tile-cull pairs).
    a = covs[:, 0, 0]
    b = covs[:, 0, 1]
    c = covs[:, 1, 1]
    det = np.maximum(a * c - b * b, 1e-6)
    cx = np.clip(means[:, 0], 0.0, float(tile_size))
    cy = np.clip(means[:, 1], 0.0, float(tile_size))
    dx_c = cx - means[:, 0]
    dy_c = cy - means[:, 1]
    m2_tile = (c * dx_c * dx_c - 2.0 * b * dx_c * dy_c + a * dy_c * dy_c) / det
    tile_keep = opacities * np.exp(-0.5 * m2_tile) >= CONTRIB_FLOOR
    kept_ids = np.where(tile_keep)[0].astype(np.int64)
    sorted_gaussian_ids = kept_ids
    tile_ranges = np.array([[0, len(kept_ids)]], dtype=np.int64)

    mb_header, mb_stream, _ = rasterization.microblock_cull(
        torch.from_numpy(means),
        torch.from_numpy(covs),
        torch.from_numpy(opacities),
        torch.from_numpy(sorted_gaussian_ids),
        torch.from_numpy(tile_ranges),
        tiles_x=1,
        tiles_y=1,
        tile_size=tile_size,
        mb_contrib_floor=MB_CONTRIB_FLOOR,
    )
    header = mb_header.numpy()
    stream = mb_stream.numpy()

    cov_inv = np.zeros_like(covs)
    a = covs[:, 0, 0]
    b = covs[:, 0, 1]
    c = covs[:, 1, 1]
    det = np.maximum(a * c - b * b, 1e-6)
    cov_inv[:, 0, 0] = c / det
    cov_inv[:, 0, 1] = -b / det
    cov_inv[:, 1, 0] = -b / det
    cov_inv[:, 1, 1] = a / det

    mb_origin_x = (np.arange(_NUM_MB) & 3) * 8
    mb_origin_y = (np.arange(_NUM_MB) >> 2) * 4

    active_pairs: set[tuple[int, int]] = set()
    for tile_id in range(1):
        for mb_m in range(_NUM_MB):
            off, cnt = header[tile_id, mb_m]
            for g in stream[off : off + cnt]:
                active_pairs.add((int(g), mb_m))

    for py in range(h):
        for px in range(w):
            px_c = px + 0.5
            py_c = py + 0.5
            mb_m = (py // 4) * 4 + (px // 8)
            for gi in kept_ids:
                dx = px_c - means[gi, 0]
                dy = py_c - means[gi, 1]
                ci = cov_inv[gi]
                power = -0.5 * (
                    ci[0, 0] * dx * dx + 2.0 * ci[0, 1] * dx * dy + ci[1, 1] * dy * dy
                )
                alpha = min(float(opacities[gi] * np.exp(min(power, 0.0))), 0.99)
                if alpha >= CONTRIB_FLOOR:
                    assert (gi, mb_m) in active_pairs, (
                        f"pixel ({px},{py}) g={gi} mb={mb_m} alpha={alpha}"
                    )


def test_drop_rate_under_five_percent(hero_blend_inputs):
    _, stats = _run_cull_blend(hero_blend_inputs)
    assert stats["drop_pct"] < 5.0


def test_work_reduction_above_50_percent(hero_blend_inputs):
    _, stats = _run_cull_blend(hero_blend_inputs)
    assert stats["work_reduction_pct"] >= 50.0, (
        f"work_reduction_pct {stats['work_reduction_pct']:.2f}% below 50% gate"
    )


def test_end_to_end_psnr_vs_hero_reference(hero_blend_inputs):
    ref = np.load(FIXTURES / "blend_output.npy")
    out, _ = _run_cull_blend(hero_blend_inputs)
    psnr = _psnr(ref, out)
    assert psnr >= 60.0, f"PSNR {psnr:.2f} dB < 60 dB gate"
