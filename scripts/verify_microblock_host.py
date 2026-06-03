#!/usr/bin/env python3
"""Layer 1 gate: microblock host binning vs per-tile numpy reference (hero fixture).

Validates prepare_microblock_payload + alpha_blend_microblock before metal
iter-001 reader/compute wiring lands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from gsplat.rasterization import (  # noqa: E402
    alpha_blend,
    alpha_blend_microblock,
    microblock_cull,
    prepare_microblock_payload,
)

MB_CONTRIB_FLOOR = 1.0 / 16384.0
PSNR_FLOOR = 60.0


def psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main() -> None:
    fixtures = REPO / "tests" / "fixtures" / "hero"
    inp_path = fixtures / "blend_inputs.npz"
    ref_path = fixtures / "blend_output.npy"
    if not inp_path.exists() or not ref_path.exists():
        raise SystemExit(f"missing hero fixtures in {fixtures}")

    data = dict(np.load(inp_path))
    ref = np.load(ref_path)
    H, W = int(data["H"]), int(data["W"])
    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32

    means = torch.from_numpy(data["means_2d"])
    covs = torch.from_numpy(data["covs_2d"])
    colors = torch.from_numpy(data["colors"])
    opacities = torch.from_numpy(data["opacities"])
    gids = torch.from_numpy(data["sorted_gaussian_ids"])
    ranges = torch.from_numpy(data["tile_ranges"])

    payload = prepare_microblock_payload(
        means, covs, colors, opacities, gids, ranges, H, W, MB_CONTRIB_FLOOR
    )
    mb_header = payload["mb_header"]
    mb_stream = payload["mb_stream"]
    stats = payload["mb_stats"]

    # Invariant: header counts sum to stream length.
    assert int(mb_header[:, :, 1].sum()) == int(mb_stream.shape[0])

    mb_image = alpha_blend_microblock(
        means,
        covs,
        colors,
        opacities,
        torch.from_numpy(mb_header.reshape(tiles_x * tiles_y, 32, 2)),
        torch.from_numpy(mb_stream.astype(np.int64)),
        H,
        W,
    ).numpy()

    tile_ref = alpha_blend(
        means, covs, colors, opacities, gids, ranges, H, W
    ).numpy()

    p_vs_ref = psnr(ref, mb_image)
    p_vs_tile = psnr(tile_ref, mb_image)
    ok = p_vs_ref >= PSNR_FLOOR

    out = {
        "stage": "microblock_host",
        "layer1_pass": bool(ok),
        "psnr_vs_numpy_ref_dB": float(p_vs_ref),
        "psnr_vs_tile_blend_dB": float(p_vs_tile),
        "psnr_floor_dB": PSNR_FLOOR,
        "mb_contrib_floor": MB_CONTRIB_FLOOR,
        "mb_stats": stats,
        "stream_len": int(payload["mb_stream"].shape[0]),
        "stream_local_len": int(payload["mb_stream_local"].shape[0]),
        "coeff_table_shape": list(payload["coeff_table"].shape),
    }
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
