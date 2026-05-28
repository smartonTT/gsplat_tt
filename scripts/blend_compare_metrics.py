#!/usr/bin/env python3
"""Build diff PNGs + metrics.json from saved .npy blend outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    return float("inf") if mse <= 0 else 10.0 * np.log10(1.0 / mse)


def tile_structure_ratio(ref: np.ndarray, cand: np.ndarray, tile: int = 32) -> float:
    diff = cand.astype(np.float64) - ref.astype(np.float64)
    h, w, c = diff.shape
    tile_means = diff.reshape(h // tile, tile, w // tile, tile, c).mean(axis=(1, 3))
    ratios = []
    for ch in range(c):
        pix_std = float(diff[:, :, ch].std())
        if pix_std > 0:
            ratios.append(float(tile_means[:, :, ch].std()) / (pix_std / tile))
    return float(np.mean(ratios)) if ratios else float("nan")


def write_diff10(ref: np.ndarray, cand: np.ndarray, out: Path) -> None:
    amp = np.clip(np.abs(ref.astype(np.float64) - cand.astype(np.float64)) * 10.0, 0.0, 1.0)
    Image.fromarray((amp * 255.0).astype(np.uint8)).save(out)


def pair_metrics(ref: np.ndarray, cand: np.ndarray, label: str) -> dict:
    d = ref.astype(np.float64) - cand.astype(np.float64)
    return {
        "label": label,
        "psnr_dB": psnr(ref, cand),
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "tile_structure_ratio": tile_structure_ratio(ref, cand),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args()
    d = args.dir
    ref = np.load(d / "numpy_ref.npy")
    current = np.load(d / "tt_current.npy")
    master = np.load(d / "tt_master.npy")
    metrics = {
        "pairs": [
            pair_metrics(ref, current, "tt_current_vs_numpy_ref"),
            pair_metrics(ref, master, "tt_master_vs_numpy_ref"),
            pair_metrics(master, current, "tt_current_vs_tt_master"),
        ]
    }
    write_diff10(ref, current, d / "diff10_tt_current_vs_numpy_ref.png")
    write_diff10(ref, master, d / "diff10_tt_master_vs_numpy_ref.png")
    write_diff10(master, current, d / "diff10_tt_current_vs_tt_master.png")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
