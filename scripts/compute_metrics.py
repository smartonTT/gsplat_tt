"""Aggregate per-view PSNR + timing metrics for one iter.

Reads:
  <iter_dir>/*.png                         (per-view renders, named e.g. hero.png)
  <iter_dir>/timing.jsonl                  (one JSON object per frame)
  <reference_dir>/*.png                    (matching view PNGs)

Writes:
  <iter_dir>/*_diff10.png                  (10x amplified abs-diff per view)
  <iter_dir>/metrics.json                  (per-view PSNR, drop-rate, timing summary)

PSNR computed in fp64 so >100 dB is representable. timing.jsonl format:
one JSON object per measured frame with at minimum `view` and `total_ms`.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_fp64(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr_fp64(ref: np.ndarray, cand: np.ndarray) -> float:
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def tile_structure_ratio(ref: np.ndarray, cand: np.ndarray, tile: int = 32) -> float:
    """Quantify how much the diff is tile-correlated vs uniform noise.

    Interpretation (empirical, against the iter-000 reference):
      ~1   uniform i.i.d. noise (impossible target)
      ~6   clean fp32 reference render
      >30  broken-baseline territory
    """
    if ref.shape != cand.shape:
        return float("nan")
    diff = cand - ref
    H, W, C = diff.shape
    if H % tile != 0 or W % tile != 0:
        return float("nan")
    tile_means = diff.reshape(H // tile, tile, W // tile, tile, C).mean(axis=(1, 3))
    ratios = []
    for c in range(C):
        pix_std = float(diff[:, :, c].std())
        expected = pix_std / tile
        if expected <= 0:
            continue
        ratios.append(float(tile_means[:, :, c].std()) / expected)
    return float(np.mean(ratios)) if ratios else float("nan")


def write_diff10(ref: np.ndarray, cand: np.ndarray, out: Path) -> None:
    """10× amplified per-channel absolute color difference, clipped to [0, 1]."""
    amp = np.clip(np.abs(ref - cand) * 10.0, 0.0, 1.0)
    Image.fromarray((amp * 255.0).astype(np.uint8)).save(out)


def aggregate_timing(timing_path: Path) -> dict:
    rows = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{timing_path} has no rows")

    stage_keys = ("project", "tile_assign", "sort", "blend")

    # Sum-of-all-frames is the primary metric. Per-view median is informative.
    total_ms_all = sum(r.get("total_ms", 0.0) for r in rows)
    per_view: dict[str, list[float]] = {}
    per_stage: dict[str, list[float]] = {}
    for r in rows:
        v = r.get("view", "unknown")
        per_view.setdefault(v, []).append(r.get("total_ms", 0.0))
        for k in stage_keys:
            if k in r:
                per_stage.setdefault(f"{k}_ms", []).append(float(r[k]))
        for k, val in r.items():
            if k.endswith("_ms") and k != "total_ms" and k not in per_stage:
                per_stage.setdefault(k, []).append(float(val))

    return {
        "n_frames": len(rows),
        "sum_total_ms": total_ms_all,
        "per_view_median_ms": {v: statistics.median(t) for v, t in per_view.items()},
        "per_stage_median_ms": {k: statistics.median(v) for k, v in per_stage.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--class", dest="iter_class", default="algorithm")
    ap.add_argument("--prev-best-ms", default="Infinity")
    ap.add_argument("--views", nargs="*", default=None,
                    help="optional whitelist; default = all PNGs in iter-dir matching ref-dir")
    args = ap.parse_args()

    timing = aggregate_timing(args.iter_dir / "timing.jsonl")

    # Auto-discover views: any PNG in iter-dir for which the same filename exists in ref-dir.
    view_files = sorted(args.iter_dir.glob("*.png"))
    view_files = [p for p in view_files if not p.stem.endswith("_diff10")]
    if args.views:
        view_files = [p for p in view_files if p.stem in args.views]

    psnr_per_view: dict[str, float] = {}
    tile_structure_per_view: dict[str, float] = {}
    for cand_path in view_files:
        ref_path = args.reference_dir / cand_path.name
        if not ref_path.exists():
            continue
        ref = load_rgb_fp64(ref_path)
        cand = load_rgb_fp64(cand_path)
        if ref.shape != cand.shape:
            continue
        psnr_per_view[cand_path.stem] = psnr_fp64(ref, cand)
        tile_structure_per_view[cand_path.stem] = tile_structure_ratio(ref, cand)
        write_diff10(ref, cand, args.iter_dir / f"{cand_path.stem}_diff10.png")

    try:
        prev_best = float(args.prev_best_ms)
    except ValueError:
        prev_best = float("inf")

    out = {
        "iter_dir": args.iter_dir.name,
        "class": args.iter_class,
        "n_frames": timing["n_frames"],
        "sum_total_ms": timing["sum_total_ms"],
        "per_view_median_ms": timing["per_view_median_ms"],
        "per_stage_median_ms": timing["per_stage_median_ms"],
        "psnr_per_view": psnr_per_view,
        "tile_structure_per_view": tile_structure_per_view,
        "prev_best_sum_ms": prev_best,
    }
    out_path = args.iter_dir / "metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
