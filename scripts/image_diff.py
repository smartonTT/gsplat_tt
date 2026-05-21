"""Compute PSNR / SSIM / RMSE / max-abs-diff / NaN-check between two PNGs and
emit an exit code that drives the alpha-blend optimization loop's permissive
visual-regression gate.

Usage:
  python scripts/image_diff.py <reference.png> <candidate.png>
                              [--anchor prev|baseline]
                              [--amplified-diff out.png]
                              [--json]

Exit codes:
  0  clean keep
  2  kept but NEEDS_REVIEW (perceptual drop)
  1  hard reject (broken output)

Hard reject (universal, both anchors):
  - NaN or Inf in candidate
  - SSIM < 0.75
  - PSNR < 20 dB
  - mean abs diff > 25 LSB (= mean(|a-b|) * 255 over the whole image)

NEEDS_REVIEW (anchor-dependent):
  --anchor prev (default):     SSIM drop > 0.05 OR PSNR < 35 dB vs the reference
  --anchor baseline:           SSIM < 0.92 OR PSNR < 32 dB vs the reference

`SSIM drop > 0.05` is only meaningful when calling against the previous-kept
image (the natural delta is "what changed this iteration"). For the baseline
anchor we use the absolute SSIM/PSNR thresholds since drift is cumulative.

`--amplified-diff out.png` writes a 10x-amplified gamma-2.2 diff PNG to the
given path for human audit.

`--json` prints metrics as a single line of JSON instead of human-readable
key=value (use in scripts that parse the output).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn  # type: ignore
except ImportError:
    print("ERROR: scikit-image not installed. Run: pip install scikit-image",
          file=sys.stderr)
    sys.exit(3)


# --- Thresholds ----------------------------------------------------------

HARD_REJECT_SSIM = 0.75
HARD_REJECT_PSNR_DB = 20.0
HARD_REJECT_MEAN_ABS_LSB = 25.0

# Per-anchor "kept but NEEDS_REVIEW" thresholds.
NEEDS_REVIEW_PREV_SSIM_DROP = 0.05  # interpreted as: SSIM < 1.0 - 0.05 = 0.95
NEEDS_REVIEW_PREV_PSNR_DB = 35.0
NEEDS_REVIEW_BASELINE_SSIM = 0.92
NEEDS_REVIEW_BASELINE_PSNR_DB = 32.0


def load_rgb(path: Path) -> np.ndarray:
    """Load PNG as (H, W, 3) float32 in [0, 1]. Drops alpha if present."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def compute_metrics(ref: np.ndarray, cand: np.ndarray) -> dict:
    """All metrics + raw arrays needed for the gate. Both inputs must be
    (H, W, 3) float32 in [0, 1] and the same shape.
    """
    diff = cand - ref
    abs_diff = np.abs(diff)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    psnr_db = 20.0 * np.log10(1.0 / rmse) if rmse > 0 else float("inf")
    # SSIM on luminance — fast (~10 ms at 640x640) and the perceptual safety net.
    ssim = float(ssim_fn(ref, cand, channel_axis=2, data_range=1.0))
    max_abs_per_chan = (abs_diff * 255.0).reshape(-1, 3).max(axis=0)
    frac_above_4lsb = float(((abs_diff * 255.0) > 4.0).any(axis=-1).mean())
    mean_abs_lsb = float(abs_diff.mean() * 255.0)
    has_nan_or_inf = bool(np.isnan(cand).any() or np.isinf(cand).any())
    return dict(
        psnr_db=psnr_db,
        ssim=ssim,
        rmse=rmse,
        max_abs_diff_R=int(max_abs_per_chan[0]),
        max_abs_diff_G=int(max_abs_per_chan[1]),
        max_abs_diff_B=int(max_abs_per_chan[2]),
        frac_pixels_gt_4LSB=frac_above_4lsb,
        mean_abs_lsb=mean_abs_lsb,
        has_nan_or_inf=has_nan_or_inf,
    )


def gate(metrics: dict, anchor: str) -> tuple[int, str, list[str]]:
    """Return (exit_code, label, reasons)."""
    reasons_hard, reasons_review = [], []

    if metrics["has_nan_or_inf"]:
        reasons_hard.append("NaN/Inf in candidate")
    if metrics["ssim"] < HARD_REJECT_SSIM:
        reasons_hard.append(f"SSIM {metrics['ssim']:.3f} < {HARD_REJECT_SSIM}")
    if metrics["psnr_db"] < HARD_REJECT_PSNR_DB:
        reasons_hard.append(f"PSNR {metrics['psnr_db']:.1f} dB < {HARD_REJECT_PSNR_DB}")
    if metrics["mean_abs_lsb"] > HARD_REJECT_MEAN_ABS_LSB:
        reasons_hard.append(
            f"mean-abs {metrics['mean_abs_lsb']:.1f} LSB > {HARD_REJECT_MEAN_ABS_LSB}")

    if reasons_hard:
        return 1, "hard-reject", reasons_hard

    if anchor == "prev":
        if metrics["ssim"] < (1.0 - NEEDS_REVIEW_PREV_SSIM_DROP):
            reasons_review.append(
                f"SSIM {metrics['ssim']:.3f} dropped > {NEEDS_REVIEW_PREV_SSIM_DROP} vs prev")
        if metrics["psnr_db"] < NEEDS_REVIEW_PREV_PSNR_DB:
            reasons_review.append(
                f"PSNR {metrics['psnr_db']:.1f} dB < {NEEDS_REVIEW_PREV_PSNR_DB} vs prev")
    elif anchor == "baseline":
        if metrics["ssim"] < NEEDS_REVIEW_BASELINE_SSIM:
            reasons_review.append(
                f"SSIM {metrics['ssim']:.3f} < {NEEDS_REVIEW_BASELINE_SSIM} vs baseline")
        if metrics["psnr_db"] < NEEDS_REVIEW_BASELINE_PSNR_DB:
            reasons_review.append(
                f"PSNR {metrics['psnr_db']:.1f} dB < {NEEDS_REVIEW_BASELINE_PSNR_DB} vs baseline")
    else:
        raise ValueError(f"unknown --anchor {anchor!r}")

    if reasons_review:
        return 2, "needs-review", reasons_review
    return 0, "clean-keep", []


def write_amplified_diff(ref: np.ndarray, cand: np.ndarray, out: Path,
                         amp: float = 10.0, gamma: float = 2.2) -> None:
    """Write a 10x amplified, gamma-corrected diff PNG for human audit."""
    diff = np.abs(cand - ref) * amp
    diff = np.clip(diff, 0.0, 1.0) ** (1.0 / gamma)
    Image.fromarray((diff * 255.0).astype(np.uint8)).save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--anchor", choices=("prev", "baseline"), default="prev",
                        help="Which threshold set to use for NEEDS_REVIEW. Default: prev")
    parser.add_argument("--amplified-diff", type=Path, default=None,
                        help="Write a 10x amplified diff PNG here")
    parser.add_argument("--json", action="store_true",
                        help="Emit one line of JSON instead of key=value")
    args = parser.parse_args()

    ref = load_rgb(args.reference)
    cand = load_rgb(args.candidate)
    if ref.shape != cand.shape:
        print(f"ERROR: shape mismatch ref={ref.shape} cand={cand.shape}", file=sys.stderr)
        sys.exit(3)

    metrics = compute_metrics(ref, cand)
    exit_code, label, reasons = gate(metrics, args.anchor)
    metrics["anchor"] = args.anchor
    metrics["gate"] = label
    metrics["exit_code"] = exit_code
    metrics["reasons"] = reasons
    metrics["reference"] = str(args.reference)
    metrics["candidate"] = str(args.candidate)

    if args.amplified_diff is not None:
        args.amplified_diff.parent.mkdir(parents=True, exist_ok=True)
        write_amplified_diff(ref, cand, args.amplified_diff)
        metrics["amplified_diff"] = str(args.amplified_diff)

    if args.json:
        print(json.dumps(metrics))
    else:
        print(f"=== {args.candidate} vs {args.reference} ===")
        print(f"anchor             = {args.anchor}")
        print(f"psnr_db            = {metrics['psnr_db']:.2f}")
        print(f"ssim               = {metrics['ssim']:.4f}")
        print(f"rmse               = {metrics['rmse']:.5f}")
        print(f"max_abs_diff (per chan, 0-255) = "
              f"R {metrics['max_abs_diff_R']}  "
              f"G {metrics['max_abs_diff_G']}  "
              f"B {metrics['max_abs_diff_B']}")
        print(f"mean_abs_diff      = {metrics['mean_abs_lsb']:.2f} LSB")
        print(f"frac_pixels > 4LSB = {metrics['frac_pixels_gt_4LSB']:.4f}")
        print(f"has_nan_or_inf     = {metrics['has_nan_or_inf']}")
        print("=== gate ===")
        print(f"result             = {label}")
        if reasons:
            for r in reasons:
                print(f"   - {r}")
        if "amplified_diff" in metrics:
            print(f"amplified_diff     = {metrics['amplified_diff']}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
