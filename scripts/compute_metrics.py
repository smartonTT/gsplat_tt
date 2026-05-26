"""Aggregate per-view PSNR + timing metrics for one iter.

Reads:  <iter_dir>/{hero,side,top}.png, <iter_dir>/timing.jsonl,
        <reference_dir>/stitch_{hero,side,top}.png
Writes: <iter_dir>/{hero,side,top}_diff10.png, <iter_dir>/metrics.json

PSNR computed in fp64 so >100 dB is representable. timing.jsonl format:
one JSON object per measured frame with at minimum `view` and `kernel_ms`.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


VIEWS = ("hero", "side", "top")


def load_rgb_fp64(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def psnr_fp64(ref: np.ndarray, cand: np.ndarray) -> float:
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def write_diff10(ref: np.ndarray, cand: np.ndarray, out: Path) -> None:
    """10x amplified absolute diff, clipped to [0,1], saved as 8-bit PNG."""
    amp = np.clip(np.abs(ref - cand) * 10.0, 0.0, 1.0)
    Image.fromarray((amp * 255.0).astype(np.uint8)).save(out)


def aggregate_timing(timing_path: Path) -> dict:
    rows = [json.loads(line) for line in timing_path.read_text().splitlines() if line.strip()]
    all_ms = [r["kernel_ms"] for r in rows]
    if not all_ms:
        raise ValueError(f"{timing_path} has no kernel_ms rows (did the kernel crash mid-run?)")
    per_view = {v: [r["kernel_ms"] for r in rows if r.get("view") == v] for v in VIEWS}
    out = {
        "kernel_ms_median": float(statistics.median(all_ms)),
        "kernel_ms_p99": float(np.percentile(all_ms, 99)),
        "per_view_median": {v: (float(statistics.median(per_view[v])) if per_view[v] else float("nan")) for v in VIEWS},
        "frame_count": len(all_ms),
    }
    # Optional per-stage medians (added in iter-008 trajectory work). Only emit
    # keys that exist on every row — missing on historical iters where
    # render_fixed.py wasn't capturing them yet.
    stage_keys = ("total_ms", "project_ms", "tile_assign_ms", "sort_ms", "blend_ms")
    stage_medians = {}
    for k in stage_keys:
        vals = [r[k] for r in rows if k in r]
        if len(vals) == len(rows):
            stage_medians[k] = float(statistics.median(vals))
    if stage_medians:
        out["stage_medians"] = stage_medians

    # iter-PROF: every sub-timing under "sub.*" gets a median too. Keeps
    # the persistent profiling story alive across future iters — every
    # metrics.json from this point forward records the full breakdown.
    sub_keys = sorted({k for r in rows for k in r if k.startswith("sub.")})
    sub_medians: dict[str, float] = {}
    for k in sub_keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            sub_medians[k[len("sub."):]] = float(statistics.median(vals))
    if sub_medians:
        out["sub_medians"] = sub_medians
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter-dir", required=True, type=Path)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--class", dest="class_tag", required=True)
    ap.add_argument("--prev-best-ms", required=True, type=float)
    args = ap.parse_args()

    psnr = {}
    for v in VIEWS:
        ref = load_rgb_fp64(args.reference_dir / f"stitch_{v}.png")
        cand = load_rgb_fp64(args.iter_dir / f"{v}.png")
        assert ref.shape == cand.shape, f"shape mismatch for {v}: ref {ref.shape} vs cand {cand.shape}"
        psnr[v] = psnr_fp64(ref, cand)
        write_diff10(ref, cand, args.iter_dir / f"{v}_diff10.png")

    timing = aggregate_timing(args.iter_dir / "timing.jsonl")

    # Optional: per-zone Tracy data if --profile was used.
    tracy_zones = None
    zones_csv = args.iter_dir / "zones.csv"
    if zones_csv.exists():
        import csv
        with zones_csv.open() as f:
            tracy_zones = [
                {"name": r["name"], "avg_ns": float(r["avg_ns"]), "count": int(r["count"])}
                for r in csv.DictReader(f)
                if r.get("name") and r.get("avg_ns")
            ]

    metrics = {
        "iter_dir": str(args.iter_dir.name),
        "class": args.class_tag,
        "prev_best_kernel_ms": args.prev_best_ms,
        "psnr_per_view": psnr,
        **timing,
    }
    if tracy_zones is not None:
        metrics["tracy_zones"] = tracy_zones
    (args.iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
