#!/usr/bin/env python3
"""Hero fixture: TT blend PSNR + 10× diff images vs numpy ref and vs master TT.

Usage (on bh-30 with device):
  python3 scripts/compare_blend_tt.py --out-dir opt/metal-screenshots/blend-compare-hero

Produces:
  tt_current.npy          current kernel blend output
  tt_master.npy           main-branch kernel (approx exp) blend output
  metrics.json            PSNR / max_abs / tile_structure for each pair
  tt_current_vs_ref.png   render (current)
  tt_master_vs_ref.png    render (master tt)
  diff10_tt_current_vs_ref.png
  diff10_tt_master_vs_ref.png
  diff10_tt_current_vs_master_tt.png
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backends import get_backend  # noqa: E402


def psnr_fp64(ref: np.ndarray, cand: np.ndarray) -> float:
    diff = ref.astype(np.float64) - cand.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def tile_structure_ratio(ref: np.ndarray, cand: np.ndarray, tile: int = 32) -> float:
    diff = cand.astype(np.float64) - ref.astype(np.float64)
    h, w, c = diff.shape
    if h % tile != 0 or w % tile != 0:
        return float("nan")
    tile_means = diff.reshape(h // tile, tile, w // tile, tile, c).mean(axis=(1, 3))
    ratios = []
    for ch in range(c):
        pix_std = float(diff[:, :, ch].std())
        expected = pix_std / tile
        if expected > 0:
            ratios.append(float(tile_means[:, :, ch].std()) / expected)
    return float(np.mean(ratios)) if ratios else float("nan")


def write_diff10(ref: np.ndarray, cand: np.ndarray, out: Path) -> None:
    amp = np.clip(np.abs(ref.astype(np.float64) - cand.astype(np.float64)) * 10.0, 0.0, 1.0)
    Image.fromarray((amp * 255.0).astype(np.uint8)).save(out)


def save_rgb_preview(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(img, 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def run_tt_blend(fixtures_dir: Path) -> np.ndarray:
    d = np.load(fixtures_dir / "blend_inputs.npz")
    ref_shape = np.load(fixtures_dir / "blend_output.npy").shape
    h, w = int(d["H"]), int(d["W"])
    backend = get_backend("tt")
    try:
        img, _ = backend.blend(
            torch.from_numpy(d["means_2d"]),
            torch.from_numpy(d["covs_2d"]),
            torch.from_numpy(d["colors"]),
            torch.from_numpy(d["opacities"]),
            torch.from_numpy(d["sorted_gaussian_ids"]),
            torch.from_numpy(d["tile_ranges"]),
            h,
            w,
        )
    finally:
        backend.close()
    out = np.asarray(img, dtype=np.float32)
    if out.shape != ref_shape:
        raise SystemExit(f"shape mismatch: got {out.shape} expected {ref_shape}")
    return out


def metric_pair(ref: np.ndarray, cand: np.ndarray, label: str) -> dict:
    diff = ref.astype(np.float64) - cand.astype(np.float64)
    return {
        "label": label,
        "psnr_dB": psnr_fp64(ref, cand),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "tile_structure_ratio": tile_structure_ratio(ref, cand),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-dir", type=Path, default=REPO / "tests/fixtures/hero")
    ap.add_argument("--out-dir", type=Path, default=REPO / "opt/metal-screenshots/blend-compare-hero")
    ap.add_argument("--skip-master-rebuild", action="store_true",
                    help="only run current TT (tt_master.npy must already exist)")
    args = ap.parse_args()

    gsplat_tt = Path("/proj_sw/user_dev/smarton/gsplat_tt")
    compute_rel = (
        "backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/"
        "kernels/compute/alpha_blend_compute.cpp"
    )
    compute_path = gsplat_tt / compute_rel
    build_dir = gsplat_tt / "backends/tt/tt-metal/build_Release"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(args.fixtures_dir / "blend_output.npy").astype(np.float32)
    np.save(out_dir / "numpy_ref.npy", ref)
    save_rgb_preview(ref, out_dir / "numpy_ref.png")

    os.environ.setdefault("TT_METAL_HOME", str(REPO / "backends/tt/tt-metal"))
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", os.environ["TT_METAL_HOME"])
    os.environ.setdefault("MESH_DEVICE", "P100")
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")
    os.environ.setdefault(
        "TT_METAL_CACHE", "/localdev/smarton/.cache/tt-metal-cache")

    # --- current TT (working tree compute kernel) ---
    print("[compare] running current TT blend...", flush=True)
    current = run_tt_blend(args.fixtures_dir)
    np.save(out_dir / "tt_current.npy", current)
    save_rgb_preview(current, out_dir / "tt_current.png")

    master = None
    if not args.skip_master_rebuild:
        if not compute_path.exists():
            raise SystemExit(f"missing {compute_path}")
        backup = compute_path.read_text()
        try:
            # main @ 95866e0: approx exp_tile<true>
            proc = subprocess.run(
                ["git", "-C", str(gsplat_tt), "show", f"main:{compute_rel}"],
                check=True,
                capture_output=True,
                text=True,
            )
            compute_path.write_text(proc.stdout)
            print("[compare] rebuilt master TT kernel (approx exp)...", flush=True)
            subprocess.run(
                ["sudo", "ninja", "-C", str(build_dir),
                 "metal_example_gaussian_splatting"],
                check=True,
            )
            cache = Path(os.environ["TT_METAL_CACHE"])
            if cache.exists():
                shutil.rmtree(cache)
                cache.mkdir(parents=True, exist_ok=True)
            master = run_tt_blend(args.fixtures_dir)
            np.save(out_dir / "tt_master.npy", master)
            save_rgb_preview(master, out_dir / "tt_master.png")
        finally:
            compute_path.write_text(backup)
            print("[compare] restored current compute kernel, rebuilding...", flush=True)
            subprocess.run(
                ["sudo", "ninja", "-C", str(build_dir),
                 "metal_example_gaussian_splatting"],
                check=True,
            )
            if Path(os.environ["TT_METAL_CACHE"]).exists():
                shutil.rmtree(os.environ["TT_METAL_CACHE"])
                Path(os.environ["TT_METAL_CACHE"]).mkdir(parents=True, exist_ok=True)
            # warm current kernel cache with one more blend
            _ = run_tt_blend(args.fixtures_dir)
    else:
        master_path = out_dir / "tt_master.npy"
        if not master_path.exists():
            raise SystemExit("--skip-master-rebuild but tt_master.npy missing")
        master = np.load(master_path).astype(np.float32)

    metrics = {
        "fixtures": str(args.fixtures_dir),
        "shape": list(ref.shape),
        "pairs": [
            metric_pair(ref, current, "tt_current_vs_numpy_ref"),
            metric_pair(ref, master, "tt_master_vs_numpy_ref"),
            metric_pair(master, current, "tt_current_vs_tt_master"),
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    write_diff10(ref, current, out_dir / "diff10_tt_current_vs_numpy_ref.png")
    write_diff10(ref, master, out_dir / "diff10_tt_master_vs_numpy_ref.png")
    write_diff10(master, current, out_dir / "diff10_tt_current_vs_tt_master.png")

    print(json.dumps(metrics, indent=2))
    print(f"wrote artifacts under {out_dir}")


if __name__ == "__main__":
    main()
