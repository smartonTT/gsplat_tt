"""Layer 2 gate: TT alpha-blend output vs numpy reference on hero fixture.

Usage:
  python3 scripts/verify_blend_metal.py [--backend tt] [--psnr-floor 80]

Compares backends/tt blend (via prepare_kernel_inputs + daemon) against
tests/fixtures/hero/blend_output.npy from capture_reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402


def psnr(ref: np.ndarray, cand: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - cand.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tt")
    ap.add_argument("--fixtures-dir", type=Path, default=Path("tests/fixtures/hero"))
    ap.add_argument("--psnr-floor", type=float, default=80.0)
    args = ap.parse_args()

    ref_path = args.fixtures_dir / "blend_output.npy"
    inp_path = args.fixtures_dir / "blend_inputs.npz"
    if not ref_path.exists() or not inp_path.exists():
        raise SystemExit(
            f"missing fixtures in {args.fixtures_dir}; "
            f"run scripts/capture_reference.py --scene bicycle first"
        )

    ref = np.load(ref_path)
    data = np.load(inp_path)
    H, W = int(data["H"]), int(data["W"])

    backend = get_backend(args.backend)
    image, _sub = backend.blend(
        torch.from_numpy(data["means_2d"]),
        torch.from_numpy(data["covs_2d"]),
        torch.from_numpy(data["colors"]),
        torch.from_numpy(data["opacities"]),
        torch.from_numpy(data["sorted_gaussian_ids"]),
        torch.from_numpy(data["tile_ranges"]),
        H,
        W,
    )
    cand = np.asarray(image, dtype=np.float32)
    p = psnr(ref, cand)
    max_abs = float(np.max(np.abs(ref.astype(np.float64) - cand.astype(np.float64))))
    ok = p >= args.psnr_floor
    out = {
        "stage": "blend",
        "backend": args.backend,
        "pass": ok,
        "psnr_dB": p,
        "psnr_floor_dB": args.psnr_floor,
        "max_abs_diff": max_abs,
        "ref_shape": list(ref.shape),
        "cand_shape": list(cand.shape),
    }
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
