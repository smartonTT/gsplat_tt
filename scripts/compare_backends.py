"""Visual regression: compare cpu_cpp renders against numpy cpu ground truth.

PSNR vs a frozen PNG reference misses two failure modes:
  1. Viewer-only bugs (camera/render mismatch) — offline PNG compare never runs.
  2. Systematic errors shared with an outdated/wrong reference set.

This script renders the same cameras with both backends and reports per-view
PSNR, coverage delta, and tile-correlated error (bad_tiles count).

Usage:
  python3 scripts/compare_backends.py --scene stitch --size 1024
  python3 scripts/compare_backends.py --cameras benchmarks/cameras_v2.json --view hero
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402
from gsplat.viewer import _orbit_pose  # noqa: E402
from scripts.derive_camera import look_at_c2w  # noqa: E402


def build_intrinsics(W: int, H: int, fov_deg: float = 50.0) -> torch.Tensor:
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(K)


def scene_orbit_cameras(
    ply_path: Path, azimuths: list[float], elev: float = 0.0, dist_mul: float = 1.2,
) -> dict[str, np.ndarray]:
    gauss = load_ply(str(ply_path))
    means = gauss.means.numpy()
    opacities = gauss.opacities.numpy()
    visible = means[opacities > 0.1]
    if visible.shape[0] < 100:
        visible = means
    lo = np.percentile(visible, 5, axis=0)
    hi = np.percentile(visible, 95, axis=0)
    center = (lo + hi) * 0.5
    diagonal = float(np.linalg.norm(hi - lo))
    dist = diagonal * dist_mul
    out: dict[str, np.ndarray] = {}
    for az in azimuths:
        pos, look, up = _orbit_pose(center, dist, azim_deg=az, elev_deg=elev)
        name = f"az{int(round(az)):03d}"
        out[name] = look_at_c2w(pos, look, up)
    return out


def render_one(
    backend_name: str, gauss, c2w: np.ndarray, W: int, H: int, fov_deg: float,
) -> np.ndarray:
    pipe = Pipeline(get_backend(backend_name), tile_size=32)
    extr = c2w_to_w2c(torch.from_numpy(c2w.astype(np.float32)))
    K = build_intrinsics(W, H, fov_deg)
    res = pipe.render(gauss, extr, K, H, W)
    if res.image is None:
        return np.zeros((H, W, 3), dtype=np.float32)
    return np.asarray(res.image, dtype=np.float64)


def analyze(ref: np.ndarray, cand: np.ndarray, tile: int = 32) -> dict:
    diff = ref - cand
    mse = float(np.mean(diff * diff))
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    H, W, _ = diff.shape
    tile_means = diff.reshape(H // tile, tile, W // tile, tile, 3).mean(axis=(1, 3))
    pix_std = float(diff.std())
    t_ratio = float(tile_means.std() / (pix_std / tile)) if pix_std > 0 else 0.0
    tile_max = np.abs(diff).reshape(H // tile, tile, W // tile, tile, 3).max(axis=(1, 3)).max(axis=2)
    bad_tiles = int((tile_max > 0.1).sum())
    cov_ref = float((ref.sum(axis=2) > 0.02).mean())
    cov_cand = float((cand.sum(axis=2) > 0.02).mean())
    return {
        "psnr": psnr,
        "max_abs": float(np.abs(diff).max()),
        "frac_gt_5pct": float((np.abs(diff).max(axis=2) > 0.05).mean()),
        "tile_structure_ratio": t_ratio,
        "bad_tiles_gt_0.1": bad_tiles,
        "coverage_ref": cov_ref,
        "coverage_cand": cov_cand,
        "coverage_delta": cov_ref - cov_cand,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stitch")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--cameras", default=None, help="optional cameras_v2.json")
    ap.add_argument("--views", nargs="*", default=None)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fov-deg", type=float, default=50.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--orbit-sweep",
        action="store_true",
        help="also test azimuths 130..280 step 15 (catches viewer orbit angles)",
    )
    args = ap.parse_args()

    ply_path = Path(args.ply) if args.ply else Path(f"scenes/{args.scene}_doll.ply")
    if not ply_path.exists() and args.scene == "stitch":
        ply_path = Path("scenes/stitch_doll.ply")
    gauss = load_ply(str(ply_path))
    W = H = args.size

    views: dict[str, np.ndarray] = {}
    if args.cameras:
        cams = json.loads(Path(args.cameras).read_text())
        scene_key = args.scene if args.scene in cams else next(iter(cams))
        entry = cams[scene_key]
        for name, spec in entry["views"].items():
            if args.views and name not in args.views:
                continue
            views[name] = np.asarray(spec["c2w"], dtype=np.float64)
    else:
        azs = [220.0]
        if args.orbit_sweep:
            azs = list(range(130, 281, 15))
        views = scene_orbit_cameras(ply_path, azs)

    if not views:
        raise SystemExit("no views to compare")

    out_dir = args.out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_backends] ply={ply_path}  size={W}  views={len(views)}")
    print(f"{'view':14s}  {'PSNR':>7s}  {'max':>6s}  {'bad_tiles':>9s}  {'cov_Δ':>7s}")
    worst_psnr = float("inf")
    worst_bad = 0
    summary_rows = []
    for name, c2w in views.items():
        ref = render_one("cpu", gauss, c2w, W, H, args.fov_deg)
        cand = render_one("cpu_cpp", gauss, c2w, W, H, args.fov_deg)
        m = analyze(ref, cand)
        worst_psnr = min(worst_psnr, m["psnr"])
        worst_bad = max(worst_bad, m["bad_tiles_gt_0.1"])
        summary_rows.append({"view": name, **m})
        print(
            f"{name:14s}  {m['psnr']:7.2f}  {m['max_abs']:6.4f}  "
            f"{m['bad_tiles_gt_0.1']:9d}  {m['coverage_delta']:7.5f}"
        )
        if out_dir:
            Image.fromarray((np.clip(cand, 0, 1) * 255).astype(np.uint8)).save(out_dir / f"{name}_cpp.png")
            amp = np.clip(np.abs(ref - cand) * 10.0, 0.0, 1.0)
            Image.fromarray((amp * 255).astype(np.uint8)).save(out_dir / f"{name}_diff10.png")

    print(f"[compare_backends] worst_psnr={worst_psnr:.2f} dB  worst_bad_tiles={worst_bad}")
    worst_max = max(r["max_abs"] for r in summary_rows)
    worst_cov = max(abs(r["coverage_delta"]) for r in summary_rows)
    print(f"[compare_backends] worst_max_abs={worst_max:.4f}  worst|coverage_Δ|={worst_cov:.5f}")
    if worst_psnr < 55.0 or worst_max > 0.2 or worst_cov > 0.01:
        print("[compare_backends] FAIL — cpu_cpp diverges from numpy cpu ground truth")
        raise SystemExit(1)
    print("[compare_backends] PASS")


if __name__ == "__main__":
    main()
