"""Render the 30-view training-pattern benchmark for a given backend.

Reads cameras_v2.json, renders every view at its declared size with the
specified backend, writes one PNG per view + timing.jsonl into out-dir.

The supervisor's run_iter.sh calls this after a successful build.

Usage:
  python3 scripts/render_30frame.py --backend cpu_cpp --cameras benchmarks/cameras_v2.json --out-dir opt/screenshots/iter-005-...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend, REGISTRY  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


def build_intrinsics(W: int, H: int, fov_deg: float) -> torch.Tensor:
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    help=f"one of: {', '.join(sorted(REGISTRY))}")
    ap.add_argument("--cameras", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--scene", default="stitch")
    ap.add_argument("--warmup", type=int, default=1, help="discard-then-render warmup runs")
    ap.add_argument("--contrib-floor", type=float, default=None,
                    help="override cameras.json contrib_floor")
    args = ap.parse_args()

    cam_data = json.loads(args.cameras.read_text())
    scene = cam_data[args.scene]
    fov_deg = float(scene["fov_deg"])
    ply_path = Path(scene["ply"])
    W, H = scene["image_size"]
    contrib_floor = args.contrib_floor if args.contrib_floor is not None else scene.get("contrib_floor", 1.0 / 255.0)
    order = scene["order"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render_30frame] backend={args.backend} scene={args.scene} size={W}x{H} contrib_floor={contrib_floor:.5f}")
    print(f"[render_30frame] loading PLY: {ply_path}")
    gauss = load_ply(str(ply_path))

    backend = get_backend(args.backend)
    pipeline = Pipeline(backend, tile_size=32)

    # Warmup: load + project the first view once to settle any lazy caches.
    if args.warmup > 0:
        c2w_warm = np.asarray(scene["views"][order[0]]["c2w"], dtype=np.float32)
        extr = c2w_to_w2c(torch.from_numpy(c2w_warm))
        K = build_intrinsics(W, H, fov_deg)
        for _ in range(args.warmup):
            _ = pipeline.render(gauss, extr, K, H, W)

    timing_rows = []
    total_t0 = time.perf_counter()
    for i, name in enumerate(order):
        c2w = np.asarray(scene["views"][name]["c2w"], dtype=np.float32)
        extr = c2w_to_w2c(torch.from_numpy(c2w))
        K = build_intrinsics(W, H, fov_deg)
        t0 = time.perf_counter()
        res = pipeline.render(gauss, extr, K, H, W)
        wall_ms = (time.perf_counter() - t0) * 1000.0

        img = res.image
        if hasattr(img, "numpy"):
            img = img.numpy()
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(img_u8).save(args.out_dir / f"{name}.png")

        sub_timings = dict(res.timings) if hasattr(res, "timings") and res.timings else {}
        row = {"view": name, "view_idx": i, "total_ms": wall_ms, **sub_timings}
        timing_rows.append(row)
        print(f"[render_30frame] {i+1:2d}/{len(order)} {name:14s} {wall_ms:7.1f} ms")

    sum_ms = (time.perf_counter() - total_t0) * 1000.0
    (args.out_dir / "timing.jsonl").write_text("\n".join(json.dumps(r) for r in timing_rows) + "\n")
    print(f"[render_30frame] sum_total_ms = {sum_ms:.1f}  ({len(order)} frames)")


if __name__ == "__main__":
    main()
