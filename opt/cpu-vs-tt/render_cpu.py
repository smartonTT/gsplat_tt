"""Headless CPU render of the gstt2 bicycle bench through backend="cpu_cpp".

Mirrors the production render convention (render/run.py + gsplat.viewer):
  * scene  = bicycle point cloud, force-square 1024x1024
  * camera = the 30 views in benchmarks/cameras_v2.json (scene "bicycle"),
             order[0] == "hero", intrinsics = _intrinsics_from_fov(W,H,fov),
             extrinsics = c2w_to_w2c(c2w).

It renders the 30 views in order, EXCLUDES the first (warmup) view from the
timing stats, and reports avg/p50/min/max frame ms over the remaining 29. It
also computes the hero PSNR vs the project's golden reference image
(benchmarks/reference_v2/hero.png) and writes a hero screenshot.

CPU ONLY: backend="cpu_cpp" runs entirely on the host CPU (no TT device). The
_gsplat_cpu extension must have been built with -DGSPLAT_WITH_TT=OFF.

Usage:
  python render_cpu.py \
      --repo-root <gstt2 repo with built backends/cpu_cpp/_gsplat_cpu*.so> \
      --gsplat-root <dir containing the `gsplat` python package> \
      --ply <scene.ply> --cameras <cameras_v2.json> \
      --reference <hero.png> --out-dir <dir> --label <mac|x86>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path


def psnr(a, b):
    import numpy as np
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True,
                    help="gstt2 repo root (provides `backends` + built _gsplat_cpu)")
    ap.add_argument("--gsplat-root", required=True,
                    help="dir containing the `gsplat` python package")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--reference", required=True, help="golden hero PNG")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    # repo-root FIRST so `import backends` resolves to the gstt2 HEAD backend
    # (which owns the freshly built _gsplat_cpu). gsplat-root provides the
    # shared `gsplat` python frontend (not tracked in the gstt2 worktree).
    sys.path.insert(0, str(Path(args.repo_root).resolve()))
    sys.path.append(str(Path(args.gsplat_root).resolve()))

    import numpy as np
    import torch
    from PIL import Image

    from backends import get_backend
    from backends.cpu_cpp import _gsplat_cpu
    from gsplat.loading_gaussians import load_ply
    from gsplat.pipeline import Pipeline
    from gsplat.utils import c2w_to_w2c

    simd = _gsplat_cpu.simd_backend()
    has_tt = _gsplat_cpu.has_tt_support()
    print(f"[render] label={args.label} simd_backend={simd} has_tt_support={has_tt}",
          flush=True)
    if has_tt:
        print("[render] WARNING: _gsplat_cpu was built WITH_TT=ON; expected CPU-only",
              flush=True)

    cam_all = json.loads(Path(args.cameras).read_text())
    cam = cam_all[args.scene]
    fov_deg = float(cam["fov_deg"])
    W = H = int(args.size)
    contrib_floor = float(cam.get("contrib_floor", 1.0 / 255.0))
    order = cam["order"]
    hero_name = order[0]

    # K via the viewer/render convention (_intrinsics_from_fov == build_intrinsics).
    longer = max(W, H)
    f = 0.5 * longer / math.tan(0.5 * math.radians(fov_deg))
    K = torch.from_numpy(np.array(
        [[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32))

    print(f"[render] loading {args.ply}", flush=True)
    gauss = load_ply(str(args.ply))
    print(f"[render] {gauss.num_gaussians:,} gaussians; scene={args.scene} "
          f"{W}x{H} fov={fov_deg} contrib_floor={contrib_floor:.3e} "
          f"hero='{hero_name}' n_views={len(order)}", flush=True)

    backend = get_backend("cpu_cpp")
    pipeline = Pipeline(backend, tile_size=32, contrib_floor=contrib_floor)

    def render_view_timed(c2w):
        extr = c2w_to_w2c(torch.from_numpy(np.asarray(c2w, dtype=np.float32)))
        t = time.perf_counter()
        res = pipeline.render(gauss, extr, K, H, W)
        wall_ms = (time.perf_counter() - t) * 1000.0
        img = res.image
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
        return img, wall_ms

    per_view = []
    hero_img = None
    for i, name in enumerate(order):
        img, wall_ms = render_view_timed(cam["views"][name]["c2w"])
        per_view.append({"view": name, "idx": i, "ms": wall_ms,
                         "warmup": i == 0})
        if name == hero_name:
            hero_img = img
        tag = " (warmup, excluded)" if i == 0 else ""
        print(f"[render]   view={name:<13} {wall_ms:8.1f} ms{tag}", flush=True)

    timed = [v["ms"] for v in per_view if not v["warmup"]]
    avg_ms = sum(timed) / len(timed)
    p50_ms = statistics.median(timed)
    min_ms = min(timed)
    max_ms = max(timed)

    # Hero PSNR vs the project's golden reference image.
    ref = np.asarray(Image.open(args.reference).convert("RGB"),
                     dtype=np.float32) / 255.0
    hero_vs_ref = psnr(hero_img, ref)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hero_png = out_dir / f"{args.label}_hero.png"
    Image.fromarray((hero_img * 255.0).astype(np.uint8)).save(hero_png)

    result = {
        "label": args.label,
        "simd_backend": simd,
        "has_tt_support": has_tt,
        "scene": args.scene,
        "size": [W, H],
        "fov_deg": fov_deg,
        "contrib_floor": contrib_floor,
        "n_views": len(order),
        "n_timed": len(timed),
        "warmup_view": hero_name,
        "avg_frame_ms": avg_ms,
        "p50_ms": p50_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "hero_vs_ref_db": hero_vs_ref,
        "hero_png": str(hero_png),
        "per_view": per_view,
    }
    (out_dir / f"{args.label}_result.json").write_text(json.dumps(result, indent=2))

    print(f"SUMMARY label={args.label} simd={simd} "
          f"avg_frame_ms={avg_ms:.1f} p50={p50_ms:.1f} min={min_ms:.1f} "
          f"max={max_ms:.1f} (n_timed={len(timed)}, warmup={hero_name} excluded) "
          f"hero_vs_ref={hero_vs_ref:.2f}dB hero_png={hero_png}", flush=True)


if __name__ == "__main__":
    main()
