"""Live viewer (gsplat/viewer.py) on the fastest kept TT path, render_clean.

Registers render/run.py's CleanBackend as backend "tt_clean" (the same
render_clean .so the 30-view bicycle bench times), renders --selftest frames of
the bicycle hero view through the viewer's own pipeline and prints their FPS,
then serves the viewer. Started and stopped by opt/viewer/viewer.sh.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# render_clean JIT kernels get their own cache (as in render/run.py).
os.environ.setdefault(
    "TT_METAL_CACHE", f"/localdev/{os.environ.get('USER', 'smarton')}/.cache/tt-metal-cache-viewer")


def _load_run_py():
    spec = importlib.util.spec_from_file_location("render_run", REPO / "render" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ply_path", nargs="?", default=str(REPO / "scenes" / "bicycle.ply"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--selftest", type=int, default=30,
                    help="hero-view frames to time before serving (0 = skip)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    run = _load_run_py()
    import backends
    backends.REGISTRY["tt_clean"] = run.CleanBackend

    import torch
    from gsplat.loading_gaussians import load_ply
    from gsplat.utils import c2w_to_w2c
    from gsplat.viewer import GaussianViewer

    cam = json.loads((REPO / "benchmarks" / "cameras_v2.json").read_text())["bicycle"]
    W, H = cam["image_size"]
    print(f"[viewer_clean] sha={os.environ.get('GSPLAT_SHA', '?')} loading {args.ply_path}", flush=True)
    gauss = load_ply(args.ply_path)
    viewer = GaussianViewer(gauss, host=args.host, port=args.port, backend="tt_clean",
                            render_width=W, render_height=H, verbose=args.verbose,
                            scene_path=args.ply_path,
                            # Same contribution floor as the bench (1/255), not the
                            # slider's 1/16384: that kept more pairs (slower, other image).
                            contrib_floor=float(cam["contrib_floor"]))

    if args.selftest > 0:
        # Same call the viewer makes per frame (pipeline.render), hero view of
        # the bench camera set; excludes JPEG encode and websocket transfer.
        hero = cam["views"][cam["order"][0]]
        K = run.build_intrinsics(W, H, float(cam["fov_deg"]))
        extr = c2w_to_w2c(torch.tensor(hero["c2w"], dtype=torch.float32))
        for _ in range(3):
            viewer.pipeline.render(gauss, extr, K, H, W)
        ms = []
        for _ in range(args.selftest):
            t = time.perf_counter()
            viewer.pipeline.render(gauss, extr, K, H, W)
            ms.append((time.perf_counter() - t) * 1000.0)
        med = statistics.median(ms)
        print(f"[viewer_clean] SELFTEST hero {W}x{H} n={len(ms)} median={med:.2f} ms "
              f"({1000.0 / med:.1f} FPS) min={min(ms):.2f} max={max(ms):.2f}", flush=True)
        # Save the device frame and score it against the reference render.
        import numpy as np
        from PIL import Image
        img = viewer.pipeline.render(gauss, extr, K, H, W).image
        img8 = (img if img.dtype == np.uint8 else
                (np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8))
        out = REPO.parent / "hero_viewer.png"
        Image.fromarray(img8).save(out)
        ref_p = REPO / "benchmarks" / "reference_v2" / "hero.png"
        if ref_p.exists():
            ref = np.asarray(Image.open(ref_p).convert("RGB"), dtype=np.float64)
            mse = float(np.mean((img8.astype(np.float64) - ref) ** 2))
            psnr = 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else float("inf")
            print(f"[viewer_clean] HERO saved {out} PSNR vs reference_v2/hero.png = {psnr:.2f} dB",
                  flush=True)

    print(f"[viewer_clean] READY port={args.port}", flush=True)
    viewer.run()


if __name__ == "__main__":
    main()
