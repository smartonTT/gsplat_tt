"""Render one view of a scene through the clean TT renderer and report PSNR.

Renders the bicycle hero view through render/ (the clean extract of the
production pipeline), renders the same view through the cpu_cpp_mb CPU
reference, and prints:

    SUMMARY hero_vs_ref=<dB>

The gate is hero_vs_ref >= 63.6 dB (production hits ~63.85 dB).

Run it through tt-workflows devrun so the per-host device lock is held:

    devrun.sh --tag render-clean -- \\
        "python3 render/run.py --iter-dir render-clean"

Artifacts are written under tmp/render-clean/ (gitignored scratch).
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

# IMPORTANT: render_clean needs NO GSPLAT_TT_* env flags. Its production
# configuration is fully baked into the C++ (config.h + env_config.h constants
# + the stage drivers), and its output is bit-identical with or without these
# flags set (verified: clean(env) vs clean(no-env) mse == 0).
#
# The flags below exist ONLY so the cpu_cpp_mb REFERENCE reproduces the exact
# production `verify_cmd` measurement (hero_vs_ref ~= 63.85 dB). The reference
# backend's render_full reads these flags and, under them, takes the production
# device->CPU-fallback path; without them it takes a slightly different CPU path
# and the anchor shifts (~47 dB vs a *different* reference). We set them so the
# comparison is apples-to-apples with production. render_clean ignores them.
_REF_VERIFY_ENV = {
    "GSPLAT_TT_JIT_WARMUP": "1",
    "GSPLAT_TT_BLEND_MODE": "2",
    "GSPLAT_TT_MB_KERNEL": "1",
    "GSPLAT_TT_DEVICE_PROJECT": "1",
    "GSPLAT_TT_RESIDENT_PROJECT": "1",
    "GSPLAT_TT_RESIDENT_GATHER": "1",
    "GSPLAT_TT_DEVICE_TILE_ASSIGN": "1",
    "GSPLAT_TT_RESIDENT_TA_IN": "1",
    "GSPLAT_TT_DEVICE_SORT": "1",
    "GSPLAT_TT_RESIDENT_PAIRS": "1",
    "GSPLAT_TT_RESIDENT_BLEND": "1",
    "GSPLAT_TT_SORT_DEVICE_PUBLISH": "1",
    "GSPLAT_TT_TA_DEVICE_SCAN": "1",
    "GSPLAT_TT_PROJ_DEVICE_SCAN": "1",
    "GSPLAT_TT_SFPU_CULL": "1",
    "GSPLAT_TT_TILE_BUCKET": "1",
    "GSPLAT_TT_BUCKET_FIT": "8192",
    "GSPLAT_TT_FUSED_TILE": "0",
    "GSPLAT_TT_L1_RECORD": "1",
}
for _k, _v in _REF_VERIFY_ENV.items():
    os.environ.setdefault(_k, _v)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from backends import get_backend  # noqa: E402
from backends.cpu_cpp.backend import CpuCppBackend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


def _load_render_clean():
    """Import the compiled render_clean module from render/."""
    here = Path(__file__).resolve().parent
    for so in sorted(here.glob("render_clean*.so")):
        spec = importlib.util.spec_from_file_location("render_clean", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(
        f"render_clean*.so not found in {here}; build it first:\n"
        "  cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build render/build-tt -j 16")


class CleanBackend(CpuCppBackend):
    """cpu_cpp prep + the clean TT device render_view (render_clean module)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("microblock", True)
        kwargs.setdefault("fused", True)
        kwargs["render_fused"] = True
        super().__init__(**kwargs)
        self._clean = _load_render_clean()

    def has_render_fused(self) -> bool:
        return True

    def render_fused(self, gaussians, extrinsics, intrinsics, image_height,
                     image_width, contrib_floor=None, k_cap=3.0,
                     use_isoellipse=False):
        gnp = self._cached_gauss_np(gaussians)
        extr_np = extrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        intr_np = intrinsics.detach().cpu().numpy().astype(np.float32, copy=False)
        cov3d = self._cached_cov3d(gnp["scales"], gnp["rotations"])
        effective_contrib_floor = (
            float(self._mb_contrib_floor) if self.contrib_floor_override is None
            else float(self.contrib_floor_override))
        image, stats = self._clean.render_view(
            gnp["means"], cov3d, gnp["opacities"], gnp["colors"],
            extr_np, intr_np, int(image_height), int(image_width), 32,
            float(self.min_opacity), effective_contrib_floor,
            float(self._mb_contrib_floor), bool(self.cull_disabled),
            float(self.transmittance_threshold), int(self.max_radius),
            float(self.k_cap), bool(self.use_isoellipse), 2)
        return np.asarray(image), dict(stats)

    def close(self):
        if hasattr(self._clean, "device_shutdown"):
            try:
                self._clean.device_shutdown()
            except Exception:
                pass


def build_intrinsics(W, H, fov_deg):
    longer = max(W, H)
    f = 0.5 * longer / math.tan(0.5 * math.radians(fov_deg))
    K = np.array([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]],
                 dtype=np.float32)
    return torch.from_numpy(K)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return -10.0 * math.log10(mse)


def render_hero(backend, gauss, view, fov_deg, W, H, contrib_floor):
    pipeline = Pipeline(backend, tile_size=32, contrib_floor=contrib_floor)
    c2w = np.asarray(view["c2w"], dtype=np.float32)
    extr = c2w_to_w2c(torch.from_numpy(c2w))
    K = build_intrinsics(W, H, fov_deg)
    # Warmup (JIT compile + caches), then the timed render.
    _ = pipeline.render(gauss, extr, K, H, W)
    res = pipeline.render(gauss, extr, K, H, W)
    img = res.image
    if hasattr(img, "numpy"):
        img = img.numpy()
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--cameras", type=Path,
                    default=REPO_ROOT / "benchmarks/cameras_v2.json")
    ap.add_argument("--iter-dir", default="render-clean")
    args = ap.parse_args()

    # Device-lock discipline: render_clean opens the TT device, so only run
    # under tt-workflows devrun (which holds the per-host lock).
    if os.environ.get("TTW_DEVRUN") != "1" and os.environ.get("TTW_ALLOW_DIRECT") != "1":
        print("[run] REFUSING to open the TT device: not launched via devrun.sh "
              "(no TTW_DEVRUN). Use devrun.sh, or set TTW_ALLOW_DIRECT=1 for a "
              "one-off manual human run.", file=sys.stderr, flush=True)
        sys.exit(3)

    import json
    cam = json.loads(args.cameras.read_text())[args.scene]
    fov_deg = float(cam["fov_deg"])
    W, H = cam["image_size"]
    contrib_floor = cam.get("contrib_floor", 1.0 / 255.0)
    hero_name = cam["order"][0]
    hero_view = cam["views"][hero_name]
    gauss = load_ply(str(Path(cam["ply"])))

    out_dir = REPO_ROOT / "tmp" / args.iter_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] rendering hero='{hero_name}' {W}x{H} scene={args.scene}", flush=True)
    t0 = time.time()
    clean = render_hero(CleanBackend(), gauss, hero_view, fov_deg, W, H, contrib_floor)
    t1 = time.time()
    ref = render_hero(get_backend("cpu_cpp_mb"), gauss, hero_view, fov_deg, W, H,
                      contrib_floor)
    t2 = time.time()

    hero_vs_ref = psnr(clean, ref)

    Image.fromarray((clean * 255.0).astype(np.uint8)).save(out_dir / "hero_clean.png")
    Image.fromarray((ref * 255.0).astype(np.uint8)).save(out_dir / "hero_ref.png")
    diff = np.clip(np.abs(clean - ref) * 10.0, 0.0, 1.0)
    Image.fromarray((diff * 255.0).astype(np.uint8)).save(out_dir / "hero_diff10.png")

    def fmt(x):
        return "inf" if x == float("inf") else f"{x:.2f}"

    print(f"SUMMARY scene={args.scene} hero='{hero_name}' "
          f"hero_vs_ref={fmt(hero_vs_ref)}dB "
          f"clean_render_s={t1 - t0:.1f} ref_render_s={t2 - t1:.1f} "
          f"out={out_dir}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
