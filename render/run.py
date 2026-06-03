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
import contextlib
import importlib.util
import math
import os
import statistics
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


@contextlib.contextmanager
def silence_os_fd_output():
    """Redirect OS-level stdout+stderr (fd 1/2) to /dev/null for the block.

    The cpu_cpp_mb reference is rendered in-process while CleanBackend still
    holds the TT device, so the reference's gsplat_tt device-init attempts
    fail and fall back to CPU — that CPU fallback is exactly the oracle that
    yields the 63.85 dB measurement. tt-metal logs those init failures
    (TT_FATAL / context_id / "device init failed") to fd 2 directly from C++,
    so a Python-level redirect (contextlib.redirect_stderr) cannot catch them;
    we dup2 /dev/null over the real fds for the duration of the reference
    render only. No computation changes — only the noisy log is suppressed.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(devnull_fd)
        os.close(saved_out)
        os.close(saved_err)


def _to_image(res):
    img = res.image
    if hasattr(img, "numpy"):
        img = img.numpy()
    return np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)


def render_hero(backend, gauss, view, fov_deg, W, H, contrib_floor):
    """Render a single view through `backend` (used for the CPU reference)."""
    pipeline = Pipeline(backend, tile_size=32, contrib_floor=contrib_floor)
    c2w = np.asarray(view["c2w"], dtype=np.float32)
    extr = c2w_to_w2c(torch.from_numpy(c2w))
    K = build_intrinsics(W, H, fov_deg)
    # Warmup (JIT compile + caches), then the render we keep.
    _ = pipeline.render(gauss, extr, K, H, W)
    return _to_image(pipeline.render(gauss, extr, K, H, W))


def render_clean_view_timed(pipeline, gauss, c2w, K, H, W):
    """Render one view through render_clean; return (image, wall_ms).

    The timed window wraps the whole `pipeline.render` call. render_clean is
    single-path TT (host orchestrates, device computes) and the heavy host prep
    (cov3d repack, gaussian np cast) is cached on the backend after warmup, so
    this wall time is a faithful per-frame number dominated by the device
    render + the thin per-view host orchestration. The PURE device makespan is
    measured separately under Tracy (TASK 3); the two should agree within noise.
    """
    extr = c2w_to_w2c(torch.from_numpy(np.asarray(c2w, dtype=np.float32)))
    t = time.perf_counter()
    res = pipeline.render(gauss, extr, K, H, W)
    wall_ms = (time.perf_counter() - t) * 1000.0
    return _to_image(res), wall_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--cameras", type=Path,
                    default=REPO_ROOT / "benchmarks/cameras_v2.json")
    ap.add_argument("--iter-dir", default="render-clean")
    ap.add_argument("--dump-views", default=None,
                    help="if set, save every bench view's render_clean output as "
                         "viewNN_<name>.png under tmp/<this dir> (8-bit RGB)")
    ap.add_argument("--no-ref", action="store_true",
                    help="skip the cpu_cpp_mb reference render + PSNR gate; time "
                         "the 30 render_clean views only (used for a clean Tracy "
                         "device-profiler capture). Does not change render_clean.")
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
    order = cam["order"]            # the 30-view bench set (order[0] == hero)
    hero_name = order[0]
    hero_view = cam["views"][hero_name]
    gauss = load_ply(str(Path(cam["ply"])))

    out_dir = REPO_ROOT / "tmp" / args.iter_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_dir = None
    if args.dump_views:
        dump_dir = REPO_ROOT / "tmp" / args.dump_views
        dump_dir.mkdir(parents=True, exist_ok=True)

    K = build_intrinsics(W, H, fov_deg)
    clean_backend = CleanBackend()
    clean_pipeline = Pipeline(clean_backend, tile_size=32, contrib_floor=contrib_floor)

    # --- Warmup (EXCLUDED from timing): JIT-compile every device program +
    # warm the backend host caches (cov3d repack, gaussian np cast). One hero
    # render is enough to compile the resident chain and populate the caches
    # that are reused by all 30 views.
    print(f"[run] warmup (hero='{hero_name}', {W}x{H}, scene={args.scene})",
          flush=True)
    t_warm = time.perf_counter()
    _ = render_clean_view_timed(clean_pipeline, gauss, hero_view["c2w"], K, H, W)
    warmup_s = time.perf_counter() - t_warm

    # --- Timed loop over ALL 30 bench views (pure render_clean device path;
    # the CPU reference is NOT rendered here — only for the hero PSNR gate).
    print(f"[run] timing {len(order)} views (warmup excluded)", flush=True)
    per_view_ms = []
    hero_clean = None
    for i, name in enumerate(order):
        img, wall_ms = render_clean_view_timed(
            clean_pipeline, gauss, cam["views"][name]["c2w"], K, H, W)
        per_view_ms.append(wall_ms)
        if name == hero_name:
            hero_clean = img
        if dump_dir is not None:
            png = dump_dir / f"view{i:02d}_{name}.png"
            Image.fromarray((img * 255.0).astype(np.uint8)).save(png)
            print(f"[run]   view={name} {wall_ms:.1f}ms saved={png.name}", flush=True)
        else:
            print(f"[run]   view={name} {wall_ms:.1f}ms", flush=True)

    avg_ms = sum(per_view_ms) / len(per_view_ms)
    p50_ms = statistics.median(per_view_ms)
    min_ms = min(per_view_ms)
    max_ms = max(per_view_ms)

    # --- Hero PSNR gate ONLY: render the cpu_cpp_mb reference in-process
    # (identical oracle pixels => identical hero_vs_ref). Its device-init
    # attempts fail (device held by the clean backend) and fall back to CPU;
    # silence the C++ fd-2 TT_FATAL spam so the "clean renderer" log stays clean.
    # Skipped under --no-ref (clean device-profiler capture: render_clean only).
    if args.no_ref:
        hero_vs_ref = float("nan")
        Image.fromarray((hero_clean * 255.0).astype(np.uint8)).save(out_dir / "hero_clean.png")
    else:
        with silence_os_fd_output():
            ref = render_hero(get_backend("cpu_cpp_mb"), gauss, hero_view, fov_deg,
                              W, H, contrib_floor)

        hero_vs_ref = psnr(hero_clean, ref)

        Image.fromarray((hero_clean * 255.0).astype(np.uint8)).save(out_dir / "hero_clean.png")
        Image.fromarray((ref * 255.0).astype(np.uint8)).save(out_dir / "hero_ref.png")
        diff = np.clip(np.abs(hero_clean - ref) * 10.0, 0.0, 1.0)
        Image.fromarray((diff * 255.0).astype(np.uint8)).save(out_dir / "hero_diff10.png")

    def fmt(x):
        return "inf" if x == float("inf") else f"{x:.2f}"

    # Headline metrics: hero_vs_ref (dB, the 63.85 gate on the hero view) and
    # avg_frame_ms (the canonical frame-time = mean render_clean wall time over
    # all 30 bench views, warmup excluded). warmup_s is reported only as
    # context (it includes JIT + cache fill) and is NOT the frame-time metric.
    print(f"SUMMARY scene={args.scene} hero='{hero_name}' "
          f"hero_vs_ref={fmt(hero_vs_ref)}dB "
          f"avg_frame_ms={avg_ms:.1f} p50_ms={p50_ms:.1f} "
          f"min_ms={min_ms:.1f} max_ms={max_ms:.1f} n_views={len(per_view_ms)} "
          f"warmup_s={warmup_s:.1f} "
          f"out={out_dir}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
