"""Render one view of a scene through the clean TT renderer and report PSNR.

Renders the bicycle hero view through render/ (the clean extract of the
production pipeline), renders the same view through the cpu_cpp_mb CPU
reference, and prints:

    SUMMARY hero_vs_ref=<dB>

The gate metric (NEW REF iter-132) is hero_vs_ref = 8-bit PSNR vs the committed
golden frame tests/fixtures/hero/hero_golden_8bit.png. Bit-identical 8-bit output
reports 100.0 dB; the loop gate is hero_vs_ref >= 50 dB (see ttw.toml). The CPU
reference is still rendered as a secondary float diagnostic (hero_vs_cpu).

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
import statistics
import subprocess
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

_CACHE_RENDER = os.environ.get(
    "TT_METAL_CACHE_RENDER",
    "/localdev/smarton/.cache/tt-metal-cache-render",
)

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
    here = Path(__file__).resolve().parent
    for so in sorted(here.glob("render_clean*.so")):
        spec = importlib.util.spec_from_file_location("render_clean", so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    raise ImportError(
        f"render_clean*.so not found in {here}; build render/build-tt first")


class CleanBackend(CpuCppBackend):
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


def _to_u8(img01):
    """Quantize a [0,1] float image to uint8 with floor — matches the saved PNG."""
    return (np.clip(np.asarray(img01, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def psnr8(img01, ref01):
    """8-bit PSNR: quantize BOTH operands to uint8 (the displayed/clamped output)
    then compare. This is the user-directed gate metric (NEW REF iter-132): it
    measures drift in the actual 8-bit pixels we ship, not float-level noise.
    Bit-identical 8-bit output -> capped 100.0 dB (finite, so the loop gate parses
    it as a number instead of 'inf')."""
    a = _to_u8(img01).astype(np.float64)
    b = _to_u8(ref01).astype(np.float64)
    mse = float(np.mean((a - b) ** 2)) / (255.0 ** 2)
    if mse <= 0.0:
        return 100.0
    return min(100.0, -10.0 * math.log10(mse))


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


def _spawn_ref_hero(out_npy: Path, scene: str, cameras: Path, iter_dir: str) -> None:
    """Render cpu_cpp_mb hero reference in a child process (exclusive device).

    render_clean and cpu_cpp_mb each embed gsplat_tt in a different .so; running
    the reference in-process after render_clean has opened the device yields a
    saturated-white ref (~3 dB) on some hosts. Subprocess ref then exit avoids
    dual MetalContext corruption.
    """
    env = os.environ.copy()
    env.setdefault("TTW_ALLOW_DIRECT", "1")
    env.pop("TT_METAL_CACHE_RENDER", None)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--ref-only",
        str(out_npy),
        "--scene", scene,
        "--cameras", str(cameras),
        "--iter-dir", iter_dir,
    ]
    print(f"[run] spawning cpu_cpp_mb reference subprocess -> {out_npy.name}",
          flush=True)
    subprocess.run(cmd, env=env, check=True)


_HOST_PROFILE = bool(os.environ.get("GSPLAT_TT_HOST_PROFILE", "").strip() not in ("", "0"))


def render_clean_view_timed(pipeline, gauss, c2w, K, H, W):
    t_c2w = time.perf_counter()
    extr = c2w_to_w2c(torch.from_numpy(np.asarray(c2w, dtype=np.float32)))
    c2w_ms = (time.perf_counter() - t_c2w) * 1000.0
    t = time.perf_counter()
    res = pipeline.render(gauss, extr, K, H, W)
    wall_ms = (time.perf_counter() - t) * 1000.0
    img = _to_image(res)
    if _HOST_PROFILE:
        to_img_ms = (time.perf_counter() - t) * 1000.0 - wall_ms
        print(f"HPPY c2w_ms={c2w_ms:.3f} to_image_ms={to_img_ms:.3f}", flush=True)
    return img, wall_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--cameras", type=Path,
                    default=REPO_ROOT / "benchmarks/cameras_v2.json")
    ap.add_argument("--iter-dir", default="render-clean")
    ap.add_argument("--dump-views", default=None)
    ap.add_argument("--no-ref", action="store_true",
                    help="skip the cpu_cpp_mb reference render + PSNR gate; time "
                         "the 30 render_clean views only (used for a clean Tracy "
                         "device-profiler capture). Does not change render_clean.")
    ap.add_argument("--ref-only", nargs=1, metavar="OUT_NPY",
                    help=argparse.SUPPRESS)
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
    order = cam["order"]
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

    if args.ref_only is not None:
        out_npy = Path(args.ref_only[0])
        ref = render_hero(get_backend("cpu_cpp_mb"), gauss, hero_view, fov_deg,
                          W, H, contrib_floor)
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, ref.astype(np.float32))
        print(f"[run] ref-only wrote {out_npy} mean={ref.mean():.4f}", flush=True)
        os._exit(0)

    ref = None
    ref_npy = out_dir / "hero_ref.npy"
    if not args.no_ref:
        _spawn_ref_hero(ref_npy, args.scene, args.cameras, args.iter_dir)
        ref = np.load(ref_npy)

    # render_clean JIT cache must not share prod kernels.
    os.environ["TT_METAL_CACHE"] = _CACHE_RENDER
    clean_backend = CleanBackend()
    clean_pipeline = Pipeline(clean_backend, tile_size=32, contrib_floor=contrib_floor)

    print(f"[run] warmup (hero='{hero_name}', {W}x{H}, scene={args.scene})",
          flush=True)
    t_warm = time.perf_counter()
    _ = render_clean_view_timed(clean_pipeline, gauss, hero_view["c2w"], K, H, W)
    warmup_s = time.perf_counter() - t_warm

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

    # NEW REF (iter-132): the gate metric is 8-bit PSNR vs a committed golden
    # frame, not float PSNR vs the freshly-rendered CPU reference. The CPU
    # reference is still rendered (when available) as a SECONDARY ground-truth
    # diagnostic (hero_vs_cpu) so we don't lose the float-level correctness anchor.
    GOLDEN_REF = REPO_ROOT / "tests" / "fixtures" / "hero" / "hero_golden_8bit.png"

    # Secondary diagnostic: float PSNR vs the freshly-rendered CPU reference.
    hero_vs_cpu = float("nan")
    if not args.no_ref and ref is not None:
        hero_vs_cpu = psnr(hero_clean, ref)
        ref_mean = float(ref.mean())
        if ref_mean > 0.95 or ref_mean < 0.05:
            print(f"[run] FATAL: reference mean={ref_mean:.4f} looks invalid "
                  f"(expected ~0.33); aborting gate", file=sys.stderr, flush=True)
            sys.exit(4)

    Image.fromarray(_to_u8(hero_clean)).save(out_dir / "hero_clean.png")

    # PRIMARY gated metric: 8-bit PSNR vs the committed golden reference.
    hero_vs_ref = float("nan")
    if GOLDEN_REF.exists():
        golden8 = np.asarray(Image.open(GOLDEN_REF).convert("RGB"), dtype=np.uint8)
        hero_vs_ref = psnr8(hero_clean, golden8.astype(np.float32) / 255.0)
        d = np.clip(np.abs(_to_u8(hero_clean).astype(np.int16)
                           - golden8.astype(np.int16)) * 10, 0, 255).astype(np.uint8)
        Image.fromarray(d).save(out_dir / "hero_diff10.png")
    elif not args.no_ref and ref is not None:
        # Golden missing -> fall back to legacy float-vs-CPU behavior.
        hero_vs_ref = hero_vs_cpu

    # CPU reference artifacts (ground-truth visibility, regardless of golden).
    if not args.no_ref and ref is not None:
        Image.fromarray(_to_u8(ref)).save(out_dir / "hero_ref.png")
        cpu_diff = np.clip(np.abs(hero_clean - ref) * 10.0, 0.0, 1.0)
        cpu_diff_name = "hero_diff10_cpu.png" if GOLDEN_REF.exists() else "hero_diff10.png"
        Image.fromarray((cpu_diff * 255.0).astype(np.uint8)).save(out_dir / cpu_diff_name)

    def fmt(x):
        return "inf" if x == float("inf") else f"{x:.2f}"

    print(f"SUMMARY scene={args.scene} hero='{hero_name}' "
          f"hero_vs_ref={fmt(hero_vs_ref)}dB(8bit-vs-golden) "
          f"hero_vs_cpu={fmt(hero_vs_cpu)}dB(float-vs-cpu) "
          f"avg_frame_ms={avg_ms:.1f} p50_ms={p50_ms:.1f} "
          f"min_ms={min_ms:.1f} max_ms={max_ms:.1f} n_views={len(per_view_ms)} "
          f"warmup_s={warmup_s:.1f} "
          f"out={out_dir}", flush=True)
    if hero_vs_ref == hero_vs_ref:
        print(f"TTW_METRIC hero_vs_ref={fmt(hero_vs_ref)}", flush=True)
    print(f"TTW_TIMING ms_view={avg_ms:.3f}", flush=True)
    print(f"TTW_TIMING blend={avg_ms:.3f}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
