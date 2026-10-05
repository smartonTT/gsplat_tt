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
    """Quantize a [0,1] float image to uint8 with floor — matches the saved PNG.
    render_clean already returns this uint8 image (packed on device, task #61)."""
    if isinstance(img01, np.ndarray) and img01.dtype == np.uint8:
        return img01
    return (np.clip(np.asarray(img01, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def _to_f01(img):
    """Float [0,1] view of an image (uint8 render_clean output or float ref)."""
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img


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
    if isinstance(img, np.ndarray) and img.dtype == np.uint8:
        return img
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


def _back_to_back(args, pipeline, gauss, cam, order, K, H, W, hero_name,
                  hero_clean, dump_dir, out_dir):
    """Throughput mode: render `order` back to back, 1 + args.b2b_passes times.

    The timed window of a pass holds only pipeline.render() + _to_image() per
    view (extrinsics are built before it). Pass 0 is the check pass: its frames
    are md5'd after the pass and written to --dump-views with the latency
    mode's file names, so the sweep md5 is computed the same way. Passes 1..N
    are the measured ones (ms_frame = their mean): by default they keep their
    frames and must match pass 0 byte for byte; --b2b-drop drops each frame as
    soon as render() returns, like a viewer (no compare). Returns the exit code
    (5 when a measured pass differs from the check pass)."""
    import hashlib
    extrs = [c2w_to_w2c(torch.from_numpy(np.asarray(cam["views"][n]["c2w"],
                                                     dtype=np.float32)))
             for n in order]
    pass_ms = []
    digests0 = None
    identical = True
    for p in range(1 + max(0, args.b2b_passes)):
        keep = p == 0 or not args.b2b_drop
        imgs = []
        t0 = time.perf_counter()
        if keep:
            for extr in extrs:
                imgs.append(_to_image(pipeline.render(gauss, extr, K, H, W)))
        else:
            for extr in extrs:
                _to_image(pipeline.render(gauss, extr, K, H, W))
        wall_ms = (time.perf_counter() - t0) * 1000.0
        pass_ms.append(wall_ms / len(order))
        if keep:
            digests = [hashlib.md5(_to_u8(im).tobytes()).hexdigest() for im in imgs]
        if p == 0:
            digests0 = digests
            for i, (name, img) in enumerate(zip(order, imgs)):
                if name == hero_name:
                    hero_clean = img
                if dump_dir is not None:
                    Image.fromarray(_to_u8(img)).save(dump_dir / f"view{i:02d}_{name}.png")
        elif keep and digests != digests0:
            identical = False
            bad = [order[i] for i, (a, b) in enumerate(zip(digests, digests0)) if a != b]
            print(f"[run] B2B pass {p}: {len(bad)} views differ from pass 0: {bad[:5]}",
                  file=sys.stderr, flush=True)
        print(f"[run] B2B pass {p}{' (check)' if p == 0 else ''}"
              f"{'' if keep else ' (drop)'}: {wall_ms:.1f} ms for {len(order)} views "
              f"= {pass_ms[-1]:.3f} ms/frame", flush=True)
        del imgs
    Image.fromarray(_to_u8(hero_clean)).save(out_dir / "hero_clean.png")
    timed = pass_ms[1:] or pass_ms
    ms_frame = statistics.mean(timed)
    raw_md5 = hashlib.md5("".join(digests0).encode()).hexdigest()[:8]
    print(f"B2B scene={args.scene} n_views={len(order)} passes={len(timed)} "
          f"drop={'yes' if args.b2b_drop else 'no'} "
          f"ms_frame={ms_frame:.3f} fps={1000.0 / ms_frame:.2f} "
          f"pass_ms_frame={','.join(f'{x:.3f}' for x in timed)} "
          f"check_pass_ms_frame={pass_ms[0]:.3f} raw_md5={raw_md5} "
          f"identical_across_passes="
          f"{('yes' if identical else 'NO') if not args.b2b_drop else 'unchecked'} "
          f"out={out_dir}", flush=True)
    print(f"TTW_TIMING b2b_ms_frame={ms_frame:.3f}", flush=True)
    return 0 if identical else 5


def _main():
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
    ap.add_argument("--view-range", default=None, metavar="START:END",
                    help="time only order[START:END] (python slice). Used by the "
                         "chunked Tracy capture so each devrun job fits the 600 s "
                         "ceiling. The warmup still renders the hero view.")
    ap.add_argument("--back-to-back", action="store_true",
                    help="throughput mode (task #275): render the sweep continuously, "
                         "no per-view host work beyond render() (no prints, PNG saves "
                         "or hashing inside the timed window), and report wall / views "
                         "as ms/frame. Secondary metric; ms_view latency stays primary.")
    ap.add_argument("--b2b-passes", type=int, default=3,
                    help="measured back-to-back passes over the sweep, after one "
                         "check pass (each timed separately)")
    ap.add_argument("--b2b-drop", action="store_true",
                    help="measured passes drop each frame when render() returns "
                         "(viewer-like; no compare against the check pass)")
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
    # Opt-in accuracy mode (task #260): GSPLAT_TT_CONTRIB_FLOOR_INV=N sets the floor
    # to 1/N, same as editing contrib_floor in the cameras json. Bicycle, N=1024:
    # 12.63 vs 11.65 ms/view, 45.83 vs 41.16 dB vs reference_v2. Unset = default 1/255.
    if os.environ.get("GSPLAT_TT_CONTRIB_FLOOR_INV"):
        contrib_floor = 1.0 / float(os.environ["GSPLAT_TT_CONTRIB_FLOOR_INV"])
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
    warm_img, _ = render_clean_view_timed(clean_pipeline, gauss, hero_view["c2w"], K, H, W)
    warmup_s = time.perf_counter() - t_warm

    if args.view_range:
        a, b = args.view_range.split(":")
        order = order[int(a) if a else None:int(b) if b else None]
        if not order:
            sys.exit(f"[run] --view-range {args.view_range} selects no views")
    print(f"[run] timing {len(order)} views (warmup excluded)", flush=True)
    # Zero the C++ per-stage accumulators so they cover the timed views only.
    if hasattr(clean_backend._clean, "reset_stage_timings"):
        clean_backend._clean.reset_stage_timings()
    per_view_ms = []
    # The warmup is a hero render; a --view-range chunk may not contain the hero.
    hero_clean = warm_img
    # GSPLAT_PER_VIEW_STAGES=1: print each view's stage-timer deltas, to find
    # one-off per-view costs (buffer regrowth) that the averages hide.
    pv_stages = (os.environ.get("GSPLAT_PER_VIEW_STAGES") == "1"
                 and hasattr(clean_backend._clean, "stage_timings"))
    st_prev = clean_backend._clean.stage_timings() if pv_stages else None
    if args.back_to_back:
        rc = _back_to_back(args, clean_pipeline, gauss, cam, order, K, H, W,
                           hero_name, hero_clean, dump_dir, out_dir)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    for i, name in enumerate(order):
        img, wall_ms = render_clean_view_timed(
            clean_pipeline, gauss, cam["views"][name]["c2w"], K, H, W)
        per_view_ms.append(wall_ms)
        if pv_stages:
            st_now = clean_backend._clean.stage_timings()
            d = {k: float(st_now[k]) - float(st_prev[k]) for k in st_now
                 if k != "views" and abs(float(st_now[k]) - float(st_prev[k])) >= 0.05}
            st_prev = st_now
            print(f"VIEW_STAGES i={i} name={name} wall={wall_ms:.2f} "
                  + " ".join(f"{k}={v:.2f}" for k, v in d.items()), flush=True)
        if name == hero_name:
            hero_clean = img
        if dump_dir is not None:
            png = dump_dir / f"view{i:02d}_{name}.png"
            Image.fromarray(_to_u8(img)).save(png)
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
    if not args.no_ref and ref is not None and ref.shape == _to_f01(hero_clean).shape:
        hero_vs_cpu = psnr(_to_f01(hero_clean), ref)
        ref_mean = float(ref.mean())
        if ref_mean > 0.95 or ref_mean < 0.05:
            print(f"[run] FATAL: reference mean={ref_mean:.4f} looks invalid "
                  f"(expected ~0.33); aborting gate", file=sys.stderr, flush=True)
            sys.exit(4)

    Image.fromarray(_to_u8(hero_clean)).save(out_dir / "hero_clean.png")

    # PRIMARY gated metric: 8-bit PSNR vs the committed golden reference.
    hero_vs_ref = float("nan")
    golden8 = None
    if not args.no_ref and GOLDEN_REF.exists():
        golden8 = np.asarray(Image.open(GOLDEN_REF).convert("RGB"), dtype=np.uint8)
        if golden8.shape != _to_u8(hero_clean).shape:
            print(f"[run] WARNING: golden {golden8.shape} != hero "
                  f"{_to_u8(hero_clean).shape}; skipping golden compare",
                  file=sys.stderr, flush=True)
            golden8 = None
    if golden8 is not None:
        hero_vs_ref = psnr8(hero_clean, golden8.astype(np.float32) / 255.0)
        d = np.clip(np.abs(_to_u8(hero_clean).astype(np.int16)
                           - golden8.astype(np.int16)) * 10, 0, 255).astype(np.uint8)
        Image.fromarray(d).save(out_dir / "hero_diff10.png")
    elif not args.no_ref and ref is not None:
        # Golden missing -> fall back to legacy float-vs-CPU behavior.
        hero_vs_ref = hero_vs_cpu

    # CPU reference artifacts (ground-truth visibility, regardless of golden).
    if not args.no_ref and ref is not None and ref.shape == _to_f01(hero_clean).shape:
        Image.fromarray(_to_u8(ref)).save(out_dir / "hero_ref.png")
        cpu_diff = np.clip(np.abs(_to_f01(hero_clean) - ref) * 10.0, 0.0, 1.0)
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

    # Per-stage host attribution of the frame (render/host/stage_timers.h).
    # Emitted as stage_<name> so the legacy ms_view/blend aliases above keep
    # their meaning (both = avg frame time) for the existing report tooling.
    # mat is 0 unless GSPLAT_TT_SPLIT_BLEND=1 (else blend holds mat+cull+blend).
    _STAGE_ORDER = ["head", "project", "tile_assign", "sort", "blend_setup",
                    "mat", "cull", "blend", "d2h", "assemble", "tail"]
    if hasattr(clean_backend._clean, "stage_timings"):
        st = clean_backend._clean.stage_timings()
        n = max(1, int(st.get("views", 0)))
        parts = []
        stage_sum = 0.0
        for k in _STAGE_ORDER:
            v = float(st.get(k, 0.0)) / n
            stage_sum += v
            parts.append(f"{k}={v:.3f}")
            print(f"TTW_TIMING stage_{k}={v:.3f}", flush=True)
        view_total = float(st.get("view_total", 0.0)) / n
        # view_total - sum(stages) is unaccounted C++ time inside render_view;
        # avg_ms - view_total is the Python/pybind marshal residual.
        print(f"TTW_TIMING stage_view_total={view_total:.3f}", flush=True)
        print(f"TTW_TIMING stage_pybind={max(0.0, avg_ms - view_total):.3f}",
              flush=True)
        print(f"STAGES n={st.get('views', 0)} " + " ".join(parts)
              + f" | sum={stage_sum:.3f} view_total={view_total:.3f}"
              + f" resid_in_view={view_total - stage_sum:+.3f}"
              + f" avg_frame_ms={avg_ms:.3f}"
              + f" resid_vs_frame={avg_ms - stage_sum:+.3f}", flush=True)

        # Leaf split of `sort` (SortCallTimings, render/host/sort.h). bin_* are
        # the Pass A count kernel, the histogram D2H, the host layout and the
        # Pass B emit kernel; publish_wait is the drain of radix+publish+dir.
        _SORT_ORDER = ["pread", "bin_count", "bin_hist_d2h", "bin_layout",
                       "upload", "bin_emit", "kernel", "d2h", "compact",
                       "publish_host", "publish_wait", "mat"]
        sort_parts = []
        sort_sum = 0.0
        for k in _SORT_ORDER:
            v = float(st.get(f"sort_{k}", 0.0)) / n
            sort_sum += v
            sort_parts.append(f"{k}={v:.3f}")
            print(f"TTW_TIMING stage_sort_{k}={v:.3f}", flush=True)
        sort_ms = float(st.get("sort", 0.0)) / n
        print(f"TTW_TIMING stage_sort_other={sort_ms - sort_sum:.3f}", flush=True)
        print(f"SORT_STAGES n={st.get('views', 0)} " + " ".join(sort_parts)
              + f" | sum={sort_sum:.3f} sort={sort_ms:.3f}"
              + f" resid={sort_ms - sort_sum:+.3f}", flush=True)

        # Leaf split of `project` and `tile_assign` (stage_timers.h): host setup
        # / SetRuntimeArgs / enqueue / Finish-or-blocking-read per device driver.
        _SUB_ORDER = {
            "project": ["cov3d", "pfwc_setup", "pfwc_rtargs", "pfwc_enqueue",
                        "pfwc_finish", "gather_setup", "gather_rtargs",
                        "gather_enqueue", "gather_wait", "gather_result"],
            "tile_assign": ["setup", "rtargs", "enqueue", "scan_finish",
                            "p_d2h", "k2_finish", "publish"],
        }
        for stage, keys in _SUB_ORDER.items():
            sub_parts = []
            sub_sum = 0.0
            for k in keys:
                v = float(st.get(f"{stage}_{k}", 0.0)) / n
                sub_sum += v
                sub_parts.append(f"{k}={v:.3f}")
                print(f"TTW_TIMING stage_{stage}_{k}={v:.3f}", flush=True)
            stage_ms = float(st.get(stage, 0.0)) / n
            print(f"TTW_TIMING stage_{stage}_other={stage_ms - sub_sum:.3f}",
                  flush=True)
            print(f"{stage.upper()}_STAGES n={st.get('views', 0)} "
                  + " ".join(sub_parts)
                  + f" | sum={sub_sum:.3f} {stage}={stage_ms:.3f}"
                  + f" resid={stage_ms - sub_sum:+.3f}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    """Run _main and always leave through os._exit (task #292). Once the device is
    open, a normal interpreter exit runs native static teardown, which crashes
    (MeshWorkload destructors, then a double free in tt-metal's
    ShmResourceTracker::cleanup_all), so an uncaught error would dump core after
    its traceback. The success paths in _main already call os._exit."""
    rc = 0
    try:
        _main()
    except SystemExit as e:
        if isinstance(e.code, int):
            rc = e.code
        elif e.code is not None:
            print(e.code, file=sys.stderr)
            rc = 1
    except BaseException:
        import traceback
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
