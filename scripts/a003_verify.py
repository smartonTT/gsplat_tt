"""amendment-003 per-iteration gate: render the TT backend over the 30-view
bicycle bench and report PSNR against (a) the cpu_cpp_mb reference render and
(b) the reference_v2 ground-truth PNGs, plus render timing.

PSNR gates (from the plan):
  hero(tt vs cpu_cpp_mb)        >= ref_hero - 0.5 dB
  min over 30 (tt vs cpu_cpp_mb) >= 60 dB   (preferred; 40 dB hard floor)

Usage:
  python3 scripts/a003_verify.py [--backend tt] [--ref-backend cpu_cpp_mb]
      [--cameras benchmarks/cameras_v2.json] [--gt-dir benchmarks/reference_v2]
      [--views N] [--out opt/a003-verify-last.json]

Prints a one-line SUMMARY and writes a JSON blob for the supervisor loop.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


def build_intrinsics(W: int, H: int, fov_deg: float) -> torch.Tensor:
    longer = max(W, H)
    f = 0.5 * longer / math.tan(0.5 * math.radians(fov_deg))
    K = np.array([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]],
                 dtype=np.float32)
    return torch.from_numpy(K)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR for float images in [0, 1]. Returns inf for identical inputs."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse)


def to_u8(a: np.ndarray) -> np.ndarray:
    """Quantize a float image to uint8 EXACTLY as the saved PNG does
    (truncation via astype(uint8)), so PSNR and the report's diff agree."""
    return (np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)


def psnr_u8(a: np.ndarray, b_u8_floats: np.ndarray) -> float:
    """PSNR against a uint8 ground-truth PNG (already loaded as float/255).

    Quantizes the candidate float render to uint8 the SAME way the PNG is
    saved (truncation) so the comparison is in the displayed-pixel domain and
    matches the report's 10x diff image. Comparing a float render directly to
    a uint8 GT floors at ~PNG rounding error (~50 dB) and is NOT a fidelity
    measure; using a different rounding mode than the GT/PNG (e.g. round vs
    floor) injects a spurious 1-LSB mismatch (~51 dB) that the eye/diff don't
    show.
    """
    a_u8 = to_u8(a).astype(np.float64) / 255.0
    return psnr(a_u8, b_u8_floats)


STAGE_KEYS = ("project", "tile_assign", "sort", "blend", "total")


_DEVICE_BACKENDS = {"tt"}


def _enforce_device_lock(*backend_names):
    """Hard guard: the TT device may only be opened through tt-workflows
    `devrun.sh`, which holds the per-host device lock and exports
    ``TTW_DEVRUN=1``. A raw ``ssh ... a003_verify.py`` bypasses that lock and
    can collide with another device job -> firmware wedge. Refuse to open the
    device unless we were launched via devrun. A human doing a one-off manual
    run can set ``TTW_ALLOW_DIRECT=1`` (never in the loop / never an agent).
    """
    if not any(b in _DEVICE_BACKENDS for b in backend_names):
        return  # CPU-only run, no TT device touched
    if os.environ.get("TTW_DEVRUN") == "1":
        return  # launched through devrun.sh, which holds the per-host lock
    if os.environ.get("TTW_ALLOW_DIRECT") == "1":
        print("[a003] WARNING: TTW_ALLOW_DIRECT=1 — opening the TT device "
              "WITHOUT the devrun lock. Manual human use only; never in the loop.",
              file=sys.stderr, flush=True)
        return
    print(
        "[a003] REFUSING to open the TT device: not launched via devrun.sh "
        "(no TTW_DEVRUN marker).\n"
        "       Every device run MUST go through tt-workflows/scripts/devrun.sh "
        "so the per-host lock serializes the device — a raw ssh run can collide "
        "with another job and wedge the device.\n"
        "       Use:  devrun.sh --tag <name> -- \"<cmd>\"\n"
        "       (one-off human manual run only:  export TTW_ALLOW_DIRECT=1)",
        file=sys.stderr, flush=True)
    sys.exit(3)


def render_all(backend_name, gauss, order, views, fov_deg, W, H, contrib_floor):
    backend = get_backend(backend_name)
    pipeline = Pipeline(backend, tile_size=32, contrib_floor=contrib_floor)
    # warmup
    c2w0 = np.asarray(views[order[0]]["c2w"], dtype=np.float32)
    extr0 = c2w_to_w2c(torch.from_numpy(c2w0))
    K = build_intrinsics(W, H, fov_deg)
    _ = pipeline.render(gauss, extr0, K, H, W)

    imgs, stage_timings = {}, {}
    for name in order:
        c2w = np.asarray(views[name]["c2w"], dtype=np.float32)
        extr = c2w_to_w2c(torch.from_numpy(c2w))
        res = pipeline.render(gauss, extr, K, H, W)
        img = res.image
        if hasattr(img, "numpy"):
            img = img.numpy()
        imgs[name] = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
        t = dict(res.timings) if getattr(res, "timings", None) else {}
        stage_timings[name] = {k: float(t.get(k, 0.0)) for k in STAGE_KEYS}
    return imgs, stage_timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="tt")
    ap.add_argument("--ref-backend", default="cpu_cpp_mb")
    ap.add_argument("--cameras", type=Path,
                    default=Path("benchmarks/cameras_v2.json"))
    ap.add_argument("--gt-dir", type=Path,
                    default=Path("benchmarks/reference_v2"))
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--views", type=int, default=0,
                    help="limit to first N views (0 = all 30)")
    ap.add_argument("--out", type=Path,
                    default=Path("opt/a003-verify-last.json"))
    ap.add_argument("--iter-dir", default=None,
                    help="iter id; screenshots saved to opt/metal-screenshots/<iter-dir>/")
    ap.add_argument("--save-all-views", action="store_true",
                    help="also save every view PNG (default: hero only)")
    args = ap.parse_args()

    # Device-access discipline: only devrun.sh (which holds the per-host lock)
    # may open the TT device. Aborts a raw bypass run before it touches HW.
    _enforce_device_lock(args.backend, args.ref_backend)

    cam = json.loads(args.cameras.read_text())[args.scene]
    fov_deg = float(cam["fov_deg"])
    W, H = cam["image_size"]
    contrib_floor = cam.get("contrib_floor", 1.0 / 255.0)
    order = cam["order"]
    if args.views > 0:
        order = order[: args.views]
    views = cam["views"]

    gauss = load_ply(str(Path(cam["ply"])))

    tt_imgs, tt_timings = render_all(
        args.backend, gauss, order, views, fov_deg, W, H, contrib_floor)
    ref_imgs, _ = render_all(
        args.ref_backend, gauss, order, views, fov_deg, W, H, contrib_floor)

    # MANDATORY DELIVERABLE: save the hero screenshot (and optionally all
    # views) into opt/metal-screenshots/<iter-dir>/ so build_report.py shows
    # the render + a 10x diff vs ground truth on every iteration's card.
    # Screenshot is a MANDATORY deliverable — always save one. If the caller
    # didn't name the iteration, fall back to a timestamped dir so a hero.png
    # is produced regardless.
    iter_dir = args.iter_dir or time.strftime("a003-verify-%Y%m%dT%H%M%SZ",
                                              time.gmtime())
    shot_dir = Path("opt/metal-screenshots") / iter_dir
    shot_dir.mkdir(parents=True, exist_ok=True)
    to_save = order if args.save_all_views else [order[0]]
    for name in to_save:
        out_name = "hero.png" if name == "hero" else f"{name}.png"
        arr = (tt_imgs[name] * 255.0).astype(np.uint8)
        Image.fromarray(arr).save(shot_dir / out_name)

    # DIAG: dump ref + 10x diff for hero, and report top 32x32-tile error blocks
    # so payload residual corruption can be localized to tile ids.
    if os.environ.get("GSPLAT_TT_DIFF_DUMP"):
        hname = order[0]
        tt = tt_imgs[hname].astype(np.float32)
        rf = ref_imgs[hname].astype(np.float32)
        Image.fromarray((rf * 255.0).astype(np.uint8)).save(shot_dir / "ref.png")
        d = np.abs(tt - rf)
        Image.fromarray((np.clip(d * 10.0, 0, 1) * 255.0).astype(np.uint8)).save(
            shot_dir / "diff10x.png")
        err = d.mean(axis=2)
        Hh, Ww = err.shape
        TS = 32
        ty_n = (Hh + TS - 1) // TS
        tx_n = (Ww + TS - 1) // TS
        blocks = []
        for tyy in range(ty_n):
            for txx in range(tx_n):
                blk = err[tyy * TS:(tyy + 1) * TS, txx * TS:(txx + 1) * TS]
                blocks.append((float(blk.mean()), txx, tyy, int(txx + tyy * tx_n)))
        blocks.sort(reverse=True)
        print(f"DIFFDUMP hero={hname} HxW={Hh}x{Ww} tiles_x={tx_n} tiles_y={ty_n} "
              f"global_mae={err.mean():.5f}")
        for mae, txx, tyy, tid in blocks[:20]:
            print(f"DIFFDUMP_TILE tile_id={tid} tx={txx} ty={tyy} "
                  f"px=({txx*TS},{tyy*TS}) mae={mae:.5f}")

    rows = []
    for name in order:
        gt_path = args.gt_dir / f"{name}.png"
        gt = None
        if gt_path.exists():
            gt = np.asarray(Image.open(gt_path).convert("RGB"),
                            dtype=np.float32) / 255.0
        row = {
            "view": name,
            # TT vs cpu_cpp_mb: both float, in-process — the TRUE kernel
            # fidelity gate (inf = bit-identical).
            "psnr_tt_vs_ref_dB": psnr(tt_imgs[name], ref_imgs[name]),
            # vs uint8 ground truth: quantize render to uint8 first so it is
            # consistent with the displayed PNG + the report's 10x diff.
            "psnr_tt_vs_gt_dB": (psnr_u8(tt_imgs[name], gt)
                                 if gt is not None else None),
            "psnr_ref_vs_gt_dB": (psnr_u8(ref_imgs[name], gt)
                                  if gt is not None else None),
            "timings_ms": tt_timings[name],
        }
        rows.append(row)

    def _finite(vals):
        return [v for v in vals if v is not None and math.isfinite(v)]

    def _median(vals):
        vals = sorted(vals)
        n = len(vals)
        if n == 0:
            return 0.0
        return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    tt_vs_ref = [r["psnr_tt_vs_ref_dB"] for r in rows]
    tt_vs_ref_finite = _finite(tt_vs_ref)
    tt_vs_gt_finite = _finite([r["psnr_tt_vs_gt_dB"] for r in rows])
    hero = rows[0]

    per_stage_median = {
        f"{k}_ms": _median([r["timings_ms"][k] for r in rows])
        for k in STAGE_KEYS
    }
    sum_total_ms = sum(r["timings_ms"]["total"] for r in rows)

    summary = {
        "backend": args.backend,
        "ref_backend": args.ref_backend,
        "blend_mode_env": os.environ.get("GSPLAT_TT_BLEND_MODE", "0"),
        "num_views": len(rows),
        "iter_dir": iter_dir,
        "screenshot": str(shot_dir / "hero.png"),
        "hero_psnr_tt_vs_ref_dB": hero["psnr_tt_vs_ref_dB"],
        "hero_psnr_tt_vs_gt_dB": hero["psnr_tt_vs_gt_dB"],
        "hero_psnr_ref_vs_gt_dB": hero["psnr_ref_vs_gt_dB"],
        "min_psnr_tt_vs_ref_dB": (min(tt_vs_ref_finite)
                                  if tt_vs_ref_finite else float("inf")),
        "min_psnr_tt_vs_gt_dB": (min(tt_vs_gt_finite)
                                 if tt_vs_gt_finite else None),
        "all_identical_vs_ref": all(not math.isfinite(v) for v in tt_vs_ref),
        "sum_total_ms": sum_total_ms,
        "ms_per_view": sum_total_ms / max(1, len(rows)),
        "per_stage_median_ms": per_stage_median,
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    def fmt(x):
        if x is None:
            return "n/a"
        return "inf" if x == float("inf") else f"{x:.2f}"

    ps = per_stage_median
    print(f"SUMMARY backend={args.backend} blend_mode={summary['blend_mode_env']} "
          f"views={summary['num_views']} "
          f"hero_vs_gt={fmt(summary['hero_psnr_tt_vs_gt_dB'])}dB "
          f"hero_vs_ref={fmt(summary['hero_psnr_tt_vs_ref_dB'])}dB "
          f"min_vs_ref={fmt(summary['min_psnr_tt_vs_ref_dB'])}dB "
          f"min_vs_gt={fmt(summary['min_psnr_tt_vs_gt_dB'])}dB "
          f"ms/view={summary['ms_per_view']:.1f} "
          f"(proj={ps['project_ms']:.1f} ta={ps['tile_assign_ms']:.1f} "
          f"sort={ps['sort_ms']:.1f} blend={ps['blend_ms']:.1f}) "
          f"shot={summary['screenshot']}",
          flush=True)

    # TT + tt-metal: skip process-exit teardown races (ProgramImpl vs MeshDevice).
    if args.backend == "tt":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _tt_shutdown() -> None:
    try:
        from backends.cpu_cpp import _gsplat_cpu as mod
    except ImportError:
        return
    if hasattr(mod, "tt_device_shutdown"):
        mod.tt_device_shutdown()


if __name__ == "__main__":
    main()
