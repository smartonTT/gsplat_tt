"""Binary-search cull thresholds against true numpy ground truth.

Ground truth: project with min_opacity=0, max_radius=-1, fixed 3σ AABB;
tile_assign without per-pair Mahalanobis; alpha_blend without microblock cull.

Compares Mahalanobis vs isoellipse AABB assignment, then tunes transmittance,
min_opacity, and k_cap until PSNR vs GT stays >= floor with best perf.

Results append to opt/cull_tune.jsonl and print a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from scripts.compare_backends import (  # noqa: E402
    analyze,
    build_intrinsics,
    close_zoom_cameras,
    render_unculled_ground_truth,
    scene_orbit_cameras,
)
from gsplat.utils import c2w_to_w2c  # noqa: E402


MIN_PSNR = 68.0  # intrinsic C++ vs numpy GT floor (true all-culls-off ref)
MAX_ABS = 0.05
JSONL = Path(__file__).resolve().parent.parent / "opt" / "cull_tune.jsonl"


@dataclass
class CullConfig:
    assign_mode: str  # "mahalanobis" | "isoellipse"
    contrib_floor_n: float  # 1/N
    transmittance_n: float  # 1/N
    min_opacity: float
    max_radius: int
    k_cap: float


def log_row(row: dict) -> None:
    JSONL.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj):
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    with JSONL.open("a") as f:
        f.write(json.dumps(_sanitize(row)) + "\n")


def render_cpp(
    gauss,
    c2w: np.ndarray,
    W: int,
    H: int,
    fov_deg: float,
    cfg: CullConfig,
    *,
    warm: bool = False,
) -> tuple[np.ndarray, dict]:
    use_maha = cfg.assign_mode == "mahalanobis"
    contrib_floor = 1.0 / max(cfg.contrib_floor_n, 1.0)
    backend = get_backend(
        "cpu_cpp",
        cull_disabled=not use_maha,
        use_isoellipse=not use_maha,
        contrib_floor=contrib_floor,
        mb_contrib_floor=contrib_floor,
        min_opacity=cfg.min_opacity,
        max_radius=cfg.max_radius,
        transmittance_threshold=1.0 / max(cfg.transmittance_n, 1.0),
        k_cap=cfg.k_cap,
    )
    pipe = Pipeline(
        backend,
        tile_size=32,
        cull_disabled=not use_maha,
        contrib_floor=contrib_floor,
        min_opacity=cfg.min_opacity,
        max_radius=cfg.max_radius,
        transmittance_threshold=1.0 / max(cfg.transmittance_n, 1.0),
        k_cap=cfg.k_cap,
        use_isoellipse=not use_maha,
    )
    extr = c2w_to_w2c(torch.from_numpy(c2w.astype(np.float32)))
    K = build_intrinsics(W, H, fov_deg)
    if warm:
        pipe.render(gauss, extr, K, H, W)
    t0 = time.perf_counter()
    res = pipe.render(gauss, extr, K, H, W)
    ms = (time.perf_counter() - t0) * 1000.0
    img = res.image if res.image is not None else np.zeros((H, W, 3), dtype=np.float32)
    stats = {
        "total_ms": ms,
        "num_visible": res.num_visible,
        "num_entries": res.num_entries,
        **res.timings,
    }
    return np.asarray(img, dtype=np.float64), stats


def eval_config(
    gauss,
    views: dict[str, np.ndarray],
    W: int,
    H: int,
    fov_deg: float,
    cfg: CullConfig,
    *,
    warm: bool = True,
    gt_cache: dict[str, np.ndarray] | None = None,
) -> dict:
    worst_psnr = float("inf")
    worst_max = 0.0
    per_view: dict[str, dict] = {}
    total_ms = 0.0
    total_entries = 0
    for name, c2w in views.items():
        if gt_cache is not None and name in gt_cache:
            ref = gt_cache[name]
        else:
            ref = render_unculled_ground_truth(gauss, c2w, W, H, fov_deg)
            if gt_cache is not None:
                gt_cache[name] = ref
        cand, stats = render_cpp(gauss, c2w, W, H, fov_deg, cfg, warm=warm)
        m = analyze(ref, cand)
        per_view[name] = {**m, "total_ms": stats["total_ms"], "num_entries": stats["num_entries"]}
        worst_psnr = min(worst_psnr, m["psnr"])
        worst_max = max(worst_max, m["max_abs"])
        total_ms += stats["total_ms"]
        total_entries += stats["num_entries"]
    ok = worst_psnr >= MIN_PSNR and worst_max <= MAX_ABS
    return {
        "config": asdict(cfg),
        "worst_psnr": worst_psnr,
        "worst_max_abs": worst_max,
        "mean_ms": total_ms / max(len(views), 1),
        "mean_entries": total_entries / max(len(views), 1),
        "ok": ok,
        "per_view": per_view,
    }


def binary_search_floor(
    gauss,
    views: dict[str, np.ndarray],
    W: int,
    H: int,
    fov_deg: float,
    assign_mode: str,
    lo_n: float,
    hi_n: float,
    base: CullConfig,
    label: str,
    gt_cache: dict[str, np.ndarray],
) -> tuple[float, dict]:
    """Search largest contrib_floor_n (tightest cull) that still passes."""
    best_n = lo_n
    best_eval: dict | None = None
    while hi_n - lo_n > 1.0:
        mid = (lo_n + hi_n) / 2.0
        cfg = CullConfig(
            assign_mode=assign_mode,
            contrib_floor_n=mid,
            transmittance_n=base.transmittance_n,
            min_opacity=base.min_opacity,
            max_radius=base.max_radius,
            k_cap=base.k_cap,
        )
        ev = eval_config(gauss, views, W, H, fov_deg, cfg, gt_cache=gt_cache)
        ev["phase"] = label
        ev["search"] = "contrib_floor_n"
        log_row(ev)
        print(
            f"  [{label}] contrib_floor 1/{mid:.0f}  "
            f"PSNR={ev['worst_psnr']:.2f}  max={ev['worst_max_abs']:.4f}  "
            f"ms={ev['mean_ms']:.1f}  ok={ev['ok']}"
        )
        if ev["ok"]:
            best_n = mid
            best_eval = ev
            lo_n = mid
        else:
            hi_n = mid
    if best_eval is None:
        cfg = CullConfig(
            assign_mode=assign_mode,
            contrib_floor_n=lo_n,
            transmittance_n=base.transmittance_n,
            min_opacity=base.min_opacity,
            max_radius=base.max_radius,
            k_cap=base.k_cap,
        )
        best_eval = eval_config(gauss, views, W, H, fov_deg, cfg, gt_cache=gt_cache)
        best_eval["phase"] = label
        best_eval["search"] = "contrib_floor_n_fallback"
        log_row(best_eval)
    return best_n, best_eval


def tune_scalar(
    gauss,
    views: dict[str, np.ndarray],
    W: int,
    H: int,
    fov_deg: float,
    base: CullConfig,
    field: str,
    values: list,
    label: str,
    gt_cache: dict[str, np.ndarray],
) -> tuple:
    best_val = values[0]
    best_ev: dict | None = None
    for v in values:
        d = asdict(base)
        d[field] = v
        cfg = CullConfig(**d)
        ev = eval_config(gauss, views, W, H, fov_deg, cfg, gt_cache=gt_cache)
        ev["phase"] = label
        ev["search"] = field
        log_row(ev)
        print(
            f"  [{label}] {field}={v}  PSNR={ev['worst_psnr']:.2f}  "
            f"max={ev['worst_max_abs']:.4f}  ms={ev['mean_ms']:.1f}  ok={ev['ok']}"
        )
        if ev["ok"] and (best_ev is None or ev["mean_ms"] < best_ev["mean_ms"]):
            best_val = v
            best_ev = ev
    return best_val, best_ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stitch")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fov-deg", type=float, default=50.0)
    ap.add_argument("--quick", action="store_true", help="fewer views for fast iteration")
    args = ap.parse_args()

    ply_path = Path(f"scenes/{args.scene}_doll.ply")
    if not ply_path.exists():
        ply_path = Path("scenes/stitch_doll.ply")
    gauss = load_ply(str(ply_path))
    W = H = args.size

    if args.quick:
        views = scene_orbit_cameras(ply_path, [220.0, 190.0])
        views.update({k: v for k, v in close_zoom_cameras(ply_path).items() if "d030" in k})
    else:
        views = scene_orbit_cameras(ply_path, list(range(130, 281, 15)))
        views.update(close_zoom_cameras(ply_path))

    print(f"[tune] views={len(views)}  ref=true_gt  floor PSNR>={MIN_PSNR} dB  max<={MAX_ABS}")
    JSONL.write_text("")
    gt_cache: dict[str, np.ndarray] = {}
    print("[tune] precomputing ground truth...")
    for name, c2w in views.items():
        gt_cache[name] = render_unculled_ground_truth(gauss, c2w, W, H, args.fov_deg)
    print(f"[tune] cached {len(gt_cache)} GT views")

    base = CullConfig(
        assign_mode="mahalanobis",
        contrib_floor_n=3000.0,
        transmittance_n=255.0,
        min_opacity=1.0 / 255.0,
        max_radius=-1,
        k_cap=3.0,
    )

    # Phase 1: loose baseline both modes
    for mode in ("mahalanobis", "isoellipse"):
        cfg = CullConfig(
            assign_mode=mode,
            contrib_floor_n=256.0,
            transmittance_n=255.0,
            min_opacity=0.0,
            max_radius=-1,
            k_cap=0.0,
        )
        ev = eval_config(gauss, views, W, H, args.fov_deg, cfg, gt_cache=gt_cache)
        ev["phase"] = "baseline_loose"
        log_row(ev)
        print(
            f"[baseline_loose] {mode}: PSNR={ev['worst_psnr']:.2f}  "
            f"ms={ev['mean_ms']:.1f}  entries={ev['mean_entries']:.0f}"
        )

    # Phase 2: tighten contrib_floor per assign mode
    print("\n=== Phase 2: tighten contrib_floor (Mahalanobis) ===")
    maha_n, maha_ev = binary_search_floor(
        gauss, views, W, H, args.fov_deg, "mahalanobis",
        lo_n=256.0, hi_n=65536.0, base=base, label="maha_contrib", gt_cache=gt_cache,
    )

    print("\n=== Phase 2b: tighten contrib_floor (isoellipse) ===")
    iso_n, iso_ev = binary_search_floor(
        gauss, views, W, H, args.fov_deg, "isoellipse",
        lo_n=256.0, hi_n=65536.0, base=base, label="iso_contrib", gt_cache=gt_cache,
    )

    winner_mode = "isoellipse" if iso_ev["mean_ms"] <= maha_ev["mean_ms"] else "mahalanobis"
    winner_n = iso_n if winner_mode == "isoellipse" else maha_n
    print(
        f"\n[assign winner] {winner_mode}  contrib_floor=1/{winner_n:.0f}  "
        f"PSNR={min(iso_ev['worst_psnr'], maha_ev['worst_psnr']):.2f}  "
        f"iso_ms={iso_ev['mean_ms']:.1f}  maha_ms={maha_ev['mean_ms']:.1f}"
    )

    tuned = CullConfig(
        assign_mode=winner_mode,
        contrib_floor_n=winner_n,
        transmittance_n=255.0,
        min_opacity=1.0 / 255.0,
        max_radius=-1,
        k_cap=3.0,
    )

    # Phase 3: transmittance
    print("\n=== Phase 3: transmittance threshold ===")
    t_n, t_ev = tune_scalar(
        gauss, views, W, H, args.fov_deg, tuned, "transmittance_n",
        [65536.0, 16384.0, 4096.0, 1024.0, 512.0, 255.0, 128.0, 64.0],
        "transmittance", gt_cache,
    )
    tuned = CullConfig(**{**asdict(tuned), "transmittance_n": t_n})

    # Phase 4: min_opacity
    print("\n=== Phase 4: min_opacity ===")
    mo, mo_ev = tune_scalar(
        gauss, views, W, H, args.fov_deg, tuned, "min_opacity",
        [0.0, 1.0 / 512.0, 1.0 / 255.0, 1.0 / 128.0, 1.0 / 64.0],
        "min_opacity", gt_cache,
    )
    tuned = CullConfig(**{**asdict(tuned), "min_opacity": mo})

    # Phase 5: k_cap (isoellipse / diagonal AABB sizing)
    print("\n=== Phase 5: k_cap ===")
    kc, kc_ev = tune_scalar(
        gauss, views, W, H, args.fov_deg, tuned, "k_cap",
        [0.0, 2.5, 3.0, 3.5, 4.0],
        "k_cap", gt_cache,
    )
    tuned = CullConfig(**{**asdict(tuned), "k_cap": kc})

    # Phase 6: max_radius (0 = min(H,W)/2 cap; -1 = disabled)
    print("\n=== Phase 6: max_radius ===")
    mr, mr_ev = tune_scalar(
        gauss, views, W, H, args.fov_deg, tuned, "max_radius",
        [-1, 0, 512, 768],
        "max_radius", gt_cache,
    )
    tuned = CullConfig(**{**asdict(tuned), "max_radius": mr})

    final = eval_config(gauss, views, W, H, args.fov_deg, tuned, gt_cache=gt_cache)
    final["phase"] = "final"
    log_row(final)

    summary = {
        "recommended": asdict(tuned),
        "worst_psnr": final["worst_psnr"],
        "worst_max_abs": final["worst_max_abs"],
        "mean_ms": final["mean_ms"],
        "maha_contrib_n": maha_n,
        "iso_contrib_n": iso_n,
        "maha_ms": maha_ev["mean_ms"],
        "iso_ms": iso_ev["mean_ms"],
    }
    out = Path("opt/cull_tune_summary.json")
    out.write_text(json.dumps(summary, indent=2))
    print("\n=== RECOMMENDED ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {JSONL} and {out}")


if __name__ == "__main__":
    main()
