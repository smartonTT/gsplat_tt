"""iter-000: generate 30-view cameras_v2.json + render the numpy reference set.

Usage:
  python3 scripts/capture_reference.py --seed 0 --size 1024 --scene bicycle

Produces 30 views: hero + 5 hero-adjacent + 12 orbit + 6 elevation + 6 challenge,
sorted by camera-position proximity (greedy nearest-next from hero). Each view is
rendered with the cpu (numpy) backend at the requested resolution.

The reference images this writes are FROZEN at iter-000 and never regenerated.
Any later iter compares its renders against these.

Default scene is ``bicycle``. When ``benchmarks/cameras.json`` has a manual hero
for the scene (e.g. Mip-NeRF360 _DSC8679), that c2w is used verbatim and the
other 29 views orbit around its look target. Otherwise framing matches the
interactive viewer ([gsplat/viewer.py:_orbit_pose]): percentile bbox center,
distance = 1.2 × diagonal, hero @ (elev=0°, az=220°).
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

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402
from gsplat.viewer import _orbit_pose  # noqa: E402
from scripts.derive_camera import filter_positions, look_at_c2w  # noqa: E402


def build_intrinsics(W: int, H: int, fov_deg: float) -> torch.Tensor:
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(K)


def cam_pos(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3].copy()


def scene_bounds(means: np.ndarray, opacities: np.ndarray) -> tuple[np.ndarray, float]:
    """Same percentile framing as the interactive viewer."""
    visible = means[opacities > 0.1]
    if visible.shape[0] < 100:
        visible = means
    lo = np.percentile(visible, 5, axis=0)
    hi = np.percentile(visible, 95, axis=0)
    center = (lo + hi) * 0.5
    diagonal = float(np.linalg.norm(hi - lo))
    return center, diagonal


def place_orbit(
    center: np.ndarray,
    distance: float,
    elev_deg: float,
    az_deg: float,
    dist_mul: float = 1.0,
) -> np.ndarray:
    """Viewer-aligned c2w at ``distance × dist_mul`` from ``center``."""
    dist = distance * dist_mul
    position, look_at, up = _orbit_pose(center, dist, azim_deg=az_deg, elev_deg=elev_deg)
    return look_at_c2w(eye=position, target=look_at, world_up=up)


def orbit_params_from_c2w(c2w: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Derive (center, distance, elev°, az°) from an OpenCV hero c2w."""
    eye = c2w[:3, 3].copy()
    forward = c2w[:3, 2].copy()
    fn = float(np.linalg.norm(forward))
    if fn < 1e-9:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        forward = forward / fn
    center = eye + forward
    dist = float(np.linalg.norm(eye - center))
    if dist < 1e-6:
        dist = 3.0
    off = (eye - center) / dist
    elev_deg = float(np.degrees(np.arcsin(np.clip(-off[1], -1.0, 1.0))))
    az_deg = float(np.degrees(np.arctan2(off[0], off[2])))
    return center, dist, elev_deg, az_deg


def load_manual_hero(
    scene: str, repo_root: Path,
) -> tuple[np.ndarray, float] | None:
    """Return (hero c2w, fov_deg) when cameras.json marks the hero manual."""
    cam_file = repo_root / "benchmarks" / "cameras.json"
    if not cam_file.exists() or scene not in json.loads(cam_file.read_text()):
        return None
    entry = json.loads(cam_file.read_text())[scene]
    hero = (entry.get("views") or {}).get("hero") or {}
    if not hero.get("manual"):
        return None
    c2w = hero.get("c2w")
    if c2w is None:
        return None
    fov_deg = float(entry.get("fov_deg", 50.0))
    return np.asarray(c2w, dtype=np.float64), fov_deg


def generate_views(
    center: np.ndarray,
    distance: float,
    seed: int,
    base_elev_deg: float = 0.0,
    base_az_deg: float = 220.0,
    *,
    hero_c2w: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return {name: c2w} for 30 views using viewer orbit conventions."""
    rng = np.random.default_rng(seed)
    views: dict[str, np.ndarray] = {}

    def place(elev_deg: float, az_deg: float, dist_mul: float = 1.0) -> np.ndarray:
        return place_orbit(center, distance, elev_deg, az_deg, dist_mul)

    if hero_c2w is not None:
        views["hero"] = hero_c2w.copy()
    else:
        views["hero"] = place(base_elev_deg, base_az_deg)

    # 5 hero-adjacent: ±8° elev/az jitter, ±5% distance.
    for i in range(5):
        d_el = float(rng.uniform(-8.0, 8.0))
        d_az = float(rng.uniform(-8.0, 8.0))
        d_r = 1.0 + float(rng.uniform(-0.05, 0.05))
        views[f"adj_{i+1}"] = place(base_elev_deg + d_el, base_az_deg + d_az, d_r)

    # 12 orbit at hero elevation, every 30° (named by absolute az for clarity).
    for k in range(1, 13):
        abs_az = (base_az_deg + 30.0 * k) % 360.0
        views[f"orbit_{int(round(abs_az)):03d}"] = place(base_elev_deg, base_az_deg + 30.0 * k)

    # 6 elevation steps. Clamp to (-85, +85) to avoid look-at gimbal.
    for d_el in (-30, -20, -10, 10, 20, 30):
        el = max(min(base_elev_deg + d_el, 85.0), -85.0)
        sign = "p" if d_el > 0 else "n"
        views[f"elev_{sign}{abs(d_el):02d}"] = place(el, base_az_deg)

    # 6 challenge views.
    views["chal_top"]     = place(85.0, base_az_deg)
    views["chal_bottom"]  = place(-85.0, base_az_deg)
    views["chal_close"]   = place(base_elev_deg, base_az_deg, dist_mul=0.50)
    views["chal_far"]     = place(base_elev_deg, base_az_deg, dist_mul=2.2)
    views["chal_quarter"] = place(base_elev_deg + 15.0, base_az_deg + 45.0, dist_mul=0.8)
    views["chal_behind"]  = place(base_elev_deg - 10.0, base_az_deg + 180.0, dist_mul=1.3)

    return views


def proximity_sort(views: dict[str, np.ndarray]) -> list[str]:
    """Greedy nearest-neighbour starting from hero."""
    if "hero" not in views:
        raise ValueError("hero view required")
    order = ["hero"]
    remaining = [k for k in views if k != "hero"]
    while remaining:
        last_pos = cam_pos(views[order[-1]])
        next_name = min(remaining,
                        key=lambda k: float(np.linalg.norm(cam_pos(views[k]) - last_pos)))
        order.append(next_name)
        remaining.remove(next_name)
    return order


def render_one(pipeline, gauss, c2w: np.ndarray, W: int, H: int, fov_deg: float) -> tuple[np.ndarray, float, dict]:
    extr = c2w_to_w2c(torch.from_numpy(c2w.astype(np.float32)))
    K = build_intrinsics(W, H, fov_deg)
    t0 = time.perf_counter()
    res = pipeline.render(gauss, extr, K, H, W)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return res.image, wall_ms, dict(res.timings) if hasattr(res, "timings") else {}


def dump_per_stage_fixtures(
    fixtures_dir: Path,
    backend, gauss, c2w: np.ndarray, W: int, H: int, fov_deg: float,
    contrib_floor: float,
) -> None:
    """Step through the pipeline manually, dump inputs+outputs of each stage.

    Used by Phase 2+ C++ ports to verify equivalence against numpy.
    The fixture set is built from the hero view; one snapshot is enough — the
    tests are about per-stage I/O equivalence, not viewpoint coverage.
    """
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    extr = c2w_to_w2c(torch.from_numpy(c2w.astype(np.float32)))
    K = build_intrinsics(W, H, fov_deg)
    means, scales, rotations, opacities, colors = (
        gauss.means, gauss.scales, gauss.rotations, gauss.opacities, gauss.colors
    )

    means_2d, covs_2d, depths, radii, valid_mask = backend.project(
        means, scales, rotations, extr, K, H, W, opacities=opacities,
    )
    np.savez_compressed(fixtures_dir / "project_inputs.npz",
                        means=means.numpy(), scales=scales.numpy(),
                        rotations=rotations.numpy(), extrinsics=extr.numpy(),
                        intrinsics=K.numpy(), H=H, W=W,
                        opacities=opacities.numpy())
    np.savez_compressed(fixtures_dir / "project_outputs.npz",
                        means_2d=means_2d.numpy(), covs_2d=covs_2d.numpy(),
                        depths=depths.numpy(), radii=radii.numpy(),
                        valid_mask=valid_mask.numpy())

    colors_v = colors[valid_mask]
    opacities_v = opacities[valid_mask]

    gaussian_ids, tile_ids, tiles_per_g = backend.tile_assign(
        means_2d, radii, H, W, tile_size=32,
        covs_2d=covs_2d, opacities=opacities_v,
    )
    np.savez_compressed(fixtures_dir / "tile_assign_inputs.npz",
                        means_2d=means_2d.numpy(), radii=radii.numpy(),
                        covs_2d=covs_2d.numpy(), opacities=opacities_v.numpy(),
                        H=H, W=W, tile_size=32, contrib_floor=contrib_floor)
    np.savez_compressed(fixtures_dir / "tile_assign_outputs.npz",
                        gaussian_ids=gaussian_ids.numpy(),
                        tile_ids=tile_ids.numpy(),
                        tiles_per_gaussian=tiles_per_g.numpy())

    tiles_x = (W + 31) // 32
    tiles_y = (H + 31) // 32
    sorted_gids, tile_ranges = backend.sort(
        gaussian_ids, tile_ids, depths, tiles_x, tiles_y,
    )
    np.savez_compressed(fixtures_dir / "sort_inputs.npz",
                        gaussian_ids=gaussian_ids.numpy(),
                        tile_ids=tile_ids.numpy(),
                        depths=depths.numpy(),
                        tiles_x=tiles_x, tiles_y=tiles_y)
    np.savez_compressed(fixtures_dir / "sort_outputs.npz",
                        sorted_gaussian_ids=sorted_gids.numpy(),
                        tile_ranges=tile_ranges.numpy())

    image, blend_sub = backend.blend(
        means_2d, covs_2d, colors_v, opacities_v,
        sorted_gids, tile_ranges, H, W,
    )
    np.savez_compressed(fixtures_dir / "blend_inputs.npz",
                        means_2d=means_2d.numpy(), covs_2d=covs_2d.numpy(),
                        colors=colors_v.numpy(), opacities=opacities_v.numpy(),
                        sorted_gaussian_ids=sorted_gids.numpy(),
                        tile_ranges=tile_ranges.numpy(),
                        H=H, W=W, tile_size=32)
    np.save(fixtures_dir / "blend_output.npy", image)
    (fixtures_dir / "meta.json").write_text(json.dumps({
        "fov_deg": fov_deg, "contrib_floor": contrib_floor,
        "n_visible": int(valid_mask.sum().item()),
        "n_entries": int(sorted_gids.numel()),
    }, indent=2))
    print(f"[capture_reference] dumped fixtures to {fixtures_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=1024, help="square render size")
    ap.add_argument("--scene", default="bicycle")
    ap.add_argument("--ply", default=None, help="override scenes/<scene>.ply path")
    ap.add_argument("--out-cameras", default="benchmarks/cameras_v2.json")
    ap.add_argument("--out-renders", default="benchmarks/reference_v2")
    ap.add_argument("--backend", default="cpu", help="cpu (numpy) or cpu_cpp")
    ap.add_argument("--contrib-floor", type=float, default=1.0 / 255.0)
    ap.add_argument("--fov-deg", type=float, default=50.0)
    ap.add_argument("--hero-elev", type=float, default=0.0,
                    help="hero camera elevation in degrees (viewer orbit)")
    ap.add_argument("--hero-az", type=float, default=220.0,
                    help="hero camera azimuth in degrees (viewer orbit)")
    args = ap.parse_args()

    # Discover PLY: use --ply if given, else derive_camera.SCENE_PLY entry.
    if args.ply:
        ply_path = Path(args.ply)
    else:
        from scripts.derive_camera import SCENE_PLY
        if args.scene not in SCENE_PLY:
            raise SystemExit(f"unknown scene {args.scene!r}; known: {list(SCENE_PLY)}")
        ply_path = Path(SCENE_PLY[args.scene])

    print(f"[capture_reference] scene={args.scene}  ply={ply_path}  size={args.size}  seed={args.seed}")
    print(f"[capture_reference] loading PLY ...")
    gauss = load_ply(str(ply_path))
    means_np = gauss.means.numpy() if hasattr(gauss.means, "numpy") else np.asarray(gauss.means)
    opacities_np = gauss.opacities.numpy() if hasattr(gauss.opacities, "numpy") else np.asarray(gauss.opacities)
    scales_np = gauss.scales.numpy() if hasattr(gauss.scales, "numpy") else np.asarray(gauss.scales)
    pos = filter_positions(means_np, opacities_np, scales_np)
    print(f"[capture_reference] after filter: {len(pos):,} / {len(means_np):,} Gaussians")

    repo_root = Path(__file__).resolve().parent.parent
    manual = load_manual_hero(args.scene, repo_root)
    hero_c2w = None
    fov_deg = args.fov_deg
    if manual is not None:
        hero_c2w, fov_deg = manual
        center, distance, base_elev, base_az = orbit_params_from_c2w(hero_c2w)
        print(f"[capture_reference] manual hero from cameras.json  fov={fov_deg}°")
        print(f"[capture_reference] orbit center={center.round(3)}  dist={distance:.3f}")
        print(f"[capture_reference] derived hero orbit: elev={base_elev:.1f}° az={base_az:.1f}°")
        hero_elev, hero_az = base_elev, base_az
    else:
        center, diagonal = scene_bounds(means_np, opacities_np)
        distance = diagonal * 1.2
        hero_elev, hero_az = args.hero_elev, args.hero_az
        print(f"[capture_reference] center={center.round(3)}  diagonal={diagonal:.3f}")
        print(f"[capture_reference] hero: elev={hero_elev}° az={hero_az}° (viewer orbit)")

    views = generate_views(
        center, distance, args.seed,
        base_elev_deg=hero_elev,
        base_az_deg=hero_az,
        hero_c2w=hero_c2w,
    )
    order = proximity_sort(views)
    print(f"[capture_reference] {len(views)} views generated, proximity-sorted")

    out_cam = {
        args.scene: {
            "ply": str(ply_path),
            "image_size": [args.size, args.size],
            "fov_deg": fov_deg,
            "contrib_floor": args.contrib_floor,
            "order": order,
            "views": {name: {"c2w": views[name].tolist(), "manual": False} for name in order},
        }
    }
    Path(args.out_cameras).write_text(json.dumps(out_cam, indent=2))
    print(f"[capture_reference] wrote {args.out_cameras}")

    out_dir = Path(args.out_renders)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop PNGs from a prior scene/view set so reference_v2 stays 1:1 with order.
    order_set = set(order)
    for stale in out_dir.glob("*.png"):
        if stale.stem not in order_set:
            stale.unlink()

    backend = get_backend(args.backend)
    pipeline = Pipeline(backend, tile_size=32)

    hero_c2w = views["hero"]
    fixtures_dir = repo_root / "tests" / "fixtures" / "hero"
    dump_per_stage_fixtures(fixtures_dir, backend, gauss, hero_c2w, args.size, args.size,
                             fov_deg, args.contrib_floor)

    timing_rows = []
    total_t0 = time.perf_counter()
    for i, name in enumerate(order):
        c2w = views[name]
        img, wall_ms, timings = render_one(pipeline, gauss, c2w, args.size, args.size, fov_deg)
        if img is None:
            img = np.zeros((args.size, args.size, 3), dtype=np.float32)
            empty_flag = " [EMPTY-FRAME]"
        else:
            empty_flag = ""
        if hasattr(img, "numpy"):
            img = img.numpy()
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(img_u8).save(out_dir / f"{name}.png")
        row = {"view": name, "view_idx": i, "total_ms": wall_ms,
               "empty_frame": empty_flag != "", **timings}
        timing_rows.append(row)
        print(f"[capture_reference] {i+1:2d}/{len(order)} {name:14s} {wall_ms:7.1f} ms{empty_flag}")

    sum_ms = (time.perf_counter() - total_t0) * 1000.0
    timing_path = out_dir / "timing.jsonl"
    timing_path.write_text("\n".join(json.dumps(r) for r in timing_rows) + "\n")
    print(f"[capture_reference] wrote {timing_path}")
    print(f"[capture_reference] sum_total_ms = {sum_ms:.1f}  ({len(order)} frames)")
    summary = {"sum_total_ms": sum_ms, "n_frames": len(order),
               "backend": args.backend, "size": args.size,
               "contrib_floor": args.contrib_floor, "seed": args.seed}
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
