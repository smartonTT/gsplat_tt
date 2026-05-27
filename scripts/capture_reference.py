"""iter-000: generate 30-view cameras_v2.json + render the numpy reference set.

Usage:
  python3 scripts/capture_reference.py --seed 0 --size 512 --scene stitch \
      --out-cameras benchmarks/cameras_v2.json --out-renders benchmarks/reference_v2

Produces 30 views: hero + 5 hero-adjacent + 12 orbit + 6 elevation + 6 challenge,
sorted by camera-position proximity (greedy nearest-next from hero). Each view is
rendered with the cpu (numpy) backend at the requested resolution.

The reference images this writes are FROZEN at iter-000 and never regenerated.
Any later iter compares its renders against these.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

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
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov_deg))
    K = np.array([[f, 0.0, W * 0.5], [0.0, f, H * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(K)


def look_at_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV-style c2w: camera looks toward -Z in camera frame.

    The image was authored with +Y pointing DOWN (see camera_controls.py),
    so by default we use world_up = -Y for the up vector.
    """
    z = (eye - target)
    z = z / max(float(np.linalg.norm(z)), 1e-9)   # camera +Z = away from target
    x = np.cross(up, z)
    x_norm = float(np.linalg.norm(x))
    if x_norm < 1e-9:
        # up parallel to z — pick an arbitrary perpendicular
        alt = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(alt, z)
        x_norm = float(np.linalg.norm(x))
    x = x / x_norm
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def cam_pos(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3].copy()


def infer_scene_center(hero_c2w: np.ndarray) -> tuple[np.ndarray, float]:
    """Scene center = point the hero camera looks at, assumed at distance ||eye||.

    Returns (center, distance).
    """
    eye = hero_c2w[:3, 3]
    # camera-forward in world = -camera-Z. c2w's 3rd column is camera-Z.
    cam_z = hero_c2w[:3, 2]
    distance = float(np.linalg.norm(eye))
    center = eye - cam_z * distance
    return center, distance


def generate_views(hero_c2w: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Return {name: c2w} for 30 views."""
    rng = np.random.default_rng(seed)
    world_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    center, dist = infer_scene_center(hero_c2w)
    views: dict[str, np.ndarray] = {"hero": hero_c2w.copy()}

    hero_eye = cam_pos(hero_c2w)
    hero_dir = (hero_eye - center) / dist   # unit vector from center toward hero
    azimuth0 = math.atan2(hero_dir[0], hero_dir[2])    # XY-plane azimuth around -Y
    elevation0 = math.asin(np.clip(-hero_dir[1], -1.0, 1.0))  # because up = -Y

    def from_az_el(az: float, el: float, r: float) -> np.ndarray:
        """Place camera at (az, el, r) relative to center; look at center, world_up=-Y."""
        # Using the convention: world_up = -Y, so +elevation tilts camera "above" scene (more -Y).
        cos_el, sin_el = math.cos(el), math.sin(el)
        sin_az, cos_az = math.sin(az), math.cos(az)
        # Unit direction from center to camera:
        d = np.array([cos_el * sin_az, -sin_el, cos_el * cos_az], dtype=np.float64)
        eye = center + d * r
        return look_at_matrix(eye, center, world_up)

    # 5 hero-adjacent (small jitter in az/el and ±5% distance)
    for i in range(5):
        d_az = rng.uniform(-math.radians(8.0), math.radians(8.0))
        d_el = rng.uniform(-math.radians(8.0), math.radians(8.0))
        d_r = 1.0 + rng.uniform(-0.05, 0.05)
        views[f"adj_{i+1}"] = from_az_el(azimuth0 + d_az, elevation0 + d_el, dist * d_r)

    # 12 orbit (every 30° starting at +30°), keep hero elevation
    for k in range(1, 13):
        az = azimuth0 + math.radians(30.0 * k)
        views[f"orbit_{int(round(math.degrees(az - azimuth0)) % 360):03d}"] = \
            from_az_el(az, elevation0, dist)

    # 6 elevation (-30, -20, -10, +10, +20, +30 deg of pitch from hero), keep hero azimuth
    for d_el_deg in (-30, -20, -10, 10, 20, 30):
        el = elevation0 + math.radians(d_el_deg)
        # Clamp to (-89, +89) to avoid look_at gimbal
        el = max(min(el, math.radians(89.0)), math.radians(-89.0))
        sign = "p" if d_el_deg > 0 else "n"
        views[f"elev_{sign}{abs(d_el_deg):02d}"] = from_az_el(azimuth0, el, dist)

    # 6 challenge views
    views["chal_top"]      = from_az_el(azimuth0, math.radians(85.0), dist)
    views["chal_bottom"]   = from_az_el(azimuth0, math.radians(-85.0), dist)
    views["chal_close"]    = from_az_el(azimuth0, elevation0, dist * 0.35)
    views["chal_far"]      = from_az_el(azimuth0, elevation0, dist * 2.2)
    views["chal_quarter"]  = from_az_el(azimuth0 + math.radians(45.0),
                                        elevation0 + math.radians(15.0), dist * 0.6)
    views["chal_behind"]   = from_az_el(azimuth0 + math.radians(180.0),
                                        elevation0 - math.radians(10.0), dist * 1.3)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512, help="square render size")
    ap.add_argument("--scene", default="stitch")
    ap.add_argument("--source-cameras", default="benchmarks/cameras.json",
                    help="provides the hero pose for the scene")
    ap.add_argument("--out-cameras", default="benchmarks/cameras_v2.json")
    ap.add_argument("--out-renders", default="benchmarks/reference_v2")
    ap.add_argument("--backend", default="cpu", help="cpu (numpy) or cpu_cpp")
    ap.add_argument("--contrib-floor", type=float, default=1.0 / 255.0)
    args = ap.parse_args()

    src_cameras = json.loads(Path(args.source_cameras).read_text())
    scene_entry = src_cameras[args.scene]
    fov_deg = float(scene_entry["fov_deg"])
    ply_path = Path(scene_entry["ply"])
    hero_c2w = np.asarray(scene_entry["views"]["hero"]["c2w"], dtype=np.float64)

    views = generate_views(hero_c2w, args.seed)
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

    print(f"[capture_reference] loading PLY: {ply_path}")
    gauss = load_ply(str(ply_path))

    backend = get_backend(args.backend)
    pipeline = Pipeline(backend, tile_size=32)

    timing_rows = []
    total_t0 = time.perf_counter()
    for i, name in enumerate(order):
        c2w = views[name]
        img, wall_ms, timings = render_one(pipeline, gauss, c2w, args.size, args.size, fov_deg)
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(img_u8).save(out_dir / f"{name}.png")
        row = {"view": name, "view_idx": i, "total_ms": wall_ms, **timings}
        timing_rows.append(row)
        print(f"[capture_reference] {i+1:2d}/{len(order)} {name:14s} {wall_ms:7.1f} ms")

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
