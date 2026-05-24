"""Derive the viewer's default-camera c2w and patch benchmarks/cameras.json.

The interactive viewer (gsplat/viewer.py) places its initial camera at
azim=180°, elev=0° around _scene_center at distance _camera_distance, where:

  visible = means[opacities > 0.1]  (fallback to all means if <100 visible)
  lo, hi  = np.percentile(visible, 5/95, axis=0)
  _scene_center   = (lo + hi) * 0.5
  _camera_distance = norm(hi - lo) * 1.2

The _orbit_pose function then returns (position, look_at, up_direction) using
world_up=-Y, initial_offset=+Z, initial_right=+X.

This script computes the c2w for the viewer-default pose and patches the
hero view in benchmarks/cameras.json. It also renames the existing hero
to "back" so the old back-of-head reference is preserved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.viewer import _orbit_pose  # noqa: E402
from scripts.derive_camera import look_at_c2w  # noqa: E402


def viewer_default_c2w(ply_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (c2w, scene_center, camera_distance) for the viewer's default
    azim=180°, elev=0° pose."""
    gauss = load_ply(str(ply_path))
    means = gauss.means.numpy()
    opacities = gauss.opacities.numpy()
    visible = means[opacities > 0.1]
    if visible.shape[0] < 100:
        visible = means
    lo = np.percentile(visible, 5, axis=0)
    hi = np.percentile(visible, 95, axis=0)
    scene_center = (lo + hi) * 0.5
    camera_distance = float(np.linalg.norm(hi - lo)) * 1.2

    position, look_at, up_direction = _orbit_pose(
        scene_center, camera_distance, azim_deg=180.0, elev_deg=0.0,
    )
    # viewer.py uses _WORLD_UP_INITIAL = -Y for viser's coordinate system, but
    # render_fixed.py / cameras.json convention is world_up = +Y (Stitch
    # appears upright when down = world -Y). Flip the sign so the resulting
    # c2w renders right-side-up via render_fixed.
    c2w = look_at_c2w(eye=position, target=look_at, world_up=-up_direction)
    return c2w, scene_center, camera_distance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="stitch",
                    help="Scene key in benchmarks/cameras.json (default: stitch)")
    ap.add_argument("--cameras-json", default="benchmarks/cameras.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print c2w but do not modify cameras.json")
    args = ap.parse_args()

    cameras_path = Path(args.cameras_json)
    if not cameras_path.exists():
        sys.exit(f"cameras.json not found: {cameras_path}")
    cameras = json.loads(cameras_path.read_text())
    if args.scene not in cameras:
        sys.exit(f"scene '{args.scene}' missing in {cameras_path}")
    scene_entry = cameras[args.scene]
    ply_path = Path(scene_entry["ply"])
    if not ply_path.exists():
        sys.exit(f"ply not found: {ply_path}")

    c2w, center, distance = viewer_default_c2w(ply_path)
    print(f"scene:           {args.scene}  ({ply_path})")
    print(f"scene_center:    {center.round(4)}")
    print(f"camera_distance: {distance:.4f}")
    print(f"c2w (viewer-default azim=180, elev=0):")
    for row in c2w:
        print("  [" + ", ".join(f"{v:+.6f}" for v in row) + "]")

    if args.dry_run:
        return

    views = scene_entry.setdefault("views", {})
    old_hero = views.get("hero")
    if old_hero is not None and "back" not in views:
        views["back"] = old_hero
        print("renamed existing 'hero' -> 'back' (back-of-head preserved)")

    views["hero"] = {
        "c2w": [[float(v) for v in row] for row in c2w],
        "manual": False,
    }

    cameras_path.write_text(json.dumps(cameras, indent=2) + "\n")
    print(f"wrote new hero c2w to {cameras_path}")


if __name__ == "__main__":
    main()
