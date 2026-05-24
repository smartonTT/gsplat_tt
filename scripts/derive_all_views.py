"""Regenerate hero/side/top c2w for every scene at 1024x1024 with full-subject framing.

Uses the same scene-bounds logic as the interactive viewer
([gsplat/viewer.py:_orbit_pose](../gsplat/viewer.py)):

  visible      = means[opacity > 0.1]   (fallback to all means if <100)
  lo,hi        = np.percentile(visible, 5/95, axis=0)
  scene_center = (lo + hi) * 0.5
  diagonal     = norm(hi - lo)

The viewer uses distance = diagonal * 1.2 for its default front view; we keep
that exact value for "hero" so REPORT.html matches what the user sees in the
browser. For "side" and "top" we use 1.3 so the off-axis projection of the
bounding box still fits inside the square 1024x1024 frame with margin.

View angles (relative to scene_center, using viewer's orbit conventions):

  hero  azim=180  elev= 0    front
  side  azim=90   elev= 0    profile
  top   azim=180  elev=60    above-front

All scenes are forced to image_size=[1024, 1024], fov_deg=50.

Existing 'back' view (preserved from previous hero rework) is left untouched.
Bicycle is skipped: it's an outdoor garden scene where the viewer is meant to
be *inside* the percentile box, not orbiting it. Tackle separately if needed.
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

# (azim_deg, elev_deg, distance_factor)
VIEWS: dict[str, tuple[float, float, float]] = {
    "hero": (180.0, 0.0, 1.2),
    "side": (90.0, 0.0, 1.3),
    "top":  (180.0, 60.0, 1.3),
}


def scene_bounds(ply_path: Path) -> tuple[np.ndarray, float]:
    g = load_ply(str(ply_path))
    means = g.means.numpy()
    opacities = g.opacities.numpy()
    visible = means[opacities > 0.1]
    if visible.shape[0] < 100:
        visible = means
    lo = np.percentile(visible, 5, axis=0)
    hi = np.percentile(visible, 95, axis=0)
    center = (lo + hi) * 0.5
    diagonal = float(np.linalg.norm(hi - lo))
    return center, diagonal


def build_c2w(center: np.ndarray, distance: float, azim_deg: float, elev_deg: float) -> np.ndarray:
    position, look_at, up = _orbit_pose(center, distance, azim_deg=azim_deg, elev_deg=elev_deg)
    # viewer.py uses _WORLD_UP_INITIAL = -Y for viser; cameras.json/render_fixed
    # uses world_up = +Y. Negate so the resulting c2w renders right-side-up.
    return look_at_c2w(eye=position, target=look_at, world_up=-up)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras-json", default="benchmarks/cameras.json")
    ap.add_argument("--scenes", nargs="+", default=["stitch", "luigi", "strawberry"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cameras_path = Path(args.cameras_json)
    cams = json.loads(cameras_path.read_text())

    for key in args.scenes:
        if key not in cams:
            print(f"[skip] {key} not in cameras.json")
            continue
        ply = Path(cams[key]["ply"])
        if not ply.exists():
            print(f"[skip] {key}: ply not found at {ply}")
            continue
        center, diagonal = scene_bounds(ply)
        print(f"\n=== {key} ({ply}) ===")
        print(f"  center:   {center.round(4)}")
        print(f"  diagonal: {diagonal:.4f}")

        cams[key]["image_size"] = [1024, 1024]
        cams[key]["fov_deg"] = 50.0
        views = cams[key].setdefault("views", {})

        for vname, (az, el, factor) in VIEWS.items():
            dist = diagonal * factor
            c2w = build_c2w(center, dist, az, el)
            print(f"  {vname}: azim={az} elev={el} dist={dist:.3f} (factor={factor})")
            views[vname] = {
                "c2w": [[float(v) for v in row] for row in c2w],
                "manual": False,
            }

    if args.dry_run:
        print("\n[dry-run] not writing cameras.json")
        return
    cameras_path.write_text(json.dumps(cams, indent=2) + "\n")
    print(f"\nwrote {cameras_path}")


if __name__ == "__main__":
    main()
