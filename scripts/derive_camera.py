"""Derive deterministic hero/side/top cameras per scene from Gaussian distribution.

Usage:
  python scripts/derive_camera.py [scene ...] [--out benchmarks/cameras.json]
  python scripts/derive_camera.py --all

Filter pass (drops Gaussians that would skew framing):
  - opacity > 0.1 (faint outliers)
  - max-scale in bottom 90% (drops giant background splats that inflate bbox)

Geometry:
  - PCA on filtered positions → 3 principal axes
  - up = axis with smallest variance, sign-aligned to +world-Y (or replaced by world-Y
    if too tilted, dot < 0.7)
  - forward_base = (longest-variance axis) projected onto plane perpendicular to up,
    renormalized
  - right = up × forward_base
  - center = 50th-percentile of positions per principal axis (mapped back to world)
  - half_extents = (95th - 5th) / 2 per principal axis; diagonal = norm(half_extents)

Three presets per scene:
  hero: elev=15° (above horizon, looking down at center), az=0
  side: elev=15°, az=90° (90° around the up axis from hero)
  top:  elev=60°, az=0

Output (one entry per scene per view in benchmarks/cameras.json):
  {
    "<scene>": {
      "ply": "scenes/<file>.ply",
      "image_size": [W, H],   # both multiples of 32 (matches kernel tile size)
      "fov_deg": 50.0,
      "views": {
        "hero": {"c2w": [[...]], "manual": false},
        "side": {...},
        "top":  {...}
      }
    },
    ...
  }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make `from gsplat...` imports work whether the script is run from the project
# root or anywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsplat.loading_gaussians import load_ply  # noqa: E402


# Canonical short-name → .ply path map. Scenes outside this list can be passed
# as full paths to --ply (see CLI).
SCENE_PLY = {
    "stitch": "scenes/stitch_doll.ply",
    "luigi": "scenes/luigi.ply",
    "strawberry": "scenes/strawberry.ply",
    "bicycle": "scenes/point_cloud.ply",
}

# Default render size for the locked cameras. Width chosen as 640; the height
# is computed once per scene from the actual bounding-box aspect (see below)
# and both dims are then snapped down to multiples of 32 so the kernel sees
# whole tiles.
DEFAULT_LONGEST_EDGE = 640
DEFAULT_FOV_DEG = 50.0

# Filter thresholds (see module docstring).
OPACITY_MIN = 0.1
SCALE_TOP_REJECT_FRAC = 0.10  # drop top 10% largest Gaussians by max scale

# Preset orbit angles (degrees). Distance is always 1.5 * diagonal.
PRESETS = {
    "hero": dict(elev_deg=15.0, az_deg=0.0),
    "side": dict(elev_deg=15.0, az_deg=90.0),
    "top":  dict(elev_deg=60.0, az_deg=0.0),
}


def filter_positions(means: np.ndarray, opacities: np.ndarray,
                     scales: np.ndarray) -> np.ndarray:
    """Apply the two-stage filter for framing derivation."""
    mask = opacities > OPACITY_MIN
    if mask.sum() < 100:
        # Synthetic / very sparse scene: keep everything, no filter.
        print(f"  filter: only {int(mask.sum())} > opacity {OPACITY_MIN}; "
              f"keeping all {len(means)} Gaussians for framing")
        mask = np.ones(len(means), dtype=bool)
    # Of the opacity-passing set, drop the top 10% largest (max-scale).
    max_scale = scales.max(axis=-1)
    keep_scales = max_scale[mask]
    if len(keep_scales) > 100:
        cutoff = np.percentile(keep_scales, 100 * (1.0 - SCALE_TOP_REJECT_FRAC))
        mask = mask & (max_scale < cutoff)
    return means[mask]


def derive_basis(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PCA on positions; return (center, half_extents, up, forward, right).

    `up` is forced into the +Y world half-space (sign flipped if needed). If
    the PCA-up is too tilted (dot with world +Y < 0.7), world +Y is used.

    `forward` is the largest-variance principal axis projected to the plane
    perpendicular to `up` and renormalized — i.e. the longest horizontal
    direction of the scene. `right = up × forward` completes the basis.

    `center` and `half_extents` are computed in the principal-axis frame and
    mapped back to world: center is the mean of the 5th/95th percentile
    projections, half_extents the (95th - 5th)/2 per axis.
    """
    mean = positions.mean(axis=0)
    centered = positions - mean
    # PCA via SVD (more numerically stable than eigendecomposition).
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt rows are right singular vectors == principal axes, sorted by variance
    # descending. So Vt[0] is the longest, Vt[2] the shortest (= up_candidate).
    axes = Vt  # (3, 3)
    variances = S * S / max(len(positions) - 1, 1)

    up_candidate = axes[2].copy()
    # Align sign to world +Y.
    if up_candidate[1] < 0:
        up_candidate = -up_candidate
    world_y = np.array([0.0, 1.0, 0.0])
    if up_candidate @ world_y < 0.7:
        # Scene is tilted in capture; trust world-Y instead.
        up = world_y
    else:
        up = up_candidate

    # Forward: longest axis projected onto plane ⊥ up.
    forward_raw = axes[0]
    forward = forward_raw - (forward_raw @ up) * up
    n = np.linalg.norm(forward)
    if n < 1e-6:
        # Degenerate; pick world +X (or +Z if up=X).
        forward = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        forward = forward - (forward @ up) * up
        forward = forward / np.linalg.norm(forward)
    else:
        forward = forward / n

    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)

    # Percentile box in principal-axis frame, then map back to world.
    proj = centered @ np.stack([right, up, forward], axis=1)  # (N, 3) along [right, up, forward]
    lo = np.percentile(proj, 5, axis=0)
    hi = np.percentile(proj, 95, axis=0)
    center_pa = (lo + hi) * 0.5      # in [right, up, forward] frame
    half_pa = (hi - lo) * 0.5
    center = mean + center_pa[0] * right + center_pa[1] * up + center_pa[2] * forward

    print(f"  pca variances:   {variances.round(3)}")
    print(f"  up:              {up.round(3)} (PCA-up dot worldY={up_candidate @ world_y:+.3f})")
    print(f"  forward:         {forward.round(3)}")
    print(f"  right:           {right.round(3)}")
    print(f"  center (world):  {center.round(3)}")
    print(f"  half_extents:    R={half_pa[0]:.3f} U={half_pa[1]:.3f} F={half_pa[2]:.3f}  "
          f"diag={np.linalg.norm(half_pa):.3f}")
    return center, half_pa, up, forward, right


def look_at_c2w(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray) -> np.ndarray:
    """Build a (4,4) camera-to-world matrix using the project's OpenCV
    convention (+Z forward, +Y down) — matches `gsplat.utils.c2w_to_w2c` and
    `gsplat.rasterization.project_gaussians`. Returns float64.

    Columns: 0 = right, 1 = down (image y), 2 = forward, 3 = eye.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(world_up, forward)
    n = np.linalg.norm(right)
    if n < 1e-9:
        # Degenerate: forward is collinear with world_up. Pick an arbitrary
        # perpendicular axis as right.
        alt_up = np.array([1.0, 0.0, 0.0]) if abs(world_up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        right = np.cross(alt_up, forward)
        right = right / np.linalg.norm(right)
    else:
        right = right / n
    down = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def derive_camera_for_preset(center: np.ndarray, half: np.ndarray,
                             up: np.ndarray, forward: np.ndarray,
                             right: np.ndarray, elev_deg: float,
                             az_deg: float) -> np.ndarray:
    """Place an orbit camera at given (elev, az) looking at center.

    elev: angle ABOVE the horizontal plane (in degrees). 0 = horizon, 90 = directly above.
    az:   angle around the up axis, in the horizontal plane. 0 = looking along
          -forward (i.e. eye is on +forward side of center). 90 = looking
          along -right (eye is on +right side).
    """
    diag = float(np.linalg.norm(half))
    distance = 1.5 * diag
    e = np.deg2rad(elev_deg)
    a = np.deg2rad(az_deg)
    # Horizontal eye direction in the (forward, right) plane.
    horiz = np.cos(a) * forward + np.sin(a) * right
    eye_dir = np.cos(e) * horiz + np.sin(e) * up
    eye = center + distance * eye_dir
    c2w = look_at_c2w(eye, center, up)
    return c2w


def derive_for_scene(scene_name: str, ply_path: Path) -> dict:
    """Derive all three preset cameras for one scene."""
    print(f"\n=== {scene_name}  ({ply_path}) ===")
    gauss = load_ply(str(ply_path))
    means = gauss.means.numpy()
    opacities = gauss.opacities.numpy()
    scales = gauss.scales.numpy()
    print(f"  loaded {len(means):,} Gaussians  "
          f"opacity range [{opacities.min():.3f}, {opacities.max():.3f}]  "
          f"max-scale [{scales.max(axis=-1).min():.4f}, {scales.max(axis=-1).max():.4f}]")
    pos = filter_positions(means, opacities, scales)
    print(f"  after filter: {len(pos):,} Gaussians remain  "
          f"({100.0 * len(pos) / max(len(means), 1):.1f}%)")

    center, half, up, forward, right = derive_basis(pos)

    # Image size: keep aspect ratio = horizontal extent / vertical extent
    # (right vs up), with the longer edge fixed at DEFAULT_LONGEST_EDGE and
    # both dims snapped to multiples of 32.
    aspect = (half[0] + 1e-6) / (half[1] + 1e-6)  # right / up
    if aspect >= 1.0:
        W = DEFAULT_LONGEST_EDGE
        H = int(DEFAULT_LONGEST_EDGE / aspect)
    else:
        H = DEFAULT_LONGEST_EDGE
        W = int(DEFAULT_LONGEST_EDGE * aspect)
    W = max(32, (W // 32) * 32)
    H = max(32, (H // 32) * 32)
    print(f"  image size:      {W}x{H}  (aspect {aspect:.2f})")

    views = {}
    for name, params in PRESETS.items():
        c2w = derive_camera_for_preset(
            center, half, up, forward, right,
            elev_deg=params["elev_deg"], az_deg=params["az_deg"],
        )
        eye = c2w[:3, 3]
        print(f"  {name:5s}: elev={params['elev_deg']:5.1f}° az={params['az_deg']:5.1f}°  "
              f"eye={eye.round(2)}  d={np.linalg.norm(eye - center):.3f}")
        views[name] = {"c2w": c2w.tolist(), "manual": False}

    return {
        "ply": str(ply_path).replace(str(Path.cwd()) + "/", ""),
        "image_size": [W, H],
        "fov_deg": DEFAULT_FOV_DEG,
        "views": views,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*",
                        help=f"Scenes to derive (default: --all). Known: {', '.join(SCENE_PLY)}")
    parser.add_argument("--all", action="store_true", help="Derive all known scenes")
    parser.add_argument("--out", default="benchmarks/cameras.json",
                        help="Output cameras.json (default: benchmarks/cameras.json)")
    parser.add_argument("--preserve-manual", action="store_true", default=True,
                        help="Skip scenes/views marked manual=true in existing cameras.json")
    args = parser.parse_args()

    if args.all or not args.scenes:
        scene_names = list(SCENE_PLY.keys())
    else:
        scene_names = args.scenes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if args.preserve_manual and out_path.exists():
        existing = json.loads(out_path.read_text())

    result = dict(existing)
    for name in scene_names:
        if name not in SCENE_PLY:
            print(f"!! unknown scene '{name}' (known: {list(SCENE_PLY)}); skipping")
            continue
        ply = Path(SCENE_PLY[name])
        if not ply.exists():
            print(f"!! {ply} not found; skipping {name}")
            continue
        new_entry = derive_for_scene(name, ply)

        # Honor manual entries: per-view, copy existing if manual=true.
        old_entry = existing.get(name, {})
        old_views = old_entry.get("views", {})
        for view_name, old_view in old_views.items():
            if old_view.get("manual", False):
                print(f"  preserving manual override for {name}.{view_name}")
                new_entry["views"][view_name] = old_view
        result[name] = new_entry

    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path} with {len(result)} scene(s)")


if __name__ == "__main__":
    main()
