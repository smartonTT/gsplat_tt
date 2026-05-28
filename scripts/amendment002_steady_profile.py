"""Steady-state per-stage profile via backend.render_fused (skips Pipeline overhead)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


def main(scene_name: str = "bicycle", view_name: str = "hero", iters: int = 10):
    cams = json.loads(Path("benchmarks/cameras_v2.json").read_text())
    scene = cams[scene_name]
    gauss = load_ply(scene["ply"])
    W, H = scene.get("image_size", [1024, 1024])
    view = scene["views"][view_name]
    fov = view.get("fov_deg", 50.0)
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov))
    K = torch.tensor(
        [[f, 0, W * 0.5], [0, f, H * 0.5], [0, 0, 1]], dtype=torch.float32
    )
    w2c = c2w_to_w2c(torch.tensor(view["c2w"], dtype=torch.float32))

    backend = get_backend("tt")
    for _ in range(3):
        backend.render_fused(gauss, w2c.unsqueeze(0), K.unsqueeze(0), H, W)

    per_stage = {
        "project_ms": [],
        "tile_assign_ms": [],
        "sort_ms": [],
        "blend_ms": [],
        "total_ms": [],
    }
    wall = []
    stats = {}
    for _ in range(iters):
        t0 = time.perf_counter()
        img, stats = backend.render_fused(gauss, w2c.unsqueeze(0), K.unsqueeze(0), H, W)
        wall.append((time.perf_counter() - t0) * 1000)
        for k in per_stage:
            per_stage[k].append(float(stats.get(k, 0)))

    print(f"scene={scene_name} view={view_name} H={H} W={W} N_gauss={gauss.means.shape[0]}")
    print(f"wall_ms median: {sorted(wall)[len(wall) // 2]:.1f}")
    for k, v in per_stage.items():
        med = sorted(v)[len(v) // 2]
        mn = min(v)
        print(f"  {k:<14} median={med:7.2f}  min={mn:7.2f}")
    print(f"  N_visible: {stats.get('num_visible', 0)}")
    print(f"  N_entries (pairs): {stats.get('num_entries', 0)}")
    print(f"  pairs_in (mb-stage): {stats.get('pairs_in', 0)}")
    print(f"  pairs_kept_per_mb: {stats.get('pairs_kept_per_mb', 0)}")
    print(f"  pairs_dropped: {stats.get('pairs_dropped', 0)}")

    backend.close()


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "bicycle"
    view = sys.argv[2] if len(sys.argv) > 2 else "hero"
    main(scene, view)
