"""Profile per-stage cost on bicycle hero with TT backend (delegates to cpu_cpp_mb)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backends import get_backend  # noqa: E402
from gsplat.loading_gaussians import load_ply  # noqa: E402
from gsplat.pipeline import Pipeline, format_timings  # noqa: E402
from gsplat.utils import c2w_to_w2c  # noqa: E402


def main(scene_name: str = "bicycle", view_name: str = "hero", iters: int = 10):
    cams = json.loads(Path("benchmarks/cameras_v2.json").read_text())
    scene = cams[scene_name]
    gauss = load_ply(scene["ply"])
    W, H = scene.get("image_size", [1024, 1024])

    backend = get_backend("tt")
    has_fused = hasattr(backend, "has_render_fused") and backend.has_render_fused()
    print(f"backend=tt has_render_fused={has_fused}")
    p = Pipeline(backend, contrib_floor=scene.get("contrib_floor", 1 / 16384))

    view = scene["views"][view_name]
    c2w = torch.tensor(view["c2w"], dtype=torch.float32)
    fov = view.get("fov_deg", 50.0)
    longer = max(W, H)
    f = 0.5 * longer / np.tan(0.5 * np.deg2rad(fov))
    K = torch.tensor(
        [[f, 0, W * 0.5], [0, f, H * 0.5], [0, 0, 1]], dtype=torch.float32
    )
    w2c = c2w_to_w2c(c2w)

    for warm in range(2):
        r = p.render(gauss, w2c.unsqueeze(0), K.unsqueeze(0), H, W)
        t = r.timings.get("total", 0.0)
        b = r.timings.get("blend", 0.0)
        print(f"warm {warm}: total={t:.1f}ms blend={b:.1f}")

    samples = {"project": [], "tile_assign": [], "sort": [], "blend": [], "total": []}
    for _ in range(iters):
        r = p.render(gauss, w2c.unsqueeze(0), K.unsqueeze(0), H, W)
        for k in samples:
            samples[k].append(r.timings.get(k, 0.0))

    print(format_timings(r))
    print(f"\nmedians ({iters} iters, {scene_name}/{view_name}):")
    for k, v in samples.items():
        med = sorted(v)[len(v) // 2]
        print(f"  {k:<14} {med:8.2f} ms")

    backend.close()


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "bicycle"
    view = sys.argv[2] if len(sys.argv) > 2 else "hero"
    main(scene, view)
