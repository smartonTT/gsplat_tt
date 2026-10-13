"""Task #596 (spec A, structural-plan-3) CPU gate: per-gaussian cheap-gate survival in pfwc deal order.

Cheap gate (phase 1 of the split pfwc, needs only the mean and the cov3d diagonal, which pfwc
already streams): z > Z_NEAR and the screen rect of a static radius
    r = 3 * sqrt(c00 + c11 + c22) * (f / z) * sqrt(1 + min(|x/z|, lim)^2) + 3 * sqrt(0.3) + 1 + PAD
meets the image. sqrt(trace cov3d) >= max scale, |row of J| = f/z * sqrt(1 + (t/z)^2) with t clipped
at lim like pfwc, + 0.3 dilation, +1 for the ceil: r >= ceil(3 sqrt(a)), so the gate is conservative.
Keep set = the #140/#169 visibility model (cullshare.visible), a superset of what the fused writer keeps.
Scene = the B3 opacity-pruned scene in pfwc order (opacity_prune.prune_layout), C pfwc cores, core c
owns tiles c, c + C, ... of 1024.
Reports per view: mean survival, max survival per core, phase-2 tile ratio on the critical core
(max_c ceil(surv_c / 1024) / max_c tiles_c), gaussians the gate drops that the model keeps (must be 0).
Usage (repo root): python3 docs/pfwc-compact-t596/gate.py [ply] [--cores 110] [--views N]
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
spec = importlib.util.spec_from_file_location("cullshare", os.path.join(HERE, "..", "pfwc-chunk-cull-t466", "cullshare.py"))
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)
from render.opacity_prune import prune_layout  # noqa: E402

TILE = 1024
Z_NEAR = 0.01
PAD = 0.0


def gate(means, rho, w2c, W, H, f):
    pc = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    ok = z > Z_NEAR
    zs = np.where(ok, z, 1.0)
    lim = 1.3 * 0.5 * W / f
    ux = np.minimum(np.abs(pc[:, 0] / zs), lim); uy = np.minimum(np.abs(pc[:, 1] / zs), lim)
    s = 3.0 * rho * f / zs
    rx = s * np.sqrt(1 + ux * ux) + 3 * math.sqrt(0.3) + 1 + PAD
    ry = s * np.sqrt(1 + uy * uy) + 3 * math.sqrt(0.3) + 1 + PAD
    mx = f * pc[:, 0] / zs + 0.5 * W; my = f * pc[:, 1] / zs + 0.5 * H
    return ok & (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H)


def main():
    args = sys.argv[1:]
    C, nviews = 110, None
    if "--cores" in args:
        i = args.index("--cores"); C = int(args[i + 1]); del args[i:i + 2]
    if "--views" in args:
        i = args.index("--views"); nviews = int(args[i + 1]); del args[i:i + 2]
    cam = json.load(open(cs.CAM))["bicycle"]
    W, H = cam["image_size"]; floor = cam["contrib_floor"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op = cs.load_ply(args[0] if args else cam["ply"])
    src = prune_layout(op.astype(np.float32), floor, C)
    means, scales, q, op = means[src], scales[src], q[src], op[src]
    n = len(means)
    rho = np.sqrt((scales ** 2).sum(1))  # sqrt(trace cov3d)
    core = (np.arange(n) // TILE) % C
    tiles_c = np.bincount(np.arange((n + TILE - 1) // TILE) % C, minlength=C)
    print(f"pruned N {n}, cores {C}, tiles/core max {tiles_c.max()}")
    print("view     vis    gate   gate/core max  p2 tile ratio (crit core)  dropped-kept")
    rows = []
    for v in cam["order"][:nviews]:
        w2c = np.linalg.inv(np.array(cam["views"][v]["c2w"], np.float64))
        vis = cs.visible(means, scales, q, op, w2c, W, H, f, floor)
        g = gate(means, rho, w2c, W, H, f)
        sc = np.bincount(core, weights=g, minlength=C)
        nc = np.bincount(core, minlength=C)
        ratio = np.ceil(sc / TILE).max() / tiles_c.max()
        lost = int((vis & ~g).sum())
        rows.append([vis.mean(), g.mean(), (sc / nc).max(), ratio, lost])
        print(f"{v:8s} {rows[-1][0]:.3f}  {rows[-1][1]:.3f}  {rows[-1][2]:.3f}          {ratio:.3f}"
              f"                     {lost}", flush=True)
    r = np.array(rows)
    print(f"MEAN     {r[:, 0].mean():.3f}  {r[:, 1].mean():.3f}  {r[:, 2].mean():.3f}          {r[:, 3].mean():.3f}")
    print(f"MAX      {r[:, 0].max():.3f}  {r[:, 1].max():.3f}  {r[:, 2].max():.3f}          {r[:, 3].max():.3f}")
    print(f"dropped-kept total {int(r[:, 4].sum())}; kill (mean gate > 0.6): {r[:, 1].mean() > 0.6}")


if __name__ == "__main__":
    main()
