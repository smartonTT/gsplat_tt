"""Task #169 step 1: share of gaussians a chunk AABB frustum test removes per view (host, no device).

Per chunk (C consecutive gaussians, in ply order or 3D Morton order of the means), the world AABB
of mean +- 3 * max(scale). A chunk is culled when all 8 box corners, in camera space, fail the same
frustum plane: z >= 0.01, |x| <= k z, |y| <= k z with k = (W/2 + MARGIN_PX) / f. Visible = the
per-gaussian test of docs/precull-t140/estimate.py (z > 0.01, 3-sigma rect meets the image,
op > floor, det > 0). "lost" counts visible gaussians in culled chunks (must be 0).
Usage (repo root): python3 docs/chunk-cull-t169/skipshare.py [ply]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r16-record"))
from aniso_hero import load_ply, quat_to_rot  # noqa: E402

CAM = "benchmarks/cameras_v2.json"
MARGIN_PX = 4.0
SIZES = (256, 1024, 4096)


def visible(means, scales, q, op, w2c, W, H, f, floor):
    pc = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    vis = np.zeros(len(means), bool)
    fr = np.nonzero(z > 0.01)[0]
    pc, z = pc[fr], z[fr]
    lim = 1.3 * 0.5 * W / f
    tx = np.clip(pc[:, 0] / z, -lim, lim) * z
    ty = np.clip(pc[:, 1] / z, -lim, lim) * z
    j00 = f / z; j02 = -f * tx / (z * z); j11 = f / z; j12 = -f * ty / (z * z)
    M = quat_to_rot(q[fr]) * scales[fr][:, None, :]
    Rc = w2c[:3, :3]
    covc = Rc[None] @ (M @ np.transpose(M, (0, 2, 1))) @ Rc.T[None]
    a = j00 * j00 * covc[:, 0, 0] + 2 * j00 * j02 * covc[:, 0, 2] + j02 * j02 * covc[:, 2, 2] + 0.3
    b = (j00 * j11 * covc[:, 0, 1] + j00 * j12 * covc[:, 0, 2] + j02 * j11 * covc[:, 2, 1]
         + j02 * j12 * covc[:, 2, 2])
    c = j11 * j11 * covc[:, 1, 1] + 2 * j11 * j12 * covc[:, 1, 2] + j12 * j12 * covc[:, 2, 2] + 0.3
    mx = f * pc[:, 0] / z + 0.5 * W
    my = f * pc[:, 1] / z + 0.5 * H
    rx = np.ceil(3 * np.sqrt(a)); ry = np.ceil(3 * np.sqrt(c))
    v = (mx + rx > 0) & (mx - rx < W) & (my + ry > 0) & (my - ry < H) & (op[fr] > floor)
    v &= (a * c - b * b) > 0
    vis[fr] = v
    return vis


def morton_order(means):
    lo = np.percentile(means, 0.5, axis=0); hi = np.percentile(means, 99.5, axis=0)
    u = np.clip((means - lo) / (hi - lo), 0, 1)
    qi = (u * 1023).astype(np.uint64)

    def spread(x):
        x = (x | (x << 16)) & np.uint64(0x030000FF)
        x = (x | (x << 8)) & np.uint64(0x0300F00F)
        x = (x | (x << 4)) & np.uint64(0x030C30C3)
        x = (x | (x << 2)) & np.uint64(0x09249249)
        return x
    code = spread(qi[:, 0]) | (spread(qi[:, 1]) << np.uint64(1)) | (spread(qi[:, 2]) << np.uint64(2))
    return np.argsort(code, kind="stable")


def chunk_boxes(means, rad, order, C):
    n = len(order)
    nc = (n + C - 1) // C
    pad = nc * C - n
    m = means[order]; r = rad[order]
    lo = m - r[:, None]; hi = m + r[:, None]
    if pad:
        lo = np.concatenate([lo, np.repeat(lo[-1:], pad, 0)])
        hi = np.concatenate([hi, np.repeat(hi[-1:], pad, 0)])
    return lo.reshape(nc, C, 3).min(1), hi.reshape(nc, C, 3).max(1)


def chunk_culled(lo, hi, w2c, k):
    corners = np.stack([np.where(np.array([(i >> d) & 1 for d in range(3)], bool), hi, lo)
                        for i in range(8)], 1)  # nc x 8 x 3
    pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
    x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
    out = (z < 0.01).all(1)
    for s in (x - k * z, -x - k * z, y - k * z, -y - k * z):
        out |= (s > 0).all(1)
    return out


def main():
    ply = sys.argv[1] if len(sys.argv) > 1 else "scenes/bicycle.ply"
    cam = json.load(open(CAM))["bicycle"]
    W, H = cam["image_size"]
    floor = cam["contrib_floor"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    k = (0.5 * W + MARGIN_PX) / f
    means, scales, q, op, _ = load_ply(ply)
    means = means.astype(np.float64)
    N = len(means)
    rad = 3.0 * scales.max(1).astype(np.float64)
    orders = {"ply": np.arange(N), "morton": morton_order(means)}
    boxes = {(o, C): chunk_boxes(means, rad, orders[o], C) for o in orders for C in SIZES}
    keys = list(boxes)
    print(f"N {N}; op <= floor (static, never visible): {(op <= floor).mean():.4f}; margin {MARGIN_PX} px")
    print("view    vis   " + " ".join(f"{o[:3]}{C:>5}" for o, C in keys) + "   (skip share of N)")
    rows = []
    for v in cam["order"]:
        w2c = np.linalg.inv(np.array(cam["views"][v]["c2w"], np.float64))
        vis = visible(means, scales, q, op, w2c, W, H, f, floor)
        row = [vis.mean()]
        lost_all = 0
        for key in keys:
            o, C = key
            lo, hi = boxes[key]
            cul = chunk_culled(lo, hi, w2c, k)
            g_cul = np.repeat(cul, C)[:N]
            row.append(g_cul.mean())
            lost_all += int(vis[orders[o]][g_cul].sum())
        rows.append(row)
        print(f"{v:7s} {row[0]:.3f} " + " ".join(f"{x:9.3f}" for x in row[1:]) + f"   lost {lost_all}")
    mean = np.mean(rows, 0)
    print(f"{'MEAN':7s} {mean[0]:.3f} " + " ".join(f"{x:9.3f}" for x in mean[1:]))
    print("x 2.98 ms: " + " ".join(f"{o[:3]}{C}={x * 2.98:.2f}" for (o, C), x in zip(keys, mean[1:])))


if __name__ == "__main__":
    main()
