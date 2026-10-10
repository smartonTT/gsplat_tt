"""Task #479: per-pfwc-core and per-screen-tile work spread, PLY vs Morton gid order (host, no device).

pfwc deals 1024-gaussian tiles strided over C cores (tile t -> core t % C). Per core: visible
gaussians and pairs (32x32 screen tiles hit by the 3-sigma rect). Downstream the storage index
s (gid) is core-segment major, so for each screen tile we also count how many distinct pfwc
segments its pairs come from and the share of its pairs in its top segment.
Usage (repo root): python3 docs/chunk-cull-t479/imbalance.py <ply> [--views N]
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "docs/pfwc-chunk-cull-t466")
from cullshare import CAM, TILE, load_ply, morton_order, quat_to_rot  # noqa: E402


def rects(means, scales, q, op, w2c, W, H, f, floor):
    """Per gaussian: visible flag and the 32 px tile rect [x0, x1) x [y0, y1)."""
    pc = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    n = len(means)
    x0 = np.zeros(n, np.int64); x1 = x0.copy(); y0 = x0.copy(); y1 = x0.copy()
    vis = np.zeros(n, bool)
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
    T = 32
    x0[fr] = np.clip(np.floor((mx - rx) / T), 0, W // T); x1[fr] = np.clip(np.ceil((mx + rx) / T), 0, W // T)
    y0[fr] = np.clip(np.floor((my - ry) / T), 0, H // T); y1[fr] = np.clip(np.ceil((my + ry) / T), 0, H // T)
    return vis, x0, x1, y0, y1


def stats(order, vis, x0, x1, y0, y1, C, W):
    n = len(order)
    core = (np.arange(n) // TILE) % C           # pfwc core of each processing slot
    v = vis[order]
    npair = np.where(v, (x1 - x0)[order] * (y1 - y0)[order], 0)
    vc = np.bincount(core, v, C); pcnt = np.bincount(core, npair, C)
    # per screen tile: pairs per segment (core)
    tx = W // 32
    idx = np.nonzero(v)[0]
    segs = []
    sx0, sx1, sy0, sy1 = (a[order][idx] for a in (x0, x1, y0, y1))
    cc = core[idx]
    tile_seg = np.zeros((tx * tx, C), np.int64)
    for dy in range(int((sy1 - sy0).max())):
        for dx in range(int((sx1 - sx0).max())):
            m = (sy0 + dy < sy1) & (sx0 + dx < sx1)
            if not m.any():
                continue
            t = (sy0[m] + dy) * tx + sx0[m] + dx
            np.add.at(tile_seg, (t, cc[m]), 1)
    tot = tile_seg.sum(1)
    big = tot >= np.percentile(tot, 90)
    nseg = (tile_seg > 0).sum(1)
    top = tile_seg.max(1) / np.maximum(tot, 1)
    return dict(vis_max_mean=vc.max() / vc.mean(), pair_max_mean=pcnt.max() / pcnt.mean(),
                pairs=int(pcnt.sum()), seg_per_tile_big=float(np.median(nseg[big])),
                top_share_big=float(np.median(top[big])), top_share_max=float(top[big].max()))


def main():
    args = sys.argv[1:]
    nviews = 3
    if "--views" in args:
        i = args.index("--views"); nviews = int(args[i + 1]); del args[i:i + 2]
    cam = json.load(open(CAM))["bicycle"]
    W, H = cam["image_size"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op = load_ply(args[0])
    orders = {"ply": np.arange(len(means)), "morton": morton_order(means)}
    C = 120
    print("view   order  vis_max/mean pair_max/mean pairs   big-tile: segs  top_share(med,max)")
    for vname in cam["order"][::max(1, len(cam["order"]) // nviews)][:nviews]:
        w2c = np.linalg.inv(np.array(cam["views"][vname]["c2w"], np.float64))
        r = rects(means, scales, q, op, w2c, W, H, f, cam["contrib_floor"])
        for k, o in orders.items():
            s = stats(o, *r, C, W)
            print(f"{vname:6s} {k:6s} {s['vis_max_mean']:.3f}        {s['pair_max_mean']:.3f}        "
                  f"{s['pairs']:8d} {s['seg_per_tile_big']:5.0f}  {s['top_share_big']:.3f} {s['top_share_max']:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
