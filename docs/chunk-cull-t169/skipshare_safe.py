"""Task #169: skip share of the conservative chunk test the device path uses (host, no device).

Per gaussian rho = 3 sqrt(trace cov3d) >= 3 sigma_max. Chunk box = union of mean +- K rho; chunk
also keeps rho_max. A chunk is culled when its box is all behind z = k_near, or all 8 corners
are beyond one side plane x = U0 z (U0 = (W - cx + 5) / fx, 5 px for the dilation, ceil and
rounding) AND rho_max / max(zmin, k_near) <= s_K = (u_K - U0) / sqrt(1 + u_K^2), u_K^2 =
K^2 (1 + U0^2) - 1: then no gaussian's linearised 3-sigma rect can reach the image
(docs/chunk-cull-t169/README.md). Morton order of the means, 3 chunk sizes, K in KS.
Usage (repo root): python3 docs/chunk-cull-t169/skipshare_safe.py <ply>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r16-record"))
from aniso_hero import load_ply  # noqa: E402
from skipshare import CAM, morton_order, visible  # noqa: E402

SIZES = (256, 1024, 4096)
KS = (1.25, 1.5, 2.0)
K_NEAR = 0.2
PAD_PX = 5.0


def tables(means, rho, order, C, K):
    n = len(order)
    nc = (n + C - 1) // C
    pad = nc * C - n
    m = means[order]; r = rho[order]
    lo = m - K * r[:, None]; hi = m + K * r[:, None]
    rr = r
    if pad:
        lo = np.concatenate([lo, np.repeat(lo[-1:], pad, 0)])
        hi = np.concatenate([hi, np.repeat(hi[-1:], pad, 0)])
        rr = np.concatenate([rr, np.repeat(rr[-1:], pad)])
    return lo.reshape(nc, C, 3).min(1), hi.reshape(nc, C, 3).max(1), rr.reshape(nc, C).max(1)


def culled(lo, hi, rmax, w2c, f, cx, cy, W, H, K):
    corners = np.stack([np.where(np.array([(i >> d) & 1 for d in range(3)], bool), hi, lo)
                        for i in range(8)], 1)
    pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
    x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
    out = (z <= K_NEAR).all(1)
    zmin = np.maximum(z.min(1), K_NEAR)
    s = rmax / zmin
    for U0, val in (((W - cx + PAD_PX) / f, x), ((cx + PAD_PX) / f, -x),
                    ((H - cy + PAD_PX) / f, y), ((cy + PAD_PX) / f, -y)):
        uk = math.sqrt(K * K * (1 + U0 * U0) - 1)
        sk = (uk - U0) / math.sqrt(1 + uk * uk)
        out |= (val - U0 * z > 0).all(1) & (s <= sk)
    return out


def main():
    ply = sys.argv[1]
    cam = json.load(open(CAM))["bicycle"]
    W, H = cam["image_size"]
    floor = cam["contrib_floor"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op, _ = load_ply(ply)
    means64 = means.astype(np.float64)
    rho = 3.0 * np.sqrt((scales.astype(np.float64) ** 2).sum(1))
    N = len(means)
    order = morton_order(means64)
    keys = [(C, K) for C in SIZES for K in KS]
    tabs = {k: tables(means64, rho, order, *k) for k in keys}
    print("view    vis   " + " ".join(f"C{C:<4}K{K:<4}" for C, K in keys))
    rows = []
    for v in cam["order"]:
        w2c = np.linalg.inv(np.array(cam["views"][v]["c2w"], np.float64))
        vis = visible(means64, scales, q, op, w2c, W, H, f, floor)[order]
        row = [vis.mean()]
        lost = 0
        for (C, K) in keys:
            cul = culled(*tabs[(C, K)], w2c, f, 0.5 * W, 0.5 * H, W, H, K)
            g = np.repeat(cul, C)[:N]
            row.append(g.mean())
            lost += int(vis[g].sum())
        rows.append(row)
        print(f"{v:7s} {row[0]:.3f} " + " ".join(f"{x:10.3f}" for x in row[1:]) + f"  lost {lost}")
    mean = np.mean(rows, 0)
    print(f"{'MEAN':7s} {mean[0]:.3f} " + " ".join(f"{x:10.3f}" for x in mean[1:]))
    print("x 2.98 ms: " + " ".join(f"C{C}K{K}={x * 2.98:.2f}" for (C, K), x in zip(keys, mean[1:])))


if __name__ == "__main__":
    main()
