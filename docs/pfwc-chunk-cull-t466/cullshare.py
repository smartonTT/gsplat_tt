"""Task #466: can a pfwc-only chunk cull keep today's gaussian order? (host, no device)

Per view, for 1024-gaussian pfwc tiles in PLY (training) order and in Morton order:
  cull  share of tiles the conservative chunk_cull.h test removes (K = 1.5, 5 px pad),
  empty share of tiles with no visible gaussian at all (the bound for ANY per-tile list),
  lost  visible gaussians inside culled tiles (must be 0).
Visible = the #140/#169 per-gaussian model (z > 0.01, 3-sigma rect meets the image,
opacity > floor, det > 0). Also times the per-view list build (vectorised box test).
Usage (repo root): python3 docs/pfwc-chunk-cull-t466/cullshare.py [ply] [--views N]
"""
import json
import math
import sys
import time

import numpy as np

CAM = "benchmarks/cameras_v2.json"
TILE = 1024
K = 1.5
K_NEAR = 0.2
PAD_PX = 5.0


def load_ply(path):
    head = open(path, "rb").read(16384)
    hl = head.index(b"end_header\n") + len(b"end_header\n")
    lines = head[:hl].split(b"\n")
    n = int(next(l for l in lines if l.startswith(b"element vertex")).split()[-1])
    props = [l.split()[-1].decode() for l in lines if l.startswith(b"property")]
    a = np.memmap(path, dtype=np.float32, mode="r", offset=hl, shape=(n, len(props)))
    ix = {p: i for i, p in enumerate(props)}
    col = lambda names: np.array(a[:, [ix[p] for p in names]], np.float64)  # noqa: E731
    means = col(["x", "y", "z"])
    scales = np.exp(col(["scale_0", "scale_1", "scale_2"]))
    q = col(["rot_0", "rot_1", "rot_2", "rot_3"])
    op = 1.0 / (1.0 + np.exp(-col(["opacity"])[:, 0]))
    return means, scales, q, op


def quat_to_rot(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


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
    qi = (np.clip((means - lo) / (hi - lo), 0, 1) * 1023).astype(np.uint64)

    def spread(x):
        x = (x | (x << np.uint64(16))) & np.uint64(0x030000FF)
        x = (x | (x << np.uint64(8))) & np.uint64(0x0300F00F)
        x = (x | (x << np.uint64(4))) & np.uint64(0x030C30C3)
        x = (x | (x << np.uint64(2))) & np.uint64(0x09249249)
        return x
    code = spread(qi[:, 0]) | (spread(qi[:, 1]) << np.uint64(1)) | (spread(qi[:, 2]) << np.uint64(2))
    return np.argsort(code, kind="stable")


def tables(means, rho, order, C=TILE, k=K):
    """chunk_cull::build: per tile the box of mean +- k rho and rho_max (rho = 3 sqrt(tr cov))."""
    n = len(order)
    nc = (n + C - 1) // C
    pad = nc * C - n
    m = means[order]; r = rho[order]
    lo = m - k * r[:, None]; hi = m + k * r[:, None]
    if pad:
        lo = np.concatenate([lo, np.repeat(lo[-1:], pad, 0)])
        hi = np.concatenate([hi, np.repeat(hi[-1:], pad, 0)])
        r = np.concatenate([r, np.repeat(r[-1:], pad)])
    return lo.reshape(nc, C, 3).min(1), hi.reshape(nc, C, 3).max(1), r.reshape(nc, C).max(1)


def culled(lo, hi, rmax, w2c, f, W, H, k=K):
    """chunk_cull::survivors negated (cx = W / 2, cy = H / 2, fx = fy = f)."""
    cx, cy = 0.5 * W, 0.5 * H
    corners = np.stack([np.where(np.array([(i >> d) & 1 for d in range(3)], bool), hi, lo)
                        for i in range(8)], 1)
    pc = corners @ w2c[:3, :3].T + w2c[:3, 3]
    x, y, z = pc[..., 0], pc[..., 1], pc[..., 2]
    out = (z <= K_NEAR).all(1)
    s = rmax / np.maximum(z.min(1), K_NEAR)
    for U0, val in (((W - cx + PAD_PX) / f, x), ((cx + PAD_PX) / f, -x),
                    ((H - cy + PAD_PX) / f, y), ((cy + PAD_PX) / f, -y)):
        uk = math.sqrt(max(0.0, k * k * (1 + U0 * U0) - 1))
        sk = (uk - U0) / math.sqrt(1 + uk * uk)
        out |= (val - U0 * z > 0).all(1) & (s <= sk)
    return out


def core_local_order(means, num_cores, C=TILE):
    """pfwc core c keeps its tiles c, c + num_cores, ... (vis_tile::SeqMap deal), but its own
    gaussians are Morton-sorted, so its chunks are compact. Each core's list is padded to whole
    tiles by repeating its last gid; returns (order, real) with real False on the padding."""
    n = len(means)
    nt = (n + C - 1) // C
    code_rank = np.empty(n, np.int64)
    code_rank[morton_order(means)] = np.arange(n)
    parts, real = [], []
    for c in range(num_cores):
        g = np.concatenate([np.arange(t * C, min(n, (t + 1) * C)) for t in range(c, nt, num_cores)])
        g = g[np.argsort(code_rank[g], kind="stable")]
        pad = (-len(g)) % C
        parts.append(np.concatenate([g, np.repeat(g[-1:], pad)]))
        real.append(np.concatenate([np.ones(len(g), bool), np.zeros(pad, bool)]))
    return np.concatenate(parts), np.concatenate(real)


def view_stats(vis, order, tab, w2c, f, W, H, real=None, C=TILE):
    """(tile cull share, gaussian cull share, empty-tile share, lost visible, list build s)."""
    if real is None:
        real = np.ones(len(order), bool)
    n = len(order)
    t0 = time.perf_counter()
    cul = culled(*tab, w2c, f, W, H)
    dt = time.perf_counter() - t0
    nc = len(cul)
    g = np.repeat(cul, C)[:n] & real
    v = vis[order] & real
    vt = np.zeros(nc * C, bool); vt[:n] = v
    empty = ~vt.reshape(nc, C).any(1)
    return cul.mean(), g.sum() / real.sum(), empty.mean(), int(v[g].sum()), dt


def fmt(name, x):
    return (f"{name:8s} {x[0]:.3f} | {x[1]:.3f} {x[2]:.3f} {int(x[3]):4d} | "
            f"{x[4]:.3f} {x[5]:.3f} {x[6]:.3f} {int(x[7]):4d} | "
            f"{x[8]:.3f} {x[9]:.3f} {x[10]:.3f} {int(x[11]):4d} | {x[12]:.2f}")


def main():
    args = sys.argv[1:]
    nviews = None
    ncores = 120  # p150 pfwc grid 12x10
    if "--cores" in args:
        i = args.index("--cores"); ncores = int(args[i + 1]); del args[i:i + 2]
    if "--views" in args:
        i = args.index("--views"); nviews = int(args[i + 1]); del args[i:i + 2]
    cam = json.load(open(CAM))["bicycle"]
    ply = args[0] if args else cam["ply"]
    W, H = cam["image_size"]
    floor = cam["contrib_floor"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op = load_ply(ply)
    rho = 3.0 * np.sqrt((scales ** 2).sum(1))  # = 3 sqrt(trace cov3d)
    N = len(means)
    orders = {"ply": (np.arange(N), None), "morton": (morton_order(means), None),
              "core": core_local_order(means, ncores)}
    tabs = {k: tables(means, rho, o) for k, (o, _) in orders.items()}
    print(f"N {N}, tiles {len(tabs['ply'][2])}, K {K}, pad {PAD_PX} px")
    print(f"pfwc cores {ncores} (core: per-core Morton order inside the SeqMap deal)")
    print("view     vis   | ply: cull  empty lost | morton: tcull gcull empty lost | "
          "core: tcull gcull empty lost | build_ms")
    rows = []
    for v in cam["order"][:nviews]:
        w2c = np.linalg.inv(np.array(cam["views"][v]["c2w"], np.float64))
        vis = visible(means, scales, q, op, w2c, W, H, f, floor)
        p, m, c = (view_stats(vis, o, tabs[k], w2c, f, W, H, real) for k, (o, real) in orders.items())
        rows.append([vis.mean(), p[0], p[2], p[3], m[0], m[1], m[2], m[3],
                     c[0], c[1], c[2], c[3], m[4] * 1e3])
        print(fmt(v, rows[-1]), flush=True)
    r = np.array(rows)
    for name, fn in (("MIN", np.min), ("MEDIAN", np.median), ("MAX", np.max)):
        print(fmt(name, fn(r, 0)))
    print(f"lost total: ply {int(r[:, 3].sum())}, morton {int(r[:, 7].sum())}, core {int(r[:, 11].sum())}")


if __name__ == "__main__":
    main()
