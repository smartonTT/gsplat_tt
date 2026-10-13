"""Task #631 (ledger U3): CPU sizing of a load-time Morton reorder + per-chunk frustum cull for pfwc.

Scene as pfwc sees it today: bicycle.ply after the #536 static opacity prune (opacity < 1/255
dropped at load). Then a per-scene Morton reorder (render/run.py morton_order, same code), chunks of
1024 gaussians (one pfwc tile), and per chunk two conservative tests, built once per scene:
  box    : render/host/chunk_cull.h (box of mean +- K rho, K = 1.5, rho = 3 sqrt(tr cov3d))
  sphere : bounding sphere of the same points (centre = box centre, R = max |m - c| + K rho),
           culled when it is behind z = k_near, or fully outside one 5 px-padded side plane with
           the same rho_max / zmin <= s_K guard as chunk_cull.h. Contained in the box test.
Per view: share of the pruned N in culled chunks, the empty-chunk bound (no visible gaussian),
and visible gaussians lost (must be 0; visibility = the #140/#169 per-gaussian model).
Nothing is fitted to the bench views: the order and the chunk bounds depend only on the scene.
Usage (repo root): python3 docs/morton-cull-t631/size631.py <bicycle.ply> [--views N]
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pfwc-chunk-cull-t466"))
from cullshare import (CAM, K, K_NEAR, PAD_PX, TILE, culled,  # noqa: E402
                       morton_order, visible)


def load_ply(path):
    """cullshare.load_ply with one sequential read (its memmap column picks are slow on 1.5 GB)."""
    head = open(path, "rb").read(16384)
    hl = head.index(b"end_header\n") + len(b"end_header\n")
    lines = head[:hl].split(b"\n")
    n = int(next(l for l in lines if l.startswith(b"element vertex")).split()[-1])
    props = [l.split()[-1].decode() for l in lines if l.startswith(b"property")]
    with open(path, "rb") as fh:
        fh.seek(hl)
        a = np.fromfile(fh, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    ix = {p: i for i, p in enumerate(props)}
    col = lambda names: a[:, [ix[p] for p in names]].astype(np.float64)  # noqa: E731
    means = col(["x", "y", "z"])
    scales = np.exp(col(["scale_0", "scale_1", "scale_2"]))
    q = col(["rot_0", "rot_1", "rot_2", "rot_3"])
    op = 1.0 / (1.0 + np.exp(-col(["opacity"])[:, 0]))
    del a
    return means, scales, q, op

MIN_OPACITY = 1.0 / 255.0


def sphere_tables(means, rho, order, C=TILE, k=K):
    n = len(order)
    nc = (n + C - 1) // C
    pad = nc * C - n
    m = means[order]; r = rho[order]
    if pad:
        m = np.concatenate([m, np.repeat(m[-1:], pad, 0)])
        r = np.concatenate([r, np.repeat(r[-1:], pad)])
    m = m.reshape(nc, C, 3); r = r.reshape(nc, C)
    lo = (m - k * r[..., None]).min(1); hi = (m + k * r[..., None]).max(1)
    c = 0.5 * (lo + hi)
    R = (np.linalg.norm(m - c[:, None, :], axis=2) + k * r).max(1)
    return lo, hi, r.max(1), c, R


def sphere_culled(c, R, rmax, w2c, f, W, H, k=K):
    pc = c @ w2c[:3, :3].T + w2c[:3, 3]
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    out = z + R <= K_NEAR
    s = rmax / np.maximum(z - R, K_NEAR)
    cx, cy = 0.5 * W, 0.5 * H
    for U0, val in (((W - cx + PAD_PX) / f, x), ((cx + PAD_PX) / f, -x),
                    ((H - cy + PAD_PX) / f, y), ((cy + PAD_PX) / f, -y)):
        uk = math.sqrt(max(0.0, k * k * (1 + U0 * U0) - 1))
        sk = (uk - U0) / math.sqrt(1 + uk * uk)
        out |= ((val - U0 * z) / math.sqrt(1 + U0 * U0) > R) & (s <= sk)
    return out


def main():
    args = sys.argv[1:]
    nviews = None
    if "--views" in args:
        i = args.index("--views"); nviews = int(args[i + 1]); del args[i:i + 2]
    cam = json.load(open(CAM))["bicycle"]
    W, H = cam["image_size"]
    floor = cam["contrib_floor"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op = load_ply(args[0] if args else cam["ply"])
    N0 = len(means)
    keep = ~(op.astype(np.float32) < np.float32(MIN_OPACITY))
    means, scales, q, op = means[keep], scales[keep], q[keep], op[keep]
    N = len(means)
    rho = 3.0 * np.sqrt((scales ** 2).sum(1))
    t0 = time.perf_counter()
    order = morton_order(means)
    lo, hi, rmax, c, R = sphere_tables(means, rho, order)
    t_build = time.perf_counter() - t0
    nc = len(R)
    print(f"N ply {N0}, after #536 prune {N} ({N0 - N} dropped), chunks {nc} x {TILE}, "
          f"K {K}, pad {PAD_PX} px, per-scene order+bounds {t_build:.2f} s (numpy)")
    print(f"sphere radius m: median {np.median(R):.3f}  p90 {np.percentile(R, 90):.3f}  "
          f"box half-diag median {np.median(np.linalg.norm(hi - lo, axis=1) / 2):.3f}")
    print("view      vis  | box: gskip  lost | sphere: gskip  lost | empty | ms box sphere")
    rows = []
    for v in cam["order"][:nviews]:
        w2c = np.linalg.inv(np.array(cam["views"][v]["c2w"], np.float64))
        vis = visible(means, scales, q, op, w2c, W, H, f, floor)
        t1 = time.perf_counter(); cb = culled(lo, hi, rmax, w2c, f, W, H); t2 = time.perf_counter()
        cs = sphere_culled(c, R, rmax, w2c, f, W, H); t3 = time.perf_counter()
        vo = np.zeros(nc * TILE, bool); vo[:N] = vis[order]
        vt = vo.reshape(nc, TILE)
        real = np.zeros(nc * TILE, bool); real[:N] = True
        realt = real.reshape(nc, TILE).sum(1)
        empty = ~vt.any(1)
        gb = realt[cb].sum() / N; gs = realt[cs].sum() / N
        lb = int(vt[cb].sum()); ls = int(vt[cs].sum())
        ge = realt[empty].sum() / N
        rows.append([vis.mean(), gb, lb, gs, ls, ge, (t2 - t1) * 1e3, (t3 - t2) * 1e3])
        print(f"{v:8s} {vis.mean():.3f} | {gb:.3f} {lb:5d} | {gs:.3f} {ls:5d} | {ge:.3f} | "
              f"{rows[-1][6]:.2f} {rows[-1][7]:.2f}", flush=True)
    r = np.array(rows)
    for name, fn in (("MIN", np.min), ("MEDIAN", np.median), ("MEAN", np.mean), ("MAX", np.max)):
        x = fn(r, 0)
        print(f"{name:8s} {x[0]:.3f} | {x[1]:.3f} {int(x[2]):5d} | {x[3]:.3f} {int(x[4]):5d} | "
              f"{x[5]:.3f} |")
    print(f"lost total: box {int(r[:, 2].sum())}, sphere {int(r[:, 4].sum())}")
    print(f"skip as share of ply N (mean): box {r[:, 1].mean() * N / N0:.3f}, "
          f"sphere {r[:, 3].mean() * N / N0:.3f}")


if __name__ == "__main__":
    main()
