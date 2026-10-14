#!/usr/bin/env python3
"""Task #710: per-tile work counts for the low-res blend sizing (Mac CPU, no device).

For a sample of the 169 train views (benchmarks/cameras_train_nn.json) and each resolution
s in (1, 2, 4), this projects bicycle.ply exactly as D1's d1prep.py does (same loader, same
gsplat/rasterization.py project + 32x32 tile assignment + per-pair contribution cull, floor 1/255)
and records, per D1 proxy N (rank.npy prefix, the #691 importance order):
  * n32[t]  : tile-Gaussian pairs of 32x32 screen tile t (== D1's pair count)
  * mb32[t] : 4x8 microblocks touched by those pairs (pixel-centre test: op*exp(-m^2/2) >= floor)
  * n16/mb16, n8/mb8: the same for 16x16 and 8x8 tiles (a pair kept iff one of the sub-tile's
    microblocks is touched)
Microblocks are a fixed 4-row x 8-col grid of the image, so mb totals do not depend on the tile size.
Writes OUT/tiles.npz (object arrays keyed by s, N, view).
Usage: tilecounts.py --ply scenes/bicycle.ply --cams benchmarks/cameras_train_nn.json
                     --rank <t691 out>/rank.npy --out DIR [--every 7] [--workers 4]
"""
import argparse, json, os, sys, time
from multiprocessing import get_context
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../d1-scaling"))
SCALES = (1, 2, 4)
NS = (6131954, 3000000, 1500000, 1000000, 500000)
G = {}


def init_worker(cache, rank, threads):
    import torch
    torch.set_num_threads(threads)
    z = np.load(cache)
    G.update(xyz_t=torch.from_numpy(z["xyz"]), op_t=torch.from_numpy(z["op"]),
             cov_t=torch.from_numpy(z["cov"]), N=len(z["op"]))
    G["dummy3"] = torch.zeros(G["N"], 3); G["dummy4"] = torch.zeros(G["N"], 4)
    r = np.load(rank)
    pos = np.empty(len(r), np.int32); pos[r] = np.arange(len(r), dtype=np.int32)
    G["pos"] = pos


def mb_masks(mx, my, ca, cb, cc, thr, tx0, ty0, W, H):
    """32-bit mask of touched 4x8 microblocks of the 32x32 tile at (tx0, ty0) per pair.
    bit = band*4 + col (band = 4-row group 0..7, col = 8-col group 0..3)."""
    P = len(mx)
    mask = np.zeros(P, np.int64)
    for r in range(32):
        py = ty0 + r + 0.5
        dy = py - my
        # power: ca*dx^2 + 2 cb dx dy + cc dy^2 <= thr  (conic = inverse cov)
        disc = cb * cb * dy * dy - ca * (cc * dy * dy - thr)
        ok = (disc >= 0) & (ty0 + r < H)
        sq = np.sqrt(np.maximum(disc, 0))
        xc = mx - cb * dy / ca
        lo = xc - sq / ca - tx0 - 0.5   # pixel centre x = tx0 + c + 0.5 with c in [lo, hi]
        hi = xc + sq / ca - tx0 - 0.5
        clo = np.ceil(lo); chi = np.floor(hi)
        clo = np.maximum(clo, 0); chi = np.minimum(chi, np.minimum(31, W - 1 - tx0))
        ok &= chi >= clo
        band = r // 4
        for col in range(4):
            c0, c1 = col * 8, col * 8 + 7
            hit = ok & (chi >= c0) & (clo <= c1)
            mask |= hit.astype(np.int64) << (band * 4 + col)
    return mask


def popcount(m):
    x = m.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def one(c2w, s):
    import torch
    import gsplat.rasterization as R
    W0, H0, K0, floor = G["W"], G["H"], G["K"], G["floor"]
    W, H = W0 // s, H0 // s
    fx, fy, cx, cy = K0[0] * W / W0, K0[1] * H / H0, K0[2] * W / W0, K0[3] * H / H0
    c2w = np.asarray(c2w, np.float64)
    w2c = np.eye(4); w2c[:3, :3] = c2w[:3, :3].T; w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
    Kt = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    R.build_covariance_3d = lambda a, b: G["cov_t"]
    m2d, c2d, dep, rad, valid = R.project_gaussians(
        G["xyz_t"], G["dummy3"], G["dummy4"], torch.from_numpy(w2c.astype(np.float32)), Kt, H, W,
        opacities=G["op_t"], contrib_floor=float(floor))
    opv = G["op_t"][valid]
    gid, tid, _ = R.get_tile_assignments(m2d, rad, H, W, 32, covs_2d=c2d, opacities=opv,
                                         contrib_floor=float(floor))
    ids = torch.nonzero(valid).squeeze(1).numpy()
    gid = gid.numpy(); tid = tid.numpy().astype(np.int64)
    tx = (W + 31) // 32; ty = (H + 31) // 32
    tx16 = (W + 15) // 16; ty16 = (H + 15) // 16
    c = c2d.numpy().astype(np.float64)
    a, b, cc = c[:, 0, 0], c[:, 0, 1], c[:, 1, 1]
    det = a * cc - b * b; det[det == 0] = 1e-30
    ica, icb, icc = cc / det, -b / det, a / det
    op = opv.numpy().astype(np.float64)
    thr = 2.0 * np.log(np.maximum(op / floor, 1.0))
    mx = m2d[:, 0].numpy().astype(np.float64); my = m2d[:, 1].numpy().astype(np.float64)
    masks = np.empty(len(gid), np.int64)
    step = 1 << 19
    for k in range(0, len(gid), step):
        g = gid[k:k + step]; t = tid[k:k + step]
        masks[k:k + step] = mb_masks(mx[g], my[g], ica[g], icb[g], icc[g], thr[g],
                                     (t % tx) * 32, (t // tx) * 32, W, H)
    pos = G["pos"][ids[gid]]
    # sub-tile masks: 16x16 = 4 bands x 2 cols, 8x8 = 2 bands x 1 col of the 32x32 tile
    subs = {}
    for ts, nb, nc in ((16, 4, 2), (8, 2, 1)):
        lst = []
        for sy in range(32 // ts):
            for sx in range(32 // ts):
                bits = 0
                for band in range(sy * nb, sy * nb + nb):
                    for col in range(sx * nc, sx * nc + nc):
                        bits |= 1 << (band * 4 + col)
                lst.append((sy, sx, popcount(masks & bits)))
        subs[ts] = lst
    mbc = popcount(masks)
    t32x, t32y = tid % tx, tid // tx
    out = {}
    for N in NS:
        keep = pos < N
        n32 = np.bincount(tid[keep], minlength=tx * ty)
        mb32 = np.bincount(tid[keep], weights=mbc[keep], minlength=tx * ty)
        o = dict(n32=n32.astype(np.int32), mb32=mb32.astype(np.int32),
                 zero_mb=int((keep & (mbc == 0)).sum()))
        for ts, lst in subs.items():
            k = 32 // ts
            txs = (W + ts - 1) // ts; tys = (H + ts - 1) // ts
            n = np.zeros(txs * tys, np.int64); mb = np.zeros(txs * tys)
            for sy, sx, qc in lst:
                q = (t32y * k + sy) * txs + (t32x * k + sx)
                sel = keep & (qc > 0)
                n += np.bincount(q[sel], minlength=txs * tys)
                mb += np.bincount(q[sel], weights=qc[sel], minlength=txs * tys)
            o[f"n{ts}"] = n.astype(np.int32); o[f"mb{ts}"] = mb.astype(np.int32)
        out[N] = o
    return out


def work(chunk):
    res = []
    for i in chunk:
        t0 = time.time()
        row = {s: one(G["views"][G["names"][i]]["c2w"], s) for s in SCALES}
        print(f"[t710] view {i} {time.time()-t0:.1f}s pairs32 s1/s2/s4 "
              f"{[int(row[s][NS[0]]['n32'].sum()) for s in SCALES]}", flush=True)
        res.append((i, row))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True); ap.add_argument("--cams", required=True)
    ap.add_argument("--rank", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=7); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cam = json.load(open(a.cams))["bicycle"]
    names = cam["order"]
    K = cam["intrinsics"]
    G.update(names=names, views=cam["views"], W=cam["image_size"][0], H=cam["image_size"][1],
             K=(K["fx"], K["fy"], K["cx"], K["cy"]), floor=cam["contrib_floor"])
    cache = os.path.join(a.out, "scene.npz")
    if not os.path.exists(cache):
        import d1prep
        xyz, op, cov = d1prep.load_ply(a.ply)
        np.savez(cache, xyz=xyz, op=op, cov=cov)
    sel = list(range(0, len(names), a.every))
    chunks = [sel[k::a.workers] for k in range(a.workers)]
    t0 = time.time()
    with get_context("fork").Pool(a.workers, init_worker, (cache, a.rank, a.threads)) as pool:
        res = sorted((r for rr in pool.map(work, chunks) for r in rr), key=lambda x: x[0])
    import pickle
    with open(os.path.join(a.out, "tiles.pkl"), "wb") as f:
        pickle.dump(dict(views=[r[0] for r in res], scales=SCALES, Ns=NS,
                         W=G["W"], H=G["H"], data=[r[1] for r in res]), f)
    print(f"[t710] done {len(res)} views {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
    main()
