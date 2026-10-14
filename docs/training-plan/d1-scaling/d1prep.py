#!/usr/bin/env python3
"""Task #691 (D1): importance ranking + CPU pair counts for the N x resolution sweep (Mac, no device).

Over the 169 train views of benchmarks/cameras_train_nn.json (1237x822, dataset intrinsics):
  * importance[g] = sum over views of alpha*T (blendscore.c; pixels sampled on a 2x2 stride at
    full resolution), the pruning score for the timing proxies (a timing workload, not a lever);
  * per resolution s in (1, 2, 4) (W,H = 1237//s, 822//s, intrinsics scaled by W'/W, H'/H):
    per-Gaussian sums over views of visible flags and of tile pairs. Proxy N's mean pairs/view is
    then sum(pairs[keep]) / 169, exact for the reference CPU pipeline (pairs are culled per
    Gaussian-tile, independent of the other Gaussians).
Visibility and pairs follow gsplat/rasterization.py (opacity-aware radius, contrib floor 1/255,
k_cap 3, per-pair contribution cull), as in #688 coherence.py (loader/cov3d copied from there).

Writes OUT/rank.npy (int32 ids, most important first), OUT/counts.npz, OUT/prep.json.
Usage: d1prep.py --ply ../../scenes/bicycle.ply --cams benchmarks/cameras_train_nn.json --out DIR
                 [--workers 3] [--limit V]
"""
import argparse, ctypes, json, os, subprocess, sys, time
from multiprocessing import get_context
import numpy as np

TILE = 32
SCALES = (1, 2, 4)
G = {}
HERE = os.path.dirname(os.path.abspath(__file__))


def build_lib(out):
    so = os.path.join(out, "blendscore.so")
    if not os.path.exists(so):
        base = ["cc", "-O3", "-ffast-math", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "blendscore.c")]
        sdkd = "/Library/Developer/CommandLineTools/SDKs"   # the default SDK may not link (#688)
        for sdk in [None, "MacOSX15.4.sdk", "MacOSX15.sdk"]:
            r = subprocess.run(base if sdk is None else base[:1] + ["-isysroot", os.path.join(sdkd, sdk)] + base[1:],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
        else:
            sys.exit("cc failed: " + r.stderr)
    lib = ctypes.CDLL(so)
    lib.blendscore.argtypes = [ctypes.c_void_p, ctypes.c_int64] + [ctypes.c_void_p] * 9 + \
        [ctypes.c_int] * 4 + [ctypes.c_void_p]
    return lib


def load_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        start = f.tell()
    lines = hdr.decode().splitlines()
    n = int([l for l in lines if l.startswith("element vertex")][0].split()[-1])
    props = [l.split()[-1] for l in lines if l.startswith("property float")]
    col = {p: i for i, p in enumerate(props)}
    want = ["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    out = np.empty((n, len(want)), np.float32)
    step = 1 << 18
    with open(path, "rb") as f:
        f.seek(start)
        for r in range(0, n, step):
            m = min(step, n - r)
            blk = np.frombuffer(f.read(m * len(props) * 4), "<f4").reshape(m, len(props))
            out[r:r + m] = blk[:, [col[w] for w in want]]
    xyz = np.ascontiguousarray(out[:, 0:3])
    op = (1.0 / (1.0 + np.exp(-out[:, 3]))).astype(np.float32)
    sc = np.exp(out[:, 4:7])
    q = out[:, 7:11]
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)
    w, x, y, z = q.T
    R = np.empty((n, 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    M = R * sc[:, None, :]
    cov = np.einsum("nij,nkj->nik", M, M).astype(np.float32)
    return xyz, op, cov


def init_worker(cache, out, threads):
    import torch
    torch.set_num_threads(threads)
    z = np.load(cache)
    G.update(xyz_t=torch.from_numpy(z["xyz"]), op_t=torch.from_numpy(z["op"]),
             cov_t=torch.from_numpy(z["cov"]), N=len(z["op"]))
    G["dummy3"] = torch.zeros(G["N"], 3); G["dummy4"] = torch.zeros(G["N"], 4)
    G["lib"] = build_lib(out)


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
    gid, tid, _ = R.get_tile_assignments(m2d, rad, H, W, TILE, covs_2d=c2d, opacities=opv,
                                         contrib_floor=float(floor))
    ids = torch.nonzero(valid).squeeze(1).numpy().astype(np.int32)
    gid = gid.numpy().astype(np.int32); tid = tid.numpy().astype(np.int32)
    N = G["N"]
    pairs = np.bincount(ids[gid], minlength=N).astype(np.int32)
    vis = valid.numpy()
    if s != 1:
        return vis, pairs, None, len(gid)
    # importance: front-to-back blend per tile (depth, id stable), sampled pixels
    d = dep.numpy()
    o = np.lexsort((d[gid], tid))
    gs = np.ascontiguousarray(gid[o]); ts = tid[o]
    ut, st = np.unique(ts, return_index=True)
    off = np.append(st, len(ts)).astype(np.int64)
    c = c2d.numpy().astype(np.float64)
    a, b, cc = c[:, 0, 0], c[:, 0, 1], c[:, 1, 1]
    det = a * cc - b * b
    det[det == 0] = 1e-30
    ca = np.ascontiguousarray(cc / det, np.float32); cb = np.ascontiguousarray(-b / det, np.float32)
    ccn = np.ascontiguousarray(a / det, np.float32)
    mx = np.ascontiguousarray(m2d[:, 0].numpy(), np.float32); my = np.ascontiguousarray(m2d[:, 1].numpy(), np.float32)
    opn = np.ascontiguousarray(opv.numpy(), np.float32)
    ut = ut.astype(np.int32)
    sc = G["score"]
    G["lib"].blendscore(off.ctypes.data, len(ut), ut.ctypes.data, gs.ctypes.data, mx.ctypes.data,
                        my.ctypes.data, ca.ctypes.data, cb.ctypes.data, ccn.ctypes.data, opn.ctypes.data,
                        ids.ctypes.data, W, H, (W + TILE - 1) // TILE, G["stride"], sc.ctypes.data)
    return vis, pairs, None, len(gid)


def work(chunk):
    N = G["N"]
    G["score"] = np.zeros(N, np.float64)
    vis = np.zeros((len(SCALES), N), np.int32); pairs = np.zeros((len(SCALES), N), np.int32)
    per = []
    for i in chunk:
        t0 = time.time(); row = {"i": i}
        for k, s in enumerate(SCALES):
            v, p, _, npairs = one(G["views"][G["names"][i]]["c2w"], s)
            vis[k] += v; pairs[k] += p
            row[f"pairs_s{s}"] = npairs; row[f"vis_s{s}"] = int(v.sum())
        row["sec"] = round(time.time() - t0, 1)
        print(f"[d1] view {i} {row}", flush=True)
        per.append(row)
    return G["score"], vis, pairs, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True); ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cam = json.load(open(a.cams))["bicycle"]
    names = cam["order"][: a.limit or None]
    K = cam["intrinsics"]
    G.update(names=names, views=cam["views"], W=cam["image_size"][0], H=cam["image_size"][1],
             K=(K["fx"], K["fy"], K["cx"], K["cy"]), floor=cam["contrib_floor"], stride=a.stride)
    cache = os.path.join(a.out, "scene.npz")
    t0 = time.time()
    if not os.path.exists(cache):
        xyz, op, cov = load_ply(a.ply)
        np.savez(cache, xyz=xyz, op=op, cov=cov)
        print(f"[d1] cached scene N={len(op)} {time.time()-t0:.0f}s", flush=True)
    build_lib(a.out)
    chunks = [list(range(len(names)))[k::a.workers] for k in range(a.workers)]
    with get_context("fork").Pool(a.workers, init_worker, (cache, a.out, a.threads)) as pool:
        res = pool.map(work, chunks)
    score = sum(r[0] for r in res); vis = sum(r[1] for r in res); pairs = sum(r[2] for r in res)
    per = sorted((row for r in res for row in r[3]), key=lambda x: x["i"])
    rank = np.argsort(-score, kind="stable").astype(np.int32)
    np.save(os.path.join(a.out, "rank.npy"), rank)
    np.savez_compressed(os.path.join(a.out, "counts.npz"), score=score, vis=vis, pairs=pairs)
    json.dump(dict(views=len(names), N=int(len(score)), stride=a.stride, scales=SCALES,
                   nonzero_score=int((score > 0).sum()), per_view=per, sec=round(time.time() - t0)),
              open(os.path.join(a.out, "prep.json"), "w"), indent=1)
    print(f"[d1] done views={len(names)} nonzero={int((score>0).sum())} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "../../..")))
    main()
