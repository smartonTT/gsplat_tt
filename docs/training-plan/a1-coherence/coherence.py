#!/usr/bin/env python3
"""Task #688 (A1): cross-view coherence on the bicycle NN training tour (CPU only).

For consecutive views i -> i+1 of benchmarks/cameras_train_nn.json at 1237x822:
  L1 (sort warm start): per 32x32 tile, take the pairs present in both views, put them in
     view i's depth order and measure how far that is from view i+1's exact order
     (descents/runs, inversions, n - LIS = keys an insertion fix-up must move), plus the
     pairs new to the tile (must be sorted and merged in).
  L2 (batch-union pre-cull): |union of visible sets over K consecutive views| / |one view|,
     K = 2, 4, 8, non-overlapping batches along the tour.
Visibility and pairs follow gsplat/rasterization.py (project_gaussians with the opacity-aware
radius, contrib floor 1/255, k_cap 3, cap min(H,W)//2, min_opacity 1/255; get_tile_assignments
with the per-pair contribution cull), in fp32. Depth key = view-space z, ties by index
(a stable sort), so "exact order" is the order a full sort of (depth, id) gives.

Usage: coherence.py --ply scenes/bicycle.ply --cams benchmarks/cameras_train_nn.json
                    --out OUT.json [--proxy 1200000] [--workers 7] [--cache DIR]
"""
import argparse, ctypes, json, os, subprocess, sys, time
from multiprocessing import get_context
import numpy as np

TILE = 32
G = {}  # globals shared with forked workers


def build_lib(cache):
    so = os.path.join(cache, "segstats.so")
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segstats.c")
    if not os.path.exists(so):
        sdks = [None] + sorted(
            [os.path.join("/Library/Developer/CommandLineTools/SDKs", s) for s in
             ("MacOSX15.4.sdk", "MacOSX15.sdk")])
        for sdk in sdks:
            cmd = ["cc", "-O2", "-shared", "-fPIC", "-o", so, src]
            if sdk:
                cmd[1:1] = ["-isysroot", sdk]
            if subprocess.run(cmd, capture_output=True).returncode == 0:
                break
        else:
            sys.exit("could not build segstats.so")
    lib = ctypes.CDLL(so)
    p = ctypes.c_void_p
    lib.segstats.argtypes = [p, p, ctypes.c_int64, p, p, p]
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
    with open(path, "rb") as f:                    # plain sequential reads (memmap faults are slow)
        f.seek(start)
        for r in range(0, n, step):
            m = min(step, n - r)
            blk = np.frombuffer(f.read(m * len(props) * 4), "<f4").reshape(m, len(props))
            out[r:r + m] = blk[:, [col[w] for w in want]]
    xyz = np.ascontiguousarray(out[:, 0:3])
    op = 1.0 / (1.0 + np.exp(-out[:, 3]))
    sc = np.exp(out[:, 4:7])
    q = out[:, 7:11]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return xyz, op.astype(np.float32), sc, q.astype(np.float32)


def cov3d(sc, q):
    w, x, y, z = q.T
    R = np.empty((len(q), 3, 3), np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    M = R * sc[:, None, :]
    return np.einsum("nij,nkj->nik", M, M).astype(np.float32)


def view(c2w):
    """Visible ids, depth rank (int32, -1 = not visible), pair keys tile*N+gid (sorted).
    Uses the reference CPU pipeline itself (gsplat/rasterization.py, torch, fp32), with the
    cached cov3d fed in place of build_covariance_3d."""
    import torch
    import gsplat.rasterization as R
    xyz, op, K, W, H, floor = (G[k] for k in ("xyz", "op", "K", "W", "H", "floor"))
    N = len(xyz)
    c2w = np.asarray(c2w, np.float64)
    w2c = np.eye(4); w2c[:3, :3] = c2w[:3, :3].T; w2c[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]
    fx, fy, cx, cy = (float(k) for k in K)
    Kt = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    R.build_covariance_3d = lambda s, q: G["cov_t"]
    opt = G["op_t"]
    m2d, c2d, dep, rad, valid = R.project_gaussians(
        G["xyz_t"], G["dummy3"], G["dummy4"], torch.from_numpy(w2c.astype(np.float32)), Kt, H, W,
        opacities=opt, contrib_floor=float(floor))
    gid, tid, _ = R.get_tile_assignments(m2d, rad, H, W, TILE, covs_2d=c2d, opacities=opt[valid],
                                         contrib_floor=float(floor))
    ids = torch.nonzero(valid).squeeze(1).numpy()
    tz = dep.numpy()
    order = ids[np.argsort(tz, kind="stable")]          # (depth, id): ids ascending, stable
    rank = np.full(N, -1, np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)
    pkey = np.sort(tid.numpy().astype(np.int64) * N + ids[gid.numpy()])
    vis = valid.numpy()
    return dict(vis=vis, rank=rank, pkey=pkey, nvis=len(ids), npairs=len(pkey))


def init_worker(cache, sel_path, threads):
    import torch
    torch.set_num_threads(threads)
    z = np.load(os.path.join(cache, "scene.npz"))
    xyz, op, cov = z["xyz"], z["op"], z["cov"]
    if sel_path:
        sel = np.load(sel_path)
        xyz, op, cov = xyz[sel], op[sel], cov[sel]
    G.update(xyz=xyz, op=op, xyz_t=torch.from_numpy(np.ascontiguousarray(xyz)),
             op_t=torch.from_numpy(np.ascontiguousarray(op)),
             cov_t=torch.from_numpy(np.ascontiguousarray(cov)),
             dummy3=torch.zeros(len(xyz), 3), dummy4=torch.zeros(len(xyz), 4))
    G["lib"] = build_lib(cache)


def segstats(keys, off):
    n = len(off) - 1
    d = np.zeros(n, np.int64); inv = np.zeros(n, np.int64); lis = np.zeros(n, np.int64)
    keys = np.ascontiguousarray(keys, np.int32); off = np.ascontiguousarray(off, np.int64)
    G["lib"].segstats(keys.ctypes.data, off.ctypes.data, n, d.ctypes.data, inv.ctypes.data,
                      lis.ctypes.data)
    return d, inv, lis


def transition(p, c):
    N = len(G["xyz"])
    pos = np.searchsorted(p["pkey"], c["pkey"])
    pos = np.minimum(pos, len(p["pkey"]) - 1)
    same_tile = int((p["pkey"][pos] == c["pkey"]).sum())   # (tile, gid) pair also in view i
    # Warm start for view i+1's tile lists: every pair whose Gaussian was visible in view i
    # ("carried") is placed by its view-i depth rank; pairs of Gaussians new to the view are "new".
    gall = (c["pkey"] % N).astype(np.int64)
    carried = p["rank"][gall] >= 0
    ck = c["pkey"][carried]
    tile = ck // N; g = gall[carried]
    o = np.lexsort((p["rank"][g], tile))           # warm-start order: by tile, then old rank
    tile_o = tile[o]; keys = c["rank"][g[o]]
    ut, start = np.unique(tile_o, return_index=True)
    off = np.append(start, len(tile_o))
    d, inv, lis = segstats(keys, off)
    n = np.diff(off)
    ntile_new = np.bincount((c["pkey"] // N), minlength=G["ntiles"])
    ntile_carr = np.bincount(tile, minlength=G["ntiles"])
    new_per_tile = ntile_new - ntile_carr
    # global (per-Gaussian) order: visible in both, old rank order, new rank keys
    both = np.nonzero((p["rank"] >= 0) & (c["rank"] >= 0))[0]
    go = both[np.argsort(p["rank"][both])]
    gk = c["rank"][go]
    gd, ginv, glis = segstats(gk, np.array([0, len(go)]))
    disp = np.abs(np.argsort(np.argsort(gk, kind="stable"), kind="stable") - np.arange(len(gk)))
    pairs_per_tile = ntile_new[ntile_new > 0]
    return dict(
        pairs=int(c["npairs"]), carried=int(len(ck)), same_tile=same_tile, new=int(c["npairs"] - len(ck)),
        runs=int((d + 1).sum()), descents=int(d.sum()), inversions=int(inv.sum()),
        moved=int((n - lis).sum()), tiles=int(len(n)),
        tiles_sorted=int((d == 0).sum()),
        # pair-weighted means of per-tile quantities
        runlen_mean=float(n.sum() / (d + 1).sum()),
        log2runs_pairwt=float((n * np.log2(d + 1)).sum() / max(n.sum(), 1)),
        log2n_pairwt=float((n * np.log2(np.maximum(n, 1))).sum() / max(n.sum(), 1)),
        new_frac_p50_tile=float(np.median(new_per_tile[ntile_new > 0] / pairs_per_tile)),
        max_tile_pairs=int(ntile_new.max()),
        g_both=int(len(go)), g_new=int(c["nvis"] - len(go)), g_runs=int(gd[0] + 1),
        g_inversions=int(ginv[0]), g_moved=int(len(go) - glis[0]), nvis=int(c["nvis"]),
        g_disp_p50=float(np.median(disp)), g_disp_p99=float(np.percentile(disp, 99)),
        g_disp_max=int(disp.max()))


def work(chunk):
    names = G["names"]; out = []; masks = {}; hits = np.zeros(len(G["xyz"]), np.int32)
    prev = None
    for i in chunk:
        tv = time.time()
        cur = view(G["views"][names[i]]["c2w"])
        if G.get("verbose"):
            print(f"[a1] view {i} {time.time()-tv:.1f}s vis={cur['nvis']} pairs={cur['npairs']}", flush=True)
        masks[i] = np.packbits(cur["vis"])
        if G["score"] and (i != chunk[0] or i == 0):  # boundary views once
            N = len(G["xyz"])
            hits += np.bincount((cur["pkey"] % N).astype(np.int64), minlength=N).astype(np.int32)
        if prev is not None:
            r = transition(prev, cur); r["i"] = i - 1; r["j"] = i
            r["from"] = names[i - 1]; r["to"] = names[i]
            out.append(r)
        prev = cur
    return out, masks, (hits if G["score"] else None)


def run(N, cache, sel_path, workers, score):
    G.update(score=score)
    n = len(G["names"])
    step = (n + workers - 1) // workers
    chunks = [list(range(max(0, s - 1), min(n, s + step))) for s in range(0, n, step)]
    G.pop("lib", None)
    with get_context("fork").Pool(workers, initializer=init_worker,
                                  initargs=(cache, sel_path, max(1, 12 // workers))) as pool:
        res = pool.map(work, chunks)
    trans = sorted([t for r in res for t in r[0]], key=lambda t: t["i"])
    masks = {}
    for r in res:
        masks.update(r[1])
    hits = sum(r[2] for r in res) if score else None
    union = {}
    for K in (1, 2, 4, 8):
        rows = []
        for s in range(0, n - K + 1, K):
            ms = [np.unpackbits(masks[i], count=N).astype(bool) for i in range(s, s + K)]
            u = np.logical_or.reduce(ms)
            rows.append(dict(start=s, union=int(u.sum()), mean_single=float(np.mean([m.sum() for m in ms])),
                             max_single=int(max(m.sum() for m in ms))))
        if not rows:
            continue
        union[K] = dict(
            batches=len(rows),
            union_mean=float(np.mean([r["union"] for r in rows])),
            ratio_mean=float(np.mean([r["union"] / r["mean_single"] for r in rows])),
            ratio_p90=float(np.percentile([r["union"] / r["mean_single"] for r in rows], 90)),
            ratio_max=float(max(r["union"] / r["mean_single"] for r in rows)),
            union_over_N=float(np.mean([r["union"] for r in rows]) / N))
    return trans, union, hits


def summarize(trans, union, N):
    T = lambda k: np.array([t[k] for t in trans], np.float64)
    pairs, carried, new = T("pairs"), T("carried"), T("new")
    s = dict(
        N=N, transitions=len(trans),
        nvis_mean=float(T("nvis").mean()), pairs_mean=float(pairs.mean()),
        pairs_min=int(pairs.min()), pairs_max=int(pairs.max()),
        max_tile_pairs=int(T("max_tile_pairs").max()),
        carried_frac=float(carried.sum() / pairs.sum()),
        same_tile_frac=float(T("same_tile").sum() / pairs.sum()),
        carried_frac_p10=float(np.percentile(carried / pairs, 10)),
        new_frac=float(new.sum() / pairs.sum()),
        inversions_per_carried=float(T("inversions").sum() / carried.sum()),
        inversions_per_carried_p50=float(np.median(T("inversions") / carried)),
        inversions_per_carried_p90=float(np.percentile(T("inversions") / carried, 90)),
        runlen_mean=float(carried.sum() / T("runs").sum()),
        runs_per_tile=float(T("runs").sum() / T("tiles").sum()),
        moved_frac=float(T("moved").sum() / carried.sum()),
        moved_frac_p50=float(np.median(T("moved") / carried)),
        moved_frac_p90=float(np.percentile(T("moved") / carried, 90)),
        tiles_fully_sorted_frac=float(T("tiles_sorted").sum() / T("tiles").sum()),
        log2runs_pairwt=float((T("log2runs_pairwt") * carried).sum() / carried.sum()),
        log2n_pairwt=float((T("log2n_pairwt") * carried).sum() / carried.sum()),
        g_new_frac=float(T("g_new").sum() / T("nvis").sum()),
        g_inversions_per_both=float(T("g_inversions").sum() / T("g_both").sum()),
        g_runlen_mean=float(T("g_both").sum() / T("g_runs").sum()),
        g_moved_frac=float(T("g_moved").sum() / T("g_both").sum()),
        g_disp_p50=float(np.median(T("g_disp_p50"))), g_disp_p99=float(np.median(T("g_disp_p99"))),
        g_disp_max=float(np.median(T("g_disp_max"))),
        union={str(k): v for k, v in union.items()})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True); ap.add_argument("--cams", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--proxy", type=int, default=0)
    ap.add_argument("--workers", type=int, default=7); ap.add_argument("--cache", default="/tmp")
    ap.add_argument("--limit", type=int, default=0, help="first N views only (smoke test)")
    a = ap.parse_args()
    t0 = time.time()
    G["lib"] = build_lib(a.cache)
    G["verbose"] = bool(a.limit)
    cam = json.load(open(a.cams))["bicycle"]
    names = cam["order"][: a.limit or None]
    W, H = cam["image_size"]; I = cam["intrinsics"]
    G.update(names=names, views=cam["views"], W=W, H=H, floor=np.float32(cam["contrib_floor"]),
             K=tuple(np.float32(I[k]) for k in ("fx", "fy", "cx", "cy")),
             ntiles=((W + TILE - 1) // TILE) * ((H + TILE - 1) // TILE))
    cf = os.path.join(a.cache, "scene.npz")
    if os.path.exists(cf):
        z = np.load(cf); xyz, op, cov, nall = z["xyz"], z["op"], z["cov"], int(z["nall"])
    else:
        xyz, op, sc, q = load_ply(a.ply)
        print(f"[a1] ply read {time.time()-t0:.1f}s", flush=True)
        nall = len(xyz)
        keep = op >= np.float32(1.0 / 255.0)      # load-time opacity prune (render/opacity_prune.py)
        xyz, op, sc, q = xyz[keep], op[keep], sc[keep], q[keep]
        cov = cov3d(sc, q)
        np.savez(cf, xyz=xyz, op=op, cov=cov, nall=nall)
    print(f"[a1] loaded {len(xyz)} gaussians (of {nall}) in {time.time()-t0:.1f}s", flush=True)
    result = {}
    del cov
    trans, union, hits = run(len(xyz), a.cache, None, a.workers, score=bool(a.proxy))
    result["full"] = dict(summary=summarize(trans, union, len(xyz)), transitions=trans)
    print(f"[a1] full done {time.time()-t0:.1f}s", json.dumps(result["full"]["summary"])[:2000], flush=True)
    if a.proxy:
        # Proxy: keep the a.proxy gaussians with the largest opacity x (pairs summed over the
        # train views). A footprint score, not D1's alpha*T importance; a timing-shape proxy only.
        score = op * hits.astype(np.float32)
        sel = np.sort(np.argsort(-score, kind="stable")[: a.proxy])
        sel_path = os.path.join(a.cache, f"proxy_{a.proxy}.npy"); np.save(sel_path, sel)
        trans, union, _ = run(len(sel), a.cache, sel_path, a.workers, score=False)
        result["proxy"] = dict(summary=summarize(trans, union, len(sel)), transitions=trans,
                               rule="top-N by opacity x sum over train views of tile pairs")
        print(f"[a1] proxy done {time.time()-t0:.1f}s", json.dumps(result["proxy"]["summary"])[:2000], flush=True)
    result["meta"] = dict(ply=a.ply, cams=a.cams, views=len(names), W=W, H=H, tile=TILE,
                          seconds=round(time.time() - t0, 1))
    json.dump(result, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
