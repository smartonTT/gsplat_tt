"""Task #637: CPU model of the one-launch sort emit stream for spatially ordered gids (U3 precondition).

Scene: bicycle.ply after the #536 opacity prune (N = 5,559,909), 1024x1024, benchmarks/cameras_v2.json.
Per view and gid order ("option") it builds the pair stream the onelaunch emit sees and runs the
sequential ring model in emit_sim.c on every mover stream:

  gid order  pfwc deals 1024-slot tiles strided over C cores (tile j -> core j % C); a core's visible
             gaussians, in tile then lane order, get consecutive gids, core segments concatenated.
  pairs      per visible gid its 32 px screen tiles (3-sigma rect, row-major), in gid order.
  emit       sort core c takes pair pages [c P / C, (c+1) P / C) (16 pairs per page), mover 0 the
             first half, mover 1 the second (GSPLAT_TT_SORT_EMIT_SPLIT 500). One stream per mover.
  town       sim3: TRISC t % 3 owns tile t (task #202), 3 units per record, batches of 128
             records (OL_PB 8 x 16), 4 batch slots; sim: one packer, 1 unit per record.
  ring       per tile R records per run, D records of ring (OL_RING / OL_RING_DEPTH); a run start
             waits until the run D records back left L1, L record-times after its last record.

Options:
  today     pruned PLY order, all chunks
  morton    Morton order (render/run.py morton_order), chunk 1024, no skip (t479 arm "ro")
  mcull     Morton, chunk 1024, sphere cull (t631), survivors dealt strided
  mcull_s8 / mcull_s64   mcull, each mover reads its pages k-way interleaved (page j of the
            mover's range -> part j % k, page j // k of that part)
  u64       Morton, 64-gaussian cull units (sphere test per unit), each pfwc tile gathers 16
            surviving units T apart (T = ceil(M / 16)); lanes unit-contiguous
  <opt>_s<k>  any order above with the k-way page interleave of each mover range (e.g. u64g1_s8)
  u64g<g>   as u64, lanes in groups of g interleaved over the 16 units (g 32 = one pfwc
            visibility mask word; g 1 = slot s -> unit s % 16, offset s // 16)

Usage (repo root): python3 docs/emit-burst-model/model.py <bicycle.ply> [--views a:b] [--cores C]
  [--opts today,morton,...] [--jobs J] [--tag T]   (needs emit_sim.so: see README)
"""
import ctypes
import json
import math
import os
import sys
import time
import multiprocessing as mp

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "morton-cull-t631"))
sys.path.insert(0, os.path.join(HERE, "..", "pfwc-chunk-cull-t466"))
from cullshare import CAM, morton_order, quat_to_rot  # noqa: E402
from size631 import MIN_OPACITY, load_ply, sphere_culled, sphere_tables  # noqa: E402

TILE = 1024
PAGE = 16          # pairs per 64 B pair page
RD = [(8, 8), (4, 8), (8, 16), (4, 16)]   # (R, D): 8/8 today, 4/8 fits L1, D 16 does not (t493)
LS = [int(x) for x in os.environ.get("EMIT_LS", "4,6,8,12,16,32").split(",")]  # flush latency, record-times
SO = os.environ.get("EMIT_SIM_SO", os.path.join(HERE, "emit_sim.so"))

G = {}


def rects(means, scales, q, op, w2c, W, H, f, floor):
    """imbalance.py rects (t479): visible flag and the 32 px tile rect per gaussian."""
    pc = means @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    n = len(means)
    x0 = np.zeros(n, np.int32); x1 = x0.copy(); y0 = x0.copy(); y1 = x0.copy()
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


def deal(tiles_slots, C):
    """tiles_slots: (nt, 1024) gaussian index per slot (-1 = empty). Core-major gid order."""
    nt = len(tiles_slots)
    j = np.arange(nt)
    o = np.lexsort((j, j % C))
    return tiles_slots[o].ravel()


def pad_tiles(seq, unit=TILE):
    pad = (-len(seq)) % unit
    return np.concatenate([seq, np.full(pad, -1, seq.dtype)]).reshape(-1, unit)


def unit_tiles(order, survive, grp):
    """64-gaussian units of `order`; surviving units gathered 16 per pfwc tile, T apart. Lanes in
    groups of `grp` (64: unit-contiguous; 32: one visibility mask word; 1: slot s -> unit s % 16)
    interleaved over the tile's 16 units."""
    units = pad_tiles(order, 64)[survive]          # (M, 64)
    M = len(units)
    T = (M + 15) // 16
    units = np.concatenate([units, np.full((T * 16 - M, 64), -1, units.dtype)])
    tiles = units.reshape(16, T, 64 // grp, grp).transpose(1, 2, 0, 3)   # tile j: units j + i T
    return tiles.reshape(T, TILE)


def gid_seq(opt, w2c, C):
    g = G
    if opt == "today":
        s = deal(pad_tiles(np.arange(g["N"])), C)
    elif opt == "morton":
        s = deal(pad_tiles(g["mo"]), C)
    elif opt.startswith("mcull"):
        cul = sphere_culled(g["c1"], g["R1"], g["rm1"], w2c, g["f"], g["W"], g["H"])
        s = deal(pad_tiles(g["mo"])[~cul], C)
    elif opt.startswith("u64"):
        cul = sphere_culled(g["c64"], g["R64"], g["rm64"], w2c, g["f"], g["W"], g["H"])
        s = deal(unit_tiles(g["mo"], ~cul, int(opt[4:].split("_")[0]) if opt.startswith("u64g") else 64), C)
    else:
        raise ValueError(opt)
    return s[s >= 0]


def streams(seq, r, C, W, interleave=0):
    vis, x0, x1, y0, y1 = r
    s = seq[vis[seq]]
    w = (x1 - x0)[s].astype(np.int64); h = (y1 - y0)[s].astype(np.int64)
    n = w * h
    keep = n > 0
    s, w, n = s[keep], w[keep], n[keep]
    st = np.cumsum(n) - n
    P = int(n.sum())
    rep = np.repeat(np.arange(len(s)), n)
    j = np.arange(P) - st[rep]
    tx = W // 32
    tiles = ((y0[s][rep] + j // w[rep]) * tx + x0[s][rep] + j % w[rep]).astype(np.int32)
    npg = (P + PAGE - 1) // PAGE
    cut = [(c * npg) // C for c in range(C + 1)]
    bounds = []
    for c in range(C):
        lo, hi = cut[c], cut[c + 1]
        mid = lo + (hi - lo) // 2
        bounds += [(lo, mid), (mid, hi)]
    parts, off = [], [0]
    for lo, hi in bounds:
        a = tiles[lo * PAGE:min(hi * PAGE, P)]
        if interleave and hi > lo:
            npp = hi - lo
            pg = np.arange(npp)
            k = interleave
            per = (npp + k - 1) // k
            src = (pg % k) * per + pg // k
            src = src[src < npp]
            pa = np.concatenate([a, np.full(npp * PAGE - len(a), -1, np.int32)]).reshape(npp, PAGE)
            a = pa[src].ravel()
            a = a[a >= 0]
        parts.append(a)
        off.append(off[-1] + len(a))
    return np.concatenate(parts), np.array(off, np.int64), P, len(s)


def run_sim(tiles, off, R, D, L, ntiles):
    lib = ctypes.CDLL(SO)
    ns = len(off) - 1
    end = np.zeros(ns, np.int64); ev = np.zeros(ns, np.int64); hist = np.zeros(1024, np.int64)
    p = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))  # noqa: E731
    lib.sim(p(tiles, ctypes.c_int32), p(off, ctypes.c_int64), ctypes.c_int32(ns), ctypes.c_int32(ntiles),
            ctypes.c_int32(R), ctypes.c_int32(D), ctypes.c_int64(L), p(end, ctypes.c_int64),
            p(ev, ctypes.c_int64), p(hist, ctypes.c_int64))
    return end, ev, hist


def run_sim3(tiles, off, R, D, L, ntiles, B=128, slots=4):
    lib = ctypes.CDLL(SO)
    ns = len(off) - 1
    end = np.zeros(ns, np.int64)
    p = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))  # noqa: E731
    lib.sim3(p(tiles, ctypes.c_int32), p(off, ctypes.c_int64), ctypes.c_int32(ns), ctypes.c_int32(ntiles),
             ctypes.c_int32(R), ctypes.c_int32(D), ctypes.c_int64(L), ctypes.c_int32(B), ctypes.c_int32(slots),
             p(end, ctypes.c_int64))
    return end


def one_view(args):
    v, opts, C = args
    g = G
    w2c = np.linalg.inv(np.array(g["cam"]["views"][v]["c2w"], np.float64))
    r = rects(g["means"], g["scales"], g["q"], g["op"], w2c, g["W"], g["H"], g["f"], g["floor"])
    ntiles = (g["W"] // 32) * (g["H"] // 32)
    out = {}
    for opt in opts:
        k = int(opt.split("_s")[1]) if "_s" in opt else 0
        seq = gid_seq(opt, w2c, C)
        tiles, off, P, nv = streams(seq, r, C, g["W"], k)
        recs = np.diff(off)
        res = {"pairs": P, "vis": nv, "slots": int(len(seq)), "rec_max": int(recs.max()),
               "rec_mean": float(recs.mean()), "sim": {}}
        for R, D in RD:
            for L in LS:
                end, ev, hist = run_sim(tiles, off, R, D, L, ntiles)
                res["sim"][f"{R}/{D}/{L}"] = {"end_max": int(end.max()), "end_mean": float(end.mean()),
                                              "stall_mean": float((end - recs).mean()),
                                              "ev": int(ev.sum()), "hist": hist.tolist()}
                e3 = run_sim3(tiles, off, R, D, L, ntiles)
                res["sim"][f"{R}/{D}/{L}"].update(e3_max=int(e3.max()), e3_mean=float(e3.mean()))
        out[opt] = res
    return v, out


def main():
    args = sys.argv[1:]
    views, C, jobs, tag = "0:30", 120, 6, ""
    opts = "today,morton,mcull,mcull_s8,mcull_s64,u64,u64g32,u64g16,u64g1"
    for flag in ("--views", "--cores", "--opts", "--jobs", "--tag"):
        if flag in args:
            i = args.index(flag); val = args[i + 1]; del args[i:i + 2]
            if flag == "--views": views = val
            elif flag == "--cores": C = int(val)
            elif flag == "--opts": opts = val
            elif flag == "--tag": tag = "-" + val
            else: jobs = int(val)
    cam = json.load(open(CAM))["bicycle"]
    W, H = cam["image_size"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    t0 = time.perf_counter()
    means, scales, q, op = load_ply(args[0])
    keep = ~(op.astype(np.float32) < np.float32(MIN_OPACITY))
    means, scales, q, op = means[keep], scales[keep], q[keep], op[keep]
    rho = 3.0 * np.sqrt((scales ** 2).sum(1))
    mo = morton_order(means)
    _, _, rm1, c1, R1 = sphere_tables(means, rho, mo, C=TILE)
    _, _, rm64, c64, R64 = sphere_tables(means, rho, mo, C=64)
    G.update(means=means, scales=scales, q=q, op=op, N=len(means), mo=mo, c1=c1, R1=R1, rm1=rm1,
             c64=c64, R64=R64, rm64=rm64, cam=cam, W=W, H=H, f=f, floor=cam["contrib_floor"])
    a, b = (int(x) for x in views.split(":"))
    vl = cam["order"][a:b]
    ol = opts.split(",")
    print(f"# N {len(means)} C {C} views {a}:{b} opts {ol} load {time.perf_counter() - t0:.1f}s", flush=True)
    res = {}
    with mp.get_context("fork").Pool(jobs) as pool:
        for v, out in pool.imap(one_view, [(v, ol, C) for v in vl]):
            res[v] = out
            print(v, " ".join(f"{o}:{out[o]['pairs']}/{out[o]['rec_max']}" for o in ol),
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    path = os.path.join(HERE, f"out-C{C}-v{a}-{b}{tag}.json")
    json.dump({"C": C, "views": vl, "res": res}, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    main()
