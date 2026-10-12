"""Task #534 (T-B): CPU model of B1 (coalesced bin emit) for the 30 bicycle views.

Builds each view's (gaussian, tile) pair list the way the device does today (PRECULL=2
pixel-centre rect, docs/precull-t157/analyze.py's "pc" arm; gaussian-major in ply order,
row-major tiles inside a rect), splits it over the cores and the two movers, and reports:
kept pairs per core, records per (core, tile), the L1 bytes to stage them as 32 B records
(whole runs, or in per-tile chunks of K records), the burst-length distribution (bursts are
also cut at the bucket's 2 KB / 64-record DRAM pages, from the device prefix's base), and
the write count. Host float64 model, not the device: estimates, not measurements.
Usage (repo root): python3 docs/coalesced-emit-tB/model.py [views|all|0:10] [ply] [part.pkl]
  then python3 docs/coalesced-emit-tB/model.py --report part1.pkl part2.pkl ...
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "precull-t140"))
sys.path.insert(0, str(HERE.parents[1] / "precull-t157"))
from estimate import CAM, COND, TILE, cells, project  # noqa: E402
from analyze import SL, t_dev  # noqa: E402

NCORES = 110          # bh-33 p100a run of t224 (#528); p150 eth-dispatch grid is 120
PAGE = 16             # pairs per 64 B pair-list page (sort_onelaunch_layout.h kElemsPerPage)
REC = 32              # bytes per blend record (PFWC_REC32)
DPAGE = 64            # records per 2 KB bucket DRAM page (task #365)
KS = (8, 16, 32, 64, 0)   # 0 = whole (core, tile) run staged contiguously
BINS = (1, 2, 4, 8, 16, 32, 64)
# docs/hw-ceilings.md (p100a yyzo-bh-07, measured): 1.35 GHz; one noc_async_write ~78 cycles
# on its mover (40.6 issue + 15 addr + ~20 barrier share); chip-wide scattered-write cap
# 1.78-2.27 G writes/s; DRAM write 259.7 GB/s on 110 cores, one NoC (p150 published peak 512).
F = 1.35e9
ISSUE = 78
CHIP = (1.78e9, 2.27e9)
BW = (259.7e9, 512e9)  # burst-length histogram upper edges (records)


def pairs(view, ply):
    (a, b, c, mx, my, op, rx, ry), W, H, floor = project(view, ply)
    txn, tyn = W // TILE, H // TILE
    det = a * c - b * b
    ok = (COND * det >= a * c) & (np.maximum(rx, ry) <= 4096)
    td = t_dev(op, floor)
    ex = np.maximum(np.sqrt(td * a) - 0.5 + SL, SL)
    ey = np.maximum(np.sqrt(td * c) - 0.5 + SL, SL)
    x0, x1 = cells(mx, np.where(ok, np.minimum(rx, ex), rx), txn)
    y0, y1 = cells(my, np.where(ok, np.minimum(ry, ey), ry), tyn)
    w = x1 - x0 + 1
    n = w * (y1 - y0 + 1)
    P = int(n.sum())
    g = np.repeat(np.arange(n.size), n)
    loc = np.arange(P) - np.repeat(np.cumsum(n) - n, n)
    tid = (y0[g] + loc // w[g]) * txn + x0[g] + loc % w[g]
    return tid.astype(np.int64), g, txn * tyn, n.size


def split_pairs(P):
    """Even page split (k2_range_speed with equal weights); mover 0 gets the low half."""
    pages = (P + PAGE - 1) // PAGE
    lo = np.array([pages * c // NCORES for c in range(NCORES + 1)])
    mid = (lo[:-1] + lo[1:]) // 2
    pg = np.arange(P) // PAGE
    core = np.searchsorted(lo, pg, side="right") - 1
    mover = (pg >= mid[core]).astype(np.int64)
    return core, mover


def split_gauss(g, M):
    """pfwc-style split: equal gaussian counts per core (pairs per core then vary)."""
    gl = np.array([M * c // NCORES for c in range(NCORES + 1)])
    core = np.searchsorted(gl, g, side="right") - 1
    return core


def pieces(start, L, K):
    """Bursts of a run [start, start + L): chunks of K records (K = 0: whole run), each
    also cut at 64-record DRAM page edges. Returns the burst lengths."""
    out = []
    keep = L > 0
    start, L = start[keep], L[keep]
    if K == 0:
        cs, cl = start, L
    else:
        nch = (L + K - 1) // K
        r = np.repeat(np.arange(L.size), nch)
        j = np.arange(nch.sum()) - np.repeat(np.cumsum(nch) - nch, nch)
        cs = start[r] + j * K
        cl = np.minimum(K, L[r] - j * K)
    # cut each chunk at page edges
    first = cs // DPAGE
    last = (cs + cl - 1) // DPAGE
    npc = last - first + 1
    r = np.repeat(np.arange(cs.size), npc)
    j = np.arange(npc.sum()) - np.repeat(np.cumsum(npc) - npc, npc)
    s = np.maximum(cs[r], (first[r] + j) * DPAGE)
    e = np.minimum(cs[r] + cl[r], (first[r] + j + 1) * DPAGE)
    out = e - s
    return out


def hist(lens):
    edges = (0,) + BINS + (1 << 30,)
    h = np.histogram(lens, bins=np.array(edges) + 0.5)[0]
    return h


def one(view, ply, acc):
    tid, g, T, M = pairs(view, ply)
    P = tid.size
    core, mover = split_pairs(P)
    key = core * T + tid
    Hc = np.bincount(key, minlength=NCORES * T).reshape(NCORES, T)
    H0 = np.bincount(key[mover == 0], minlength=NCORES * T).reshape(NCORES, T)
    base = np.cumsum(Hc, axis=0) - Hc  # device_prefix: exclusive over cores
    npc = Hc.sum(1)
    acc["P"].append(P)
    acc["M"].append(M)
    acc["core_pairs_max"].append(int(npc.max()))
    acc["core_tiles"].append((Hc > 0).sum(1))
    acc["tile_max"].append(int(Hc.sum(0).max()))
    run = Hc[Hc > 0]
    acc["run_len"].append(run)
    # pfwc-style (equal gaussians) split: pairs per core
    cg = split_gauss(g, M)
    Hg = np.bincount(cg * T + tid, minlength=NCORES * T).reshape(NCORES, T)
    acc["gcore_pairs_max"].append(int(Hg.sum(1).max()))
    for lab, HH in (("p", Hc), ("g", Hg)):
        for K in KS:
            if K == 0:
                l1 = HH.sum(1) * REC
            else:
                l1 = ((HH + K - 1) // K * K).sum(1) * REC + ((HH + K - 1) // K).sum(1) * 4
            acc[f"l1_{lab}_{K}"].append(l1)
    b = base.ravel(); h = Hc.ravel(); h0 = H0.ravel()
    for K in KS:
        if K == 0:  # whole core run, one mover writes it (or split in halves below)
            lens = pieces(b, h, 0)
        else:      # per-mover rings / chunks of K, as today's emit with OL_RING = K
            lens = np.concatenate([pieces(b, h0, K), pieces(b + h0, h - h0, K)])
        acc[f"w_{K}"].append(lens.size)
        acc[f"h_{K}"] += hist(lens)
    print(f"{view}: P {P} M {M} core max {npc.max()} gsplit max {Hg.sum(1).max()} "
          f"tiles/core {np.mean((Hc > 0).sum(1)):.0f} run mean {run.mean():.1f} "
          f"writes K8 {acc['w_8'][-1]} full {acc['w_0'][-1]}", flush=True)


def main():
    import pickle
    if sys.argv[1] == "--report":
        acc = None
        for f in sys.argv[2:]:
            a = pickle.load(open(f, "rb"))
            if acc is None:
                acc = a
                continue
            for k, v in a.items():
                acc[k] = acc[k] + v
        report(acc, len(acc["P"]))
        return
    views = sys.argv[1].split(",") if len(sys.argv) > 1 else ["hero"]
    ply = sys.argv[2] if len(sys.argv) > 2 else "scenes/bicycle.ply"
    order = json.load(open(CAM))["bicycle"]["order"]
    if views == ["all"]:
        views = order
    elif ":" in views[0]:
        i, j = map(int, views[0].split(":"))
        views = order[i:j]
    from collections import defaultdict
    acc = defaultdict(list)
    for K in KS:
        acc[f"h_{K}"] = np.zeros(len(BINS) + 1, np.int64)
    for v in views:
        one(v, ply, acc)
    if len(sys.argv) > 3:
        pickle.dump(dict(acc), open(sys.argv[3], "wb"))
    report(acc, len(views))


def report(acc, V):
    P = np.array(acc["P"])
    print(f"\nTOTAL {V} views, {NCORES} cores, even page split")
    print(f"pairs/view: median {np.median(P):.0f} mean {P.mean():.0f} max {P.max()}; "
          f"visible gaussians median {np.median(acc['M']):.0f}; max tile {max(acc['tile_max'])}")
    print(f"pairs/core (even split): mean {P.mean() / NCORES:.0f}, max {max(acc['core_pairs_max'])}; "
          f"equal-gaussian split max {max(acc['gcore_pairs_max'])}")
    ct = np.concatenate(acc["core_tiles"])
    rl = np.concatenate(acc["run_len"])
    print(f"tiles/core: mean {ct.mean():.0f} p95 {np.percentile(ct, 95):.0f} max {ct.max()}")
    print(f"records per (core, tile) run: mean {rl.mean():.1f} median {np.median(rl):.0f} "
          f"p95 {np.percentile(rl, 95):.0f} max {rl.max()}; record-weighted mean "
          f"{(rl * rl).sum() / rl.sum():.1f}")
    print("\nL1 staging bytes per core (KB): K = chunk records (0 = whole runs, contiguous)")
    for lab, nm in (("p", "even pairs"), ("g", "equal gaussians")):
        for K in KS:
            l1 = np.concatenate(acc[f"l1_{lab}_{K}"]) / 1024
            print(f"  {nm:15s} K={K:2d}: mean {l1.mean():7.1f} p95 {np.percentile(l1, 95):7.1f} "
                  f"max {l1.max():7.1f}  cores>920KB {(l1 > 920).sum()} of {l1.size}")
    print("\nbursts per view and length histogram (records; edges " + ",".join(map(str, BINS)) + ",>64)")
    for K in KS:
        w = np.array(acc[f"w_{K}"])
        h = acc[f"h_{K}"]
        print(f"  K={K:2d}: writes/view mean {w.mean():9.0f} max {w.max()}  mean burst "
              f"{P.sum() / w.sum():5.1f} rec ({P.sum() * REC / w.sum():6.0f} B)  hist {h.tolist()}")
    print("\npredicted write time per view, mean view (ms): mover issue = writes / (2 movers x "
          f"{NCORES}) x {ISSUE} cyc; chip = writes / {CHIP[0] / 1e9:.2f}-{CHIP[1] / 1e9:.2f} G/s; "
          f"bw = bytes / {BW[0] / 1e9:.0f}-{BW[1] / 1e9:.0f} GB/s")
    byt = P.mean() * REC
    bw = (byt / BW[0] * 1e3, byt / BW[1] * 1e3)
    rows = [("32 B scatter (no ring)", P.mean())] + [
        (f"K={K}" if K else "whole runs", np.mean(acc[f"w_{K}"])) for K in KS]
    for nm, w in rows:
        iss = w / (2 * NCORES) * ISSUE / F * 1e3
        chip = (w / CHIP[1] * 1e3, w / CHIP[0] * 1e3)
        lo = max(iss, chip[0], bw[1])
        hi = max(iss, chip[1], bw[0])
        print(f"  {nm:22s} writes {w:9.0f}: mover issue {iss:.3f}, chip {chip[0]:.3f}-{chip[1]:.3f}, "
              f"bw {bw[1]:.3f}-{bw[0]:.3f} -> write floor {lo:.3f}-{hi:.3f}")


if __name__ == "__main__":
    main()
