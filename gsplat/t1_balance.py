"""T1 blend backward load balance (#775, plan docs/training-plan/t773/PLAN.md L2).

The device bench (render/train/blend_bwd_bench.cpp) gives each core a contiguous block of tile
rows balanced by pair count. Core time is mostly bodies (one visited SFPU vector of one pair,
~200 cycles), and bodies per pair vary by tile, so the hero's busiest core runs 1.66x the mean
(#747 stats). This module models and builds two lossless fixes, host side only:

1. LPT: tile rows sorted by modelled cycles, each given to the least loaded core. The kernels
   take any contiguous run of tile rows (rows carry rec_start; planes are indexed by row), so
   the plan is a row permutation plus per-core row boundaries. Outputs do not change: each slot
   is written by one pair of one row, whatever the row order.
2. Microblock split of the heaviest tiles: a tile row becomes k part rows, part p visiting only
   the vectors in its microblock (a contiguous run of 32/k vectors: k=4 is one 16x16 face).
   Pixels are independent in the backward (T and BH are per pixel), so a part's per-pixel math
   is the unsplit math for its pixels. A pair is kept in every part its mask touches (pairs with
   an empty mask stay in part 0, so every slot is still written). The first part a pair appears
   in writes its real slot, later parts write partial slots after n_slots; T2 adds the partials
   in fixed part order (merge_partials). A pair seen by one part is bit-identical to the unsplit
   run; a pair seen by several parts is a fixed-order sum of per-part sums, within the T0 tolerance.

Cost model (cycles at 1.35 GHz per core): COST_TILE per row + COST_PAIR per pair + COST_BODY per
body, least squares on the 110 cores of #747's hero stats (max residual 0.9%).
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys

import numpy as np

COST_TILE, COST_PAIR, COST_BODY = 53994.3, 275.25, 200.32
CLOCK_HZ = 1.35e9
N_VEC = 32


def popcount32(x) -> np.ndarray:
    x = np.asarray(x, np.uint32).astype(np.uint64)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) & 0xFFFFFFFF) >> 24


def microblocks(k: int) -> list[int]:
    """Vector masks of the k microblocks of a tile (contiguous vector runs, k in 1, 2, 4, 8, 16, 32)."""
    assert k in (1, 2, 4, 8, 16, 32), k
    n = N_VEC // k
    return [((1 << n) - 1) << (n * i) for i in range(k)]


def cost(nt, npairs, nbodies):
    return COST_TILE * np.asarray(nt, np.float64) + COST_PAIR * np.asarray(npairs, np.float64) \
        + COST_BODY * np.asarray(nbodies, np.float64)


def tile_loads(tiles: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairs and bodies per tile row (tiles: rows of tile_id, rec_start, rec_count, ...)."""
    start, cnt = tiles[:, 1].astype(np.int64), tiles[:, 2].astype(np.int64)
    pc = np.concatenate([[0], np.cumsum(popcount32(masks), dtype=np.int64)])
    return cnt, pc[start + cnt] - pc[start]


def contiguous_split(pairs: np.ndarray, n_cores: int) -> np.ndarray:
    """Row boundaries (n_cores + 1) of blend_bwd_bench.cpp: contiguous blocks by pairs + 1."""
    n = len(pairs)
    beg = np.full(n_cores + 1, n, np.int64)
    w = np.asarray(pairs, np.int64) + 1
    total, acc, c = int(w.sum()), 0, 0
    beg[0] = 0
    for t in range(n):
        if c + 1 >= n_cores:
            break
        acc += int(w[t])
        if acc * n_cores >= total * (c + 1):
            c += 1
            beg[c] = t + 1
    return beg


def lpt(weights: np.ndarray, n_cores: int) -> np.ndarray:
    """Longest processing time first: core of each item. Deterministic (ties: lower item, then
    lower core)."""
    w = np.asarray(weights, np.float64)
    core = np.zeros(len(w), np.int64)
    heap = [(0.0, c) for c in range(n_cores)]
    for i in np.argsort(-w, kind="stable"):
        load, c = heapq.heappop(heap)
        core[i] = c
        heapq.heappush(heap, (load + w[i], c))
    return core


def split_rows(tiles: np.ndarray, masks: np.ndarray, ks: np.ndarray) -> dict:
    """Part rows for tile rows split into ks[row] microblocks (ks 1 = unsplit).
    Returns per part row: src (tile row), part, vmask, pairs (global rec indices, in order),
    their restricted masks and rank (0: this part writes the real slot; k: the pair's k-th
    later part, the k-th partial in the merge)."""
    src, part, vmask, pairs, pmasks, rank = [], [], [], [], [], []
    for r, (tid, start, cnt) in enumerate(tiles[:, :3].astype(np.int64)):
        idx = np.arange(start, start + cnt)
        m = masks[idx].astype(np.uint32)
        seen = np.zeros(cnt, np.int64)
        for p, vm in enumerate(microblocks(int(ks[r]))):
            mp = m & np.uint32(vm)
            keep = (mp != 0) | ((m == 0) & (p == 0))
            src.append(r)
            part.append(p)
            vmask.append(vm)
            pairs.append(idx[keep])
            pmasks.append(mp[keep])
            rank.append(seen[keep])
            seen += keep
    return {"src": np.asarray(src, np.int64), "part": np.asarray(part, np.int64),
            "vmask": np.asarray(vmask, np.uint64), "pairs": pairs, "masks": pmasks, "rank": rank}


def part_loads(sp: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([len(x) for x in sp["pairs"]], np.int64),
            np.array([int(popcount32(m).sum()) for m in sp["masks"]], np.int64))


def choose_splits(tiles: np.ndarray, masks: np.ndarray, n_cores: int, kmax: int = 8) -> np.ndarray:
    """Microblock count per tile row (1 = whole). Greedy: LPT the parts; while the heaviest part
    belongs to a row with k < kmax, double that row's k, and keep it only if the modelled
    busiest core drops (each part pays its own tile and pair overhead)."""
    ks = np.ones(len(tiles), np.int64)
    memo = {}

    def row_costs(r, k):
        if (r, k) not in memo:
            npr, nb = part_loads(split_rows(tiles[r:r + 1], masks, np.array([k])))
            memo[r, k] = cost(1, npr, nb)
        return memo[r, k]

    pairs, bodies = tile_loads(tiles, masks)
    whole = cost(1, pairs, bodies)

    def items(ks):
        c = [row_costs(r, k) if k > 1 else whole[r:r + 1] for r, k in enumerate(ks)]
        return np.concatenate(c), np.repeat(np.arange(len(ks)), [len(x) for x in c])

    c, row = items(ks)
    span = np.bincount(lpt(c, n_cores), c, n_cores).max()
    tried = np.zeros(len(tiles), bool)
    while True:
        cand = np.flatnonzero(~tried[row] & (ks[row] < kmax))
        if cand.size == 0:
            break
        r = row[cand[np.argmax(c[cand])]]
        ks2 = ks.copy()
        ks2[r] *= 2
        c2, row2 = items(ks2)
        span2 = np.bincount(lpt(c2, n_cores), c2, n_cores).max()
        if span2 < span:
            ks, c, row, span = ks2, c2, row2, span2
        else:
            tried[r] = True
    return ks


def plan(tiles: np.ndarray, masks: np.ndarray, n_cores: int, split: bool = False,
         kmax: int = 8) -> dict:
    """Split (choose_splits; split False = whole tiles) then LPT. Returns the part rows, the
    core per part row, the row permutation (part rows grouped by core, original order inside a
    core) and the per-core boundaries, pairs, bodies and modelled cycles."""
    ks = choose_splits(tiles, masks, n_cores, kmax) if split else np.ones(len(tiles), np.int64)
    sp = split_rows(tiles, masks, ks)
    npr, nb = part_loads(sp)
    core = lpt(cost(1, npr, nb), n_cores)
    perm = np.lexsort((np.arange(len(core)), core))
    beg = np.searchsorted(core[perm], np.arange(n_cores + 1))
    per = lambda v: np.bincount(core, v, n_cores)  # noqa: E731
    return {"ks": ks, "split": sp, "core": core, "perm": perm, "beg": beg,
            "core_tiles": np.bincount(core, minlength=n_cores), "core_pairs": per(npr),
            "core_bodies": per(nb), "core_cycles": per(cost(1, npr, nb)),
            "n_extra": int(sum((rk > 0).sum() for rk in sp["rank"]))}


def contiguous_loads(tiles: np.ndarray, masks: np.ndarray, n_cores: int) -> dict:
    pairs, bodies = tile_loads(tiles, masks)
    beg = contiguous_split(pairs, n_cores)
    nt = np.diff(beg)
    cp = np.add.reduceat(np.append(pairs, 0), beg[:-1])[:n_cores] * (nt > 0)
    cb = np.add.reduceat(np.append(bodies, 0), beg[:-1])[:n_cores] * (nt > 0)
    return {"beg": beg, "core_tiles": nt, "core_pairs": cp, "core_bodies": cb,
            "core_cycles": cost(nt, cp, cb)}


def summary(loads: dict) -> dict:
    cyc, b = loads["core_cycles"], loads["core_bodies"]
    return {"busiest_bodies": int(b.max()), "mean_bodies": round(float(b.mean()), 1),
            "bodies_busiest_over_mean": round(float(b.max() / b.mean()), 3),
            "busiest_pairs": int(loads["core_pairs"].max()),
            "busiest_ms": round(float(cyc.max()) / CLOCK_HZ * 1e3, 2),
            "mean_ms": round(float(cyc.mean()) / CLOCK_HZ * 1e3, 2),
            "mean_over_busiest": round(float(cyc.mean() / cyc.max()), 3)}


def apply_plan(p: dict, pl: dict) -> dict:
    """Device prep (blend_backward_dev.prep() layout) of the plan: part rows in core order, recs
    rebuilt per part row (restricted masks; slot = real slot or n_slots + extra index), planes
    copied from the source tile. p['n_slots'] grows by the partial slots; merge_dst maps each
    partial slot to its real slot and merge_rank gives its place in the fixed merge order."""
    sp, n0 = pl["split"], p["n_slots"]
    tiles, recs, planes, dst, rank = [], [], [], [], []
    extra, nrec = 0, 0
    for i in pl["perm"]:
        r, idx = int(sp["src"][i]), sp["pairs"][i]
        rr = p["recs"][idx].copy()
        rr[:, 12] = sp["masks"][i]
        rk = sp["rank"][i]
        j = np.flatnonzero(rk > 0)
        dst.append(rr[j, 13].astype(np.int64))
        rank.append(rk[j])
        rr[j, 13] = n0 + extra + np.arange(j.size)
        extra += j.size
        tiles.append([p["tiles"][r, 0], nrec, len(idx), -(-len(idx) // 16)])
        recs.append(rr)
        planes.append(p["planes"][r])
        nrec += len(idx)
    q = dict(p)
    q.update(tiles=np.asarray(tiles, np.uint32).reshape(-1, 4),
             recs=np.concatenate(recs) if recs else p["recs"][:0],
             planes=np.asarray(planes, np.float32).reshape(-1, *p["planes"].shape[1:]),
             n_slots=n0 + extra, n_slots_base=n0, merge_dst=np.concatenate(dst + [np.zeros(0, np.int64)]),
             merge_rank=np.concatenate(rank + [np.zeros(0, np.int64)]), core_beg=pl["beg"].astype(np.int64))
    return q


def merge_partials(slots: np.ndarray, n_base: int, dst: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """T2 merge: real slot s += its partials in fixed order (rank 1, 2, ...), fp32."""
    out = np.array(slots[:n_base], np.float32, copy=True)
    ext = np.asarray(slots[n_base:], np.float32)
    for k in range(1, int(rank.max()) + 1 if rank.size else 1):
        sel = np.flatnonzero(rank == k)
        out[dst[sel]] = out[dst[sel]] + ext[sel]       # dst unique within one rank
    return out


def write_plan(q: dict, out: str) -> None:
    """blend_backward_dev.write() plus cores.u32 (row boundaries per core) and merge.i64
    (partial slot -> real slot, rank)."""
    from gsplat.blend_backward_dev import write
    write(q, out)
    q["core_beg"].astype(np.uint32).tofile(os.path.join(out, "cores.u32"))
    np.stack([q["merge_dst"], q["merge_rank"]], 1).astype(np.int64).tofile(os.path.join(out, "merge.i64"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bodies_npz", help="docs/t775/hero_bodies.py output (tiles, masks)")
    ap.add_argument("--cores", type=int, default=110)
    ap.add_argument("--kmax", type=int, nargs="*", default=[2, 4, 8])
    ap.add_argument("--stats", help="device stats.json of the contiguous run, checked against the model")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    z = np.load(a.bodies_npz)
    tiles, masks = z["tiles"], z["masks"]
    pairs, bodies = tile_loads(tiles, masks)
    o = np.argsort(-bodies, kind="stable")
    rep = {"n_tiles": int(len(tiles)), "n_pairs": int(pairs.sum()), "n_bodies": int(bodies.sum()),
           "cores": a.cores, "model": {"tile": COST_TILE, "pair": COST_PAIR, "body": COST_BODY},
           "heaviest_tiles": [{"tile_id": int(tiles[i, 0]), "pairs": int(pairs[i]), "bodies": int(bodies[i])}
                              for i in o[:10]],
           "ideal_ms": round(float(cost(len(tiles), pairs.sum(), bodies.sum())) / a.cores / CLOCK_HZ * 1e3, 2),
           "contiguous_by_pairs": summary(contiguous_loads(tiles, masks, a.cores)),
           "lpt": summary(plan(tiles, masks, a.cores))}
    if a.stats:
        st = json.load(open(a.stats))
        ix = {k: i for i, k in enumerate(st["cols"])}
        dv = np.array(st["cores"], np.float64)[:, [ix["nt"], ix["np"], ix["nv"]]]
        cl = contiguous_loads(tiles, masks, a.cores)
        md = np.stack([cl["core_tiles"], cl["core_pairs"], cl["core_bodies"]], 1)
        tot = np.array(st["cores"], np.float64)[:, ix["tot"]]
        rep["device_check"] = {"stats": a.stats, "nt_np_nv_equal": bool(np.array_equal(dv, md)),
                               "model_vs_device_cycles_max_rel": round(float(np.abs(cl["core_cycles"] / tot - 1).max()), 4),
                               "device_busiest_ms": round(float(tot.max()) / CLOCK_HZ * 1e3, 2)}
    for km in a.kmax:
        pl = plan(tiles, masks, a.cores, True, km)
        rep[f"lpt_split_kmax{km}"] = {**summary(pl), "split_tiles": int((pl["ks"] > 1).sum()),
                                     "ks": {str(k): int((pl["ks"] == k).sum()) for k in (2, 4, 8, 16) if (pl["ks"] == k).any()},
                                     "part_rows": int(len(pl["core"])), "partial_slots": pl["n_extra"],
                                     "pairs_total": int(pl["core_pairs"].sum())}
    print(json.dumps(rep, indent=1))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
