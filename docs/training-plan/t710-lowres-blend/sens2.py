#!/usr/bin/env python3
"""Task #710 add-on: B1 sensitivity to the skip cost (absolute ns/record), and B1 with
per-sub-tile lists (no skip, more sort lists): max list length vs the 65,472-record bucket.
Usage: sens2.py --tiles out/tiles.pkl --model out/model.json --out out/sens2.json"""
import argparse, json, pickle
import numpy as np
from model import CAP, makespan, tile_cost, subgrid, q_to8, sort_ms


def pieces(p, o, sg, r):
    """Like model.hybrid but also returns each piece's list length (own sub-tile list)."""
    n32, mb32 = o["n32"], o["mb32"]
    base = tile_cost(p, n32, mb32)
    total = base.sum(); costs = list(base); lens = list(n32)
    for t in np.argsort(-base):
        if base[t] <= total / 120:
            break
        N16, M16 = sg[16][0][t], sg[16][1][t]
        c16 = tile_cost(p, N16, M16) + np.where(N16 > 0, r * (n32[t] - N16), 0)
        costs[t] = 0.0; lens[t] = 0
        share = (total + c16.sum() - base[t]) / 120
        for q in range(4):
            if c16[q] > share:
                i8 = q_to8(q)
                N8, M8 = sg[8][0][t, i8], sg[8][1][t, i8]
                c8 = tile_cost(p, N8, M8) + np.where(N8 > 0, r * (n32[t] - N8), 0)
                costs.extend(c8.tolist()); lens.extend(N8.tolist())
            else:
                costs.append(c16[q]); lens.append(N16[q])
        total = sum(costs)
    return np.array(costs), np.array(lens)


ap = argparse.ArgumentParser()
ap.add_argument("--tiles"); ap.add_argument("--model"); ap.add_argument("--out")
a = ap.parse_args()
T = pickle.load(open(a.tiles, "rb")); M = json.load(open(a.model))
p = np.array(M["params"]); add = M["other_ms"] + M["b2b_gap_ms"]
W0, H0 = T["W"], T["H"]
res = {}
for s in (2, 4):
    for N in T["Ns"]:
        vs = [v[s][N] for v in T["data"]]
        W, H = W0 // s, H0 // s
        t32 = len(vs[0]["n32"])
        row = {"max_list_32": int(max(o["n32"].max() for o in vs)),
               "max_list_16": int(max(o["n16"].max() for o in vs)),
               "max_list_8": int(max(o["n8"].max() for o in vs))}
        for r in (0.0, 1.0, 2.0, 4.0):
            ms, ml, nl = [], [], []
            for o in vs:
                c, l = pieces(p, o, subgrid(o, W, H), r)
                ms.append(p[0] + makespan(c) * 1e-6)
                ml.append(int(l.max())); nl.append(int((l > 0).sum()))
            # r>0: walk the parent's list (one sort list per 32x32 tile, parent list must fit the bucket)
            # r=0 own lists: each piece sorted separately (more lists, smaller max list)
            row[f"B1 walk-parent r={r:g}ns"] = round(float(np.mean(ms) + sort_ms(t32) + add), 3)
            if r == 0.0:
                row["B1 own sub-tile lists"] = round(float(np.mean(ms) + sort_ms(np.mean(nl)) + add), 3)
                row["B1 own lists max_list"] = int(max(ml))
                row["B1 lists"] = float(np.mean(nl))
        res[f"{N}_s{s}"] = row
        print(N, s, row, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
