#!/usr/bin/env python3
"""Task #710: fit a per-tile blend cost model to D1's measured blend times, then size the
low-res blend options. CPU only; inputs are tilecounts.py's tiles.pkl and D1's summary.json.

Blend model (one view): blend_ms = b0 + makespan over C cores of tile costs, list-scheduled in
descending-cost order (what blend_claim_order.h + late claim do), with
    tile cost = f + s*n + s_ov*max(n - 8192, 0) + m*mb
n = pairs of the tile, mb = 4x8 microblocks those pairs touch, 8192 = kBucketFit (tiles above it
leave the L1 bucket path). The view's forward = sort(tiles) + blend + other.
Usage: model.py --tiles out/tiles.pkl --d1 ../d1-scaling/out/summary.json --out out/model.json
"""
import argparse, heapq, json, pickle
import numpy as np
from scipy.optimize import least_squares

CORES = 120
FIT = 8192
CAP = 65472


def makespan(costs, cores=CORES):
    c = np.sort(np.asarray(costs, np.float64))[::-1]
    c = c[c > 0]
    if len(c) <= cores:
        return c[0] if len(c) else 0.0
    h = [0.0] * cores
    for x in c:
        heapq.heapreplace(h, h[0] + x)
    return max(h)


def tile_cost(p, n, mb):
    b0, f, s, sov, m = p
    n = np.asarray(n, np.float64)
    return np.where(n > 0, f + s * n + sov * np.maximum(n - FIT, 0) + m * np.asarray(mb, np.float64), 0.0)


def sort_ms(lists):
    # D1 (M): 1014 tiles 0.378-0.393 ms, 260 tiles 0.118-0.129, 70 tiles 0.063-0.065
    return 0.040 + 0.00034 * lists


def blend_view(p, n, mb):
    return p[0] + makespan(tile_cost(p, n, mb)) * 1e-6  # costs in ns -> ms


def subgrid(o, W, H):
    """Per 32x32 tile: its 16x16 and 8x8 sub-tile (n, mb) arrays, shape (T32, 4) and (T32, 16)."""
    tx, ty = (W + 31) // 32, (H + 31) // 32
    out = {}
    for ts in (16, 8):
        k = 32 // ts
        txs = (W + ts - 1) // ts
        n = o[f"n{ts}"]; mb = o[f"mb{ts}"]
        tys = len(n) // txs
        N = np.zeros((tx * ty, k * k), np.int64); M = np.zeros((tx * ty, k * k), np.int64)
        for t in range(tx * ty):
            x, y = t % tx, t // tx
            j = 0
            for sy in range(k):
                for sx in range(k):
                    X, Y = x * k + sx, y * k + sy
                    if X < txs and Y < tys:
                        N[t, j] = n[Y * txs + X]; M[t, j] = mb[Y * txs + X]
                    j += 1
        out[ts] = (N, M)
    return out


def hybrid(p, o, sg, r_skip, max_depth=8):
    """B1: split the heaviest 32x32 tiles spatially (16x16, then 8x8 quarters) until no tile costs
    more than the balanced share. A sub-tile core walks its parent's sorted list and skips the
    pairs that miss it (r_skip ns per skipped record). Returns (costs, lists, max list)."""
    n32, mb32 = o["n32"], o["mb32"]
    base = tile_cost(p, n32, mb32)
    total = base.sum()
    order = np.argsort(-base)
    costs = list(base)
    nsplit = 0
    for t in order:
        if base[t] <= total / CORES:
            break
        N16, M16 = sg[16][0][t], sg[16][1][t]
        c16 = tile_cost(p, N16, M16) + np.where(N16 > 0, r_skip * (n32[t] - N16), 0)
        costs[t] = 0.0
        share = (total + c16.sum() - base[t]) / CORES
        for q in range(4):
            if max_depth >= 8 and c16[q] > share:
                N8, M8 = sg[8][0][t, q_to8(q)], sg[8][1][t, q_to8(q)]
                c8 = tile_cost(p, N8, M8) + np.where(N8 > 0, r_skip * (n32[t] - N8), 0)
                costs.extend(c8.tolist()); nsplit += 4
            else:
                costs.append(c16[q]); nsplit += 1
        total = sum(costs)
    return np.array(costs), nsplit


def q_to8(q):
    # 16x16 quadrant q (row-major 2x2) -> its four 8x8 sub-tiles in the 4x4 row-major grid
    qy, qx = divmod(q, 2)
    return [(qy * 2 + a) * 4 + (qx * 2 + b) for a in range(2) for b in range(2)]


def depth_split(p, o, merge_ns):
    """B2: cut a heavy tile's depth-sorted list into k chunks on k cores, merge (C, T) after."""
    n32, mb32 = o["n32"], o["mb32"]
    base = tile_cost(p, n32, mb32)
    share = base.sum() / CORES
    costs = []
    for t in range(len(n32)):
        if n32[t] == 0:
            continue
        k = int(np.ceil(base[t] / share)) if base[t] > share else 1
        if k == 1:
            costs.append(base[t]); continue
        cn = n32[t] / k; cm = mb32[t] / k
        c = tile_cost(p, [cn], [cm])[0] + merge_ns
        costs.extend([c] * k)
    return np.array(costs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True); ap.add_argument("--d1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = pickle.load(open(a.tiles, "rb"))
    W0, H0 = T["W"], T["H"]
    rows = [r for r in json.load(open(a.d1))["rows"] if r.get("b2b")]
    data = T["data"]

    def pred_blend(p, s, N):
        return np.mean([blend_view(p, v[s][N]["n32"], v[s][N]["mb32"]) for v in data])

    def resid(p, which):
        return [pred_blend(p, r["s"], r["N"]) - r["stages"]["blend"] for r in rows if which(r)]

    fits = {}
    for name, free in (("pairs only (f, s)", [1, 1, 1, 0, 0]),
                       ("pairs + microblocks (f, s, m)", [1, 1, 1, 0, 1]),
                       ("pairs + microblocks + over-FIT (f, s, s_ov, m)", [1, 1, 1, 1, 1])):
        x0 = np.array([0.1, 1000.0, 300.0, 100.0, 50.0])
        idx = [i for i in range(5) if free[i]]

        def fun(q):
            p = np.zeros(5); p[idx] = q
            return resid(p, lambda r: True)
        sol = least_squares(fun, x0[idx], bounds=(0, np.inf))
        p = np.zeros(5); p[idx] = sol.x
        res = np.array(fun(sol.x))
        # leave-resolution-out check: fit on s=1,2, predict s=4
        def fun12(q):
            pp = np.zeros(5); pp[idx] = q
            return resid(pp, lambda r: r["s"] != 4)
        s12 = least_squares(fun12, x0[idx], bounds=(0, np.inf))
        p12 = np.zeros(5); p12[idx] = s12.x
        lro = resid(p12, lambda r: r["s"] == 4)
        fits[name] = dict(p=p.tolist(), rms=float(np.sqrt(np.mean(res ** 2))), max=float(np.abs(res).max()),
                          resid={f'{r["cfg"]}': round(float(x), 3) for r, x in zip(rows, res)},
                          s4_from_s12=[round(float(x), 3) for x in lro])
        print(name, "p(b0 ms, f ns, s ns, s_ov ns, m ns) =", np.round(p, 3).tolist(),
              "rms %.3f max %.3f" % (fits[name]["rms"], fits[name]["max"]), "s=4 out-of-sample resid", fits[name]["s4_from_s12"])
    best = min(fits, key=lambda k: fits[k]["rms"])
    p = np.array(fits[best]["p"])
    print("using", best)

    # other (project, tile_assign, xview, ...) and b2b - latency, from D1
    other = np.mean([r["stages"]["view_total"] - r["stages"]["sort"] - r["stages"]["blend"] for r in rows])
    b2b_gap = np.mean([r["b2b"] - r["lat"] for r in rows])
    full = {r["N"]: r["b2b"] for r in rows if r["s"] == 1}

    opts = {}
    Ns = T["Ns"]
    for s in (1, 2, 4):
        for N in Ns:
            vs = [v[s][N] for v in data]
            W, H = W0 // s, H0 // s
            rec = {}
            P = np.mean([o["n32"].sum() for o in vs])
            mx = max(int(o["n32"].max()) for o in vs)
            mx16 = max(int(o["n16"].max()) for o in vs)
            mx8 = max(int(o["n8"].max()) for o in vs)
            t32 = len(vs[0]["n32"]); t16 = len(vs[0]["n16"]); t8 = len(vs[0]["n8"])

            def fwd(blend, lists):
                return blend + sort_ms(lists) + other + b2b_gap
            # O0: today's 32x32 split (fails if a list > CAP)
            b = np.mean([blend_view(p, o["n32"], o["mb32"]) for o in vs])
            rec["O0 today 32x32"] = dict(ms=fwd(b, t32), ok=mx <= CAP, max_list=mx)
            # ideal: same work, perfect balance, no per-tile cost growth
            work = np.mean([tile_cost(p, o["n32"], o["mb32"]).sum() for o in vs]) * 1e-6 / CORES
            rec["ideal 32x32 (perfect balance)"] = dict(ms=fwd(p[0] + work, t32), ok=True, max_list=0)
            # A: 16x16 tiles everywhere
            b = np.mean([blend_view(p, o["n16"], o["mb16"]) for o in vs])
            rec["A 16x16 tiles"] = dict(ms=fwd(b, t16), ok=mx16 <= CAP, max_list=mx16)
            b = np.mean([blend_view(p, o["n8"], o["mb8"]) for o in vs])
            rec["A8 8x8 tiles"] = dict(ms=fwd(b, t8), ok=mx8 <= CAP, max_list=mx8)
            # B1: hybrid spatial split of heavy tiles (parent list walked, misses skipped)
            if s > 1 or N == Ns[0]:
                for rs, lab in ((0.0, "r=0"), (0.25 * p[2], "r=s/4"), (0.5 * p[2], "r=s/2")):
                    bb, ns = [], []
                    for o in vs:
                        sg = subgrid(o, W, H)
                        c, k = hybrid(p, o, sg, rs)
                        bb.append(p[0] + makespan(c) * 1e-6); ns.append(k)
                    rec[f"B1 heavy-tile spatial split ({lab})"] = dict(
                        ms=fwd(np.mean(bb), t32), ok=True, max_list=mx, subtiles=float(np.mean(ns)))
                # B2: depth split with a (C,T) merge (16 KB fp32 RGBT per extra chunk, ~2 us)
                b = np.mean([p[0] + makespan(depth_split(p, o, 2000.0)) * 1e-6 for o in vs])
                rec["B2 heavy-tile depth split + merge"] = dict(ms=fwd(b, t32), ok=True, max_list=0)
            for k_, v_ in rec.items():
                v_["ms"] = round(float(v_["ms"]), 3)
            opts[f"{N}_s{s}"] = dict(N=N, s=s, pairs=float(P), full_res_b2b_M=full.get(N), options=rec)
            print(N, s, "pairs %.3fM" % (P / 1e6), "full-res b2b", full.get(N))
            for k_, v_ in rec.items():
                print("   %-45s %7.3f ms %s" % (k_, v_["ms"], "" if v_["ok"] else "FAILS (list %d > %d)" % (v_["max_list"], CAP)),
                      ("subtiles %.0f" % v_["subtiles"]) if "subtiles" in v_ else "")
    # model check against D1 b2b
    chk = {r["cfg"]: dict(M=r["b2b"], model=opts[f'{r["N"]}_s{r["s"]}']["options"]["O0 today 32x32"]["ms"]) for r in rows}
    json.dump(dict(fits=fits, used=best, params=p.tolist(), other_ms=other, b2b_gap_ms=b2b_gap,
                   check_b2b=chk, options=opts, views=T["views"]), open(a.out, "w"), indent=1)
    for k_, v_ in chk.items():
        print("check", k_, v_)


if __name__ == "__main__":
    main()
