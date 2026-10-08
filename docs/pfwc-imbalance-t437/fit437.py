#!/usr/bin/env python3
"""t437: per-core pfwc time vs CPU-model work (tilework.npz) under the strided deal.
Fits end[core, view] ~ a*chunks + b*visible + c*pairs (+ const), reports explained variance,
and simulates alternative deals (contiguous, visible-balanced) for the predicted max core."""
import numpy as np, csv
W = np.load("tilework.npz"); vis, prs = W["vis"].astype(float), W["pairs"].astype(float)
T = vis.shape[1]
def lin(tag, gx):
    R = list(csv.DictReader(open(f"core_view-{tag}.csv")))
    C = gx * 10
    V = sorted({(int(r["chunk"]), int(r["view"])) for r in R})
    end = np.zeros((C, len(V))); t1 = np.zeros((C, len(V)))
    for r in R:
        c = int(r["c"]); j = V.index((int(r["chunk"]), int(r["view"])))
        end[c, j] = max(float(r["BRISC_e"]), float(r["NCRISC_e"])) / 1000
        t1[c, j] = (float(r["TRISC_1_e"]) - float(r["TRISC_1_s"])) / 1000
    # view index into cameras order: chunk*10 + view-1
    vidx = [ch // 10 * 10 + (v - 1) if ch in (0, 10, 20) else ch * 10 + v - 1 for ch, v in V]
    return end, t1, [ (0 if ch == 0 else (10 if ch == 1 else 20)) + v - 1 for ch, v in V]
for tag, gx in (("dev-E12", 12), ("dev-E11", 11)):
    end, t1, vi = lin(tag, gx); C = gx * 10
    core = np.arange(T) % C
    nch = np.bincount(core, minlength=C).astype(float)
    cv = np.stack([np.bincount(core, weights=vis[v], minlength=C) for v in vi], 1)
    cp = np.stack([np.bincount(core, weights=prs[v], minlength=C) for v in vi], 1)
    X = np.stack([np.repeat(nch[:, None], len(vi), 1).ravel(), cv.ravel(), cp.ravel(), np.ones(end.size)], 1)
    # per-view centering removes the per-view constant; fit on deviations
    def dev(a): return (a - a.mean(0)).ravel()
    for nm, cols in (("chunks", [0]), ("visible", [1]), ("pairs", [2]), ("chunks+visible", [0, 1]), ("chunks+vis+pairs", [0, 1, 2])):
        Xd = np.stack([dev(X[:, k].reshape(end.shape)) for k in cols], 1)
        y = dev(end)
        b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        r2 = 1 - ((y - Xd @ b) ** 2).sum() / (y ** 2).sum()
        print(f"{tag} end ~ {nm}: R2 {r2:.3f} coef {np.round(b * 1e3, 5)} (us per unit)")
    # model-predicted slowest cores
    Xd = np.stack([dev(X[:, k].reshape(end.shape)) for k in (0, 1)], 1); y = dev(end)
    b, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = (Xd @ b).reshape(end.shape) + end.mean(0)
    o = np.argsort(-pred.mean(1)); om = np.argsort(-end.mean(1))
    print(f"  measured top-6 cores (linear c): {om[:6].tolist()}  model top-6: {o[:6].tolist()}")
    print(f"  measured max/mean {end.max(0).mean()/end.mean(0).mean():.3f} max {end.max(0).mean():.3f}; model max {pred.max(0).mean():.3f}")
    resid = end - pred
    print(f"  residual std per core-view {resid.std()*1e3:.1f} us; worst-core residual mean {resid[om[0]].mean()*1e3:.1f} us; max |core mean resid| {np.abs(resid.mean(1)).max()*1e3:.1f} us")
    np.save(f"pred-{tag}.npy", pred); np.save(f"coef-{tag}.npy", b)
    # visible per core stats
    print(f"  visible/core per view: mean {cv.mean():.0f}, max/mean {(cv.max(0)/cv.mean(0)).mean():.3f}; chunks/core {nch.min():.0f}..{nch.max():.0f}")
