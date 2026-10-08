#!/usr/bin/env python3
"""t437: predicted pfwc program end (max core) under alternative tile deals, using the
per-core linear fit (fit437.py: end ~ a*chunks + b*visible, per-view centred) plus the
measured per-core-view residuals (kept on the same core index, so unexplained per-core
slowness stays). E12 only (12x10)."""
import numpy as np, csv
W = np.load("tilework.npz"); vis = W["vis"].astype(float)
nv, T = vis.shape
tag, C = "dev-E12", 120
R = list(csv.DictReader(open(f"core_view-{tag}.csv")))
V = sorted({(int(r["chunk"]), int(r["view"])) for r in R})
end = np.zeros((C, len(V)))
for r in R:
    end[int(r["c"]), V.index((int(r["chunk"]), int(r["view"])))] = max(float(r["BRISC_e"]), float(r["NCRISC_e"])) / 1000
vi = [ch * 10 + v - 1 for ch, v in V]  # chunk index 0/1/2 -> views 0-9, 10-19, 20-29
a, b = np.load(f"coef-{tag}.npy") / 1.0  # ms per chunk, ms per visible
def loads(assign, v):
    nch = np.bincount(assign, minlength=C); cv = np.bincount(assign, weights=vis[v], minlength=C)
    return nch, cv
cur = np.arange(T) % C
# model baseline for the current deal -> residual
pred_cur = np.zeros_like(end)
for j, v in enumerate(vi):
    nch, cv = loads(cur, v)
    pred_cur[:, j] = a * (nch - nch.mean()) + b * (cv - cv.mean())
mu = end.mean(0)
resid = end - (pred_cur + mu)
def lpt(w):
    o = np.argsort(-w); s = np.zeros(C); asg = np.empty(T, int)
    import heapq
    h = [(0.0, c) for c in range(C)]; heapq.heapify(h)
    for t in o:
        l, c = heapq.heappop(h); asg[t] = c; heapq.heappush(h, (l + w[t], c))
    return asg
def evaluate(name, asg_of_view, keep_resid=True):
    mx = []
    for j, v in enumerate(vi):
        asg = asg_of_view(v)
        nch, cv = loads(asg, v)
        p = mu[j] + a * (nch - nch.mean()) + b * (cv - cv.mean()) + (resid[:, j] if keep_resid else 0)
        mx.append(p.max())
    mx = np.array(mx)
    print(f"{name:42s} max-core {mx.mean():.3f} ms  gain vs measured {end.max(0).mean() - mx.mean():+.3f}")
    return mx
print(f"measured: max-core {end.max(0).mean():.3f}, mean-core {mu.mean():.3f}, model coef a={a*1e3:.1f} us/chunk b={b*1e3:.4f} us/visible")
print(f"residual: std {resid.std()*1e3:.1f} us, per-core mean std {resid.mean(1).std()*1e3:.1f} us")
evaluate("strided (current, model+resid)", lambda v: cur)
evaluate("strided, no residual", lambda v: cur, False)
wt = lambda vv: a + b * vv
evaluate("LPT oracle (same view weights)", lambda v: lpt(wt(vis[v])))
evaluate("LPT oracle, no residual", lambda v: lpt(wt(vis[v])), False)
evaluate("LPT prev view (temporal, view v-1)", lambda v: lpt(wt(vis[max(v - 1, 0)])))
static_mean = lpt(wt(vis.mean(0)))
evaluate("LPT static, mean of 30 views", lambda v: static_mean)
static_hero = lpt(wt(vis[0]))
evaluate("LPT static, hero view only", lambda v: static_hero)
# view-independent proxy: count of opacity>=1/255 gaussians per tile is ~constant; skip.
cont = np.minimum(np.arange(T) * C // T, C - 1)
evaluate("contiguous blocks (legacy split)", lambda v: cont)
rng = np.random.default_rng(1); perm = rng.permutation(T)
evaluate("random static shuffle (strided over perm)", lambda v: (np.argsort(perm) % C))
# Equal-count variant (keeps per-core tile counts and seg_base as today): tiles sorted by
# weight, dealt in rounds of C, each round's heaviest tile to the least-loaded core.
def lpt_eq(w, init=None):
    load = np.zeros(C) if init is None else init.copy(); asg = np.empty(T, int)
    o = np.argsort(-w); cnt = np.bincount(cur, minlength=C)
    for r in range(0, T, C):
        blk = o[r:r + C]
        elig = np.argsort(load)[:len(blk)] if len(blk) == C else np.argsort(load)[np.argsort(load) < C][:len(blk)]
        if len(blk) < C:  # remainder: cores that get the extra tile today
            elig = [c for c in np.argsort(load) if cnt[c] > T // C][:len(blk)]
        for t, c in zip(blk, elig): asg[t] = c; load[c] += w[t]
    return asg
eq = lpt_eq(wt(vis[0]))
assert (np.bincount(eq, minlength=C) == np.bincount(cur, minlength=C)).all()
evaluate("equal-count LPT static hero", lambda v: eq)
evaluate("equal-count LPT static hero, no residual", lambda v: eq, False)
