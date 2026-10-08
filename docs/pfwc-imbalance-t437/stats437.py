#!/usr/bin/env python3
"""t437: slow vs median core breakdown from core_view-*.csv (ana437.py output). us -> ms."""
import sys, csv
import numpy as np
from collections import defaultdict
def load(fn):
    R = list(csv.DictReader(open(fn)))
    for r in R:
        for k in r:
            r[k] = float(r[k]) if '.' in r[k] else int(r[k])
    return R
for tag in sys.argv[1:]:
    R = load(f"core_view-{tag}.csv")
    V = sorted({(r["chunk"], r["view"]) for r in R}); cores = sorted({(r["x"], r["y"]) for r in R})
    ci = {c: i for i, c in enumerate(cores)}; vi = {v: i for i, v in enumerate(V)}
    F = ["BRISC_s","BRISC_e","NCRISC_s","NCRISC_e","TRISC_0_s","TRISC_0_e","TRISC_1_s","TRISC_1_e","TRISC_2_s","TRISC_2_e"]
    A = np.zeros((len(cores), len(V), len(F)))
    lin = {}
    for r in R:
        A[ci[(r["x"], r["y"])], vi[(r["chunk"], r["view"])]] = [r[f] for f in F]
        lin[(r["x"], r["y"])] = r["c"]
    A /= 1000.0  # ms
    g = lambda n: A[:, :, F.index(n)]
    end = np.maximum(g("BRISC_e"), g("NCRISC_e"))
    tend = np.maximum.reduce([g("TRISC_0_e"), g("TRISC_1_e"), g("TRISC_2_e")])
    start = np.minimum(g("BRISC_s"), g("NCRISC_s"))
    t1 = g("TRISC_1_e") - g("TRISC_1_s")
    btail = g("BRISC_e") - tend; ntail = g("NCRISC_e") - tend
    print(f"== {tag}: cores {len(cores)} views {len(V)}")
    print(f"program end (max core, all RISCs) mean {end.max(0).mean():.3f} ms; mean core end {end.mean(0).mean():.3f}; "
          f"max/mean {end.max(0).mean()/end.mean(0).mean():.3f}; TRISC-only max {tend.max(0).mean():.3f} mean {tend.mean(0).mean():.3f}")
    print(f"launch skew: core start max {start.max():.4f} ms (mean {start.mean():.4f})")
    # which RISC ends last per core-view
    last = np.argmax(np.stack([g("BRISC_e"), g("NCRISC_e"), tend]), 0)
    print("last RISC to end (BRISC/NCRISC/TRISC) share over all core-views:", np.bincount(last.ravel(), minlength=3) / last.size)
    ls = last[np.argmax(end, 0), np.arange(len(V))]
    print("  on the slowest core per view:", np.bincount(ls, minlength=3))
    cm = end.mean(1); o = np.argsort(-cm)
    slow = np.argmax(end, 0)
    cnt = defaultdict(int)
    for s in slow: cnt[cores[s]] += 1
    print("slowest core per view (by full end):", sorted(cnt.items(), key=lambda x: -x[1])[:8])
    # rank stability: per view rank of each core; correlation of per-core end across views
    Z = (end - end.mean(0)) / end.std(0)
    cc = np.corrcoef(Z.T)  # view x view
    print(f"mean view-to-view correlation of per-core end: {cc[np.triu_indices(len(V),1)].mean():.3f}")
    # split halves
    h1, h2 = end[:, :15].mean(1), end[:, 15:].mean(1)
    print(f"per-core mean end, views 1-15 vs 16-30 corr: {np.corrcoef(h1,h2)[0,1]:.3f}")
    med = o[len(o)//2]
    def row(i, name):
        return (f"| {name} {cores[i]} c={lin[cores[i]]} | {start[i].mean()*1e3:.1f} | {(g('TRISC_0_e')-g('TRISC_0_s'))[i].mean():.3f} | {t1[i].mean():.3f} | "
                f"{(g('TRISC_2_e')-g('TRISC_2_s'))[i].mean():.3f} | {(g('NCRISC_e')-g('NCRISC_s'))[i].mean():.3f} | {(g('BRISC_e')-g('BRISC_s'))[i].mean():.3f} | "
                f"{btail[i].mean()*1e3:.0f} | {ntail[i].mean()*1e3:.0f} | {end[i].mean():.3f} | {(np.argmax(end,0)==i).sum()} |")
    print("| core | start us | TRISC0 | TRISC1 math | TRISC2 | NCRISC kernel | BRISC kernel | BRISC tail us | NCRISC tail us | end | slowest in views |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for i in o[:6]: print(row(i, "slow"))
    print(row(med, "median"))
    for i in o[-2:]: print(row(i, "fast"))
    # per-view slowest-core averages vs mean core
    sel = (slow, np.arange(len(V)))
    print(f"per-view slowest core: TRISC1 {t1[sel].mean():.3f} vs mean core {t1.mean():.3f}; "
          f"tail(max BR/NC after TRISC) {np.maximum(btail,ntail)[sel].mean()*1e3:.0f} us vs mean {np.maximum(btail,ntail).mean()*1e3:.0f} us")
    # correlation between TRISC1 duration and end per view
    r_t1 = np.mean([np.corrcoef(t1[:, v], end[:, v])[0, 1] for v in range(len(V))])
    r_tl = np.mean([np.corrcoef(np.maximum(btail,ntail)[:, v], end[:, v])[0, 1] for v in range(len(V))])
    print(f"per-view corr(end, TRISC1 dur) {r_t1:.3f}; corr(end, writer tail) {r_tl:.3f}")
    # column/row
    for ax, nm in ((0, "x"), (1, "y")):
        keys = sorted({c[ax] for c in cores})
        print(f"  by {nm}: " + " ".join(f"{k}:{np.mean([cm[i] for i,c in enumerate(cores) if c[ax]==k]):.3f}" for k in keys))
    np.save(f"end-{tag}.npy", end); np.save(f"t1-{tag}.npy", t1)
    with open(f"cores-{tag}.txt", "w") as f:
        for i, c in enumerate(cores): f.write(f"{c[0]} {c[1]} {lin[c]} {cm[i]:.4f} {t1[i].mean():.4f}\n")
