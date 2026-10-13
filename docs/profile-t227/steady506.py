#!/usr/bin/env python3
"""t506: steady-state b2b accounting on top of ana506 (same CSV from prep506.py).

ana506 means include the first kept view of each chunk, whose pfwc ran before a ~26 ms
host pause between passes (pfwc -> k2 window). This script drops what that pause touches:
  - pfwc -> k2 boundaries of the first view of each chunk,
  - periods that end in another chunk (the last view of each chunk).
Output: per-program window vs mean/max core length (end spread), per-boundary all-core and
per-core idle, the per-view period split, the lever bounds, and a Gantt PNG for 3 views.
Usage: steady506.py CSV [--mhz 1350] [--gantt out.png] [--gantt-views 1,2,3]
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ana506 as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--mhz", type=float, default=1350.0)
    ap.add_argument("--gantt")
    ap.add_argument("--gantt-views", default="1,2,3")
    a = ap.parse_args()
    A.CYC_MS = a.mhz * 1000.0
    ms = A.ms
    iv, td = A.load_dev(a.csv)
    FWI, Z, D, NV, cores = A.assign(iv, td)
    VPC = A.VIEWS_PER_CHUNK
    first = lambda v: v % VPC == 0
    nxt = [v for v in range(NV - 1) if v // VPC == (v + 1) // VPC]

    def core_len(v, p):
        d = defaultdict(lambda: [1 << 62, 0])
        for c, r, s, e in FWI[(v, p)]:
            d[c][0] = min(d[c][0], s)
            d[c][1] = max(d[c][1], e)
        return d

    CL = {(v, p): core_len(v, p) for v in range(NV) for p in range(A.NP)}
    PW = {(v, p): (min(s for s, _ in CL[(v, p)].values()), max(e for _, e in CL[(v, p)].values()))
          for v in range(NV) for p in range(A.NP)}
    print(f"# steady506: {a.csv}  views={NV} cores={len(cores)}  ms at {a.mhz:.0f} MHz")
    print(f"steady periods (pfwc N -> pfwc N+1, same chunk, N not first of chunk): "
          f"{len([v for v in nxt if not first(v)])}")

    # program windows and end spread (window - core length = load-imbalance idle inside it)
    print("\n## Programs (mean over views; window = first core start -> last core end)\n")
    print(f"{'prog':<10}{'window':>8}{'core mean':>10}{'core max':>9}{'start spr':>10}{'end spr':>9}{'idle/core':>10}")
    prog = {}
    for p in range(A.NP):
        vs = [v for v in range(NV) if not (p == 0 and first(v))] if p else list(range(NV))
        w = [ms(PW[(v, p)][1] - PW[(v, p)][0]) for v in vs]
        cm = [np.mean([ms(e - s) for s, e in CL[(v, p)].values()]) for v in vs]
        cx = [np.max([ms(e - s) for s, e in CL[(v, p)].values()]) for v in vs]
        ss = [np.mean([ms(s - PW[(v, p)][0]) for s, _ in CL[(v, p)].values()]) for v in vs]
        es = [np.mean([ms(PW[(v, p)][1] - e) for _, e in CL[(v, p)].values()]) for v in vs]
        prog[A.PROGS[p]] = dict(window=np.mean(w), core_mean=np.mean(cm), core_max=np.mean(cx),
                                start_spread=np.mean(ss), end_spread=np.mean(es))
        print(f"{A.PROGS[p]:<10}{np.mean(w):8.3f}{np.mean(cm):10.3f}{np.mean(cx):9.3f}{np.mean(ss):10.3f}"
              f"{np.mean(es):9.3f}{np.mean(w) - np.mean(cm):10.3f}")

    # boundaries: all-core gap (no core runs anything) and per-core idle (that core idle)
    print("\n## Boundaries (steady views; ms)\n")
    print(f"{'boundary':<26}{'all-core':>9}{'max':>7}{'per-core':>9}{'pc max':>8}{'n':>4}")
    bnd = {}
    labels = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for p, q in labels:
        lab = f"{A.PROGS[p]} -> {A.PROGS[q]}" + (" N+1" if q == 0 else "")
        vs = [v for v in (nxt if q == 0 else range(NV)) if not (p == 0 and first(v))]
        g, pc, pcx = [], [], []
        for v in vs:
            w = v + 1 if q == 0 else v
            g.append(ms(PW[(w, q)][0] - PW[(v, p)][1]))
            x = [ms(CL[(w, q)][c][0] - CL[(v, p)][c][1]) for c in cores]
            pc.append(np.mean(x))
            pcx.append(np.max(x))
        bnd[lab] = dict(all_core=np.mean(g), all_core_max=np.max(g), per_core=np.mean(pc), per_core_max=np.mean(pcx))
        print(f"{lab:<26}{np.mean(g):9.3f}{np.max(g):7.3f}{np.mean(pc):9.3f}{np.mean(pcx):8.3f}{len(vs):4d}")
    tot_all = sum(b["all_core"] for b in bnd.values())
    tot_pc = sum(b["per_core"] for b in bnd.values())
    print(f"{'sum':<26}{tot_all:9.3f}{'':>7}{tot_pc:9.3f}")

    # steady period split
    sv = [v for v in nxt if not first(v)]
    per = [ms(PW[(v + 1, 0)][0] - PW[(v, 0)][0]) for v in sv]
    print(f"\n## Steady period: mean {np.mean(per):.3f} median {np.median(per):.3f} min {np.min(per):.3f} "
          f"max {np.max(per):.3f} ms (n={len(sv)})")
    wsum = {A.PROGS[p]: np.mean([ms(PW[(v, p)][1] - PW[(v, p)][0]) for v in sv]) for p in range(A.NP)}
    gsum = np.mean([ms(PW[(v + 1, 0)][0] - PW[(v, 0)][0]) - sum(ms(PW[(v, p)][1] - PW[(v, p)][0]) for p in range(A.NP))
                    for v in sv])
    print("  windows: " + "  ".join(f"{k} {x:.3f}" for k, x in wsum.items()) + f"  all-core gaps {gsum:.3f}")
    cm = {A.PROGS[p]: np.mean([np.mean([ms(e - s) for s, e in CL[(v, p)].values()]) for v in sv]) for p in range(A.NP)}
    print("  mean core busy: " + "  ".join(f"{k} {x:.3f}" for k, x in cm.items())
          + f"  sum {sum(cm.values()):.3f}  -> per-core idle per period {np.mean(per) - sum(cm.values()):.3f}")

    # lever bounds
    mb = prog["mat+blend"]
    a_bound = mb["end_spread"] + bnd["mat+blend -> pfwc N+1"]["all_core"]
    b_bound = tot_all
    print("\n## Bounds (ms/view, upper)\n")
    print(f"(a) pfwc N+1 into the mat+blend N tail: mean end spread {mb['end_spread']:.3f} + all-core gap "
          f"{bnd['mat+blend -> pfwc N+1']['all_core']:.3f} = {a_bound:.3f}  "
          f"(per-core idle at that boundary {bnd['mat+blend -> pfwc N+1']['per_core']:.3f})")
    print(f"(b) summed all-core inter-program idle: {b_bound:.3f}; summed per-core boundary idle "
          f"(spread, needs program overlap to reclaim): {tot_pc:.3f}")

    if a.gantt:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gv = [int(x) for x in a.gantt_views.split(",")]
        col = {"pfwc": "#4c72b0", "k2": "#dd8452", "sort": "#55a868", "mat+blend": "#c44e52"}
        t0 = PW[(gv[0], 0)][0]
        fig, ax = plt.subplots(figsize=(14, 8))
        order = sorted(cores, key=lambda c: (c[1], c[0]))
        for i, c in enumerate(order):
            for v in gv:
                for p in range(A.NP):
                    s, e = CL[(v, p)][c]
                    ax.barh(i, ms(e - s), left=ms(s - t0), height=0.9, color=col[A.PROGS[p]], lw=0)
        for p, n in enumerate(A.PROGS):
            ax.barh(-5, 0, color=col[n], label=n)
        ax.set_ylim(-1, len(order))
        ax.set_xlabel(f"ms from view {gv[0]} pfwc start (device clock {a.mhz:.0f} MHz)")
        ax.set_ylabel("core (sorted by y, x)")
        ax.set_title(f"t506 best-iter-223 bh-30 p150 b2b: per-core program windows, views {a.gantt_views}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(a.gantt, dpi=110)
        print(f"\ngantt: {a.gantt}")


if __name__ == "__main__":
    main()
