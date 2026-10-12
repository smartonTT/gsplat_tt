#!/usr/bin/env python3
"""t508: per-view bound of the big-tile split lever from the t506 MATCULL capture.

Per view and core: mat start = first mat_* zone start, mat end = last mat_cull_mask end (TRISC)
or last mover zone end. A pure-big core is one whose NCRISC runs mat_ol_keys. Reports, in mat ms
(per-view makespan = max over cores of mat end - min start):
  base       today's makespan
  ideal      big items free: a pure-big core ends at its BRISC end (hard upper bound of any split)
  no-waits+big-free  every mover's done-waits removed and big items free: ceiling of the
             combined lever (fill jobs + rank-ordered gather + big-tile split)
  no-waits   every done-wait removed, big items as today (finer fill jobs alone, ideal)
  split2     each big subchunk in 2 parts on 2 cores: NCRISC part time = keys + select + sort/2 +
             gather/2 measured on that core, + half its fill wait; the 2nd part displaces another
             core's NCRISC work (+DISP ms on every other core)
Usage: bound508.py dev-M0.csv.gz [...]"""
import sys, os
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "b2b-timeline-p150-223"))
import ana506 as A

DISP = 0.04  # 6 cores' displaced NCRISC work (~13k recs each) over ~220 slots

def views(path):
    iv, td = A.load_dev(path)
    FWI, Z, D, nv, cores = A.assign(iv, td)
    for v in range(nv):
        per = defaultdict(lambda: defaultdict(list))  # core -> risc -> [(z, s, e)]
        for (vv, p, z), L in Z.items():
            if vv != v or not z.startswith("mat_"):
                continue
            for c, r, s, e in L:
                per[c][r].append((z, s, e))
        if per:
            yield per

def main():
    rows = []
    for path in sys.argv[1:]:
        for per in views(path):
            t0 = min(s for c in per for r in per[c] for z, s, e in per[c][r])
            ends, big = {}, {}
            for c, R in per.items():
                ends[c] = A.ms(max(e for r in R for z, s, e in R[r]) - t0)
                nc = R.get("NCRISC", [])
                if any(z == "mat_ol_keys" for z, s, e in nc):
                    zt = defaultdict(float)
                    for z, s, e in nc:
                        zt[z] += A.ms(e - s)
                    br = A.ms(max((e for z, s, e in R.get("BRISC", [])), default=t0) - t0)
                    part = zt["mat_ol_keys"] + zt["mat_ol_sort"] * 0.75 + zt["mat_ol_gather"] * 0.6 \
                        + zt["mat_cull_wait"] / 2 + zt["mat_ol_wr"] / 2 + zt["mat_meta"]
                    nstart = A.ms(min(s for z, s, e in nc) - t0)
                    big[c] = (br, nstart + part, zt)
            def nowait(R):  # each mover's end minus its done-waits; the core ends with its later mover
                out = 0.0
                for r in ("NCRISC", "BRISC"):
                    L = R.get(r, [])
                    if L:
                        out = max(out, A.ms(max(e for z, s, e in L) - t0)
                                  - sum(A.ms(e - s) for z, s, e in L if z == "mat_cull_wait"))
                return out
            nw = {c: nowait(R) for c, R in per.items()}
            base = max(ends.values())
            others = [ends[c] for c in ends if c not in big]
            ideal = max(others + [big[c][0] for c in big])
            split = max([x + DISP for x in others] + [max(b[0], b[1]) for b in big.values()])
            # combined ceiling: no done-waits anywhere, big items free (a big core keeps its BRISC side)
            def brisc_nowait(R):
                L = R.get("BRISC", [])
                return (A.ms(max(e for z, s, e in L) - t0)
                        - sum(A.ms(e - s) for z, s, e in L if z == "mat_cull_wait")) if L else 0.0
            comb = max([nw[c] for c in nw if c not in big] + [brisc_nowait(per[c]) for c in big])
            jobs = max(nw.values())  # fill jobs alone: every done-wait gone, big items as today
            rows.append((base, ideal, split, comb, jobs, len(big)))
    n = len(rows)
    m = [sum(r[i] for r in rows) / n for i in range(5)]
    print(f"views {n}, pure-big cores/view {sum(r[5] for r in rows) / n:.1f}")
    print(f"mat makespan  base {m[0]:.3f}  ideal(big free) {m[1]:.3f}  split2 {m[2]:.3f}  no-waits+big-free {m[3]:.3f}  no-waits {m[4]:.3f}  (ms)")
    for k, x in (("ideal", m[1]), ("split2", m[2]), ("comb", m[3]), ("nowait", m[4])):
        g = m[0] - x
        print(f"{k:7s} gain {g:.3f} model ms -> frame x0.28 {g * 0.28:.3f}, x0.40 {g * 0.40:.3f}")

if __name__ == "__main__":
    main()

def detail(path, nview=6):
    for i, per in enumerate(views(path)):
        if i >= nview:
            break
        t0 = min(s for c in per for r in per[c] for z, s, e in per[c][r])
        ends = {c: A.ms(max(e for r in R for z, s, e in R[r]) - t0) for c, R in per.items()}
        bigc = {c for c, R in per.items() if any(z == "mat_ol_keys" for z, s, e in R.get("NCRISC", []))}
        top = sorted(ends.items(), key=lambda x: -x[1])[:6]
        print(f"view {i}: " + "  ".join(f"{c[0]}-{c[1]}{'*' if c in bigc else ''} {e:.3f}" for c, e in top))
