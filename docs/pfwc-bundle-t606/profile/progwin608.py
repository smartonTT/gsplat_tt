#!/usr/bin/env python3
"""t608: all 4 programs' device windows per view (FOLD=0: pfwc, k2, sort, mat+blend), from the same
t595-style csv as pfwc608.py. Window = first FW start on any core -> last FW end; gap = start of this
program - end of the previous one. Prints the mean over the last K views (minus view 0 of them).
Usage: progwin608.py DEV.csv[.gz] [--keep 5]"""
import argparse
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pfwc608 import NP, load, ms  # noqa: E402

NAMES = ["pfwc", "k2", "sort", "mat+blend"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--keep", type=int, default=5)
    a = ap.parse_args()
    iv, _ = load(a.csv)
    per_core = defaultdict(list)  # core -> list of (s, e) per program, any RISC's FW
    for (c, r), zs in iv.items():
        fw = sorted((s, e) for z, s, e in zs if z.endswith("-FW"))
        per_core[(c, r)].append(fw)
    # group FW zones of all RISCs of a core into programs by BRISC-FW order
    progs = defaultdict(lambda: [None, None])
    for (c, r), zs in iv.items():
        b = sorted((s, e) for z, s, e in zs if z == "BRISC-FW")
        others = sorted((s, e) for z, s, e in zs if z.endswith("-FW") and z != "BRISC-FW")
        for i, (s, e) in enumerate(b):
            lo, hi = s, e
            for s2, e2 in others:
                if s2 >= s and s2 <= e:
                    lo, hi = min(lo, s2), max(hi, e2)
            p = progs[i]
            p[0] = lo if p[0] is None else min(p[0], lo)
            p[1] = hi if p[1] is None else max(p[1], hi)
    n = len(progs)
    nv = n // NP
    first = nv - a.keep
    rows = []
    for v in range(first, nv):
        w = []
        for k in range(NP):
            s, e = progs[v * NP + k]
            ps = progs[v * NP + k - 1][1] if v * NP + k > 0 else s
            w.append((ms(e - s), ms(s - ps)))
        per = ms(progs[v * NP + NP - 1][1] - progs[v * NP - 1][1]) if v > 0 else 0
        rows.append((w, per))
    print(f"# {a.csv}: {nv} views; views {first}..{nv-1} (steady = all but the first)")
    print("view " + " ".join(f"{x:>9}(gap)" for x in NAMES) + "   period")
    for i, (w, per) in enumerate(rows):
        print(f"{i:>4} " + " ".join(f"{x:9.3f}({g:5.3f})" for x, g in w) + f"   {per:7.3f}")
    st = rows[1:]
    m = [(sum(r[0][k][0] for r in st) / len(st), sum(r[0][k][1] for r in st) / len(st)) for k in range(NP)]
    print("MEAN " + " ".join(f"{x:9.3f}({g:5.3f})" for x, g in m) + f"   {sum(r[1] for r in st)/len(st):7.3f}")


if __name__ == "__main__":
    main()
