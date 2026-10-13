#!/usr/bin/env python3
"""t630: FOLD=1 fixed-cost attribution from the stitched device CSV (prep506.py, ANA_PROGS=pfwc,sort,mat+blend).

Per program and view: window (first core FW start -> last core FW end), launch skew (spread of the
cores' BRISC-FW starts), slowest core span, per-core preamble (FW start -> first zone) / postamble
(last zone end -> FW end). Inside mat+blend: the critical (last-ending) core's timeline split into
launch offset, mat phase (TRISC1 FW start -> tile_blend_sfpu start), blend setup, first-counts wait,
blend SFPU, reader waits, T init, emit, post, writer drain; plus the bz_r_* / bz_c_* / bz_w_* means.
Usage: ANA_PROGS=pfwc,sort,mat+blend bz630.py STITCHED.csv.gz
"""
import os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ANA_PROGS", "pfwc,sort,mat+blend")
import ana506 as A

C = 1350.0 * 1000.0  # cycles per ms


def m(x):
    return float(np.median(x)) if len(x) else float("nan")  # t630: median (a few doubled marker sums skew means)


def main():
    iv, td = A.load_dev(sys.argv[1])
    FWI, Z, D, nv, cores = A.assign(iv, td)
    P = A.PROGS
    print(f"views {nv} cores {len(cores)} programs {P}")
    rows = defaultdict(list)
    for p, pn in enumerate(P):
        for v in range(nv):
            fw = FWI[(v, p)]
            if not fw:
                continue
            w0 = min(s for _, _, s, _ in fw); w1 = max(e for _, _, _, e in fw)
            per = defaultdict(lambda: [1 << 62, 0])
            for c, r, s, e in fw:
                per[c][0] = min(per[c][0], s); per[c][1] = max(per[c][1], e)
            bstart = [s for c, r, s, e in fw if r == "BRISC"]
            zs = defaultdict(lambda: [1 << 62, 0])
            for (vv, pp, z), L in Z.items():
                if vv != v or pp != p:
                    continue
                for c, r, s, e in L:
                    zs[c][0] = min(zs[c][0], s); zs[c][1] = max(zs[c][1], e)
            spans = {c: e - s for c, (s, e) in per.items()}
            pre = [zs[c][0] - per[c][0] for c in per if c in zs]
            post = [per[c][1] - zs[c][1] for c in per if c in zs]
            rows[pn].append(dict(win=(w1 - w0) / C, skew=(max(bstart) - min(bstart)) / C,
                                 slow=max(spans.values()) / C, mean=m(list(spans.values())) / C,
                                 endspread=(w1 - min(e for s, e in per.values())) / C,
                                 pre=m(pre) / C, post=m(post) / C))
    print("\n## per-program fixed parts (ms, median over views)\n")
    print("| program | window | slowest core | mean core | launch skew | end spread | preamble | postamble |")
    print("|---|---|---|---|---|---|---|---|")
    for pn in P:
        R = rows[pn]
        print(f"| {pn} | " + " | ".join(f"{m([r[k] for r in R]):.3f}" for k in
              ("win", "slow", "mean", "skew", "endspread", "pre", "post")) + " |")

    p = P.index("mat+blend")
    crit = defaultdict(list); allc = defaultdict(list)
    for v in range(nv):
        fw = FWI[(v, p)]
        if not fw:
            continue
        w0 = min(s for _, _, s, _ in fw); w1 = max(e for _, _, _, e in fw)
        zone = {}
        for (vv, pp, z), L in Z.items():
            if vv == v and pp == p:
                for c, r, s, e in L:
                    zone.setdefault((c, r, z), (s, e))
        fwc = {(c, r): (s, e) for c, r, s, e in fw}
        core_end = defaultdict(int)
        for (c, r), (s, e) in fwc.items():
            core_end[c] = max(core_end[c], e)
        for c in core_end:
            t1 = fwc.get((c, "TRISC_1")); br = fwc.get((c, "BRISC")); nc = fwc.get((c, "NCRISC"))
            bl = zone.get((c, "TRISC_1", "tile_blend_sfpu")); ld = zone.get((c, "NCRISC", "tile_blend_load"))
            if not (t1 and br and bl):
                continue
            d = D[v]
            g = lambda n, r: d.get((c, r, n), 0) / C
            row = dict(
                launch=(min(br[0], t1[0]) - w0) / C,
                mat=(bl[0] - t1[0]) / C,
                c_setup=g("bz_c_setup", "TRISC_1"), c_cntw1=g("bz_c_cntw1", "TRISC_1"),
                c_cntw=g("bz_c_cntw", "TRISC_1"), c_init=g("bz_c_init", "TRISC_1"),
                c_blend=g("bz_c_blend", "TRISC_1"), c_emit=g("bz_c_emit", "TRISC_1"),
                c_tot=g("bz_c_tot", "TRISC_1"), c_nt=d.get((c, "TRISC_1", "bz_c_nt"), 0),
                c_nsc=d.get((c, "TRISC_1", "bz_c_nsc"), 0),
                blend_zone=(bl[1] - bl[0]) / C, t1_post=(t1[1] - bl[1]) / C,
                drain=(core_end[c] - t1[1]) / C, span=(core_end[c] - min(br[0], t1[0])) / C,
                end_gap=(w1 - core_end[c]) / C,
                ld_start=((ld[0] - w0) / C) if ld else float("nan"),
            )
            for n in ("tot", "setup", "claim", "meta", "ready", "bulkw", "dma", "cntw", "qw"):
                row["r_" + n] = g("bz_r_" + n, "NCRISC")
            for n in ("nt", "nsc", "rdyn"):
                row["r_" + n] = d.get((c, "NCRISC", "bz_r_" + n), 0)
            for n in ("tot", "pre", "qw", "outw", "pack", "wr", "bar"):
                row["w_" + n] = g("bz_w_" + n, "BRISC")
            for n in ("nt", "npoll"):
                row["w_" + n] = d.get((c, "BRISC", "bz_w_" + n), 0)
            for k, x in row.items():
                allc[k].append(x)
            if core_end[c] == w1:
                row["window"] = (w1 - w0) / C
                for k, x in row.items():
                    crit[k].append(x)
    print("\n## mat+blend per core (ms; mean over cores x views | critical last-ending core, median over views)\n")
    print("| part | all cores | critical core |")
    print("|---|---|---|")
    for k in allc:
        print(f"| {k} | {m(allc[k]):.3f} | {m(crit[k]):.3f} |")
    print(f"| window | | {m(crit['window']):.3f} |")


if __name__ == "__main__":
    main()
