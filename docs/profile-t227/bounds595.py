#!/usr/bin/env python3
"""t595: step-0 bounds of structural plan 3 specs B and C from the always-on fz_* markers.

Input: the stitched profiler CSV of the measured b2b views (prep506.py output). Per view, inside
the mat+blend program (last program of the view), per core:
  mat end      TRISC_1 mat_cull_mask end (from the earliest mat start of the view)
  mover end    end of the mover's last sort_subchunk_mat zone; mover work = its span - fz_mv_dw
  big core     NCRISC has fz_mv_big > 0 (holds a >8192-record item)
  mb end       the core's mat+blend program end (FW zones), from the program start

Spec B (bound508 equivalent; MATCULL_PROF per-item zones are not needed):
  base = max over cores of max(mat end, mover ends)
  comb = max over cores of (non-big: max over movers of end - fz_mv_dw; big: BRISC end - BRISC fz_mv_dw)
  jobs = max over cores of max over movers of end - fz_mv_dw   (fill jobs alone)
  frame gain = (base - x) x 0.28 / x 0.40 (t335 / t508 factors).
Spec C: per core band = fz_m_band (MATH cycles in band_batch), t1 = mat_cull_mask length,
  mover = max over movers of (span - fz_mv_dw); saving_c = min(band, max(0, t1 - mover)).
  critical core = the core whose mat+blend program ends last in the view.
  Reports saving on the critical core and the makespan bound max_c(mb_end_c - saving_c).
Usage: ANA_PROGS=... bounds595.py p.csv.gz [--skip N]  (skip the first N views of the file)
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ana506 as A  # noqa: E402

MOV = ("NCRISC", "BRISC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip", type=int, default=0)
    a = ap.parse_args()
    iv, td = A.load_dev(a.csv)
    FWI, Z, D, nv, cores = A.assign(iv, td)
    pm = A.NP - 1
    R = defaultdict(list)
    for v in range(a.skip, nv):
        fw = FWI[(v, pm)]
        p0 = min(s for c, r, s, e in fw)
        mbend = defaultdict(int)
        for c, r, s, e in fw:
            mbend[c] = max(mbend[c], e)
        z = defaultdict(lambda: defaultdict(list))
        for zn in ("mat_cull_mask", "sort_subchunk_mat"):
            for c, r, s, e in Z.get((v, pm, zn), []):
                z[c][(r, zn)].append((s, e))
        mats = [s for c in z for s, e in z[c].get(("TRISC_1", "mat_cull_mask"), [])]
        if not mats:
            continue
        t0 = min(mats)
        d = D[v]
        base, comb, jobs, sav, mbe, rows = {}, {}, {}, {}, {}, {}
        for c in cores:
            m = z[c].get(("TRISC_1", "mat_cull_mask"), [])
            if not m:
                continue
            mat_end = A.ms(max(e for s, e in m) - t0)
            t1 = A.ms(sum(e - s for s, e in m))
            mend, mwork, mnw = {}, {}, {}
            for r in MOV:
                L = z[c].get((r, "sort_subchunk_mat"), [])
                if not L:
                    continue
                dw = A.ms(d.get((c, r, "fz_mv_dw"), 0))
                mend[r] = A.ms(max(e for s, e in L) - t0)
                mwork[r] = A.ms(sum(e - s for s, e in L)) - dw
                mnw[r] = mend[r] - dw
            big = d.get((c, "NCRISC", "fz_mv_big"), 0) > 0
            base[c] = max([mat_end] + list(mend.values()))
            jobs[c] = max(mnw.values(), default=mat_end)
            comb[c] = mnw.get("BRISC", 0.0) if big else jobs[c]
            band = A.ms(d.get((c, "TRISC_1", "fz_m_band"), 0))
            mov = max(mwork.values(), default=0.0)
            sav[c] = min(band, max(0.0, t1 - mov))
            mbe[c] = A.ms(mbend[c] - p0)
            rows[c] = dict(t1=t1, band=band, mov=mov, big=big, mat_end=mat_end, mb_end=mbe[c],
                           dw_n=A.ms(d.get((c, "NCRISC", "fz_mv_dw"), 0)),
                           dw_b=A.ms(d.get((c, "BRISC", "fz_mv_dw"), 0)),
                           mvsort=A.ms(d.get((c, "NCRISC", "fz_mv_sort"), 0) + d.get((c, "BRISC", "fz_mv_sort"), 0)) / 2,
                           t1_mbw=A.ms(d.get((c, "TRISC_1", "fz_m_mbw"), 0)))
        b = max(base.values())
        crit = max(mbe, key=mbe.get)
        R["base"].append(b)
        R["comb"].append(max(comb.values()))
        R["jobs"].append(max(jobs.values()))
        R["nbig"].append(sum(r["big"] for r in rows.values()))
        R["mb_make"].append(max(mbe.values()))
        R["mb_make_C"].append(max(mbe[c] - sav[c] for c in mbe))
        R["C_crit"].append(sav[crit])
        # direct frame-level form: a core's blend follows its mat on the same TRISCs, so a core whose
        # mat ends d earlier ends mat+blend at most d earlier
        R["mb_make_B"].append(max(mbe[c] - max(0.0, base[c] - comb[c]) for c in mbe))
        R["mb_make_J"].append(max(mbe[c] - max(0.0, base[c] - jobs[c]) for c in mbe))
        for k in ("t1", "band", "mov", "dw_n", "dw_b", "mvsort", "t1_mbw", "mat_end"):
            R[f"crit_{k}"].append(rows[crit][k])
            R[f"mean_{k}"].append(np.mean([r[k] for r in rows.values()]))
            R[f"max_{k}"].append(np.max([r[k] for r in rows.values()]))
        R["crit_big"].append(rows[crit]["big"])
        R["mean_C"].append(np.mean(list(sav.values())))
    m = {k: float(np.mean(x)) for k, x in R.items()}
    n = len(R["base"])
    print(f"views {n}, big cores/view {m['nbig']:.1f}")
    print("\nSpec B (bound508 equivalent from fz_*; mat makespan ms)")
    print(f"  base {m['base']:.3f}  no-waits+big-free(comb) {m['comb']:.3f}  no-waits(jobs) {m['jobs']:.3f}")
    for k in ("comb", "jobs"):
        g = m["base"] - m[k]
        print(f"  {k:5s} gain {g:.3f} model ms -> frame x0.28 {g * 0.28:.3f}, x0.40 {g * 0.40:.3f}")
    for k, n_ in (("B", "comb"), ("J", "jobs")):
        print(f"  direct mat+blend makespan ({n_}): {m['mb_make']:.3f} -> {m['mb_make_' + k]:.3f} "
              f"(gain {m['mb_make'] - m['mb_make_' + k]:.3f} ms/view)")
    print("\nSpec C (band_batch on the FPU; ms)")
    print(f"  mat+blend makespan {m['mb_make']:.3f}; critical core is a big core in {m['crit_big'] * 100:.0f}% of views")
    for k in ("t1", "band", "mov", "dw_n", "dw_b", "mvsort", "t1_mbw", "mat_end"):
        print(f"  {k:8s} critical {m['crit_' + k]:.3f}  mean {m['mean_' + k]:.3f}  max {m['max_' + k]:.3f}")
    print(f"  saving on critical core min(band, t1 - mover) = {m['C_crit']:.3f}; mean over cores {m['mean_C']:.3f}")
    print(f"  makespan bound: {m['mb_make']:.3f} -> {m['mb_make_C']:.3f} (gain {m['mb_make'] - m['mb_make_C']:.3f})")


if __name__ == "__main__":
    main()
