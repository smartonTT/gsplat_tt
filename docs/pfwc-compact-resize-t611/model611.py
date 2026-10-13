#!/usr/bin/env python3
"""t611: plan-4 spec 2 step 0 re-size of A' (pfwc survivor compaction on RECFAST), CPU only.

python3 docs/pfwc-compact-resize-t611/model611.py <t484 out/pfwc-sr.csv.gz>

Same per-core-launch model as docs/structural-plan-4/model603.py (t598 parts, t603 busy/TRISC
terms), vectorized. model603 fitted 2 parameters (lead L, TRISC offset o) to 2 measurements and
predicted the RECFAST + VIS_FAST bundle at +0.12..0.26 ms/view. t606 measured +0.070, the same as
RECFAST alone (+0.071), and VIS_FAST alone +0.022. Here the fit uses all four measurements:
  base window 1.523 (t595), RECFAST 0.071 (t599), VIS_FAST 0.022 (t606 r5), bundle 0.070 (t606)
and two model forms:
  M1: end = max(T + tl, mover + L)                 params L, o, dV (VIS_FAST TRISC cut)
  M2: M1 plus a per-launch window floor F          params L, o, dV, F
Then it reads A' (RECFAST + compaction, repack split over both movers) as gain over
best-iter-227, at 30/40/50 cyc per survivor. Spec 2 step 0 kill: incremental gain < 0.15 at 40.
"""
import os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "pfwc-mover-t598"))
import model598 as M

C = 1350.0
ms = lambda c: c / C / 1000.0
MEAS = dict(base=1.523, rec=0.071, vis=0.022, bundle=0.070)
REC_NEW = 67.0
LANES_PER_CHUNK = 1024


def main():
    ana = M.load_ana(os.path.join(here, "..", "pfwc-mover-t598", "ana.py"))
    rows = M.core_rows(ana, sys.argv[1])
    T1 = np.array([ms(P[1]) for *_, P, B, N in rows])
    means = np.array([ms(P[7]) for *_, P, B, N in rows])
    T_pure = T1 - (means - np.percentile(means, 5))
    Bp = [M.parts(B) for *_, B, N in rows]
    Np = [M.parts(N) for *_, B, N in rows]
    runs = np.array([r[0] for r in rows])
    order = np.argsort(runs, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(runs[order])) + 1]

    def busy(p, rec_cyc=None, rep_cyc=0.0, rep_share=0.5, s_core=None, other=None):
        cls = p["cls"] * M.F_PRUNE
        rec = p["rec"]
        if rec_cyc is not None:
            rec = rec - (p["rec"] - p["fl"]) + ms(rec_cyc * p["m"])
        extra = 0.0
        if rep_cyc:
            m_core = p["m"] + other["m"]
            extra = ms(rep_cyc * m_core) * rep_share - cls * (1 - s_core)
        return cls + p["pfx"] + rec + p["opn"] + p["tail"] + extra

    s = np.array([(b["m"] + n["m"]) / ((b["n"] + n["n"]) * LANES_PER_CHUNK) for b, n in zip(Bp, Np)])
    tl = 0.045

    def per_launch(T, bb, bn, L):
        return np.maximum.reduceat(np.maximum(T + tl, np.maximum(bb, bn) + L)[order], starts)

    def window(T, bb, bn, L, F=0.0):
        return float(np.maximum(per_launch(T, bb, bn, L), F).mean())

    T0 = (T_pure - M.ACQ) * M.F_PRUNE
    b0 = np.array([busy(p) for p in Bp]); n0 = np.array([busy(p) for p in Np])
    bR = np.array([busy(p, REC_NEW) for p in Bp]); nR = np.array([busy(p, REC_NEW) for p in Np])

    def arms(L, o, dV, F):
        T = T0 + o
        w0 = window(T, b0, n0, L, F)
        return w0, w0 - window(T, bR, nR, L, F), w0 - window(T - dV, b0, n0, L, F), \
            w0 - window(T - dV, bR, nR, L, F)

    def err(L, o, dV, Fs):
        # vectorized over the floor values Fs
        T = T0 + o
        a = [per_launch(T, b0, n0, L), per_launch(T, bR, nR, L),
             per_launch(T - dV, b0, n0, L), per_launch(T - dV, bR, nR, L)]
        w = [np.maximum(x[None, :], Fs[:, None]).mean(1) for x in a]
        return (abs(w[0] - MEAS["base"]) + abs(w[0] - w[1] - MEAS["rec"])
                + abs(w[0] - w[2] - MEAS["vis"]) + abs(w[0] - w[3] - MEAS["bundle"]))

    def fit(Fs, step=0.01, rng=None):
        Fs = np.asarray(Fs, float)
        best = None
        Ls, os_, ds = rng or (np.arange(-0.30, 0.30, step), np.arange(-0.30, 0.30, step),
                              np.arange(0.0, 0.41, 0.02))
        for L in Ls:
            for o in os_:
                for dV in ds:
                    e = err(L, o, dV, Fs)
                    i = int(np.argmin(e))
                    if best is None or e[i] < best[0]:
                        best = (float(e[i]), L, o, dV, Fs[i])
        return best

    def fit2(Fs, dVs=None):
        if dVs is not None:  # dV fixed (measured), search L, o and F only
            e, L0, o0, d0, F0 = fit(Fs, rng=(np.arange(-0.30, 0.30, 0.01), np.arange(-0.30, 0.30, 0.01), dVs))
            Fr = np.arange(F0 - 0.01, F0 + 0.0101, 0.0025)
            return fit(Fr, rng=(np.arange(L0 - 0.01, L0 + 0.0101, 0.0025),
                                np.arange(o0 - 0.01, o0 + 0.0101, 0.0025), dVs))
        e, L0, o0, d0, F0 = fit(Fs)
        Fr = Fs if len(Fs) == 1 else np.arange(F0 - 0.01, F0 + 0.0101, 0.0025)
        return fit(Fr, rng=(np.arange(L0 - 0.01, L0 + 0.0101, 0.0025),
                            np.arange(o0 - 0.01, o0 + 0.0101, 0.0025),
                            np.arange(max(0, d0 - 0.02), d0 + 0.0201, 0.005)))

    print(f"core-launches {len(rows)}; survival per core mean {s.mean():.3f} max {s.max():.3f}")
    print(f"measured: base {MEAS['base']} gains rec {MEAS['rec']} vis {MEAS['vis']} bundle {MEAS['bundle']}")
    # M3: dV fixed to VIS_FAST's measured TRISC cut (t608 profile, slowest-core TRISC_1 busy
    # 1.391 -> 1.340 alone, 1.376 -> 1.337 on top of RECFAST): 0.04 and 0.05.
    only = sys.argv[2:]  # optional subset of model names, e.g. M3
    Fgrid = np.arange(1.20, 1.53, 0.01)
    for name, Fs, dVs in (("M1 (no floor)", [0.0], None), ("M2 (window floor F)", Fgrid, None),
                          ("M3 (floor F, dV 0.04 measured)", Fgrid, [0.04]),
                          ("M3 (floor F, dV 0.05 measured)", Fgrid, [0.05])):
        if only and name.split()[0] not in only:
            continue
        e, L, o, dV, F = fit2(Fs, dVs)
        T = T0 + o
        w0, gr, gv, gb = arms(L, o, dV, F)
        print(f"\n== {name}: L {L:+.4f} o {o:+.4f} dV {dV:.3f} F {F:.3f}  sum|err| {e:.3f}")
        print(f"   fit: base {w0:.3f}  rec {gr:+.3f}  vis {gv:+.3f}  bundle {gb:+.3f}")
        TA = T * (0.39 + 1.34 * np.minimum(1.0, s * 1.05)) / 1.68
        wR = window(T, bR, nR, L, F)
        print("   A' = RECFAST + compaction (split movers): gain vs best-iter-227 | incremental over RECFAST")
        for rc in (0, 30, 40, 50):
            bA = np.array([busy(p, REC_NEW, rc or 1e-9, 0.5, sc, q) for p, q, sc in zip(Bp, Np, s)])
            nA = np.array([busy(p, REC_NEW, rc or 1e-9, 0.5, sc, q) for p, q, sc in zip(Np, Bp, s)])
            wA = window(TA, bA, nA, L, F)
            print(f"     rep {rc:3d} cyc/surv: {w0 - wA:+.3f} | {wR - wA:+.3f}")


if __name__ == "__main__":
    main()
