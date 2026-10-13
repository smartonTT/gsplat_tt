#!/usr/bin/env python3
"""t603: pfwc window model after RECFAST (t599), CPU only, no device.

python3 docs/structural-plan-4/model603.py <t484 out/pfwc-sr.csv.gz>

Reuses the t598 per-core-launch model (docs/pfwc-mover-t598/model598.py):
    end = max(T + tail, max(busy_B, busy_N) + lead),  window = max core end per launch.
t598 projected iter 227 with a fitted lead (0.245) that overpredicted the measured RECFAST gain
(model +0.198 at 70 cyc/survivor, measured -0.071 ms/view b2b, t599). Here two parameters are
calibrated on iter-227 measurements instead:
  - TRISC offset o (T227 = (T_pure - ACQ) * F_PRUNE + o) so that the base window = 1.523 ms
    (t595 measured pfwc window, best-iter-227, bh-33),
  - lead L so that RECFAST (rec at 67 cyc/survivor, t599 microbench) gains 0.071 ms (t599 A/B).
Then, on top of RECFAST, it sweeps a TRISC cut dT (any TRISC-side pfwc lever: VIS_FAST, spec A)
and, for spec A (survivor compaction), adds the repack cost on the movers.
"""
import os, sys
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "pfwc-mover-t598"))
import model598 as M

C = 1350.0
ms = lambda c: c / C / 1000.0
BASE_227 = 1.523   # t595 pfwc window
RECFAST_GAIN = 0.071  # t599 b2b A/B
REC_NEW = 67.0     # t599 v1 cyc/survivor
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
    uv = np.unique(runs)

    def busy(p, rec_cyc=None, rep_cyc=0.0, rep_share=0.5, s_core=None, other=None):
        cls = p["cls"] * M.F_PRUNE
        rec = p["rec"]
        if rec_cyc is not None:
            rec = rec - (p["rec"] - p["fl"]) + ms(rec_cyc * p["m"])
        extra = 0.0
        if rep_cyc:
            # spec A: classify runs on dense survivor lanes only (cls x s), and the repack moves
            # every survivor of the core (both movers' m) once; rep_share of it lands on this mover.
            m_core = p["m"] + other["m"]
            extra = ms(rep_cyc * m_core) * rep_share - cls * (1 - s_core)
        return cls + p["pfx"] + rec + p["opn"] + p["tail"] + extra

    # survival per core-launch: survivors / lanes; lanes = chunks x 1024 (n = chunks per mover)
    s = np.array([(b["m"] + n["m"]) / ((b["n"] + n["n"]) * LANES_PER_CHUNK) for b, n in zip(Bp, Np)])

    def window(T, bb, bn, L, tl):
        pred = np.maximum(T + tl, np.maximum(bb, bn) + L)
        return float(np.mean([pred[runs == v].max() for v in uv]))

    T0 = (T_pure - M.ACQ) * M.F_PRUNE
    tl = 0.045
    b0 = np.array([busy(p) for p in Bp]); n0 = np.array([busy(p) for p in Np])
    bR = np.array([busy(p, REC_NEW) for p in Bp]); nR = np.array([busy(p, REC_NEW) for p in Np])
    best = None
    for L in np.arange(-0.30, 0.30, 0.0025):
        for o in np.arange(-0.30, 0.30, 0.0025):
            w0 = window(T0 + o, b0, n0, L, tl)
            wR = window(T0 + o, bR, nR, L, tl)
            e = abs(w0 - BASE_227) + abs((w0 - wR) - RECFAST_GAIN)
            if best is None or e < best[0]:
                best = (e, L, o, w0, wR)
    e, L, o, w0, wR = best
    T = T0 + o
    print(f"core-launches {len(rows)}; survival per core mean {s.mean():.3f} p95 {np.percentile(s, 95):.3f} max {s.max():.3f}")
    print(f"calibrated: lead {L:+.4f} ms, TRISC offset {o:+.4f} ms (slowest-core T {T.max():.3f}), "
          f"base {w0:.3f} (meas {BASE_227}), RECFAST gain {w0 - wR:.3f} (meas {RECFAST_GAIN})")
    print(f"mover busy max after RECFAST: B {bR.max():.3f} N {nR.max():.3f}; before: B {b0.max():.3f} N {n0.max():.3f}")
    print(f"mover floor after RECFAST (TRISC free): {window(T * 0 - 10, bR, nR, L, tl):.3f}")

    print("\n== TRISC cut dT on top of RECFAST (gain vs best-iter-227 base, RECFAST included)")
    print("   dT    no RECFAST   with RECFAST")
    for dT in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        g0 = w0 - window(T - dT, b0, n0, L, tl)
        g1 = w0 - window(T - dT, bR, nR, L, tl)
        print(f"  {dT:4.2f}   {g0:+.3f}       {g1:+.3f}")

    # spec A: TRISC per core = fixed 0.39/1.68 share (projection + gate) + survival-scaled 1.34/1.68
    # (cov_cam, cov2d, conic, vis+precull) of today's T (t484 split, t596). Phase-2 runs on dense
    # tiles, so the scaled part uses ceil-to-tile survival; the worst core's tiles are 0.377-0.46
    # of today's (t596 gate); use s_core x 1.05 for partial-tile padding.
    TA = T * (0.39 + 1.34 * np.minimum(1.0, s * 1.05)) / 1.68
    print(f"\n== spec A on top of RECFAST: TRISC slowest core {T.max():.3f} -> {TA.max():.3f}")
    print("   rep cyc/surv   NCRISC only   split both movers")
    for rc in (0, 20, 30, 40, 50, 67, 95):
        out = []
        for share_b, share_n in ((0.0, 1.0), (0.5, 0.5)):
            bA = np.array([busy(p, REC_NEW, rc, share_b, sc, q) if rc else busy(p, REC_NEW, 1e-9, share_b, sc, q)
                           for p, q, sc in zip(Bp, Np, s)])
            nA = np.array([busy(p, REC_NEW, rc, share_n, sc, q) if rc else busy(p, REC_NEW, 1e-9, share_n, sc, q)
                           for p, q, sc in zip(Np, Bp, s)])
            out.append(w0 - window(TA, bA, nA, L, tl))
        print(f"   {rc:5d}          {out[0]:+.3f}        {out[1]:+.3f}")

    # spec A after a RECFAST + VIS_FAST bundle (TRISC already cut by dV in the survival-scaled
    # part): incremental gain of A over the bundle.
    for dV in (0.10, 0.15, 0.20):
        bund = window(T - dV, bR, nR, L, tl)
        TA2 = T * 0.39 / 1.68 + (T * 1.34 / 1.68 - dV) * np.minimum(1.0, s * 1.05)
        print(f"\n== spec A over a RECFAST + TRISC-cut {dV:.2f} bundle (bundle window {bund:.3f}, "
              f"TRISC slowest {TA2.max():.3f}): incremental gain")
        for rc in (0, 30, 40, 50, 67):
            row = []
            for share_b, share_n in ((0.0, 1.0), (0.5, 0.5)):
                bA = np.array([busy(p, REC_NEW, rc or 1e-9, share_b, sc, q) for p, q, sc in zip(Bp, Np, s)])
                nA = np.array([busy(p, REC_NEW, rc or 1e-9, share_n, sc, q) for p, q, sc in zip(Np, Bp, s)])
                row.append(bund - window(TA2, bA, nA, L, tl))
            print(f"   rep {rc:3d} cyc/surv: NCRISC only {row[0]:+.3f}   split {row[1]:+.3f}")


if __name__ == "__main__":
    main()
