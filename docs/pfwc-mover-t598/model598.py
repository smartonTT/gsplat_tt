#!/usr/bin/env python3
"""t598: pfwc mover-floor model (t484 lever 3), CPU only, no device.

python3 docs/pfwc-mover-t598/model598.py <t484 out/pfwc-sr.csv.gz> [<out/pfwc-s.csv.gz>]

Per core-launch of the t484 STEPCYC capture (iter 222 + SKIP_RGB, p100 11x10, 110 cores, 60
launches) it takes each RISC's measured parts and models the core end as

    end = max(T + tail_B, T + tail_N, busy_B + lead, busy_N + lead)

T = TRISC compute without writer back-pressure (uniform across cores, t484), busy = mover
cls+pfx+rec+opn+tail, lead = first-chunk delay, tail = the last chunk's mover work after TRISC.
lead and the tails are fitted on the measured core ends. Window = max core end per launch.

It then projects to iter 227 (ACQ_FUSE=3 cut TRISC; the B3 opacity prune cut N to 0.907, so T
and classify scale by 0.907, survivors do not) and applies the two parts of the lever:
  (a) batch the record/page writes: the flush time fl goes to 0 (upper bound) or halves;
  (b) deal chunks by running survivor count: both movers get the core's mean busy (upper
      bound, chunk granularity ~1/27 of a mover's rec ignored).
"""
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import importlib.util

C = 1350.0
ms = lambda c: c / C / 1000.0
F_PRUNE = 5559909 / 6131954  # B3 prune, iter 225
ACQ = 0.106  # iter 223 ACQ_FUSE b2b gain (t489), taken as the TRISC cut


def load_ana(path):
    spec = importlib.util.spec_from_file_location("ana", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def core_rows(ana, csv):
    raw, ts = ana.load(csv)
    runs = ana.launches(raw)
    ws = ana.windows(raw, runs)
    rows = []
    for v, t0, rs in ws:
        for c, d in rs.items():
            end = ms(max(e for _, e in d.values()) - t0)
            pc = ts.get((c, "TRISC_1", "pfwc_pc"), [])
            b = ts.get((c, "BRISC", "pfwc_ws"), [])
            n = ts.get((c, "NCRISC", "pfwc_ws"), [])
            if len(pc) <= v or len(b) <= v or len(n) <= v:
                continue
            P = [pc[v].get(i, 0) for i in range(len(ana.PC))]
            B = [b[v].get(i, 0) for i in range(len(ana.WS))]
            N = [n[v].get(i, 0) for i in range(len(ana.WS))]
            rows.append((v, c, end, P, B, N))
    return rows


def parts(W):  # WS = n wall wait cls pfx rec opn tail rd fl m
    d = dict(zip("n wall wait cls pfx rec opn tail rd fl m".split(), W))
    return {k: (ms(x) if k not in ("n", "m") else x) for k, x in d.items()}


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    ana = load_ana(os.environ.get("ANA", os.path.join(here, "ana.py")))
    rows = core_rows(ana, sys.argv[1])
    T1 = np.array([ms(P[1]) for *_, P, B, N in rows])
    # TRISC compute without back-pressure: TRISC_1 wall on cores whose step-4 (means) excess is small.
    means = np.array([ms(P[7]) for *_, P, B, N in rows])
    floor = np.percentile(means, 5)
    T_pure = T1 - (means - floor)
    Tc = float(np.median(T_pure))
    print(f"core-launches {len(rows)}; TRISC_1 wall mean {T1.mean():.3f}; minus back-pressure: median {Tc:.3f} "
          f"p5 {np.percentile(T_pure, 5):.3f} p95 {np.percentile(T_pure, 95):.3f}")
    Bp = [parts(B) for *_, B, N in rows]
    Np = [parts(N) for *_, B, N in rows]
    busy = lambda p: p["cls"] + p["pfx"] + p["rec"] + p["opn"] + p["tail"]
    bB = np.array([busy(p) for p in Bp]); bN = np.array([busy(p) for p in Np])
    mB = np.array([p["m"] for p in Bp]); mN = np.array([p["m"] for p in Np])
    nB = np.array([p["n"] for p in Bp]); nN = np.array([p["n"] for p in Np])
    end = np.array([r[2] for r in rows]); runs = np.array([r[0] for r in rows])
    print(f"mover busy mean B {bB.mean():.3f} N {bN.mean():.3f}; max B {bB.max():.3f} N {bN.max():.3f}")

    # Fit lead and tail per survivor of the last chunk: end ~ max(T + tail, busy + lead).
    best = None
    for lead in np.arange(0.0, 0.25, 0.005):
        for tl in np.arange(0.0, 0.10, 0.005):
            pred = np.maximum(T_pure + tl, np.maximum(bB, bN) + lead)
            err = np.abs(pred - end).mean()
            if best is None or err < best[0]:
                best = (err, lead, tl)
    err, lead, tl = best
    import os
    if os.environ.get("LEAD"):  # sensitivity: fix lead, refit the TRISC tail
        lead = float(os.environ["LEAD"])
        tl = min(np.arange(0.0, 0.10, 0.005), key=lambda t: np.abs(np.maximum(T_pure + t, np.maximum(bB, bN) + lead) - end).mean())
        err = np.abs(np.maximum(T_pure + tl, np.maximum(bB, bN) + lead) - end).mean()
    print(f"fit: lead {lead:.3f} ms, TRISC tail {tl:.3f} ms, mean |err| {err:.4f} ms")

    def window(T, bb, bn):
        pred = np.maximum(T + tl, np.maximum(bb, bn) + lead)
        return np.mean([pred[runs == v].max() for v in np.unique(runs)]), pred

    meas = np.mean([end[runs == v].max() for v in np.unique(runs)])
    w0, _ = window(T_pure, bB, bN)
    print(f"measured window {meas:.3f}; model window {w0:.3f}")

    def scen(label, T, bb, bn, ref=None):
        w, pred = window(T, bb, bn)
        mover = np.maximum(bb, bn) + lead > T + tl
        s = f"  {label:58s} window {w:.3f}"
        if ref is not None:
            s += f"  gain {ref - w:+.3f}"
        s += f"  mover-bound cores {100 * mover.mean():.0f}%"
        print(s)
        return w

    def proj(p, fl_scale=1.0, rec_cyc=None):
        cls = p["cls"] * F_PRUNE
        rec = p["rec"] - (1 - fl_scale) * p["fl"]
        if rec_cyc is not None:  # rec minus flush at rec_cyc cycles per survivor
            rec = rec - (p["rec"] - p["fl"]) + ms(rec_cyc * p["m"])
        return cls + p["pfx"] + rec + p["opn"] + p["tail"]

    out = {}
    for tag, dT in (("iter 222 + SKIP_RGB (measured base)", None), ("iter 227 projection", ACQ)):
        print(f"\n== {tag}")
        if dT is None:
            T = T_pure
            fB = lambda **k: np.array([busy(p) - (1 - k.get("fl_scale", 1)) * p["fl"] for p in Bp])
            fN = lambda **k: np.array([busy(p) - (1 - k.get("fl_scale", 1)) * p["fl"] for p in Np])
        else:
            T = (T_pure - dT) * F_PRUNE
            fB = lambda **k: np.array([proj(p, **k) for p in Bp])
            fN = lambda **k: np.array([proj(p, **k) for p in Np])
        b, n = fB(), fN()
        ref = scen("base", T, b, n)
        scen("TRISC free (mover floor only)", T * 0, b, n, ref)
        scen("(a) batch writes, fl x0.5", T, fB(fl_scale=0.5), fN(fl_scale=0.5), ref)
        scen("(a) batch writes, fl -> 0 (upper bound)", T, fB(fl_scale=0.0), fN(fl_scale=0.0), ref)
        avg = (b + n) / 2
        scen("(b) survivor-count deal, perfect balance (upper bound)", T, avg, avg, ref)
        b0, n0 = fB(fl_scale=0.0), fN(fl_scale=0.0)
        a0 = (b0 + n0) / 2
        out[tag] = scen("(a)+(b) upper bounds together", T, a0, a0, ref)
        for cyc in (90, 70):
            bc, nc = fB(rec_cyc=cyc), fN(rec_cyc=cyc)
            scen(f"(ref) rec at {cyc} cyc/survivor (not this lever)", T, bc, nc, ref)
        print(f"  slowest-core TRISC {np.max(T):.3f}, mover busy max B {b.max():.3f} N {n.max():.3f}, "
              f"imbalance |B-N| mean {np.abs(b - n).mean():.3f} p95 {np.percentile(np.abs(b - n), 95):.3f}")


if __name__ == "__main__":
    main()
