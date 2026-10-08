#!/usr/bin/env python3
"""t428 copy of the t419 model (branch ttp/t419-blend-interleave-model), with T419_SRC / T419_CHUNKS /
T419_PREFIX / T419_CACHE env overrides so it can run on other captures; the model itself is unchanged.

t419: CPU model of interleaving blend into TRISC1's mat between-job gaps (lever L1a).

Input: the #413 Tracy captures (docs/mat-job-zones-t413/out/dev-c{0,10,20}.csv.gz, p100a,
GSPLAT_TT_MATCULL_PROF=1). Each chunk has 11 frames; frame 0 repeats frame 1's camera (the
stitch drops it), so frames 1..10 of each chunk = the 30 views.

Per view it rebuilds, per core: the mat window and the TRISC1 mj_job intervals (gaps), the blend
window (tile_blend_sfpu) and the NCRISC rd_l1_bulk slab loads. From those it infers:
  * slab ready time = end of the mj_job that culled it + WR_LAG (mover write + ready flag);
  * per-slab blend cost on the TRISCs: the blend reader keeps 2 slabs in flight (CB_BUCKET_BULK has
    2 slots), so the end of rd_l1_bulk k+2 is the end of compute of slab k; the last two slabs
    split the rest of the blend window. Costs are matched to slabs by rank of mj_job length
    ("rank" map: bigger mat job -> bigger blend) or at random ("rand" map);
  * records per slab from mj_job length (fz_mv_recs calibrates cycles per record).

Then it simulates the mat+blend phase:
  * baseline: each core blends after its mat ends; the cores claim slabs from one queue in
    descending cost order (task #60 dynamic claim), a claim waits for the slab's ready flag;
    frame end = latest finish (the slowest core);
  * mover speedup s: start and between gaps scaled by s (s = 1.0 today, 0.7, 0.5); mat end,
    job times and ready times move with them;
  * fill policies, applied in each between gap (the start gap holds no ready slabs of this
    view's own core, and in practice none at all: see README):
      fit  - whole slabs (= whole single-slab tiles), non-preemptive, oracle best fit: take the
             largest ready slab whose cost fits the rest of the gap; slab size <= the L1 slot cap;
      pre  - sub-slab (128-record batch) granularity, preemptible: DEST blend state spilled to
             L1 and restored at each yield; fills max(0, gap - SW - SPILL) of each gap from a
             fluid pool (upper bound for any interleave);
    each filled slab adds mover time to that core's mat (NCRISC reader MU_N per slab; BRISC
    writer MU_B per tile unless the u8 pack/write is deferred to after mat), which pushes the
    core's mat end and its later ready times out;
  * post-mat: the remaining slabs are list-scheduled exactly like the baseline.
gain = baseline end - interleave end at the same s (ms/view, mean over the 30 views).

Usage: model.py [--out DIR]
"""
import argparse
import bisect
import os
import pickle
import random
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "docs", "xvpin-tracy"))

CYC_US = 1350.0          # p100a capture, 1350 MHz
SRC = os.environ.get("T419_SRC", os.path.join(ROOT, "docs", "mat-job-zones-t413", "out"))
CHUNKS = os.environ.get("T419_CHUNKS", "c0,c10,c20").split(",")
PREFIX = os.environ.get("T419_PREFIX", "dev-")
WR_LAG = 3.0             # us: mat_ol_wr median 2.0 us + ready flag write
SW = 2.0                 # us per switch mat<->blend (LLK re-init of unpack/math/pack), each way
SPILL = 4.0              # us per yield for the DEST spill+restore (t311: 2-5 us)
MU_N = 5.0               # us NCRISC per interleaved slab: ready poll, meta/dir reads, bulk DMA
MU_B = 30.0              # us BRISC per interleaved tile: img_pack_u8 (3x1024 px scalar) + 32 row writes
BUCKET_FIT = 8192        # records per subchunk


def us(c):
    return c / CYC_US


def load():
    cache = os.environ.get("T419_CACHE", os.path.join("/tmp", "t419_views.pkl"))
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    import ana407 as A
    out = {}
    for ch in CHUNKS:
        iv, td = A.load_dev(os.path.join(SRC, f"{PREFIX}{ch}.csv.gz"))
        _, Z, D, NV, cores = A.assign(iv, td)
        for v in range(NV):
            rec = {}
            for c in cores:
                g = lambda z, r: sorted((s, e) for cc, rr, s, e in Z[(v, 3, z)] if cc == c and rr == r)
                rec[c] = dict(mat=g("mat_cull_mask", "TRISC_1"), jobs=g("mj_job", "TRISC_1"),
                              blend=g("tile_blend_sfpu", "TRISC_1"), bulk=g("rd_l1_bulk", "NCRISC"),
                              fz={k[2]: val for k, val in D[v].items() if k[0] == c})
            out[(ch, v)] = rec
    pickle.dump(out, open(cache, "wb"))
    return out


def view_model(rec):
    """-> per-core dicts in us relative to the earliest mat start, plus the slab list."""
    cs = [c for c in rec if rec[c]["mat"] and rec[c]["jobs"] and rec[c]["blend"] and rec[c]["bulk"]]
    t0 = min(rec[c]["mat"][0][0] for c in cs)
    cores, costs, jobs = {}, [], []
    for c in cs:
        r = rec[c]
        ms0, me0 = (us(x - t0) for x in r["mat"][0])
        J = [(us(s - t0), us(e - t0)) for s, e in r["jobs"]]
        B0, B1 = (us(x - t0) for x in r["blend"][0])
        b = [us(e - t0) for s, e in r["bulk"]]
        n = len(b)
        ce = [b[k + 2] for k in range(n - 2)]
        if n >= 2:
            last = ce[-1] if ce else B0
            ce += [last + (B1 - last) / 2, B1]
        else:
            ce = [B1]
        prev = b[0]
        for x in ce:
            costs.append(max(1.0, x - prev))
            prev = x
        cores[c] = dict(ms=ms0, me=me0, J=J, B0=B0, B1=B1)
        for s, e in J:
            jobs.append((c, s, e))
    recs = sum(rec[c]["fz"].get("fz_mv_recs", 0) for c in cs)
    jt = sum(e - s for _, s, e in jobs)
    return cores, sorted(costs), jobs, recs / jt if jt else 0.0


def shifted(core, s):
    """job intervals and mat end with start/between gaps scaled by s"""
    J = core["J"]
    t = core["ms"] + s * (J[0][0] - core["ms"])
    out, gaps = [], []
    for i, (a, e) in enumerate(J):
        if i:
            g = s * (a - J[i - 1][1])
            gaps.append((t, t + g))
            t += g
        out.append((t, t + (e - a)))
        t += e - a
    me = t + (core["me"] - J[-1][1])
    return out, gaps, me


def simulate(cores, costs, jobs, rpu, s, policy, cap, mapping, defer_wr, rng):
    """-> (frame end us, per-core fill us mean, filled slabs per core mean, mover delay per core mean)"""
    # slab table: (core, job index) -> cost, records, ready time
    order = sorted(range(len(jobs)), key=lambda k: jobs[k][2] - jobs[k][1])
    cst = list(costs)
    if mapping == "rand":
        rng.shuffle(cst)
    cost_of = {}
    for rank, k in enumerate(order):
        cost_of[k] = cst[rank] if mapping == "rank" else cst[k]
    sh = {c: shifted(cores[c], s) for c in cores}
    idx = defaultdict(int)
    slabs = []  # [cost, recs, ready, core, jidx]
    for k, (c, a, e) in enumerate(jobs):
        j = idx[c]
        idx[c] += 1
        dur = e - a
        slabs.append([cost_of[k], dur * rpu, sh[c][0][j][1] + WR_LAG, c, j])
    taken = [False] * len(slabs)
    fill = defaultdict(float)
    nfill = defaultdict(int)
    delay = defaultdict(float)
    if policy == "fit":
        # gaps in global time order; pool = ready, unclaimed, size <= cap
        G = sorted((a, b, c) for c in cores for a, b in sh[c][1])
        byready = sorted(range(len(slabs)), key=lambda k: slabs[k][2])
        for a, b, c in G:
            t = a + SW
            while True:
                room = b - t - SW
                if room <= 0:
                    break
                best = None
                for k in byready:
                    if slabs[k][2] > t:
                        break
                    if taken[k] or slabs[k][1] > cap or slabs[k][0] > room:
                        continue
                    if best is None or slabs[k][0] > slabs[best][0]:
                        best = k
                if best is None:
                    break
                taken[best] = True
                t += slabs[best][0]
                fill[c] += slabs[best][0]
                nfill[c] += 1
        for c in cores:
            delay[c] = max(nfill[c] * MU_N, 0.0 if defer_wr else nfill[c] * MU_B)
    elif policy == "pre":
        # fluid: every between gap >= the overheads is filled, from the biggest-remaining slabs
        # (the pool is never empty after the first jobs end; see README)
        tot = 0.0
        for c in cores:
            for a, b in sh[c][1]:
                f = (b - a) - 2 * SW - SPILL
                if f > 0:
                    fill[c] += f
                    nfill[c] += 1
            tot += fill[c]
        # remove the filled work from the pool: take it from the smallest slabs that are ready
        # by mid-mat (the scheduler would pick any ready slab; small-first keeps the post-mat LPT
        # tail intact for the big ones -- the best case for the tail)
        for k in sorted(range(len(slabs)), key=lambda k: slabs[k][0]):
            if tot <= 0:
                break
            take = min(tot, slabs[k][0])
            slabs[k][0] -= take
            tot -= take
            if slabs[k][0] <= 1e-6:
                taken[k] = True
        for c in cores:
            # NCRISC reloads a partly blended slab after each yield; BRISC writes the finished tiles
            delay[c] = max(nfill[c] * MU_N, 0.0 if defer_wr else (fill[c] / 362.0) * MU_B)
    # post-mat list scheduling: descending cost, claim waits for ready
    free = sorted((sh[c][2] + delay[c], c) for c in cores)
    import heapq
    heapq.heapify(free)
    rest = sorted((k for k in range(len(slabs)) if not taken[k]), key=lambda k: -slabs[k][0])
    end = 0.0
    for k in rest:
        t, c = heapq.heappop(free)
        t = max(t, slabs[k][2] + (delay[slabs[k][3]] if slabs[k][4] >= 0 else 0))
        t += slabs[k][0]
        end = max(end, t)
        heapq.heappush(free, (t, c))
    end = max(end, max(t for t, _ in free))
    n = len(cores)
    return end, sum(fill.values()) / n, sum(nfill.values()) / n, sum(delay.values()) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load()
    keys = [k for k in sorted(d) if k[1] != 0]
    views = [view_model(d[k]) for k in keys]
    rng = random.Random(419)
    lines = []
    P = lambda *x: lines.append(" ".join(str(y) for y in x))

    # calibration and inputs
    meas = [max(c["B1"] for c in v[0].values()) for v in views]
    base = [simulate(*v, 1.0, "none", 0, "rank", True, rng)[0] for v in views]
    P(f"views {len(views)}, cores/view {np.mean([len(v[0]) for v in views]):.0f}, "
      f"slabs/view {np.mean([len(v[1]) for v in views]):.0f}")
    P(f"calibration: measured slowest-core blend end {np.mean(meas) / 1000:.3f} ms, "
      f"simulated baseline {np.mean(base) / 1000:.3f} ms (diff {np.mean(np.array(base) - meas) / 1000:+.3f})")
    gaps = np.array([b - a for v in views for c in v[0].values() for a, b in shifted(c, 1.0)[1]])
    costs = np.concatenate([v[1] for v in views])
    P("between gaps us pct 10/50/90/99/max:", np.percentile(gaps, [10, 50, 90, 99, 100]).round(0))
    P("blend cost per slab us pct 1/10/25/50/90:", np.percentile(costs, [1, 10, 25, 50, 90]).round(0),
      f"mean {costs.mean():.0f}")
    P(f"share of between-gap time in gaps >= the p10 slab ({np.percentile(costs, 10):.0f} us): "
      f"{gaps[gaps >= np.percentile(costs, 10)].sum() / gaps.sum():.2f}")
    rpu = np.mean([v[3] for v in views])
    P(f"records per mj_job us (fz_mv_recs / mj_job time): {rpu:.1f}")
    P("")
    hdr = f"{'case':<46}{'s':>5}{'mover_only':>11}{'gain':>8}{'fill/core':>10}{'slabs':>7}{'mvdelay':>8}"
    P("gain = baseline end - interleave end at the same s, ms/view (mean of 30 views);")
    P("mover_only = gain of the gap cut alone vs s=1.0 baseline; fill/core, mvdelay in us")
    P(hdr)
    rows = []
    cases = [
        ("fit whole slabs, cap 1024 rec (32 KB slot), wr now", "fit", 1024, "rank", False),
        ("fit whole slabs, cap 1024 rec, wr deferred", "fit", 1024, "rank", True),
        ("fit whole slabs, cap 4096 rec (128 KB slot), wr deferred", "fit", 4096, "rank", True),
        ("fit whole slabs, no cap, wr deferred", "fit", 10 ** 9, "rank", True),
        ("fit whole slabs, no cap, wr deferred, rand map", "fit", 10 ** 9, "rand", True),
        ("pre sub-slab + DEST spill, wr now", "pre", 10 ** 9, "rank", False),
        ("pre sub-slab + DEST spill, wr deferred", "pre", 10 ** 9, "rank", True),
    ]
    for s in (1.0, 0.7, 0.5):
        b_s = [simulate(*v, s, "none", 0, "rank", True, rng)[0] for v in views]
        mover_only = (np.mean(base) - np.mean(b_s)) / 1000
        for name, pol, cap, mp, dw in cases:
            r = [simulate(*v, s, pol, cap, mp, dw, rng) for v in views]
            g = (np.mean(b_s) - np.mean([x[0] for x in r])) / 1000
            fl = np.mean([x[1] for x in r])
            nf = np.mean([x[2] for x in r])
            dl = np.mean([x[3] for x in r])
            P(f"{name:<46}{s:5.1f}{mover_only:11.3f}{g:8.3f}{fl:10.0f}{nf:7.2f}{dl:8.0f}")
            rows.append((name, s, mover_only, g, fl, nf, dl))
        P("")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.out, "model.txt"), "w").write(txt + "\n")
    with open(os.path.join(a.out, "gain-table.csv"), "w") as f:
        f.write("case,s,mover_only_ms,gain_ms,fill_us_per_core,slabs_per_core,mover_delay_us\n")
        for r in rows:
            f.write(f"\"{r[0]}\",{r[1]},{r[2]:.3f},{r[3]:.3f},{r[4]:.0f},{r[5]:.2f},{r[6]:.0f}\n")


if __name__ == "__main__":
    main()
