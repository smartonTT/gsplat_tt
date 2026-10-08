#!/usr/bin/env python3
"""t407/t412: xvpin Tracy analysis of the stitched 30-view bh-30 capture.

Device (stitched profile_log_device.csv, 1350 cyc/us): every core runs 4 BRISC-FW programs
per view (0 pfwc, 1 k2, 2 sort, 3 mat+blend). A FW interval of any RISC belongs to the
BRISC-FW program on the same core it overlaps most; a zone or fz_* value belongs to the FW
interval of its own RISC that contains it. Views are the BRISC-FW index // 4 (30 views,
3 chunks of 10; the chunks have their own time base, so view N -> N+1 is only measured
inside a chunk: 27 transitions).

Reports: per-view program and zone windows; device idle (gaps in the union of all FW
intervals of all cores, and per-core gaps in each core's own FW union), with the
emit end -> mat start and mat+blend N end -> pfwc N+1 windows in detail; program overlap;
per-core mat / blend / mover / writer busy; mover sort/permute from fz_mv_*; (a) the TRISC1
idle-in-mat bound from fz_m_*; host zones from the tracy -u dumps; the reconciliation of the
untraced stage means with the traced device span.

Usage: ana407.py STITCHED.csv[.gz] [--out DIR]   (DIR holds r*.log, T*.log, tracy-u-c*.csv.gz)
"""
import argparse
import bisect
import csv
import glob
import gzip
import os
import re
from collections import defaultdict

import numpy as np

CYC_MS = 1350.0 * 1000.0
PROGS = ["pfwc", "k2", "sort", "mat+blend"]
FW = {"BRISC-FW", "NCRISC-FW", "TRISC-FW"}
RISCS = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]
VIEWS_PER_CHUNK = 10
TILES = 1024


def ms(c):
    return c / CYC_MS


def opener(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p)


def load_dev(path):
    """-> iv[(core, risc)] = [(zone, s, e)], td[(core, risc)] = [(name, ts, value)]"""
    iv, td, stack = defaultdict(list), defaultdict(list), {}
    with opener(path) as f:
        next(f)
        next(f)
        for p in csv.reader(f):
            if len(p) < 12:
                continue
            c, r, t, z, ty = (int(p[1]), int(p[2])), p[3].strip(), int(p[5]), p[10].strip(), p[11].strip()
            if ty == "ZONE_START":
                stack[(c, r, z)] = t
            elif ty == "ZONE_END":
                s = stack.pop((c, r, z), None)
                if s is not None:
                    iv[(c, r)].append((z, s, t))
            elif z.startswith("fz_"):
                td[(c, r)].append((z, t, int(p[6])))
    return iv, td


def assign(iv, td):
    """-> FWI[(v, p)] = [(core, risc, s, e)], Z[(v, p, zone)] = [(core, risc, s, e)],
    D[v][(core, risc, name)] = value, nviews, cores"""
    cores = sorted({c for c, _ in iv})
    FWI, Z, D = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(int))
    nviews = None
    for c in cores:
        b = sorted((s, e) for z, s, e in iv[(c, "BRISC")] if z == "BRISC-FW")
        assert len(b) % 4 == 0, (c, len(b))
        nviews = len(b) // 4 if nviews is None else nviews
        assert len(b) // 4 == nviews, (c, len(b))
        bs = [s for s, _ in b]
        for r in RISCS:
            fw = sorted((s, e) for z, s, e in iv.get((c, r), []) if z in FW)
            tags = []
            for s, e in fw:
                k0 = bisect.bisect_right(bs, s) - 1
                cand = [k for k in (k0 - 1, k0, k0 + 1) if 0 <= k < len(b)]
                k = max(cand, key=lambda k: (min(e, b[k][1]) - max(s, b[k][0]), -abs(b[k][0] - s)))
                tags.append(divmod(k, 4))
                FWI[divmod(k, 4)].append((c, r, s, e))
            starts = [s for s, _ in fw]
            for z, s, e in iv.get((c, r), []):
                if z in FW:
                    continue
                i = bisect.bisect_right(starts, s) - 1
                assert i >= 0 and e <= fw[i][1], (c, r, z)
                v, p = tags[i]
                Z[(v, p, z)].append((c, r, s, e))
            for name, t, val in td.get((c, r), []):
                i = bisect.bisect_right(starts, t) - 1
                v, p = tags[i]
                D[v][(c, r, name)] += val
    return FWI, Z, D, nviews, cores


def union(iv):
    """sorted merged [(s, e)] of [(s, e)]"""
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def idle_in(u, a, b):
    """cycles in [a, b] not covered by the merged union u"""
    if b <= a:
        return 0
    cov = sum(max(0, min(e, b) - max(s, a)) for s, e in u)
    return (b - a) - cov


def st(x):
    x = np.asarray(x, float)
    return x.mean(), x.min(), x.max()


def hdr(t):
    print(f"\n## {t}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--zone-csv", help="write per-view zone windows here")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--job-csv", help="t413: write the per-core TRISC1 idle gaps (per view) here")
    ap.add_argument("--mhz", type=float, default=1350.0, help="device clock (aiclk) in MHz")
    a = ap.parse_args()
    global CYC_MS
    CYC_MS = a.mhz * 1000.0
    iv, td = load_dev(a.csv)
    FWI, Z, D, NV, cores = assign(iv, td)
    NC = len(cores)
    chunk = lambda v: v // VIEWS_PER_CHUNK
    nxt = [v for v in range(NV - 1) if chunk(v) == chunk(v + 1)]
    print(f"# ana407: {a.csv}\nviews={NV} cores={NC} transitions inside chunks={len(nxt)}; ms at {a.mhz:.0f} MHz")

    # zone -> program check
    zp = defaultdict(set)
    for (v, p, z) in Z:
        zp[z].add(PROGS[p])
    print("\nzone -> program (from FW containment): " + "; ".join(
        f"{z}: {'/'.join(sorted(ps))}" for z, ps in sorted(zp.items()) if not z.endswith("KERNEL")))
    trisc_progs = sorted({PROGS[p] for (v, p), L in FWI.items() for c, r, s, e in L if r.startswith("TRISC")})
    print(f"TRISC-FW programs: {', '.join(trisc_progs)}")

    def win(L):
        return min(x[2] for x in L), max(x[3] for x in L)

    PW = {(v, p): win(FWI[(v, p)]) for v in range(NV) for p in range(4)}
    V0 = [PW[(v, 0)][0] for v in range(NV)]

    def core_len(v, p):
        d = defaultdict(lambda: [1 << 62, 0])
        for c, r, s, e in FWI[(v, p)]:
            d[c][0] = min(d[c][0], s)
            d[c][1] = max(d[c][1], e)
        return d

    # ---- per-view program windows
    hdr("Per-view program windows (ms from the view's first pfwc FW start; window = first start on any core -> last end on any core)")
    print(f"{'view':>4} " + " ".join(f"{n + ' s/len':>16}" for n in PROGS) + f" {'span':>7} {'period':>7}")
    rows = []
    for v in range(NV):
        cells, r = [], []
        for p in range(4):
            s, e = PW[(v, p)]
            cells.append(f"{ms(s - V0[v]):7.3f}/{ms(e - s):7.3f}")
            r += [ms(s - V0[v]), ms(e - s)]
        span = ms(PW[(v, 3)][1] - V0[v])
        per = ms(V0[v + 1] - V0[v]) if v in nxt else float("nan")
        rows.append(r + [span, per])
        print(f"{v:4d} " + " ".join(f"{x:>16}" for x in cells) + f" {span:7.3f} {per:7.3f}")
    R = np.array(rows)
    print(f"mean " + " ".join(f"{R[:, 2 * p].mean():7.3f}/{R[:, 2 * p + 1].mean():7.3f}" for p in range(4))
          + f" {R[:, 8].mean():7.3f} {np.nanmean(R[:, 9]):7.3f}")
    print("\nper-core program length (core's first FW start -> last FW end, any RISC; mean over views of mean/max over cores):")
    for p in range(4):
        m = [np.mean([e - s for s, e in core_len(v, p).values()]) for v in range(NV)]
        x = [np.max([e - s for s, e in core_len(v, p).values()]) for v in range(NV)]
        print(f"  {PROGS[p]:<10} mean {ms(np.mean(m)):6.3f}  max {ms(np.mean(x)):6.3f}")

    # ---- zone windows
    hdr("Zone windows (mean over views; ms from view start; busy = per-core summed zone time, core = max over its RISCs)")
    print(f"{'zone':<22}{'prog':>10}{'start':>8}{'end':>8}{'window':>8}{'busy mean':>10}{'busy max':>9}{'n/core':>7}")
    zcsv = open(a.zone_csv, "w") if a.zone_csv else None
    if zcsv:
        zcsv.write("view,prog,zone,start_ms,end_ms,busy_mean_ms,busy_max_ms\n")
    zrows = []
    for (v, p, z) in Z:
        if z.endswith("KERNEL"):
            continue
        zrows.append((z, p))
    for z, p in sorted(set(zrows), key=lambda k: (k[1], np.mean([min(x[2] for x in Z[(v, k[1], k[0])]) - V0[v]
                                                                     for v in range(NV) if Z.get((v, k[1], k[0]))]))):
        S, E, BM, BX, N = [], [], [], [], []
        for v in range(NV):
            L = Z.get((v, p, z))
            if not L:
                continue
            S.append(ms(min(x[2] for x in L) - V0[v]))
            E.append(ms(max(x[3] for x in L) - V0[v]))
            bc = defaultdict(lambda: defaultdict(int))
            for c, r, s, e in L:
                bc[c][r] += e - s
            per = [max(d.values()) for d in bc.values()]
            BM.append(ms(np.mean(per)))
            BX.append(ms(np.max(per)))
            if zcsv:
                zcsv.write(f"{v},{PROGS[p]},{z},{S[-1]:.4f},{E[-1]:.4f},{BM[-1]:.4f},{BX[-1]:.4f}\n")
            N.append(len(L) / len(bc) / len({r for _, r, _, _ in L}))
        print(f"{z:<22}{PROGS[p]:>10}{np.mean(S):8.3f}{np.mean(E):8.3f}{np.mean(E) - np.mean(S):8.3f}"
              f"{np.mean(BM):10.3f}{np.mean(BX):9.3f}{np.mean(N):7.2f}")
    if zcsv:
        zcsv.close()
        print(f"per-view zone windows: {a.zone_csv}")

    # ---- device idle: union of all FW intervals
    hdr("Device idle (no FW program on any core): gaps in the union of all FW intervals, per view period")
    U = {}
    for ch in range(NV // VIEWS_PER_CHUNK + 1):
        U[ch] = union([(s, e) for (v, p), L in FWI.items() if chunk(v) == ch for _, _, s, e in L])
    lab_tot = defaultdict(list)
    idle_v = []
    # label each union gap by the programs ending / starting around it
    ends = sorted((e, v, p) for (v, p), L in FWI.items() for _, _, _, e in L)
    starts = sorted((s, v, p) for (v, p), L in FWI.items() for _, _, s, _ in L)
    ends_t = [x[0] for x in ends]
    starts_t = [x[0] for x in starts]
    for v in nxt:
        u = U[chunk(v)]
        idle_v.append(ms(idle_in(u, V0[v], V0[v + 1])))
        acc = defaultdict(int)
        for gs, ge in zip([x[1] for x in u[:-1]], [x[0] for x in u[1:]]):
            if gs < V0[v] or gs >= V0[v + 1]:
                continue
            pe = ends[bisect.bisect_right(ends_t, gs) - 1]
            ns = starts[bisect.bisect_left(starts_t, ge)]
            acc[f"{PROGS[pe[2]]} end -> {PROGS[ns[2]]} start"] += ge - gs
        for k in set(lab_tot) | set(acc):
            lab_tot[k].append(ms(acc.get(k, 0)))
    m, lo, hi = st(idle_v)
    print(f"all-core idle per view period: mean {m:.3f} min {lo:.3f} max {hi:.3f} ms ({len(idle_v)} periods)")
    for k, x in sorted(lab_tot.items(), key=lambda kv: -np.mean(kv[1])):
        x = x + [0.0] * (len(idle_v) - len(x))
        print(f"  {k:<34} mean {np.mean(x):.3f}  max {np.max(x):.3f}")

    hdr("Per-core idle (core runs no FW program on any RISC), between its own programs; mean over views of mean/max over cores")
    gaps = defaultdict(lambda: defaultdict(list))  # label -> v -> [per-core gap]
    CL ={(v, p): core_len(v, p) for v in range(NV) for p in range(4)}
    for v in range(NV):
        for c in cores:
            for p in range(3):
                gaps[f"{PROGS[p]} -> {PROGS[p + 1]}"][v].append(CL[(v, p + 1)][c][0] - CL[(v, p)][c][1])
            if v in nxt:
                gaps["mat+blend N -> pfwc N+1"][v].append(CL[(v + 1, 0)][c][0] - CL[(v, 3)][c][1])
    print(f"{'gap (negative = overlap on that core)':<38}{'mean':>8}{'max':>8}{'min':>8}")
    for k, d in gaps.items():
        print(f"{k:<38}{ms(np.mean([np.mean(x) for x in d.values()])):8.3f}"
              f"{ms(np.mean([np.max(x) for x in d.values()])):8.3f}{ms(np.mean([np.min(x) for x in d.values()])):8.3f}")
    # per-core own FW union idle over the view period
    cu = {}
    for c in cores:
        for ch in range(NV // VIEWS_PER_CHUNK + 1):
            cu[(c, ch)] = union([(s, e) for (v, p), L in FWI.items() if chunk(v) == ch for cc, _, s, e in L if cc == c])
    pci = [np.mean([ms(idle_in(cu[(c, chunk(v))], V0[v], V0[v + 1])) for c in cores]) for v in nxt]
    print(f"per-core idle per view period (mean core): {np.mean(pci):.3f} ms")

    # ---- program overlap
    hdr("Program overlap")
    for p in range(3):
        g = [PW[(v, p + 1)][0] - PW[(v, p)][1] for v in range(NV)]
        print(f"  global {PROGS[p]} last end -> {PROGS[p + 1]} first start: mean {ms(np.mean(g)):.3f} min {ms(np.min(g)):.3f} max {ms(np.max(g)):.3f}")
    g = [PW[(v + 1, 0)][0] - PW[(v, 3)][1] for v in nxt]
    print(f"  global mat+blend N last end -> pfwc N+1 first start: mean {ms(np.mean(g)):.3f} min {ms(np.min(g)):.3f} max {ms(np.max(g)):.3f}")
    ov = [(k, v) for k, d in gaps.items() for v, x in d.items() if min(x) < 0]
    print(f"  per-core overlaps (a core starts a program before it finished the previous one): {len(ov)}")

    # ---- emit -> mat window
    hdr("(b) emit end -> mat start, per view (ms from view start; idle = all-core / mean per-core idle inside [last emit end, first mat start])")
    print(f"{'view':>4}{'emit_end':>9}{'town_end':>9}{'sort_end':>9}{'mb_start':>9}{'scm_st':>8}{'mat_min':>8}"
          f"{'mat_mean':>9}{'gap':>7}{'dev_idle':>9}{'core_idle':>10}{'gap_mean':>9}")
    B = []
    for v in range(NV):
        t = lambda zz, pp, f: f(x[2 if f is min else 3] for x in Z[(v, pp, zz)])
        E = t("sort_ol_emit", 2, max)
        TW = t("sort_ol_town", 2, max)
        SE = PW[(v, 2)][1]
        MB = PW[(v, 3)][0]
        SCM = t("sort_subchunk_mat", 3, min)
        mats = defaultdict(lambda: 1 << 62)
        for c, r, s, e in Z[(v, 3, "mat_cull_mask")]:
            mats[c] = min(mats[c], s)
        M, Mm = min(mats.values()), np.mean(list(mats.values()))
        u = U[chunk(v)]
        di = idle_in(u, E, M)
        ci = np.mean([idle_in(cu[(c, chunk(v))], E, M) for c in cores])
        r = [ms(x - V0[v]) for x in (E, TW, SE, MB, SCM, M, Mm)] + [ms(M - E), ms(di), ms(ci), ms(Mm - E)]
        B.append(r)
        print(f"{v:4d}" + "".join(f"{x:9.3f}" if i not in (4, 5, 7) else f"{x:8.3f}" if i != 7 else f"{x:7.3f}"
                                    for i, x in enumerate(r)))
    B = np.array(B)
    print(f"mean" + "".join(f"{x:9.3f}" if i not in (4, 5, 7) else f"{x:8.3f}" if i != 7 else f"{x:7.3f}"
                              for i, x in enumerate(B.mean(0))))

    # ---- mat+blend N -> pfwc N+1
    hdr("mat+blend N end -> pfwc N+1 start (ms; window = first core's mat+blend end -> last core's pfwc N+1 start)")
    print(f"{'view':>4}{'mb_end_min':>11}{'mb_end_mean':>12}{'mb_end_max':>11}{'pfwc1_min':>10}{'pfwc1_max':>10}{'gap':>7}{'dev_idle':>9}{'core_idle':>10}")
    G = []
    for v in nxt:
        cl0, cl1 = CL[(v, 3)], CL[(v + 1, 0)]
        ee = [cl0[c][1] for c in cores]
        ss = [cl1[c][0] for c in cores]
        a_, b_ = min(ee), max(ss)
        r = [ms(min(ee) - V0[v]), ms(np.mean(ee) - V0[v]), ms(max(ee) - V0[v]), ms(min(ss) - V0[v]), ms(max(ss) - V0[v]),
             ms(min(ss) - max(ee)), ms(idle_in(U[chunk(v)], a_, b_)),
             ms(np.mean([idle_in(cu[(c, chunk(v))], a_, b_) for c in cores]))]
        G.append(r)
        print(f"{v:4d}{r[0]:11.3f}{r[1]:12.3f}{r[2]:11.3f}{r[3]:10.3f}{r[4]:10.3f}{r[5]:7.3f}{r[6]:9.3f}{r[7]:10.3f}")
    G = np.array(G).mean(0)
    print(f"mean{G[0]:11.3f}{G[1]:12.3f}{G[2]:11.3f}{G[3]:10.3f}{G[4]:10.3f}{G[5]:7.3f}{G[6]:9.3f}{G[7]:10.3f}")

    # ---- per-core mat / blend / movers / writer
    hdr("Per-core busy in mat+blend (ms; mean over views of mean / max over cores)")
    met = defaultdict(lambda: defaultdict(list))
    for v in range(NV):
        pc = defaultdict(dict)
        zz = lambda z: Z[(v, 3, z)]
        for c, r, s, e in zz("mat_cull_mask"):
            pc[c][f"mat {r}"] = e - s
        for c, r, s, e in zz("tile_blend_sfpu"):
            pc[c][f"blend {r}"] = e - s
        for c, r, s, e in zz("tile_blend_load"):
            pc[c]["blend_load NCRISC"] = e - s
        for c, r, s, e in zz("sort_subchunk_mat"):
            pc[c][f"movers sort_subchunk_mat {r}"] = e - s
            if r == "BRISC":
                pc[c]["_scm_end"] = e
        for c, r, s, e in zz("BRISC-KERNEL"):
            pc[c]["_bk_end"] = e
        for c, r, s, e in zz("rd_l1_bulk"):
            pc[c]["n_blend_jobs"] = pc[c].get("n_blend_jobs", 0) + 1
        mb_t1 = {}
        for c, r, s, e in zz("mat_cull_mask"):
            if r == "TRISC_1":
                mb_t1[c] = e
        for c, r, s, e in zz("tile_blend_sfpu"):
            if r == "TRISC_1" and c in mb_t1:
                pc[c]["T1 mat end -> blend start"] = s - mb_t1[c]
        for c, d in pc.items():
            if "_scm_end" in d and "_bk_end" in d:
                d["writer BRISC (scm end -> kernel end)"] = d["_bk_end"] - d["_scm_end"]
            d["mat+blend core length"] = CL[(v, 3)][c][1] - CL[(v, 3)][c][0]
        for k in sorted({k for d in pc.values() for k in d if not k.startswith("_")}):
            x = np.array([d[k] for d in pc.values() if k in d], float)
            met[k]["mean"].append(x.mean())
            met[k]["max"].append(x.max())
            met[k]["min"].append(x.min())
    order = ["mat TRISC_0", "mat TRISC_1", "mat TRISC_2", "T1 mat end -> blend start", "blend TRISC_0", "blend TRISC_1",
             "blend TRISC_2", "blend_load NCRISC", "movers sort_subchunk_mat NCRISC", "movers sort_subchunk_mat BRISC",
             "writer BRISC (scm end -> kernel end)", "mat+blend core length", "n_blend_jobs"]
    print(f"{'metric':<40}{'mean':>8}{'min':>8}{'max':>8}")
    for k in order:
        if k in met:
            f = (lambda x: x) if k.startswith("n_") else ms
            print(f"{k:<40}{f(np.mean(met[k]['mean'])):8.3f}{f(np.mean(met[k]['min'])):8.3f}{f(np.mean(met[k]['max'])):8.3f}")

    # ---- movers fz
    hdr("Mover sort / permute per core from fz_mv_* (ms; mean over views of mean / max over cores)")
    print(f"{'metric':<22}" + "".join(f"{r + ' ' + s:>14}" for r in ("NCRISC", "BRISC") for s in ("mean", "max")))
    for k in ("sort", "perm", "rd", "wr", "dw", "big", "meta"):
        cells = []
        for r in ("NCRISC", "BRISC"):
            mm = [np.mean([D[v].get((c, r, f"fz_mv_{k}"), 0) for c in cores]) for v in range(NV)]
            mx = [np.max([D[v].get((c, r, f"fz_mv_{k}"), 0) for c in cores]) for v in range(NV)]
            cells += [ms(np.mean(mm)), ms(np.mean(mx))]
        print(f"{'fz_mv_' + k:<22}" + "".join(f"{x:14.3f}" for x in cells))
    sp = [np.mean([D[v].get((c, r, f"fz_mv_{k}"), 0) for c in cores for r in ("NCRISC", "BRISC")
                   for k in ("sort", "perm")]) * 2 for v in range(NV)]
    print(f"sort+perm per mover (mean over both movers): {ms(np.mean(sp)) / 1:.3f} ms")

    # ---- (a) TRISC1 idle in mat
    hdr("(a) TRISC1 idle in mat, bound from fz totals (per core; mean over views of mean / max over cores)")
    A = defaultdict(lambda: defaultdict(list))
    bound_end, real_end = [], []
    for v in range(NV):
        pc = {}
        nj = defaultdict(int)
        for c, r, s, e in Z[(v, 3, "rd_l1_bulk")]:
            nj[c] += 1
        bl = {c: e - s for c, r, s, e in Z[(v, 3, "tile_blend_sfpu")] if r == "TRISC_1"}
        for c in cores:
            g = lambda n, r="TRISC_1": D[v].get((c, r, n))
            tot, band, copy, mbw = g("fz_m_tot"), g("fz_m_band"), g("fz_m_copy"), g("fz_m_mbw")
            if tot is None or c not in bl:
                continue
            idle = tot - band - copy
            per_job = bl[c] / nj[c] if nj[c] else float("nan")
            per_tile = bl[c] / (TILES / NC)
            pc[c] = {"t1 mat total (fz_m_tot)": ms(tot), "t1 band_batch": ms(band), "t1 copy_tile": ms(copy),
                     "t1 mailbox wait (fz_m_mbw)": ms(mbw), "t1 idle bound (tot-band-copy)": ms(idle),
                     "t0 pick_job wait (fz_u_pick)": ms(g("fz_u_pick", "TRISC_0")),
                     "t1 blend tile_blend_sfpu": ms(bl[c]), "blend jobs (rd_l1_bulk)": nj[c],
                     "blend per job (ms)": ms(per_job), "blend per 32x32 tile (ms, 1024/110 tiles)": ms(per_tile),
                     "idle bound in blend jobs": idle / per_job, "idle / blend busy": idle / bl[c],
                     "mat batches (fz_u_nb)": g("fz_u_nb", "TRISC_0"), "idle per batch (us)": idle / 1350.0 / max(1, g("fz_u_nb", "TRISC_0"))}
            pc[c]["_len"] = CL[(v, 3)][c][1] - CL[(v, 3)][c][0]
            pc[c]["_save"] = min(idle, bl[c])
        for k in next(iter(pc.values())):
            if k.startswith("_"):
                continue
            x = np.array([d[k] for d in pc.values()], float)
            A[k]["mean"].append(x.mean())
            A[k]["max"].append(x.max())
        real_end.append(ms(max(d["_len"] for d in pc.values())))
        bound_end.append(ms(max(d["_len"] - d["_save"] for d in pc.values())))
    print(f"{'metric':<44}{'mean':>9}{'max':>9}")
    for k, d in A.items():
        print(f"{k:<44}{np.mean(d['mean']):9.3f}{np.mean(d['max']):9.3f}")
    print(f"slowest-core mat+blend length {np.mean(real_end):.3f} ms; if every core's T1 mat idle were filled with its own "
          f"blend work: {np.mean(bound_end):.3f} ms (upper bound on the gain: {np.mean(real_end) - np.mean(bound_end):.3f} ms/view)")

    job_gaps(Z, D, CL, NV, cores, a.job_csv)

    # ---- host zones
    host_zones(a.out)
    reconcile(a.out, NV, nxt, V0, PW)


MOVERS = ("NCRISC", "BRISC")
GAP_MIN_US = (5, 20, 50, 100)


def job_gaps(Z, D, CL, NV, cores, job_csv=None):
    """t413: where TRISC1 idles in mat, from the per-job zones (GSPLAT_TT_MATCULL_PROF=1 capture).
    Per core: mat = TRISC1 mat_cull_mask zone; jobs = TRISC1 mj_job zones. start = mat start ->
    first job start; between = sum of job[i] end -> job[i+1] start; tail = last job end -> mat end
    (the final mj_wait for both streams to end); in-job = job time - band - copy (fz_m_*). For the
    start/between/tail gaps, the mover zones (per-subchunk MAT_PZ zones) that overlap them say what
    the movers were doing. Fill: gap time blend work could take at a given granularity."""
    if not any(Z.get((v, 3, "mj_job")) for v in range(NV)):
        print("\n(no mj_job zones: capture without GSPLAT_TT_MATCULL_PROF=1; per-job gaps skipped)")
        return
    hdr("(c) TRISC1 idle in mat from per-job zones (t413; per core, ms; mean over views of mean / max over cores)")
    A = defaultdict(lambda: defaultdict(list))
    MV = defaultdict(lambda: defaultdict(float))  # part -> mover zone -> cycles (summed over cores/views)
    GT = defaultdict(float)                        # part -> gap cycles (same sums)
    fill = defaultdict(list)                       # threshold -> per view mean-core fillable
    crit = defaultdict(list)                       # threshold -> per view slowest-core mat+blend if filled
    rows, njobs_bad = [], 0
    for v in range(NV):
        mz = {c: (s, e) for c, r, s, e in Z[(v, 3, "mat_cull_mask")] if r == "TRISC_1"}
        jobs, waits = defaultdict(list), defaultdict(list)
        for c, r, s, e in Z[(v, 3, "mj_job")]:
            if r == "TRISC_1":
                jobs[c].append((s, e))
        for c, r, s, e in Z[(v, 3, "mj_wait")]:
            if r == "TRISC_1":
                waits[c].append((s, e))
        mvz = defaultdict(list)
        for (vv, p, z), L in Z.items():
            if vv == v and p == 3 and (z.startswith("mat_") and z != "mat_cull_mask"):
                for c, r, s, e in L:
                    if r in MOVERS:
                        mvz[c].append((z, r, s, e))
        bl = {c: e - s for c, r, s, e in Z[(v, 3, "tile_blend_sfpu")] if r == "TRISC_1"}
        pc, fl = {}, defaultdict(dict)
        for c in cores:
            if c not in mz or not jobs[c]:
                continue
            ms0, me0 = mz[c]
            J = sorted(jobs[c])
            g = lambda n, r="TRISC_1": D[v].get((c, r, n), 0)
            nj_fz = g("fz_u_nj", "TRISC_0")
            if nj_fz and nj_fz != len(J):
                njobs_bad += 1
            start = J[0][0] - ms0
            betw = [J[i + 1][0] - J[i][1] for i in range(len(J) - 1)]
            tail = me0 - J[-1][1]
            jt = sum(e - s for s, e in J)
            band, copy = g("fz_m_band"), g("fz_m_copy")
            injob = jt - band - copy
            parts = {"start": [(ms0, J[0][0])], "between": [(J[i][1], J[i + 1][0]) for i in range(len(J) - 1)],
                     "tail": [(J[-1][1], me0)]}
            for part, ivs in parts.items():
                for a0, b0 in ivs:
                    GT[part] += b0 - a0
                    for z, r, s, e in mvz[c]:
                        o = min(e, b0) - max(s, a0)
                        if o > 0:
                            MV[part][f"{r}:{z}"] += o
            gl = [start] + betw + [tail]
            for us in GAP_MIN_US:
                fl[us][c] = sum(x for x in gl if x >= us * CYC_MS / 1000.0)
            per_job = None
            nbj = sum(1 for cc, r, s, e in Z[(v, 3, "rd_l1_bulk")] if cc == c)
            if c in bl and nbj:
                per_job = bl[c] / nbj
                fl["blend job"][c] = sum((x // per_job) * per_job for x in gl)
            mlen = me0 - ms0
            pc[c] = {"mat length (T1 mat_cull_mask)": ms(mlen), "jobs (mj_job)": len(J),
                     "start gap (mat start -> 1st job)": ms(start), "between-job gaps (sum)": ms(sum(betw)),
                     "between-job gap max": ms(max(betw) if betw else 0),
                     "between-job gap mean": ms(np.mean(betw) if betw else 0),
                     "tail gap (last job -> mat end)": ms(tail),
                     "idle outside jobs (start+between+tail)": ms(start + sum(betw) + tail),
                     "in-job idle (job - band - copy)": ms(injob), "band_batch (fz_m_band)": ms(band),
                     "T1 idle total (outside + in-job)": ms(start + sum(betw) + tail + injob)}
            rows.append([v, c[0], c[1], len(J), ms(mlen), ms(start), ms(sum(betw)), ms(max(betw) if betw else 0),
                         ms(tail), ms(injob), ms(band)])
        if not pc:
            continue
        for k in next(iter(pc.values())):
            x = np.array([d[k] for d in pc.values()], float)
            A[k]["mean"].append(x.mean())
            A[k]["max"].append(x.max())
        L = CL[(v, 3)]
        real = max(L[c][1] - L[c][0] for c in pc)
        for key, d in fl.items():
            fill[key].append(ms(np.mean([d.get(c, 0) for c in pc])))
            crit[key].append(ms(max(L[c][1] - L[c][0] - min(d.get(c, 0), bl.get(c, 0)) for c in pc)) - ms(real))
    print(f"{'metric':<44}{'mean':>9}{'max':>9}")
    for k, d in A.items():
        print(f"{k:<44}{np.mean(d['mean']):9.3f}{np.mean(d['max']):9.3f}")
    if njobs_bad:
        print(f"WARNING: {njobs_bad} core-views where mj_job count != fz_u_nj (dropped markers?)")
    print("\nWhat the movers do during the TRISC1 gaps (share of gap time; overlap of NCRISC/BRISC MAT_PZ zones,"
          " each mover counted on its own, so a part sums to <= 200%):")
    for part in ("start", "between", "tail"):
        tot = GT[part]
        if not tot:
            continue
        top = sorted(MV[part].items(), key=lambda kv: -kv[1])[:10]
        print(f"  {part:<8} ({ms(tot) / max(1, NV) / max(1, len(cores)):.3f} ms per core-view): " +
              ", ".join(f"{k} {100 * x / tot:.0f}%" for k, x in top))
    print("\nBlend fill (per core, ms/view; gap time in gaps >= the threshold; 'blend job' = whole blend jobs"
          " (tile_blend_sfpu / rd_l1_bulk count) that fit in each gap). crit = change of the slowest core's"
          " mat+blend length if every core filled that much of its gaps with its own blend work:")
    for key in list(GAP_MIN_US) + ["blend job"]:
        if fill.get(key):
            lab = f">= {key} us" if key != "blend job" else "whole blend jobs"
            print(f"  {lab:<18} fillable {np.mean(fill[key]):.3f} mean-core; crit {np.mean(crit[key]):+.3f} ms/view")
    if job_csv:
        with open(job_csv, "w") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["view", "core_x", "core_y", "jobs", "mat_ms", "start_ms", "between_ms", "between_max_ms",
                        "tail_ms", "injob_idle_ms", "band_ms"])
            for r in rows:
                w.writerow([f"{x:.4f}" if isinstance(x, float) else x for x in r])


HOST = ["host_cq1_proj_m", "EnqueueProgram", "host_cq1_k2_rows", "host_finish_cq1_bridge", "host_blend_setup",
        "host_wait_blend_event", "host_finish_blend", "host_finish_ta_keep_fill", "FDMeshCommandQueue::finish",
        "FDMeshCommandQueue::finish_nolock", "enqueue_read_shard_from_core", "enqueue_write_shard_to_core",
        "WriteToDeviceL1", "WriteRuntimeArgsToDevice", "wait_for_fetch_q_space"]


def host_zones(out):
    """Host zones of the tracy -u dumps; views = intervals between host_cq1_proj_m starts (one per view).
    View 0 (warmup) and the last view (it holds the device profiler read) are dropped."""
    hdr("Host zones (tracy -u, host clock; per view = between consecutive host_cq1_proj_m starts; warmup and last view dropped)")
    W = defaultdict(lambda: defaultdict(list))
    nv = 0
    for path in sorted(glob.glob(os.path.join(out, "tracy-u-c*.csv.gz"))):
        rows = defaultdict(list)
        with opener(path) as f:
            for r in csv.reader(f):
                if len(r) >= 8 and r[0] in HOST:
                    s = int(r[5])
                    rows[r[0]].append((s, s + int(r[6]), r[7]))
        anc = sorted(s for s, _, _ in rows["host_cq1_proj_m"])
        for i in range(1, len(anc) - 1):
            nv += 1
            for z in HOST:
                L = [(s, e) for s, e, _ in rows[z] if anc[i] <= s < anc[i + 1]]
                W[z]["n"].append(len(L))
                W[z]["sum"].append(sum(e - s for s, e in L) / 1e6)
                if L:
                    W[z]["first"].append((min(s for s, _ in L) - anc[i]) / 1e6)
                    W[z]["last_end"].append((max(e for _, e in L) - anc[i]) / 1e6)
                    W[z]["max"].append(max(e - s for s, e in L) / 1e6)
            ep = sorted(s for s, _, _ in rows["EnqueueProgram"] if anc[i] <= s < anc[i + 1])
            for j, s in enumerate(ep):
                W[f"EnqueueProgram#{j}"]["first"].append((s - anc[i]) / 1e6)
            W["_period"]["ms"].append((anc[i + 1] - anc[i]) / 1e6)
    print(f"views={nv}; host view period (proj_m -> next proj_m) mean {np.mean(W['_period']['ms']):.3f} ms")
    print(f"{'zone':<36}{'n/view':>7}{'ms/view':>9}{'max one':>9}{'first st':>9}{'last end':>9}")
    for z in HOST:
        d = W[z]
        if not d["first"]:
            continue
        print(f"{z:<36}{np.mean(d['n']):7.1f}{np.mean(d['sum']):9.3f}{np.mean(d['max']):9.3f}"
              f"{np.mean(d['first']):9.3f}{np.mean(d['last_end']):9.3f}")
    print("EnqueueProgram starts (ms from proj_m start): " + ", ".join(
        f"#{j} {np.mean(W[f'EnqueueProgram#{j}']['first']):.3f}" for j in range(8) if W[f"EnqueueProgram#{j}"]["first"]))
    print("not Tracy zones in this build (TTW_TIMING stage timers only, see the reconciliation): xview hook, "
          "EventSynchronize, gather_wait, bin_layout, publish_host")


def stages(path):
    for ln in open(path):
        if ln.startswith("STAGES "):
            return {k: float(v) for k, v in re.findall(r"(\w+)=([-+\d.]+)", ln)}
    return {}


def view_ms(path):
    return [float(x) for x in re.findall(r"^\[run\]\s+view=\S+ ([\d.]+)ms", open(path).read(), re.M)]


def reconcile(out, NV, nxt, V0, PW):
    hdr("Reconciliation: untraced stage means vs traced device span (ms/view)")
    rs = [stages(p) for p in sorted(glob.glob(os.path.join(out, "r[0-9].log")))]
    k = lambda key: np.mean([s.get(key, 0) for s in rs])
    uv = np.mean([view_ms(p) for p in sorted(glob.glob(os.path.join(out, "r[0-9].log")))], axis=0)
    tv = []
    for c in (0, 10, 20):
        p = os.path.join(out, f"T{c}.log")
        tv += view_ms(p) if os.path.exists(p) else [float("nan")] * VIEWS_PER_CHUNK
    tv = np.array(tv)
    per = np.array([(V0[v + 1] - V0[v]) / CYC_MS for v in nxt])
    win = {p: np.mean([(PW[(v, p)][1] - PW[(v, p)][0]) / CYC_MS for v in nxt]) for p in range(4)}
    gaps = {p: np.mean([(PW[(v, p + 1)][0] - PW[(v, p)][1]) / CYC_MS for v in nxt]) for p in range(3)}
    g30 = np.mean([(PW[(v + 1, 0)][0] - PW[(v, 3)][1]) / CYC_MS for v in nxt])
    print(f"untraced: {len(rs)} rounds x {len(uv)} views; traced: the {len(nxt)} views that have a next view in their chunk")
    print(f"{'untraced host stage (r1-r3 mean)':<36}{'ms':>7}   {'traced device (stitched, same view set where noted)':<44}{'ms':>7}")
    left = [("avg_frame_ms", k("avg_frame_ms")), ("view_total", k("view_total")), ("  head", k("head")),
            ("  project (gather_wait inside)", k("project")), ("  tile_assign", k("tile_assign")), ("  sort", k("sort")),
            ("  blend (mat+blend wait)", k("blend")), ("  d2h", k("d2h")), ("  xview", k("xview")), ("  tail", k("tail")),
            ("frame - view_total (pybind etc.)", k("avg_frame_ms") - k("view_total")),
            (f"per-view mean, {len(nxt)} views", np.mean(uv[nxt])), (f"traced host per-view, {len(nxt)} views", np.nanmean(tv[nxt])), ("", None)]
    right = [("device period pfwc N -> pfwc N+1", per.mean()), ("pfwc window", win[0]), ("pfwc -> k2 gap", gaps[0]),
             ("k2 window", win[1]), ("k2 -> sort gap", gaps[1]), ("sort window", win[2]), ("sort -> mat+blend gap", gaps[2]),
             ("mat+blend window", win[3]), ("mat+blend -> pfwc N+1 gap", g30),
             ("sum of the above", win[0] + win[1] + win[2] + win[3] + sum(gaps.values()) + g30),
             ("traced device - untraced per-view", per.mean() - np.mean(uv[nxt])),
             ("traced host - traced device", np.nanmean(tv[nxt]) - per.mean()), ("", None), ("", None)]
    for (a, x), (b, y) in zip(left, right):
        print(f"{a:<36}{'' if x is None else f'{x:7.3f}':>7}   {b:<44}{'' if y is None else f'{y:7.3f}':>7}")


if __name__ == "__main__":
    main()
