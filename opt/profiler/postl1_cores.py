#!/usr/bin/env python3
"""Post-L1 per-core profile of the 4-program view (task #297).

For each view (warmup dropped): per program and RISC the kernel span (KERNEL zone,
mean and max over cores); for the fused mat+blend program per core the mat end, the
blend start, the gap, blend TRISC1 span and mat mover span; the pfwc, k2 and
sort_ol emit tails (max minus mean core end); and, when the capture has them, the
mat-phase cull counters "mc_*" from mat_cull_compute.cpp (profiler builds only):
mc_uw = UNPACK spin waiting for a mover's coefficient tile (TRISC idle inside the
mat phase), mc_utot = UNPACK cull loop total, mc_nb = batches.

Usage: postl1_cores.py <dev30.csv> [--percore out.csv]
ms at 1.35 GHz, times from the first mat start on any core.
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0
DATA, ZONE, TYPE = 6, 10, 11
PROGS = ["pfwc", "k2", "sort_ol", "matblend"]
MARK = {"pfwc": ("pfwc",), "k2": ("k2_pairs", "k2_rows"),
        "sort_ol": ("sort_ol_prefix", "sort_ol_barrier", "sort_ol_town", "sort_ol_emit"),
        "matblend": ("mat_cull_mask", "sort_subchunk_mat")}
RISCS = ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--percore")
    a = ap.parse_args()
    _, lines, ts = read_csv(a.csv)
    frame, n_frames, _ = segment_frames(lines, ts)
    stack = {}
    iv = defaultdict(list)  # (frame, core, risc, zone) -> [(s, e)]
    data = defaultdict(int)  # (frame, core, name) -> value
    for ln, t, fr in zip(lines, ts, frame):
        if fr < 1:
            continue
        p = ln.split(",", 12)
        z, r, c = p[ZONE], p[3], (p[1], p[2])
        if z.startswith("mc_") and not p[TYPE].startswith("ZONE"):
            data[(fr, c, z)] += int(p[DATA])
            continue
        key = (fr, c, r, z)
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
        elif p[TYPE] == "ZONE_END" and key in stack:
            iv[key].append((stack.pop(key), int(t)))

    span = defaultdict(list)    # (prog, risc) -> per-view [mean, max] over cores
    rows = defaultdict(list)    # summary metric -> per-view value
    pc = defaultdict(lambda: defaultdict(list))  # core -> metric -> per-view value
    for f in range(1, n_frames):
        cores = sorted({k[1] for k in iv if k[0] == f})
        if not cores:
            continue
        # kernel spans per program: a KERNEL zone belongs to the last program whose
        # first marker zone (any core) starts nearest to it (all cores launch together)
        pst = []
        for pr in PROGS:
            ss = [x[0] for k, v in iv.items() if k[0] == f and k[3] in MARK[pr] for x in v]
            pst.append(min(ss) if ss else None)
        for r in RISCS:
            kz = "TRISC-KERNEL" if r.startswith("TRISC") else f"{r}-KERNEL"
            per = defaultdict(list)
            for c in cores:
                for s, e in iv.get((f, c, r, kz), []):
                    cand = [(abs(s - p0), i) for i, p0 in enumerate(pst) if p0 is not None]
                    if cand:
                        per[PROGS[min(cand)[1]]].append((e - s) / CYC_MS)
            for pr, v in per.items():
                span[(pr, r)].append((np.mean(v), np.max(v)))

        def ends(zone, riscs):
            out = {}
            for c in cores:
                e = [x[1] for r in riscs for x in iv.get((f, c, r, zone), [])]
                s = [x[0] for r in riscs for x in iv.get((f, c, r, zone), [])]
                if e:
                    out[c] = (min(s), max(e))
            return out

        for name, zone, riscs in (("pfwc", "pfwc", ["TRISC_0", "TRISC_1", "TRISC_2"]),
                                  ("k2", "k2_pairs", ["BRISC", "NCRISC"]),
                                  ("emit", "sort_ol_emit", ["BRISC", "NCRISC"]),
                                  ("town", "sort_ol_town", ["TRISC_0", "TRISC_1", "TRISC_2"])):
            o = ends(zone, riscs)
            if o:
                e = np.array([v[1] for v in o.values()], dtype=np.float64)
                s0 = min(v[0] for v in o.values())
                rows[f"{name}_window"].append((e.max() - s0) / CYC_MS)
                rows[f"{name}_tail(max-mean end)"].append((e.max() - e.mean()) / CYC_MS)

        mat = ends("mat_cull_mask", ["TRISC_1"])
        bl = ends("tile_blend_sfpu", ["TRISC_1"])
        if not mat:
            continue
        t0 = min(v[0] for v in mat.values())
        ms = lambda x: (x - t0) / CYC_MS
        me, bs, be, idle, mbusy = [], [], [], [], []
        for c in cores:
            if c not in mat or c not in bl:
                continue
            m_s, m_e = mat[c]
            b_s, b_e = bl[c]
            mv = {r: sum(e - s for s, e in iv.get((f, c, r, "sort_subchunk_mat"), [])) / CYC_MS
                  for r in ("BRISC", "NCRISC")}
            d = {k: data.get((f, c, k)) for k in ("mc_uw", "mc_utot", "mc_nb", "mc_mw", "mc_act", "mc_tot")}
            me.append(ms(m_e)); bs.append(ms(b_s)); be.append(ms(b_e))
            g = pc[c]
            g["mat_start"].append(ms(m_s)); g["mat_end"].append(ms(m_e))
            g["blend_start"].append(ms(b_s)); g["blend_end"].append(ms(b_e))
            g["gap"].append((b_s - m_e) / CYC_MS); g["blend_len"].append((b_e - b_s) / CYC_MS)
            g["mov_br"].append(mv["BRISC"]); g["mov_nc"].append(mv["NCRISC"])
            if d["mc_uw"] is not None:
                g["t1_idle_mat"].append(d["mc_uw"] / CYC_MS)
                g["t1_busy_mat"].append((d["mc_utot"] - d["mc_uw"]) / CYC_MS)
                g["batches"].append(d["mc_nb"] or 0)
                idle.append(d["mc_uw"] / CYC_MS)
                mbusy.append((d["mc_utot"] - d["mc_uw"]) / CYC_MS)
            if d["mc_mw"] is not None:
                g["math_wait"].append(d["mc_mw"] / CYC_MS)
                g["math_act"].append(d["mc_act"] / CYC_MS)
        me, bs, be = map(np.array, (me, bs, be))
        rows["cores"].append(len(me))
        rows["mat_end_mean"].append(me.mean()); rows["mat_end_max"].append(me.max())
        rows["blend_start_mean"].append(bs.mean()); rows["blend_start_max"].append(bs.max())
        rows["gap_mean"].append((bs - me).mean())
        rows["blend_end_mean"].append(be.mean()); rows["blend_end_max"].append(be.max())
        rows["blend_tail(max-mean end)"].append(be.max() - be.mean())
        if idle:
            rows["t1_idle_in_mat_mean"].append(np.mean(idle))
            rows["t1_idle_in_mat_min"].append(np.min(idle))
            rows["t1_idle_in_mat_max"].append(np.max(idle))
            rows["t1_cull_busy_mean"].append(np.mean(mbusy))

    nv = len(rows["cores"])
    print(f"views={nv} (warmup dropped), ms, means over views")
    print("\n[kernel span per program and RISC: mean over cores / max over cores]")
    print(f"{'program':10s}" + "".join(f"{r:>17s}" for r in RISCS))
    for pr in PROGS:
        cells = []
        for r in RISCS:
            v = span.get((pr, r))
            cells.append(f"{np.mean([x[0] for x in v]):7.3f}/{np.mean([x[1] for x in v]):7.3f}" if v else f"{'-':>15s}")
        print(f"{pr:10s}" + "".join(f"  {c:>15s}" for c in cells))
    print("\n[summary]")
    for k, x in rows.items():
        print(f"{k:30s} {np.mean(x):8.3f}")
    if pc:
        keys = list(next(iter(pc.values())).keys())
        print("\n[per-core distribution of per-view means: min / p25 / median / p75 / max]")
        vals = {k: np.array([np.mean(pc[c][k]) for c in pc if pc[c][k]]) for k in keys}
        for k, v in vals.items():
            if len(v):
                q = np.percentile(v, [0, 25, 50, 75, 100])
                print(f"{k:16s} " + " ".join(f"{x:7.3f}" for x in q))
        if a.percore:
            with open(a.percore, "w") as fo:
                fo.write("core," + ",".join(keys) + "\n")
                for c in sorted(pc):
                    fo.write(f"{c[0]}-{c[1]}," + ",".join(
                        f"{np.mean(pc[c][k]):.4f}" if pc[c][k] else "" for k in keys) + "\n")


if __name__ == "__main__":
    main()
