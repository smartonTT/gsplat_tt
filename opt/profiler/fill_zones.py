#!/usr/bin/env python3
"""Mat-phase split of the MATCULL_TRISC_FILL path per core (task #319).

Reads the "fz_*" timestamped-data counters that profiler builds (PROFILE_KERNEL)
of sort_subchunk_materialize.cpp (movers, "fz_mv_*") and mat_cull_compute.cpp
(TRISC0 "fz_u_*", TRISC1 "fz_m_*", TRISC2 "fz_p_*") report once per kernel, plus
the mat_cull_mask / sort_subchunk_mat / pfwc zones, from a stitched Tracy device CSV.

Per view (warmup dropped) and core: mat end (mat_cull_mask TRISC_1 end, from the
first mat start on any core), mover busy (sort_subchunk_mat span), mover done-word
wait and per-item parts, TRISC0 pick_job wait (idle) / fill CB wait / fill copy and
records, TRISC1 mailbox wait / copy_tile / band_batch, TRISC2 mailbox wait /
tile_regs_wait / pack / patch wait on the packer / patch. Values are means over
views; "mean" and "max" are over cores. Also the pfwc tail (max - mean core end).

Usage: fill_zones.py <dev30.csv> [--percore out.csv]   (ms at 1.35 GHz)
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
MOVERS = ("NCRISC", "BRISC")
CNT = {"fz_mv_dwn", "fz_mv_dwb", "fz_mv_jobs", "fz_mv_recs", "fz_u_nb", "fz_u_nr", "fz_u_nj"}


def per_core(iv, data, f, c, t0):
    """Metrics of one core in view f (ms unless noted), or None without mat zones."""
    mat = iv.get((f, c, "TRISC_1", "mat_cull_mask"))
    if not mat:
        return None
    d = lambda r, k: data.get((f, c, r, k))
    ms = lambda x: x / CYC_MS
    m = {"mat_end": ms(max(e for _, e in mat) - t0), "mat_len": ms(sum(e - s for s, e in mat))}
    for r in MOVERS:
        t = r[:2].lower()
        m[f"{t}_busy"] = ms(sum(e - s for s, e in iv.get((f, c, r, "sort_subchunk_mat"), [])))
        if d(r, "fz_mv_dw") is None:
            continue
        for k in ("dw", "rd", "sort", "perm", "wr", "meta", "big"):
            m[f"{t}_{k}"] = ms(d(r, f"fz_mv_{k}"))
        m[f"{t}_dwm_us"] = d(r, "fz_mv_dwm") / 1350.0
        for k in ("dwn", "dwb", "jobs", "recs"):
            m[f"{t}_{k}"] = d(r, f"fz_mv_{k}")
    u = {k: d("TRISC_0", f"fz_u_{k}") for k in ("tot", "pick", "fillw", "fill", "nb", "nr", "nj")}
    if u["tot"] is not None:
        m["t0_tot"] = ms(u["tot"])
        m["t0_idle(pick)"] = ms(u["pick"])
        m["t0_fillw"] = ms(u["fillw"])
        m["t0_fill"] = ms(u["fill"])
        m["t0_other"] = ms(u["tot"] - u["pick"] - u["fillw"] - u["fill"])
        m["t0_busy_frac"] = (u["tot"] - u["pick"]) / u["tot"] if u["tot"] else 0.0
        m["t0_recs"], m["t0_batches"], m["t0_jobs"] = u["nr"], u["nb"], u["nj"]
        m["t0_fill_cyc_rec"] = u["fill"] / u["nr"] if u["nr"] else 0.0
    mt = {k: d("TRISC_1", f"fz_m_{k}") for k in ("tot", "mbw", "copy", "band")}
    if mt["tot"] is not None:
        m["t1_tot"] = ms(mt["tot"])
        m["t1_mbw"] = ms(mt["mbw"])
        m["t1_copy"] = ms(mt["copy"])
        m["t1_band"] = ms(mt["band"])
        m["t1_idle(tot-band)"] = ms(mt["tot"] - mt["band"])
    p = {k: d("TRISC_2", f"fz_p_{k}") for k in ("tot", "mbw", "regw", "pack", "patchw", "patch")}
    if p["tot"] is not None:
        m["t2_tot"] = ms(p["tot"])
        for k in ("mbw", "regw", "pack", "patchw", "patch"):
            m[f"t2_{k}"] = ms(p[k])
        m["t2_busy(pack+patch)"] = ms(p["pack"] + p["patch"])
        m["t2_idle(waits)"] = ms(p["mbw"] + p["regw"] + p["patchw"])
        if u["nr"]:
            m["t2_patch_cyc_rec"] = p["patch"] / u["nr"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--percore")
    a = ap.parse_args()
    _, lines, ts = read_csv(a.csv)
    frame, n_frames, _ = segment_frames(lines, ts)
    stack = {}
    iv = defaultdict(list)   # (frame, core, risc, zone) -> [(s, e)]
    data = defaultdict(int)  # (frame, core, risc, name) -> summed value
    for ln, t, fr in zip(lines, ts, frame):
        if fr < 1:
            continue
        p = ln.split(",", 12)
        z, r, c = p[ZONE], p[3], (p[1], p[2])
        if z.startswith("fz_") and not p[TYPE].startswith("ZONE"):
            data[(fr, c, r, z)] += int(p[DATA])
            continue
        key = (fr, c, r, z)
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
        elif p[TYPE] == "ZONE_END" and key in stack:
            iv[key].append((stack.pop(key), int(t)))

    view = defaultdict(lambda: defaultdict(list))  # metric -> "mean"/"max" -> per view
    pc = defaultdict(lambda: defaultdict(list))    # core -> metric -> per view
    tails = defaultdict(list)
    for f in range(1, n_frames):
        cores = sorted({k[1] for k in iv if k[0] == f})
        mats = [s for c in cores for s, _ in iv.get((f, c, "TRISC_1", "mat_cull_mask"), [])]
        if not mats:
            continue
        t0 = min(mats)
        rows = {}
        for c in cores:
            m = per_core(iv, data, f, c, t0)
            if m:
                rows[c] = m
                for k, v in m.items():
                    pc[c][k].append(v)
        for k in next(iter(rows.values())):
            v = np.array([m[k] for m in rows.values() if k in m], dtype=np.float64)
            view[k]["mean"].append(v.mean())
            view[k]["max"].append(v.max())
        view["cores"]["mean"].append(len(rows))
        view["cores"]["max"].append(len(rows))
        for name, zone, riscs in (("pfwc", "pfwc", ("TRISC_0", "TRISC_1", "TRISC_2")),
                                  ("mat", "mat_cull_mask", ("TRISC_1",))):
            e = [max(x[1] for r in riscs for x in iv.get((f, c, r, zone), [(0, 0)])) for c in cores]
            e = np.array([x for x in e if x > 0], dtype=np.float64)
            if len(e):
                tails[name].append((e.max() - e.mean()) / CYC_MS)

    print(f"views={len(view['cores']['mean'])} (warmup dropped); per view mean / max over cores, "
          "then mean over views; ms unless named (cyc_rec = cycles per record, *_us = us)")
    print(f"{'metric':22s} {'mean':>10s} {'max':>10s}")
    for k, v in view.items():
        print(f"{k:22s} {np.mean(v['mean']):10.3f} {np.mean(v['max']):10.3f}")
    for k, v in tails.items():
        print(f"{k + '_tail(max-mean end)':22s} {np.mean(v):10.3f}")
    if pc and a.percore:
        keys = sorted({k for c in pc for k in pc[c]})
        with open(a.percore, "w") as fo:
            fo.write("core," + ",".join(keys) + "\n")
            for c in sorted(pc):
                fo.write(f"{c[0]}-{c[1]}," + ",".join(
                    f"{np.mean(pc[c][k]):.4f}" if pc[c][k] else "" for k in keys) + "\n")


if __name__ == "__main__":
    main()
