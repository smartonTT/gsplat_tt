#!/usr/bin/env python3
"""Per-mover breakdown of sort_subchunk_mat and the SFPU cull (task #86).

Capture with GSPLAT_TT_MATCULL_PROF=1 (fine zones), then:
    matcull_breakdown.py <profile_log_device.csv>
For each program (mat, cull) and each view: the window (first start .. last
end over all cores), each RISC's busy time in every fine zone, and the
critical RISC (the one whose top zone ends last). Prints means over views
(warmup frame dropped), ms at 1.35 GHz.
"""
import os
import sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0
ZONE, TYPE = 10, 11
PROGS = {
    "mat": ("sort_subchunk_mat", ["sort_subchunk_mat"]),
    "cull": ("tile_l1_cull_rd", ["tile_l1_cull_rd", "tile_l1_mask_wr", "tile_mb_mask"]),
}
FINE = {
    "mat": ["mat_meta", "mat_rd", "mat_sort", "mat_perm", "mat_wr", "mat_ov_rd",
            "mat_ov_sort", "mat_ov_perm", "mat_ov_wr", "mat_gather"],
    "cull": ["cr_meta", "cr_tile", "cr_slot_wait", "cr_bulk", "cr_fill", "cw_tile",
             "cw_slab_wait", "cw_patch", "cw_wr"],
}


def main():
    _, lines, ts = read_csv(sys.argv[1])
    frame, n_frames, _ = segment_frames(lines, ts)
    stack = {}
    busy = defaultdict(int)          # (frame, core, risc, zone) -> cyc
    span = {}                        # (frame, core, risc, zone) -> [start, end]
    for ln, t, fr in zip(lines, ts, frame):
        if fr < 1:
            continue
        p = ln.split(",", 12)
        key = (int(fr), (p[1], p[2]), p[3], p[ZONE])
        if p[TYPE] == "ZONE_START":
            stack[key] = int(t)
        elif p[TYPE] == "ZONE_END" and key in stack:
            t0, t1 = stack.pop(key), int(t)
            busy[key] += t1 - t0
            s = span.setdefault(key, [t0, t1])
            s[0], s[1] = min(s[0], t0), max(s[1], t1)
    frames = range(1, n_frames)
    for prog, (top, tops) in PROGS.items():
        print(f"== {prog} (views={len(frames)})")
        win, crit_rows, per_risc = [], [], defaultdict(list)
        for f in frames:
            ks = [k for k in span if k[0] == f and k[3] in tops]
            if not ks:
                continue
            w0 = min(span[k][0] for k in ks)
            w1 = max(span[k][1] for k in ks)
            win.append((w1 - w0) / CYC_MS)
            movers = [k for k in ks if k[3] == top] if prog == "mat" else ks
            for k in movers:
                row = {"busy": busy[k], "start": span[k][0] - w0, "end": span[k][1] - w0}
                for z in FINE[prog]:
                    row[z] = busy.get((f, k[1], k[2], z), 0)
                per_risc[(k[2], k[3])].append(row)
            crit = max(movers, key=lambda k: span[k][1])
            row = {"risc": crit[2], "zone": crit[3], "busy": busy[crit],
                   "start": span[crit][0] - w0}
            for z in FINE[prog]:
                row[z] = busy.get((f, crit[1], crit[2], z), 0)
            crit_rows.append(row)
        print(f"window_ms {mean(win):.3f}")
        cols = ["busy", "start", "end"] + FINE[prog]
        print(f"{'risc/zone':<34}" + "".join(f"{c[:11]:>12}" for c in cols))
        for (risc, z), rows in sorted(per_risc.items()):
            n = len(rows) / len(win)
            print(f"{risc + '/' + z + ' mean':<34}" + "".join(
                f"{mean(r[c] for r in rows) / CYC_MS:12.3f}" for c in cols) + f"  n/view={n:.0f}")
            print(f"{risc + '/' + z + ' max':<34}" + "".join(
                f"{max(r[c] for r in rows) / CYC_MS:12.3f}" for c in cols))
        crs = defaultdict(int)
        for r in crit_rows:
            crs[r["risc"] + "/" + r["zone"]] += 1
        print("critical RISC per view:", dict(crs))
        print(f"{'critical mean':<34}" + "".join(
            f"{mean(r.get(c, 0) for r in crit_rows) / CYC_MS:12.3f}" for c in cols))
        print()


if __name__ == "__main__":
    main()
