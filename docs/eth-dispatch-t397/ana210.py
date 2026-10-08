#!/usr/bin/env python3
"""t382 (copied by t387; per-core table prints core-ms = sum over cores instead of sum/110): per-view device timeline, program gaps and per-core mat/blend busy from a Tracy
`csvexport -u` dump (device zones are on the host clock there; thread = 1000 + 8*core + risc,
risc 0 BRISC, 1 NCRISC, 2-4 TRISC0-2). Device frames are clusters of pfwc starts (gap > 1 ms).
Host zones are on a different clock base than the device zones in the dump (offset ~seconds and
not constant), so host zones are reported separately on the host clock, anchored at each view's
first EnqueueProgram (host frames = clusters of EnqueueProgram starts, gap > 20 ms).
Stats are means over frames 1..N-1 (frame 0 is the warm view after JIT).

Usage: ana210.py <tracy-u.csv[.gz]> [label]"""
import bisect, csv, gzip, sys
from collections import defaultdict
import numpy as np

DEV = ["pfwc", "k2_pairs", "k2_rows", "sort_ol_prefix", "sort_ol_barrier", "sort_ol_fill",
       "sort_ol_fillb", "sort_ol_emit", "sort_ol_town", "sort_subchunk_mat", "mat_cull_mask",
       "tile_blend_load", "tile_blend_sfpu", "BRISC-FW", "NCRISC-FW", "TRISC-FW"]
HOST = ["EnqueueProgram", "host_cq1_proj_m", "host_cq1_k2_rows", "host_finish_cq1_bridge",
        "host_blend_setup", "host_finish_blend"]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    rows = defaultdict(list)
    with op(path, "rt") as f:
        for r in csv.reader(f):
            if len(r) < 8 or r[0] == "name" or (r[0] not in DEV and r[0] not in HOST):
                continue
            s = int(r[5]); rows[r[0]].append((s, s + int(r[6]), int(r[7])))
    return rows


def main():
    path = sys.argv[1]; label = sys.argv[2] if len(sys.argv) > 2 else path
    rows = load(path)
    ps = sorted(s for s, _, _ in rows["pfwc"])
    cuts = [ps[0]] + [b for a, b in zip(ps, ps[1:]) if b - a > 1_000_000]
    nfr = len(cuts)
    fr = defaultdict(lambda: defaultdict(list))        # frame -> zone -> [(s, e, th)]
    for z, iv in rows.items():
        for s, e, th in iv:
            fr[bisect.bisect_right(cuts, s + 200_000) - 1][z].append((s, e, th))
    F = [f for f in range(1, nfr) if fr[f]["tile_blend_sfpu"]]
    ms = lambda v: v / 1e6
    W = defaultdict(list)                               # zone -> [(start, end)] rel. to pfwc start
    for f in F:
        t0 = cuts[f]
        for z, iv in fr[f].items():
            W[z].append((ms(min(s for s, _, _ in iv) - t0), ms(max(e for _, e, _ in iv) - t0)))
    print(f"{label}: frames={len(F)} (of {nfr}); device ms from the view's first pfwc start")
    print(f"{'zone':<24}{'start':>9}{'end':>9}{'window':>9}")
    for z in sorted(W, key=lambda k: np.mean([a for a, _ in W[k]])):
        a = np.array(W[z]); print(f"{z:<24}{a[:, 0].mean():9.3f}{a[:, 1].mean():9.3f}{(a[:, 1] - a[:, 0]).mean():9.3f}")
    # host clock: frames = clusters of EnqueueProgram starts
    es = sorted(s for s, _, _ in rows["EnqueueProgram"])
    hc = [es[0]] + [b for a, b in zip(es, es[1:]) if b - a > 20_000_000]
    hf = defaultdict(lambda: defaultdict(list))
    for z in HOST:
        for s, e, th in rows[z]:
            hf[bisect.bisect_right(hc, s) - 1][z].append((s, e))
    HF = [f for f in range(1, len(hc)) if hf[f]["host_finish_blend"]]
    print(f"\nhost zones (host clock, ms from the view's first EnqueueProgram; {len(HF)} views)")
    hw = defaultdict(list)
    for f in HF:
        for z, iv in hf[f].items():
            if z != "EnqueueProgram":
                hw[z].append((ms(min(s for s, _ in iv) - hc[f]), ms(max(e for _, e in iv) - hc[f])))
        for k, (s, e) in enumerate(sorted(hf[f]["EnqueueProgram"])):
            hw[f"EnqueueProgram#{k}"].append((ms(s - hc[f]), ms(e - hc[f])))
    for z in sorted(hw, key=lambda k: np.mean([a for a, _ in hw[k]])):
        if len(hw[z]) >= len(HF) // 2:
            a = np.array(hw[z]); print(f"{z:<24}{a[:, 0].mean():9.3f}{a[:, 1].mean():9.3f}{(a[:, 1] - a[:, 0]).mean():9.3f}")

    def edge(z, which, f):
        iv = fr[f][z]
        return (min(s for s, _, _ in iv) if which == "s" else max(e for _, e, _ in iv)) if iv else None

    def gap(a, ae, b, bs):
        v = [ms(edge(b, bs, f) - edge(a, ae, f)) for f in F if fr[f][a] and fr[f][b]]
        return np.mean(v), np.min(v), np.max(v)
    print("\ngaps (mean min max, ms; negative = overlap)")
    for a, ae, b, bs, what in [
            ("pfwc", "e", "k2_pairs", "s", "pfwc end -> k2_pairs start"),
            ("k2_rows", "e", "sort_ol_prefix", "s", "k2_rows end -> sort prefix start"),
            ("sort_ol_emit", "e", "mat_cull_mask", "s", "emit end -> mat start"),
            ("sort_ol_town", "e", "mat_cull_mask", "s", "town end -> mat start"),
            ("mat_cull_mask", "e", "tile_blend_sfpu", "e", "mat end -> blend end"),
            ("tile_blend_sfpu", "e", "BRISC-FW", "e", "blend sfpu end -> last BRISC-FW end")]:
        m, lo, hi = gap(a, ae, b, bs); print(f"  {what:<46}{m:8.3f}{lo:8.3f}{hi:8.3f}")
    span = [ms(edge("BRISC-FW", "e", f) - cuts[f]) for f in F]
    span_b = [ms(edge("tile_blend_sfpu", "e", f) - cuts[f]) for f in F]
    idle = [ms(cuts[f + 1] - edge("BRISC-FW", "e", f)) for f in F if f + 1 < nfr]
    per = [ms(cuts[f + 1] - cuts[f]) for f in F if f + 1 < nfr]
    print(f"\ndevice span pfwc start -> last blend sfpu end  {np.mean(span_b):.3f} (min {np.min(span_b):.3f} max {np.max(span_b):.3f})")
    print(f"device span pfwc start -> last BRISC-FW end    {np.mean(span):.3f} (min {np.min(span):.3f} max {np.max(span):.3f})")
    print(f"traced view period (pfwc start -> next)        {np.mean(per):.3f}; device idle between views {np.mean(idle):.3f} (traced: includes the mid-run profiler dump)")

    # per-core busy: core = (thread-1000)//8; per core = max over its RISC threads of summed zone time
    def core_busy(z, f, risc=None):
        b = defaultdict(lambda: defaultdict(int)); first = {}; last = {}
        for s, e, th in fr[f][z]:
            c, r = divmod(th - 1000, 8)
            if risc is not None and r not in risc:
                continue
            b[c][r] += e - s; first[c] = min(first.get(c, s), s); last[c] = max(last.get(c, e), e)
        return {c: max(v.values()) for c, v in b.items()}, first, last
    print("\nper-core busy, ms (mean over frames of per-frame mean/min/max over cores; core = max over its RISCs)")
    print(f"{'zone':<34}{'mean':>8}{'min':>8}{'max':>8}{'cores':>7}{'core-ms':>9}{'window':>8}")
    fused = defaultdict(list)
    for z, risc in [("mat_cull_mask", None), ("sort_subchunk_mat", None), ("tile_blend_sfpu", None),
                    ("tile_blend_load", None), ("mat+blend (TRISC busy)", None), ("mat start->blend end span", None)]:
        M, MN, MX, NC, S, WN = [], [], [], [], [], []
        for f in F:
            if z == "mat+blend (TRISC busy)":
                a, _, _ = core_busy("mat_cull_mask", f, {2, 3, 4}); b, _, _ = core_busy("tile_blend_sfpu", f, {2, 3, 4})
                v = {c: a.get(c, 0) + b.get(c, 0) for c in set(a) | set(b)}
            elif z == "mat start->blend end span":
                _, fs, _ = core_busy("mat_cull_mask", f); _, _, le = core_busy("tile_blend_sfpu", f)
                v = {c: le[c] - fs[c] for c in fs if c in le}
            else:
                v, _, _ = core_busy(z, f, risc)
            x = np.array(list(v.values()), float) / 1e6
            M.append(x.mean()); MN.append(x.min()); MX.append(x.max()); NC.append(len(x)); S.append(x.sum())
            WN.append(ms(max(edge(z2, "e", f) for z2 in ["tile_blend_sfpu"]) - edge("mat_cull_mask", "s", f))
                      if "mat" in z and "blend" in z else (ms(edge(z, "e", f) - edge(z, "s", f)) if fr[f].get(z) else 0))
        print(f"{z:<34}{np.mean(M):8.3f}{np.mean(MN):8.3f}{np.mean(MX):8.3f}{int(np.mean(NC)):7d}{np.mean(S):9.3f}{np.mean(WN):8.3f}")


if __name__ == "__main__":
    main()
