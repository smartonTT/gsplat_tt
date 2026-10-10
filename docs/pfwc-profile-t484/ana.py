#!/usr/bin/env python3
"""t484: pfwc per-core breakdown from the b2b device-profiler CSVs (remote_prof.sh arms z, s, sr).
CPU only.  ana.py out/dev-z.csv.gz out/dev-s.csv.gz [out/dev-sr.csv.gz]

Per pfwc launch (run host id holding the TRISC_1 "pfwc" zone; the first launch, the hero
warm-up, is dropped):
  window     first kernel start on any core -> last BRISC/NCRISC/TRISC end on any core
  core end   per core, last RISC end - window start; imbalance = max / mean
  per RISC   kernel span (BRISC-KERNEL, NCRISC-KERNEL, TRISC_0/1/2 "pfwc" zone)
STEPCYC arms add the in-kernel split (wall cycles summed over each core's chunks):
  pfwc_pc  TRISC0/1/2: n wall init s0..s12 (s0 input wait, s1 transform, s2 recip, s3 depth,
           s4 means, s5 cov_cam, s6 cov2d a, c + radii, s7 cov2d b + conic (P2 fused), s12 vis + pops)
  pfwc_ws  BRISC (role 0, even chunks) / NCRISC (role 1, odd chunks + most input reads):
           n wall wait cls pfx rec opn tail rd fl m
"""
import csv, gzip, sys
from collections import defaultdict
import numpy as np

C = 1350.0  # cycles per us
KZ = {"BRISC": "BRISC-KERNEL", "NCRISC": "NCRISC-KERNEL"}
PC = ["n", "wall", "init", "in_wait", "xform", "recip", "depth", "means", "cov_cam", "cov2d a,c+radii",
      "cov2d b+conic", "s8", "s9", "s10", "s11", "vis+pops"]
WS = "n wall wait cls pfx rec opn tail rd fl m".split()
ms = lambda c: c / C / 1000.0


def load(path):
    """The run host ID column is 0 in these CSVs, so launches are matched by order: launch v is
    the v-th TRISC_1 "pfwc" zone of each core; kernel zones of the other RISCs are matched by
    overlap; timestamped data records start a new launch at index 0."""
    raw = defaultdict(list)  # (core, risc, nm) -> [(start, end)]
    st = {}
    ts = defaultdict(list)  # (core, risc, zone) -> [{idx: val}] per launch
    rows = []
    with gzip.open(path, "rt") as f:
        next(f); next(f)
        for p in csv.reader(f):
            if len(p) < 12:
                continue
            nm, ty = p[10].strip(), p[11].strip()
            if ty == "TS_DATA" and nm in ("pfwc_pc", "pfwc_ws"):
                rows.append((int(p[5]), (int(p[1]), int(p[2])), p[3].strip(), nm, ty, int(p[6])))
            elif ty in ("ZONE_START", "ZONE_END") and nm in ("pfwc", "BRISC-KERNEL", "NCRISC-KERNEL"):
                rows.append((int(p[5]), (int(p[1]), int(p[2])), p[3].strip(), nm, ty, 0))
    for t, core, risc, nm, ty, v in sorted(rows, key=lambda x: x[0]):
        if ty == "TS_DATA":
            i, val = (v >> 32) & 15, v & 0xffffffff
            k = (core, risc, nm)
            if i == 0 or not ts[k]:
                ts[k].append({})
            ts[k][-1][i] = val
            continue
        k = (core, risc, nm)
        if ty == "ZONE_START":
            st[k] = t
        elif k in st:
            raw[k].append((st.pop(k), t))
    return raw, ts


def launches(raw):
    cores = sorted({k[0] for k in raw if k[1] == "TRISC_1" and k[2] == "pfwc"})
    nv = min(len(raw[(c, "TRISC_1", "pfwc")]) for c in cores) if cores else 0
    return list(range(1, nv))  # drop the warm-up launch


def windows(raw, runs):
    cores = sorted({k[0] for k in raw if k[1] == "TRISC_1" and k[2] == "pfwc"})
    out = []
    for v in runs:
        rs = {}
        for c in cores:
            s1, e1 = raw[(c, "TRISC_1", "pfwc")][v]
            d = {"TRISC_1": (s1, e1)}
            for risc in ("TRISC_0", "TRISC_2"):
                l = raw.get((c, risc, "pfwc"), [])
                if len(l) > v: d[risc] = l[v]
            for risc, z in KZ.items():
                cand = [iv for iv in raw.get((c, risc, z), []) if iv[0] <= s1 + 20000 and iv[1] >= e1 - 20000]
                if cand: d[risc] = min(cand, key=lambda iv: abs(iv[0] - s1))
            rs[c] = d
        t0 = min(s for d in rs.values() for s, _ in d.values())
        out.append((v, t0, rs))
    return out


def summarize_windows(name, ws):
    win, mean_end, imb, start_skew = [], [], [], []
    per_risc = defaultdict(list)
    crit = defaultdict(int)
    for r, t0, rs in ws:
        ends = {c: max(e for _, e in d.values()) for c, d in rs.items()}
        starts = {c: min(s for s, _ in d.values()) for c, d in rs.items()}
        e = np.array([ms(v - t0) for v in ends.values()])
        win.append(e.max()); mean_end.append(e.mean()); imb.append(e.max() / e.mean())
        start_skew.append(ms(max(starts.values()) - t0))
        for c, d in rs.items():
            for risc, (s, en) in d.items():
                per_risc[risc].append(ms(en - s))
        cmax = max(ends, key=ends.get)
        d = rs[cmax]
        crit[max(d, key=lambda k: d[k][1])] += 1
    print(f"\n== {name}: {len(ws)} pfwc launches, {len(ws[0][2])} cores")
    print(f"  window (first start -> last end) mean {np.mean(win):.3f} ms  p50 {np.median(win):.3f}  "
          f"min {np.min(win):.3f} max {np.max(win):.3f}")
    print(f"  mean core end {np.mean(mean_end):.3f} ms   imbalance max/mean {np.mean(imb):.3f}   "
          f"last-core start skew {np.mean(start_skew):.3f} ms")
    for risc in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
        v = np.array(per_risc[risc])
        if len(v):
            print(f"  {risc:8s} kernel span mean {v.mean():.3f}  p90 {np.percentile(v, 90):.3f}  max {v.max():.3f}")
    print(f"  RISC that ends last on the slowest core: {dict(crit)}")
    return np.mean(win), np.mean(mean_end)


def summarize_steps(name, ts, runs):
    rs = set(runs)
    pc = defaultdict(list); ws = {0: [], 1: []}
    for (core, risc, z), lst in ts.items():
        for r, d in enumerate(lst):
            if r not in rs:
                continue
            if z == "pfwc_pc":
                pc[risc].append((core, r, [d.get(i, 0) for i in range(len(PC))]))
            else:
                role = 0 if risc == "BRISC" else 1
                ws[role].append((core, r, [d.get(i, 0) for i in range(len(WS))]))
    print(f"\n== {name}: in-kernel split, ms per launch (mean over cores; [p90 core]; slowest-wall core)")
    for risc in sorted(pc):
        a = np.array([v for _, _, v in pc[risc]], float)
        j = a[:, 1].argmax()
        print(f"  {risc} pfwc_pc ({len(a)} core-launches, chunks/core {a[:, 0].mean():.1f})")
        for i in range(1, len(PC)):
            if a[:, i].mean() < 1 and i >= 3:
                continue
            print(f"    {PC[i]:18s} {ms(a[:, i].mean()):.3f}  [{ms(np.percentile(a[:, i], 90)):.3f}]  {ms(a[j, i]):.3f}")
    for role, rows in ws.items():
        if not rows:
            continue
        a = np.array([v for _, _, v in rows], float)
        j = a[:, 1].argmax()
        print(f"  {'BRISC' if role == 0 else 'NCRISC'} pfwc_ws role {role} ({len(a)} core-launches, "
              f"chunks/core {a[:, 0].mean():.1f}, survivors/core {a[:, 10].mean():.0f} (max {a[:, 10].max():.0f}))")
        for i in range(1, 10):
            print(f"    {WS[i]:6s} {ms(a[:, i].mean()):.3f}  [{ms(np.percentile(a[:, i], 90)):.3f}]  {ms(a[j, i]):.3f}")
        busy = a[:, 3] + a[:, 4] + a[:, 5] + a[:, 6] + a[:, 7]
        rec_ex = a[:, 5] - a[:, 9]
        print(f"    busy(cls+pfx+rec+opn+tail) {ms(busy.mean()):.3f}; rec minus flush per survivor "
              f"{(rec_ex / np.maximum(a[:, 10], 1)).mean():.0f} cyc; cls per chunk {(a[:, 3] / np.maximum(a[:, 0], 1)).mean():.0f} cyc")
        r = np.corrcoef(a[:, 1], a[:, 10])[0, 1]
        print(f"    corr(wall, survivors) {r:.2f}")
    # Per core: total survivors vs TRISC_1 wall vs writer walls.
    t1 = {(c, r): v for c, r, v in pc.get("TRISC_1", [])}
    w0 = {(c, r): v for c, r, v in ws[0]}
    w1 = {(c, r): v for c, r, v in ws[1]}
    keys = sorted(set(t1) & set(w0) & set(w1))
    if keys:
        m = np.array([w0[k][10] + w1[k][10] for k in keys], float)
        tw = np.array([t1[k][1] for k in keys], float)
        tw_noin = np.array([t1[k][1] - t1[k][3] for k in keys], float)
        bw = np.array([w0[k][1] for k in keys], float)
        nw = np.array([w1[k][1] for k in keys], float)
        print(f"  per core: survivors mean {m.mean():.0f} max {m.max():.0f} (max/mean {m.max() / m.mean():.3f}); "
              f"corr(TRISC_1 wall, survivors) {np.corrcoef(tw, m)[0, 1]:.2f}; "
              f"TRISC_1 wall minus input wait mean {ms(tw_noin.mean()):.3f} max {ms(tw_noin.max()):.3f}")
        lim = np.argmax(np.stack([tw_noin, bw - np.array([w0[k][2] for k in keys]),
                                  nw - np.array([w1[k][2] for k in keys])]), axis=0)
        print(f"  busiest unit per core-launch (TRISC_1 busy / BRISC busy / NCRISC busy): "
              f"{np.bincount(lim, minlength=3).tolist()}")


if __name__ == "__main__":
    res = {}
    for path in sys.argv[1:]:
        name = path.split("/")[-1].replace(".csv.gz", "")
        spans, ts = load(path)
        runs = launches(spans)
        if not runs:
            print(f"{name}: no pfwc launches"); continue
        res[name] = summarize_windows(name, windows(spans, runs))
        if ts:
            summarize_steps(name, ts, runs)
    if len(res) > 1:
        names = list(res)
        b = res[names[0]]
        for n in names[1:]:
            print(f"\n{n} - {names[0]}: window {res[n][0] - b[0]:+.3f} ms, mean core end {res[n][1] - b[1]:+.3f} ms")
