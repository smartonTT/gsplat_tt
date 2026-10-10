#!/usr/bin/env python3
"""K2 compute vs NoC split and the K2-into-sort_ol fusion model (task #274).

Needs a capture with GSPLAT_TT_K2_PROF=1: every K2 mover (BRISC, NCRISC of each
core) then records "k2p_*" timestamped-data markers at its end, value = cycles
summed over that launch (k2p_npg = pair pages of its range).

Part 1, per mover (ms/view): setup (count table, seg table, pub), read issue,
read wait, write issue, writes-flushed wait, the k2_pairs loop, rows, final
write barrier, kernel total. compute = pairs - (read issue + read wait + write
issue + flushed wait); NoC = the rest of the loop + final barrier.

Part 2, route (a) model: K2 and sort_ol in one launch. The sort's BRISC prefix
reads every core's count row, so it needs a global barrier after the slowest
K2 core; NCRISC's window fill needs only its own core's K2. Per view and core:
  prefix'  = Kmax + BL + (prefix_start - sort_kernel_start)
  BRISC shift d_c = prefix_start - prefix'
  release' = max_c(barrier_start_c - d_c) + (release - max_c barrier_start_c)
  emit'    = emit_start - (release - release')   (NCRISC: also >= own fill end')
  sort_end' = max over movers of emit_end - (emit_start - emit')
saving = sort_end - sort_end' (mean over views). BL = barrier latency (2 us).

Usage: k2_parts.py <dev30.csv> [n_views=30]
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0
DATA, ZONE, TYPE = 6, 10, 11
BL = 2e-3 * CYC_MS  # barrier latency, cycles
PARTS = ["k2p_setup", "k2p_riss", "k2p_rdw", "k2p_wiss", "k2p_wfl", "k2p_pairs", "k2p_rows",
         "k2p_wbar", "k2p_tot"]


def main():
    path = sys.argv[1]
    nv = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    _, lines, ts = read_csv(path)
    frame, n_frames, _ = segment_frames(lines, ts)
    val = defaultdict(lambda: defaultdict(int))  # mover -> part -> cycles
    npg = defaultdict(int)
    stack = {}
    # (frame, core, risc) -> list of (start, end, zone)
    iv = defaultdict(list)
    for ln, t, fr in zip(lines, ts, frame):
        if fr < 0:
            continue
        p = ln.split(",", 12)
        z, typ = p[ZONE], p[TYPE]
        core, risc = (p[1], p[2]), p[3]
        if z.startswith("k2p_"):
            try:
                v = int(p[DATA])
            except ValueError:
                continue
            if z == "k2p_npg":
                npg[(core, risc)] += v
            else:
                val[(core, risc)][z] += v
            continue
        key = (core, risc, z)
        if typ == "ZONE_START":
            stack[key] = int(t)
        elif typ == "ZONE_END" and key in stack:
            iv[(fr, core, risc)].append((stack.pop(key), int(t), z))

    def ms(c):
        return c / CYC_MS / nv

    movers = sorted(val)
    if movers:
        print(f"== Part 1: K2 mover split ({len(movers)} movers, ms/view)")
        groups = {"mean": movers, "BRISC": [m for m in movers if m[1] == "BRISC"],
                  "NCRISC": [m for m in movers if m[1] == "NCRISC"]}
        busiest = max(movers, key=lambda m: val[m]["k2p_tot"])
        cols = list(groups) + ["busiest"]
        print(f"{'part':<40}" + "".join(f"{c:>10}" for c in cols))

        def row(name, f):
            vs = [sum(f(m) for m in g) / max(len(g), 1) for g in groups.values()] + [f(busiest)]
            print(f"{name:<40}" + "".join(f"{v:>10.3f}" for v in vs))

        for k in PARTS:
            row(k, lambda m, k=k: ms(val[m][k]))
        noc = ["k2p_riss", "k2p_rdw", "k2p_wiss", "k2p_wfl"]
        row("loop NoC (riss+rdw+wiss+wfl)", lambda m: ms(sum(val[m][k] for k in noc)))
        row("loop compute (pairs - NoC)", lambda m: ms(val[m]["k2p_pairs"] - sum(val[m][k] for k in noc)))
        row("NoC wait only (rdw+wfl+wbar)",
            lambda m: ms(val[m]["k2p_rdw"] + val[m]["k2p_wfl"] + val[m]["k2p_wbar"]))
        tot = sum(val[m]["k2p_tot"] for m in movers)
        comp = sum(val[m]["k2p_pairs"] - sum(val[m][k] for k in noc) + val[m]["k2p_setup"]
                   + val[m]["k2p_rows"] for m in movers)
        loopc = sum(val[m]["k2p_pairs"] - sum(val[m][k] for k in noc) for m in movers)
        print(f"compute share of K2 busy: loop compute {100 * loopc / tot:.1f} %, "
              f"loop+setup+rows {100 * comp / tot:.1f} %")
        pairs = sum(npg[m] for m in movers) * 16
        if pairs:
            print(f"pairs per view {pairs / nv:.0f}; loop cycles/pair "
                  f"{sum(val[m]['k2p_pairs'] for m in movers) / pairs:.1f}, compute cycles/pair "
                  f"{loopc / pairs:.1f}")
    else:
        print("no k2p_* markers (capture with GSPLAT_TT_K2_PROF=1); Part 1 skipped")

    # Part 2: route (a) model.
    savings, ubs, gaps = [], [], []
    for f in range(n_frames):
        cores = {c for (fr, c, r) in iv if fr == f}
        k2e, sk, pst, bst, bend, est, eend, fend, skn = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for c in cores:
            for r in ("BRISC", "NCRISC"):
                zs = sorted(iv.get((f, c, r), []))
                kern = [z for z in zs if z[2].endswith("-KERNEL")]
                names = [z for z in zs if not z[2].endswith(("-KERNEL", "-FW"))]

                def owner(zname):
                    for a, b, n in names:
                        if n == zname:
                            for ka, kb, _ in kern:
                                if ka <= a <= kb:
                                    return (ka, kb), (a, b)
                    return None, None

                k, _ = owner("k2_pairs")
                if k:
                    k2e[c] = max(k2e.get(c, 0), k[1])
                k, e = owner("sort_ol_emit")
                if k:
                    est[(c, r)], eend[(c, r)] = e
                    if r == "BRISC":
                        sk[c] = k[0]
                    else:
                        skn[c] = k[0]
                if r == "BRISC":
                    _, pz = owner("sort_ol_prefix")
                    _, bz = owner("sort_ol_barrier")
                    if pz and bz:
                        pst[c] = pz[0]
                        bst[c], bend[c] = bz
                else:
                    _, fz = owner("sort_ol_fill")
                    if fz:
                        fend[c] = fz[1]
        ok = [c for c in cores if c in k2e and c in sk and c in pst and c in skn and c in fend
              and (c, "BRISC") in est and (c, "NCRISC") in est]
        if not ok:
            continue
        kmax = max(k2e[c] for c in ok)
        d = {c: pst[c] - (kmax + BL + (pst[c] - sk[c])) for c in ok}
        rel = max(bend[c] for c in ok)
        rel2 = max(bst[c] - d[c] for c in ok) + (rel - max(bst[c] for c in ok))
        sh = rel - rel2
        end0 = max(eend.values())
        end1 = 0
        for c in ok:
            for r in ("BRISC", "NCRISC"):
                e2 = est[(c, r)] - sh
                if r == "NCRISC":
                    e2 = max(e2, k2e[c] + (fend[c] - skn[c]))
                end1 = max(end1, eend[(c, r)] - (est[(c, r)] - e2))
        savings.append((end0 - end1) / CYC_MS)
        ubs.append((min(sk[c] for c in ok) - kmax) / CYC_MS)
        gaps.append(sum(kmax - k2e[c] for c in ok) / len(ok) / CYC_MS)
    if savings:
        n = len(savings)
        print(f"== Part 2: route (a) model, K2 inside the sort_ol launch ({n} views)")
        print(f"K2 end (max core) -> sort kernel start (first core): {sum(ubs) / n:.3f} ms/view")
        print(f"K2 tail: mean core end -> max core end: {sum(gaps) / n:.3f} ms/view")
        print(f"modeled saving (sort_ol end moves earlier by): mean {sum(savings) / n:.3f}, "
              f"min {min(savings):.3f}, max {max(savings):.3f} ms/view")
    else:
        print("Part 2: K2 / sort_ol zones not found")


if __name__ == "__main__":
    main()
