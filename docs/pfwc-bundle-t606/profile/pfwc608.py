#!/usr/bin/env python3
"""t608: per-RISC busy of the pfwc program window from a t595-style device csv (FOLD=0, 4 programs
per view: pfwc, k2, sort, mat+blend; 11 views per chunk, the last K=5 are the measured b2b pass).

Per view (pfwc program only):
  window      = first FW start of any RISC on any core -> last FW end
  slowest core = the core whose last RISC ends latest
  busy        = that RISC's KERNEL zone length (FW length if no KERNEL zone) in the pfwc program
  end         = that RISC's FW end - window start
  setter      = the RISC whose KERNEL zone ends last on the slowest core (BRISC-FW always ends
                last: it waits for the other RISCs, so FW ends do not name the setter)
Also the mean over cores per RISC and, over all cores, how often each RISC is the last to end.
STEPCYC captures (GSPLAT_TT_PFWC_STEPCYC=1) add the movers' pfwc_ws split (t484/t598 names:
n wall wait cls pfx rec opn tail rd fl m; wait = the color reads and the 10 compute tiles) and
the TRISCs' pfwc_pc wall and input wait, for the slowest core and the mean core.
Usage: pfwc608.py DEV.csv[.gz] [--keep 5] [--label X]
"""
import argparse
import bisect
import csv
import gzip
from collections import Counter, defaultdict

CYC_MS = 1350.0 * 1000.0
NP = 4
RISCS = ["TRISC_0", "TRISC_1", "TRISC_2", "NCRISC", "BRISC"]
FWZ = {"BRISC": "BRISC-FW", "NCRISC": "NCRISC-FW", "TRISC_0": "TRISC-FW", "TRISC_1": "TRISC-FW",
       "TRISC_2": "TRISC-FW"}
KZ = {"BRISC": "BRISC-KERNEL", "NCRISC": "NCRISC-KERNEL", "TRISC_0": "TRISC-KERNEL",
      "TRISC_1": "TRISC-KERNEL", "TRISC_2": "TRISC-KERNEL"}


def ms(c):
    return c / CYC_MS


WS = "n wall wait cls pfx rec opn tail rd fl m".split()


def load(path):
    iv, stack, ts = defaultdict(list), {}, defaultdict(list)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        next(f)
        next(f)
        for p in csv.reader(f):
            if len(p) < 12:
                continue
            c, r, t, z, ty = (int(p[1]), int(p[2])), p[3].strip(), int(p[5]), p[10].strip(), p[11].strip()
            if ty == "TS_DATA" and z in ("pfwc_ws", "pfwc_pc"):
                v = int(p[6])
                i, val = (v >> 32) & 15, v & 0xFFFFFFFF
                if i == 0 or not ts[(c, r, z)]:
                    ts[(c, r, z)].append({})
                ts[(c, r, z)][-1][i] = val
            elif ty == "ZONE_START":
                stack[(c, r, z)] = t
            elif ty == "ZONE_END":
                s = stack.pop((c, r, z), None)
                if s is not None:
                    iv[(c, r)].append((z, s, t))
    return iv, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    iv, ts = load(a.csv)
    cores = sorted({c for c, _ in iv})
    # per[(view, core)][risc] = (fw_s, fw_e, busy)
    per = defaultdict(dict)
    nprog = None
    for c in cores:
        b = sorted((s, e) for z, s, e in iv[(c, "BRISC")] if z == "BRISC-FW")
        assert len(b) % NP == 0, (c, len(b))
        nprog = len(b) if nprog is None else nprog
        assert len(b) == nprog, (c, len(b))
        first = len(b) - NP * a.keep
        bs = [s for s, _ in b]
        for r in RISCS:
            fw = sorted((s, e) for z, s, e in iv.get((c, r), []) if z == FWZ[r])
            kz = sorted((s, e) for z, s, e in iv.get((c, r), []) if z == KZ[r])
            ks = [s for s, _ in kz]
            for s, e in fw:
                k0 = bisect.bisect_right(bs, s) - 1
                cand = [k for k in (k0 - 1, k0, k0 + 1) if 0 <= k < len(b)]
                k = max(cand, key=lambda k: (min(e, b[k][1]) - max(s, b[k][0]), -abs(b[k][0] - s)))
                if k < first or k % NP != 0:
                    continue
                i = bisect.bisect_left(ks, s)
                ins = [(kst, ke) for kst, ke in kz[i:i + 4] if ke <= e and kst >= s]
                busy = sum(ke - kst for kst, ke in ins) or (e - s)
                kend = max((ke for _, ke in ins), default=e)
                per[((k - first) // NP, c)][r] = (s, e, busy, kend)
    print(f"# {a.label} {a.csv}: {len(cores)} cores, {nprog} programs/core, last {a.keep} views, pfwc only")
    hdr = "view  window | slowest core | " + " ".join(f"{r:>8}" for r in RISCS) + " | setter"
    print(hdr)
    agg = defaultdict(list)
    last_cnt = Counter()
    for v in range(a.keep):
        d = {c: per[(v, c)] for c in cores if (v, c) in per}
        w0 = min(x[0] for rr in d.values() for x in rr.values())
        w1 = max(x[1] for rr in d.values() for x in rr.values())
        for c, rr in d.items():
            last_cnt[max(rr, key=lambda r: rr[r][3])] += 1
            for r in RISCS:
                if r in rr:
                    agg[("mean", r)].append(ms(rr[r][2]))
        cs = max(d, key=lambda c: max(x[1] for x in d[c].values()))
        agg["slow_core"].append(cs)
        rr = d[cs]
        setter = max(rr, key=lambda r: rr[r][3])
        agg["window"].append(ms(w1 - w0))
        for r in RISCS:
            agg[("slow_busy", r)].append(ms(rr[r][2]))
            agg[("slow_end", r)].append(ms(rr[r][3] - w0))
        agg["setter"].append(setter)
        print(f"{v:>4} {ms(w1 - w0):7.3f} | {str(cs):>12} | "
              + " ".join(f"{ms(rr[r][2]):8.3f}" for r in RISCS) + f" | {setter}")
        print(f"{'':>4} {'end':>7} | {'':>12} | "
              + " ".join(f"{ms(rr[r][3] - w0):8.3f}" for r in RISCS))
    n = a.keep
    mean = lambda k: sum(agg[k]) / len(agg[k])
    print(f"MEAN window {mean('window'):.3f} ms; setter per view: {Counter(agg['setter']).most_common()}")
    print("slowest-core busy  " + " ".join(f"{r}={mean(('slow_busy', r)):.3f}" for r in RISCS))
    print("slowest-core kernel end (from window start) " + " ".join(f"{r}={mean(('slow_end', r)):.3f}" for r in RISCS))
    print("mean-core busy     " + " ".join(f"{r}={mean(('mean', r)):.3f}" for r in RISCS))
    tot = sum(last_cnt.values())
    print("last KERNEL to end, all cores x views: "
          + " ".join(f"{r}={last_cnt[r] / tot:.0%}" for r in RISCS) + f" (n={tot}, views={n})")
    if not ts:
        return
    print("STEPCYC (ms; slowest core per view, mean over the 5 views | mean core):")
    cores_ts = sorted({c for c, _, _ in ts})
    for r, z, names in (("BRISC", "pfwc_ws", WS), ("NCRISC", "pfwc_ws", WS)):
        rows = {c: ts[(c, r, z)][-a.keep:] for c in cores_ts if len(ts.get((c, r, z), [])) >= a.keep}
        if not rows:
            continue
        for i, nm in enumerate(names):
            if nm in ("n", "m"):
                continue
            slow = sum(rows[c][v].get(i, 0) for v, c in enumerate(agg["slow_core"]) if c in rows) / n
            allc = sum(x.get(i, 0) for L in rows.values() for x in L) / (len(rows) * a.keep)
            print(f"  {r:6s} {nm:5s} {ms(slow):.3f} | {ms(allc):.3f}")
        sm = sum(rows[c][v].get(10, 0) for v, c in enumerate(agg["slow_core"]) if c in rows) / n
        print(f"  {r:6s} survivors/launch slowest core {sm:.0f}")
    for r in ("TRISC_0", "TRISC_1", "TRISC_2"):
        rows = {c: ts[(c, r, "pfwc_pc")][-a.keep:] for c in cores_ts
                if len(ts.get((c, r, "pfwc_pc"), [])) >= a.keep}
        if not rows:
            continue
        for i, nm in ((1, "wall"), (3, "in_wait"), (15, "vis+pops")):
            slow = sum(rows[c][v].get(i, 0) for v, c in enumerate(agg["slow_core"]) if c in rows) / n
            allc = sum(x.get(i, 0) for L in rows.values() for x in L) / (len(rows) * a.keep)
            print(f"  {r:7s} {nm:8s} {ms(slow):.3f} | {ms(allc):.3f}")


if __name__ == "__main__":
    main()
