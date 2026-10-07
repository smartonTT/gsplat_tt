#!/usr/bin/env python3
"""t370: per-view device span from a device profiler log (profile_log_device.csv[.gz]).
Span = first pfwc start -> last tile_blend_sfpu end of the view (device clock). Views are split
where pfwc starts jump by > 1 ms; view 0 (warm-up) is dropped. Also the in-view device-idle time
(gaps >= 5 us in the union of all *-FW zones over all cores) and the main zone windows.
untraced ms/view - span = the per-view time the device waits for the host (d2h, rtargs, enqueue).

Usage: dev_span.py <profile_log_device.csv[.gz]>
"""
import csv, gzip, sys
from collections import defaultdict

ZONES = ["pfwc", "k2_pairs", "sort_ol_emit", "mat_cull_mask", "tile_blend_sfpu"]


def main():
    p = sys.argv[1]; op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        mhz = float(f.readline().split("CHIP_FREQ[MHz]:")[1].split(",")[0]); f.readline()
        open_ = {}; z = defaultdict(list); fw = []
        for r in csv.reader(f):
            if len(r) < 12: continue
            name, typ, t = r[10].strip(), r[11].strip(), int(r[5])
            if not (name.endswith("-FW") or name in ZONES): continue
            k = (r[1], r[2], r[3], name)
            if typ == "ZONE_START": open_[k] = t
            elif typ == "ZONE_END" and k in open_:
                s = open_.pop(k)
                (fw if name.endswith("-FW") else z[name]).append((s, t))
    us = lambda c: c / mhz
    starts = sorted(s for s, _ in z["pfwc"])
    fs = [starts[0]] + [b for a, b in zip(starts, starts[1:]) if us(b - a) > 1000]
    fw.sort(); busy = []
    for a, b in fw:
        if busy and a <= busy[-1][1]: busy[-1][1] = max(busy[-1][1], b)
        else: busy.append([a, b])
    spans, idles, win = [], [], defaultdict(list)
    for i in range(1, len(fs)):
        f0, f1 = fs[i], (fs[i + 1] if i + 1 < len(fs) else float("inf"))
        ends = [e for s, e in z["tile_blend_sfpu"] if f0 <= s < f1]
        if not ends: continue
        fe = max(ends); spans.append(us(fe - f0))
        bb = [x for x in busy if x[1] > f0 and x[0] < fe]
        idles.append(sum(us(b[0] - a[1]) for a, b in zip(bb, bb[1:]) if us(b[0] - a[1]) >= 5))
        for n in ZONES:
            iv = [(s, e) for s, e in z[n] if f0 <= s < f1]
            if iv: win[n].append((us(min(s for s, _ in iv) - f0), us(max(e for _, e in iv) - f0)))
    n = len(spans); m = lambda v: sum(v) / len(v)
    print(f"views {n} (view 0 dropped), clock {mhz:.0f} MHz")
    print(f"device span pfwc start -> blend end: mean {m(spans)/1e3:.3f} ms "
          f"(min {min(spans)/1e3:.3f}, max {max(spans)/1e3:.3f}); in-view device idle {m(idles)/1e3:.3f} ms")
    for k in ZONES:
        if win[k]: print(f"  {k:16s} {m([a for a, _ in win[k]])/1e3:6.3f} - {m([b for _, b in win[k]])/1e3:6.3f} ms")


if __name__ == "__main__":
    main()
