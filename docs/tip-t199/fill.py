# t181: per-view makespan and per-core duration of the sort's window-fill zones
# (sort_ol_fill = NCRISC, sort_ol_fillb = BRISC, profiler builds), plus prefix
# and barrier, from a stitched dev30.csv. GB/s uses 8 B per pair (gid + tid) of
# the view's P from the capture log: an upper bound on the fill bytes (a mover
# range past OL_WIN_PAGES reads the rest in the emit).
#   fill.py dev30.csv [capture.log]
import collections, re, statistics as st, sys
CLK = 1350.0  # cycles per us
f = sys.argv[1]
zones = ("sort_ol_fill", "sort_ol_fillb", "sort_ol_prefix", "sort_ol_barrier", "sort_ol_emit")
ivs = {z: collections.defaultdict(list) for z in zones}
open_ = {}
for line in open(f):
    c = line.split(",")
    if len(c) < 12: continue
    z = c[10].strip()
    if z not in ivs: continue
    key = (z, c[1], c[2], c[3])
    t = int(c[5])
    if c[11].strip() == "ZONE_START": open_[key] = t
    elif key in open_: ivs[z][key[1:]].append((open_.pop(key), t))
Ps = []
if len(sys.argv) > 2:
    Ps = [int(m) for m in re.findall(r"stage=ONELAUNCH P=(\d+)", open(sys.argv[2]).read())]
for z in zones:
    d = ivs[z]
    if not d: print(z, "no zones"); continue
    nv = min(len(v) for v in d.values())
    mk = [(max(d[k][i][1] for k in d) - min(d[k][i][0] for k in d)) / CLK for i in range(1, nv)]
    dur = [(d[k][i][1] - d[k][i][0]) / CLK for k in d for i in range(1, nv)]
    print(f"{z:16s} riscs {len(d):3d} views {nv - 1:2d} makespan us mean {st.mean(mk):7.1f}  "
          f"per-core dur us med {st.median(dur):6.1f} max {max(dur):6.1f}")
    if z == "sort_ol_fill" and Ps:
        P = st.mean(Ps[-(nv - 1):]) if len(Ps) >= nv - 1 else st.mean(Ps)
        print(f"  P mean {P:.0f}: fill bytes <= {8 * P / 1e6:.1f} MB/view")
# Both movers' fills together (union per view, start of NCRISC fill to last end).
a, b = ivs["sort_ol_fill"], ivs["sort_ol_fillb"]
if a and b and Ps:
    nv = min(min(len(v) for v in a.values()), min(len(v) for v in b.values()))
    mk = [(max(max(a[k][i][1] for k in a), max(b[k][i][1] for k in b)) -
           min(min(a[k][i][0] for k in a), min(b[k][i][0] for k in b))) / CLK for i in range(1, nv)]
    P = st.mean(Ps[-(nv - 1):]) if len(Ps) >= nv - 1 else st.mean(Ps)
    print(f"fill union makespan us {st.mean(mk):.1f}; {8 * P / 1e3 / st.mean(mk):.1f} GB/s (8 B/pair, P {P:.0f})")
