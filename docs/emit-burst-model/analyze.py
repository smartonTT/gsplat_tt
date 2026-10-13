"""Task #637: tables from model.py output (all out-C120-v*.json under this folder, merged by view).
ms scale: t479 calibration (calib.py, C=110 p100a): 1.42 ms / e3_max of the today arm at R8 D8 L
(9.86e-5 ms per record-time at L=4); same Blackhole clock, applied to C=120 record-times.
Usage: python3 docs/emit-burst-model/analyze.py [L]"""
import glob
import json
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
L = int(sys.argv[1]) if len(sys.argv) > 1 else 4
res = {}
for f in sorted(glob.glob(os.path.join(here, "out-C120-v*.json"))):
    for v, o in json.load(open(f))["res"].items():
        res.setdefault(v, {}).update(o)
cal = json.load(open(os.path.join(here, "out-C110-v0-10.json")))["res"]
u = 1.42 / np.mean([cal[v]["today"]["sim"][f"8/8/{L}"]["e3_max"] for v in cal])
views = list(res)
opts = [o for o in res[views[0]] if all(o in res[v] for v in views)]
N = res[views[0]]["today"]["slots"]
PF = 1.063  # pfwc ms/view at full N (t227 fit), scales with slots read
print(f"{len(views)} views, C=120, L={L}, {u * 1e3:.4f} us per record-time; N={N}")


def pct(h, q):
    c = np.cumsum(h)
    return int(np.searchsorted(c, q * c[-1]))


print("| order | skip (slots read) | burst p50 / p99 / max (records, same tile) | records in bursts >4 / >8 "
      "| emit Δ ms R8D8 | emit Δ ms R4D8 | gross pfwc saving ms | 1-packer stall, record-times |")
print("|---|---|---|---|---|---|---|---|")
rows = {}
for o in opts:
    skip = np.mean([1 - res[v][o]["slots"] / N for v in views])
    h = np.sum([np.array(res[v][o]["sim"][f"8/8/{L}"]["hist"], float) for v in views], axis=0)
    mx = max(int(np.nonzero(np.array(res[v][o]["sim"][f"8/8/{L}"]["hist"]))[0].max()) for v in views)
    gt4 = h[5:].sum() / h.sum()
    gt8 = h[9:].sum() / h.sum()
    d = {}
    for rd in ("8/8", "4/8"):
        a = np.mean([res[v][o]["sim"][f"{rd}/{L}"]["e3_max"] for v in views])
        b = np.mean([res[v]["today"]["sim"][f"8/8/{L}"]["e3_max"] for v in views])
        d[rd] = u * (a - b)
    st = np.mean([res[v][o]["sim"][f"8/8/{L}"]["stall_mean"] for v in views])
    rows[o] = (skip, d)
    print(f"| {o} | {skip:.3f} | {pct(h, .5)} / {pct(h, .99)} / {mx}{'+' if mx >= 1023 else ''} "
          f"| {gt4:.3f} / {gt8:.3f} | {d['8/8']:+.3f} | {d['4/8']:+.3f} | {skip * PF:.3f} | {st:.3f} |")
print()
print("per-view spread of emit Δ ms (R4D8): min / median / max")
for o in opts:
    b = [u * (res[v][o]["sim"][f"4/8/{L}"]["e3_max"] - res[v]["today"]["sim"][f"8/8/{L}"]["e3_max"])
         for v in views]
    print(f"  {o:10s} {min(b):+.3f} / {np.median(b):+.3f} / {max(b):+.3f}")
