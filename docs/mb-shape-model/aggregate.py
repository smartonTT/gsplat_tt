#!/usr/bin/env python3
"""Task #408: average mbsim outputs (out/*.txt) per shape and apply the cost models."""
import glob
import sys
from collections import defaultdict

BLEND_MS = 3.682            # blend ms/core per view (task spec)
DISP_MS = 65 / 1.35e9 / 110 * 1e3   # t148: 65 cycles per mb dispatch, 1.35 GHz, 110 cores
LIVE_MS = 0.197e-3 / 110            # t147 fit: 0.197 us per live record per tile, 110 cores
GATE_MS = 0.09

d = sys.argv[1] if len(sys.argv) > 1 else "out"
files = sorted(glob.glob(f"{d}/*.txt"))
acc = defaultdict(lambda: defaultdict(float))
shapes, exact = [], defaultdict(lambda: True)
for fn in files:
    for line in open(fn):
        t = line.split()
        if not t or t[0] != "shape":
            continue
        s = t[1]
        if s not in shapes:
            shapes.append(s)
        kv = dict(zip(t[2::2], t[3::2]))
        for k, v in kv.items():
            acc[s][k] += float(v)
        if int(kv["ch_diff_bits"]) != 0:
            exact[s] = False
nv = len(files)
b = acc["4x8"]
print(f"views={nv}")
print("| shape | dispatches/view | ratio vs 4x8 | pair ops/view | live records/view | lane fill | bit-exact (fp32) | u8 ch diffs/view | saving ms/view (spec 3.682 x ratio) | saving ms/view (t148 per-dispatch) | saving ms/view (t147 per-live-rec) |")
print("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
for s in shapes:
    a = acc[s]
    r = a["disp"] / b["disp"]
    print(f"| {s} | {a['disp']/nv:,.0f} | {r:.3f} | {a['pairs']/nv:,.0f} | {a['rec_live']/nv:,.0f} | "
          f"{a['lanes_live']/(32*a['disp']):.3f} | {'yes' if exact[s] else 'no'} | {a['ch_diff_u8']/nv:,.0f} | "
          f"{BLEND_MS*(1-r):+.3f} | {(b['disp']-a['disp'])/nv*DISP_MS:+.3f} | {(b['rec_live']-a['rec_live'])/nv*LIVE_MS:+.3f} |")
ok = [s for s in shapes if s != "4x8" and exact[s] and BLEND_MS * (1 - acc[s]["disp"] / b["disp"]) >= GATE_MS]
print(f"\nrecords/view {b['rec']/nv:,.0f}  cull-kept {b['rec_cull']/nv:,.0f}  4x8 disp without early stop {b['disp_nocut']/nv:,.0f}")
print("gate:", "PASS " + ",".join(ok) if ok else "FAIL (no shape is bit-exact and >= 0.09 ms/view)")
