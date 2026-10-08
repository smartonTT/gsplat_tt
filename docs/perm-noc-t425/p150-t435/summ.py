#!/usr/bin/env python3
"""t435: per-run view_total/frame/blend + md5 golden status from p150/out."""
import glob, os, re, sys, json
O = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"{O}/r*-[NZ].log")):
    tag = os.path.basename(f)[:-4]; s = open(f).read()
    m = re.search(r"^STAGES .*", s, re.M); st = dict(re.findall(r"(\w+)=([-+\d.]+)", m.group(0))) if m else {}
    g = re.findall(r"MD5_GOLDEN_\w+[^\n]*", s)
    rows[tag] = {k: float(st[k]) for k in ("view_total", "avg_frame_ms", "project", "sort", "blend", "xview") if k in st}
    rows[tag]["golden"] = g[-1][:80] if g else None
for t, r in rows.items(): print(t, r)
for a in "NZ":
    v = [rows[t]["view_total"] for t in sorted(rows) if t.endswith(a) and "view_total" in rows[t]]
    fr = [rows[t]["avg_frame_ms"] for t in sorted(rows) if t.endswith(a) and "avg_frame_ms" in rows[t]]
    if v: print(a, "view_total", v, "mean", round(sum(v)/len(v), 3), "fps", round(1000/(sum(v)/len(v)), 1), "frame", fr, round(sum(fr)/len(fr), 3))
