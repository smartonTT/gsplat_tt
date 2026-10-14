#!/usr/bin/env python3
"""Task #691 (D1): table + fits from the sweep logs (out/logs) and the CPU counts (out/counts.npz).
Fits b2b ms/view = floor + slope * pairs/view (all points, and per resolution), and
b2b = a + b * pairs + c * N (pfwc walks every Gaussian). Pairs are reference-CPU pairs
(gsplat/rasterization.py, mean over the 169 views), the same per-Gaussian per-tile cull for every N.
  analyze.py [OUT_DIR]   -> prints markdown, writes OUT_DIR/summary.json
"""
import json, re, sys, statistics as st
from pathlib import Path
import numpy as np

O = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "out")
L = O / "logs"
z = np.load(O / "counts.npz"); rank = np.load(O / "rank.npy")
V = json.load(open(O / "prep.json"))["views"]
N0 = len(rank)
NT = {"full": N0, "3M": 3_000_000, "1p5M": 1_500_000, "1M": 1_000_000, "0p5M": 500_000}
S = (1, 2, 4)
pairs_g, vis_g = z["pairs"], z["vis"]
order = np.argsort(np.argsort(-z["score"], kind="stable"))  # position of each id in rank
rows = []
for nt, n in NT.items():
    keep = rank[:n]
    for k, s in enumerate(S):
        cfg = f"{nt}_s{s}"
        r = dict(cfg=cfg, N=n, s=s, W=1237 // s, H=822 // s,
                 pairs=float(pairs_g[k][keep].sum()) / V, vis=float(vis_g[k][keep].sum()) / V)
        b = []
        for rr in (1, 2):
            f = L / f"run-r{rr}-b2b-{cfg}.log"
            m = f.exists() and re.search(r"TTW_TIMING b2b_ms_frame_median=([\d.]+)", f.read_text())
            if m:
                b.append(float(m.group(1)))
        r["b2b_rounds"] = b
        r["b2b"] = st.mean(b) if b else None
        f = L / f"run-r1-lat-{cfg}.log"
        if f.exists():
            t = f.read_text()
            m = re.search(r"TTW_TIMING ms_view=([\d.]+)", t)
            r["lat"] = float(m.group(1)) if m else None
            r["stages"] = {k2: float(v) for k2, v in re.findall(r"TTW_TIMING stage_([a-z_0-9]+)=([-\d.]+)", t)
                           if k2 in ("head", "project", "tile_assign", "sort", "blend_setup", "mat", "cull",
                                     "blend", "d2h", "assemble", "tail", "xview", "view_total")}
        rows.append(r)

def fit(pts, cols):
    X = np.array([[1.0] + [p[c] for c in cols] for p in pts]); y = np.array([p["b2b"] for p in pts])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return coef, float(np.sqrt((res ** 2).mean())), float(np.abs(res).max())

ok = [r for r in rows if r["b2b"] is not None]
fits = {}
if len(ok) >= 3:
    c, rms, mx = fit(ok, ["pairs"]); fits["all: floor + slope*pairs"] = dict(floor_ms=c[0], ns_per_pair=c[1] * 1e6, rms=rms, max=mx)
    c, rms, mx = fit(ok, ["pairs", "N"]); fits["all: a + b*pairs + c*N"] = dict(floor_ms=c[0], ns_per_pair=c[1] * 1e6, ns_per_N=c[2] * 1e6, rms=rms, max=mx)
    for s in S:
        p = [r for r in ok if r["s"] == s]
        if len(p) >= 3:
            c, rms, mx = fit(p, ["pairs"]); fits[f"s={s}: floor + slope*pairs"] = dict(floor_ms=c[0], ns_per_pair=c[1] * 1e6, rms=rms, max=mx)
            c, rms, mx = fit(p, ["pairs", "N"]); fits[f"s={s}: a + b*pairs + c*N"] = dict(floor_ms=c[0], ns_per_pair=c[1] * 1e6, ns_per_N=c[2] * 1e6, rms=rms, max=mx)
print("| N | res | pairs/view (CPU ref) | visible/view | b2b ms/view (r1, r2) | latency ms/view | project | tile_assign | sort | blend+mat+cull | other |")
print("|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    sg = r.get("stages") or {}
    bl = sum(sg.get(k, 0) for k in ("blend", "mat", "cull", "blend_setup"))
    oth = sum(sg.get(k, 0) for k in ("head", "d2h", "assemble", "tail", "xview"))
    b = f"{r['b2b']:.3f} ({', '.join(f'{x:.3f}' for x in r['b2b_rounds'])})" if r["b2b"] else "-"
    lat = r.get("lat")
    print(f"| {r['N']/1e6:.2f}M | {r['W']}x{r['H']} | {r['pairs']/1e6:.3f}M | {r['vis']/1e6:.3f}M | {b} | "
          f"{lat if lat is None else f'{lat:.3f}'} | {sg.get('project', 0):.3f} | {sg.get('tile_assign', 0):.3f} | "
          f"{sg.get('sort', 0):.3f} | {bl:.3f} | {oth:.3f} |")
print()
for k, v in fits.items():
    print(f"- {k}: " + ", ".join(f"{a}={b:.4g}" for a, b in v.items()))
json.dump(dict(rows=rows, fits=fits), open(O / "summary.json", "w"), indent=1)
