#!/usr/bin/env python3
"""t439: per-tile pfwc weights for GSPLAT_TT_PFWC_DEAL=lpt from the t437 CPU model.
w[t] = a + b * visible[hero, t] (us), a, b fitted on the 12x10 traces (coef-dev-E12.npy).
Usage: make_weights.py TILEWORK_NPZ COEF_NPY OUT_TXT [VIEW=hero]"""
import sys
import numpy as np
tw, coef, out = sys.argv[1:4]
view = sys.argv[4] if len(sys.argv) > 4 else "hero"
t = np.load(tw)
vi = list(t["order"]).index(view)
a, b = (float(x) * 1e3 for x in np.load(coef)[:2])  # ms -> us
w = a + b * t["vis"][vi].astype(np.float64)
with open(out, "w") as f:
    f.write(f"# t439 pfwc tile weights (us): {a:.4f} + {b:.6f} * visible, view {view}, {len(w)} tiles\n")
    for x in w: f.write(f"{x:.4f}\n")
print(f"{out}: {len(w)} tiles, w min {w.min():.2f} mean {w.mean():.2f} max {w.max():.2f}")
