#!/usr/bin/env python3
"""t523: summarize the combined-stack b2b A/B (bench523.sh output).

  ana523.py <out dir> [reference png]
Prints per-arm round medians (B2B ms_frame_median), the median per arm, per-round paired deltas
against 'off' and 'ec', raw_md5 per arm, and hero PSNR against the reference if heroes exist.
"""
import re, statistics as st, sys
from pathlib import Path

O = Path(sys.argv[1]); REF = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/reference_v2/hero.png"
B = re.compile(r"^B2B .*?ms_frame_median=([\d.]+).*?raw_md5=(\w+)", re.M)
runs = {}
for f in sorted(O.glob("run-r*-*.log")):
    m = re.match(r"run-r(\d+)-(\w+)\.log", f.name); b = B.search(f.read_text(errors="replace"))
    if m and b:
        runs[(int(m.group(1)), m.group(2))] = (float(b.group(1)), b.group(2))
arms = ["off", "ec", "st"]; rounds = sorted({r for r, _ in runs})
med = {}
for a in arms:
    v = [runs[(r, a)][0] for r in rounds if (r, a) in runs]
    md5 = sorted({runs[(r, a)][1] for r in rounds if (r, a) in runs})
    if v:
        med[a] = st.median(v)
        print(f"{a:3s} n={len(v)} rounds={' '.join(f'{x:.3f}' for x in v)} median={med[a]:.3f} raw_md5={','.join(md5)}")
for base, arm in (("off", "ec"), ("off", "st"), ("ec", "st")):
    d = [runs[(r, arm)][0] - runs[(r, base)][0] for r in rounds if (r, arm) in runs and (r, base) in runs]
    if d:
        print(f"{arm}-{base}: median-of-arms {med[arm] - med[base]:+.3f}  paired median {st.median(d):+.3f}  "
              f"rounds {' '.join(f'{x:+.3f}' for x in d)}")
try:
    import numpy as np
    from PIL import Image
    ref = np.asarray(Image.open(REF).convert("RGB"), dtype=np.float64)
    for a in arms:
        p = O / f"hero-{a}.png"
        if p.exists():
            img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
            mse = ((img - ref) ** 2).mean()
            print(f"hero-{a}: PSNR {10 * np.log10(255.0 ** 2 / mse):.2f} dB vs {REF}")
except ImportError as e:
    print("PSNR skipped:", e)
