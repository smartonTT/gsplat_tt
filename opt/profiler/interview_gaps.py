#!/usr/bin/env python3
"""Device-idle time between views: per frame, the first and last *-FW zone on any
core; prints each frame's device span and the gap from the previous frame's last
FW end to this frame's first FW start (host work + dispatch between views, as the
device sees it; inflated by the profiler's mid-run dump, so confirm untraced).

Usage: interview_gaps.py <csv> [--skip-first]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stitch_device_csv import read_csv, segment_frames  # noqa: E402

CYC_MS = 1350.0 * 1000.0
ZONE, TYPE = 10, 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--skip-first", action="store_true")
    args = ap.parse_args()
    _, lines, ts = read_csv(args.csv)
    frame, n_frames, _ = segment_frames(lines, ts)
    lo = [None] * n_frames
    hi = [None] * n_frames
    for ln, t, fr in zip(lines, ts, frame):
        p = ln.split(",", 12)
        if fr < 0 or not p[ZONE].endswith("-FW"):
            continue
        t = int(t)
        if p[TYPE] == "ZONE_START" and (lo[fr] is None or t < lo[fr]):
            lo[fr] = t
        if p[TYPE] == "ZONE_END" and (hi[fr] is None or t > hi[fr]):
            hi[fr] = t
    first = 1 if args.skip_first else 0
    spans, gaps = [], []
    for f in range(first, n_frames):
        if lo[f] is None or hi[f] is None:
            continue
        spans.append((hi[f] - lo[f]) / CYC_MS)
        if f > first and hi[f - 1] is not None:
            gaps.append((lo[f] - hi[f - 1]) / CYC_MS)
    s, g = np.array(spans), np.array(gaps)
    print(f"frames={len(s)} device span ms mean={s.mean():.3f} min={s.min():.3f} max={s.max():.3f}")
    if len(g):
        print(f"gap between views ms mean={g.mean():.3f} median={np.median(g):.3f} min={g.min():.3f} max={g.max():.3f}")


if __name__ == "__main__":
    main()
