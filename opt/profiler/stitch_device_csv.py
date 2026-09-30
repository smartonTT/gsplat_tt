#!/usr/bin/env python3
"""Stitch per-chunk tt-metal profile_log_device.csv files into one 30-view CSV.

A chunked capture (capture_tracy_chunked.sh) runs render/run.py once per view
range, each in its own devrun job. Every chunk starts with a warmup render of
the hero view, so each chunk CSV holds 1 warmup frame + its timed views.

Frames are found from the anchor zone (default proj_count, in the first program
of a frame: exactly one ZONE_START per worker core per frame). Anchor starts
cluster per frame. A frame begins at the first marker of the contiguous run
(consecutive markers < LEAD_GAP apart) that leads into its anchor cluster, which
picks up the first program's FW/pfwc lead-in but stops at the host gap between
frames. Widest-gap splitting does NOT work: the sort emit holds a ~30 ms
marker-free stretch inside a frame, and warmup JIT leaves multi-second gaps.
Checks, so a truncated/overflowed chunk or a bad split fails loudly: every frame
has the same number of anchor starts, and every zone's START/END counts balance
per frame (no zone straddles a boundary).

The warmup frame of every chunk is dropped unless --keep-warmup is given.
Timestamps of a chunk are shifted when needed so chunks never overlap in time
(the device may be reset between jobs).

Usage: stitch_device_csv.py -o OUT.csv CHUNK1.csv [CHUNK2.csv ...]
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

ANCHOR = "proj_count"
T_COL, ZONE_COL, TYPE_COL = 5, 10, 11
CLUSTER_GAP_CYC = 20 * 1350 * 1000     # 20 ms: anchors of one frame start within us
LEAD_GAP_CYC = 5 * 1350 * 1000         # 5 ms: marker gap inside a frame's lead-in
CHUNK_GAP_CYC = 100 * 1350 * 1000      # idle gap inserted between shifted chunks


def read_csv(path):
    """-> (header_lines[2], data_lines[], timestamps int64[])"""
    with open(path) as f:
        header = [f.readline(), f.readline()]
        lines = [ln for ln in f if ln.strip()]
    ts = np.empty(len(lines), dtype=np.int64)
    for i, ln in enumerate(lines):
        ts[i] = int(ln.split(",", 6)[T_COL])
    return header, lines, ts


def segment_frames(lines, ts, anchor=ANCHOR):
    """Assign every row a frame index. Returns (frame_of_row int[], n_frames, cores/frame)."""
    anchors = np.sort(np.array([t for ln, t in zip(lines, ts)
                                if ln.split(",", 12)[ZONE_COL] == anchor
                                and ln.split(",", 12)[TYPE_COL] == "ZONE_START"],
                               dtype=np.int64))
    if anchors.size == 0:
        raise ValueError(f"no {anchor} ZONE_START rows: not a render_clean capture")
    breaks = np.nonzero(np.diff(anchors) > CLUSTER_GAP_CYC)[0] + 1
    clusters = np.split(anchors, breaks)
    sizes = {c.size for c in clusters}
    if len(sizes) != 1:
        raise ValueError(f"uneven {anchor} starts per frame {[c.size for c in clusters]}"
                         " (truncated capture or profiler buffer overflow?)")
    all_sorted = np.sort(ts)
    bounds = []
    for prev, cur in zip(clusters, clusters[1:]):
        i = int(np.searchsorted(all_sorted, cur.min(), side="left"))
        while i > 0 and all_sorted[i] - all_sorted[i - 1] < LEAD_GAP_CYC:
            i -= 1
            if all_sorted[i] <= prev.max():
                raise ValueError("no host gap between frames: cannot split")
        bounds.append(all_sorted[i])
    frame = np.searchsorted(np.array(bounds, dtype=np.int64), ts, side="right")
    bal = defaultdict(int)
    for ln, f in zip(lines, frame):
        p = ln.split(",", 12)
        if p[TYPE_COL] in ("ZONE_START", "ZONE_END"):
            bal[(int(f), p[ZONE_COL])] += 1 if p[TYPE_COL] == "ZONE_START" else -1
    bad = sorted((f, z, n) for (f, z), n in bal.items() if n)
    if bad:
        raise ValueError(f"zones straddle frame boundaries (frame, zone, START-END): {bad[:8]}")
    return frame, len(clusters), sizes.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("chunks", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--keep-warmup", action="store_true")
    ap.add_argument("--anchor", default=ANCHOR)
    args = ap.parse_args()

    out_header, out_lines = None, []
    prev_max = None
    total_frames, cores = 0, None
    for path in args.chunks:
        header, lines, ts = read_csv(path)
        out_header = out_header or header
        if header[1] != out_header[1]:
            sys.exit(f"{path}: column header differs from the first chunk")
        frame, n_frames, ncores = segment_frames(lines, ts, args.anchor)
        if cores is not None and ncores != cores:
            sys.exit(f"{path}: {ncores} cores/frame, earlier chunks had {cores}")
        cores = ncores
        keep = np.ones(len(lines), bool) if args.keep_warmup else frame > 0
        kept_frames = n_frames - (0 if args.keep_warmup else 1)
        shift = 0
        kts = ts[keep]
        if prev_max is not None and kts.size and kts.min() <= prev_max:
            shift = int(prev_max + CHUNK_GAP_CYC - kts.min())
        for ln, t, k in zip(lines, ts, keep):
            if not k:
                continue
            if shift:
                p = ln.split(",", T_COL + 1)
                p[T_COL] = str(int(t) + shift)
                ln = ",".join(p)
            out_lines.append(ln)
        if kts.size:
            prev_max = int(kts.max()) + shift
        total_frames += kept_frames
        print(f"chunk {path}: rows={len(lines)} frames={n_frames} "
              f"(warmup rows={int((frame == 0).sum())}) kept_rows={int(keep.sum())} "
              f"kept_frames={kept_frames} cores/frame={ncores} shift_cyc={shift}")

    with open(args.out, "w") as f:
        f.writelines(out_header)
        f.writelines(out_lines)
    print(f"stitched {args.out}: rows={len(out_lines)} frames={total_frames} "
          f"rows/frame={len(out_lines) / max(total_frames, 1):.0f}")


if __name__ == "__main__":
    main()
