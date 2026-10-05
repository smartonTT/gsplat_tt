"""Task #157: why lever C's rect misses most dead records, and what a per-row
ellipse test in tile_assign K2 would remove. Host float64 model (estimate.py's
projection), not the device: shares are estimates.

Arms, all on the (gaussian, tile) records of the 3-sigma rect:
  dev   lever C as built (t142): t from the log2 upper bound, r' = rint(sqrt(t a) + 2)
  ideal lever C with an exact log and r' = ceil(sqrt(t a) + 1) (the t140 model)
  para  dev + per-row sheared band |u - s v| <= hw (s = b / c, hw = sqrt(T det / c) + 1)
  row   dev + per-row exact ellipse extent (== ellipse meets the tile's pixel-centre
        box at threshold T); ell_tile.h is its integer version, checked by
        tests/unit/test_ell_tile.cpp
T = the pre-cull's t (t_band + >= 0.2). A record is live when the band cull's
ellipse (m2 <= t_band = 2 ln(op / floor) + 0.05) meets the tile's pixel-centre box.
Usage (repo root): python3 docs/precull-t157/analyze.py [views|all] [ply]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "precull-t140"))
from estimate import CAM, COND, TILE, box_min_m2, cells, project  # noqa: E402

C0_MARGIN = 0.25
SL = 0.125  # pixel-centre rect slack (px)
ARMS = ("dev", "ideal", "pc", "para", "row", "pc+para", "pc+row")


def t_dev(op, floor):
    """pfwc step 11.6 t (t142): 2 ln2 * log2 upper bound + c0, clamped at 0."""
    bits = op.astype(np.float32).view(np.uint32)
    e = (bits >> 23).astype(np.int64) - 127
    m1 = ((bits & 0x7FFFFF) >> 9).astype(np.float64) / 16384.0  # 9 low bits truncated
    lg = e + m1 + 0.0860713 + 2.0 ** -14
    c0 = 2 * math.log(1 / floor) + C0_MARGIN
    return np.maximum(2 * math.log(2) * lg + c0, 0.0)


def one(view, ply, acc):
    (a, b, c, mx, my, op, rx, ry), W, H, floor = project(view, ply)
    txn, tyn = W // TILE, H // TILE
    x0, x1 = cells(mx, rx, txn)
    y0, y1 = cells(my, ry, tyn)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    n = w * h
    P = int(n.sum())
    g = np.repeat(np.arange(n.size), n)
    loc = np.arange(P) - np.repeat(np.cumsum(n) - n, n)
    tx = x0[g] + loc % w[g]
    ty = y0[g] + loc // w[g]
    det = a * c - b * b
    ia, ib, ic = c / det, -b / det, a / det
    t = 2 * np.log(op / floor)
    u_lo = tx * TILE + 0.5 - mx[g]
    u_hi = u_lo + 31.0
    v_lo = ty * TILE + 0.5 - my[g]
    v_hi = v_lo + 31.0
    m2 = box_min_m2(ia[g], ib[g], ic[g], u_lo, u_hi, v_lo, v_hi)
    live = m2 <= t[g] + 0.05
    well = COND * det >= a * c
    ok = well & (np.maximum(rx, ry) <= 4096)

    def rect_in(rpx, rpy):
        px0, px1 = cells(mx, rpx, txn)
        py0, py1 = cells(my, rpy, tyn)
        return (tx >= px0[g]) & (tx <= px1[g]) & (ty >= py0[g]) & (ty <= py1[g])

    td = t_dev(op, floor)
    dev = rect_in(np.where(ok, np.minimum(rx, np.rint(np.sqrt(td * a) + 2)), rx),
                  np.where(ok, np.minimum(ry, np.rint(np.sqrt(td * c) + 2)), ry))
    ti = np.maximum(t + C0_MARGIN, 0)
    ideal = rect_in(np.where(well, np.minimum(rx, np.ceil(np.sqrt(ti * a) + 1)), rx),
                    np.where(well, np.minimum(ry, np.ceil(np.sqrt(ti * c) + 1)), ry))
    # pixel-centre rect: a tile meets [m - e, m + e] at a pixel centre iff it meets
    # the rect of radius e - 0.5, e = sqrt(T cov); SL px slack, r' >= SL
    e_x = np.maximum(np.sqrt(td * a) - 0.5 + SL, SL)
    e_y = np.maximum(np.sqrt(td * c) - 0.5 + SL, SL)
    pc = rect_in(np.where(ok, np.minimum(rx, e_x), rx), np.where(ok, np.minimum(ry, e_y), ry))
    # per-row tests, threshold T = td, only on shrink-ok gaussians (others keep all)
    T = td[g]
    s = (b / c)[g]
    hw = np.sqrt(T * (det / c)[g]) + 1.0
    lo = np.minimum(s * v_lo, s * v_hi) - hw
    hi = np.maximum(s * v_lo, s * v_hi) + hw
    para = dev & ((u_hi >= lo) & (u_lo <= hi) | ~ok[g])
    row = dev & ((m2 <= T) | ~ok[g])
    arms = {"dev": dev, "ideal": ideal, "pc": pc, "para": para, "row": row,
            "pc+para": pc & para, "pc+row": pc & row}
    nd = P - int(live.sum())
    line = f"view {view}: records {P} dead {nd} ({nd / P:.4f})"
    for k, m in arms.items():
        lost = int((live & ~m).sum())
        rem = P - int(m.sum())
        line += f" | {k} -{rem} ({rem / P:.4f}, {rem / max(nd, 1):.3f} of dead, lost {lost})"
        acc[k] += rem
        acc[k + "_lost"] += lost
    print(line)
    acc["P"] += P
    acc["dead"] += nd
    # where the dead records the dev rect keeps sit
    dd = dev & ~live
    ddp = pc & ~live
    acc["ddp"] += int(ddp.sum())
    acc["ddp_strip"] += int((ddp & (((w == 1) | (h == 1)) & (n > 1))[g]).sum())
    acc["ddp_notok"] += int((ddp & ~ok[g]).sum())
    big = (w >= 2) & (h >= 2)
    strip = ((w == 1) | (h == 1)) & (n > 1)
    acc["dd"] += int(dd.sum())
    acc["dd_big"] += int((dd & big[g]).sum())
    acc["dd_strip"] += int((dd & strip[g]).sum())
    acc["dd_notok"] += int((dd & ~ok[g]).sum())
    # dead by the dev rect's tile-quantised bbox only (would need a tighter rect) vs
    # inside the ellipse's own bbox (corners: need an ellipse test)
    ex = np.sqrt(np.maximum(t + 0.05, 0) * a)[g]
    ey = np.sqrt(np.maximum(t + 0.05, 0) * c)[g]
    inbox = (u_hi >= -ex) & (u_lo <= ex) & (v_hi >= -ey) & (v_lo <= ey)
    acc["dd_corner"] += int((dd & inbox).sum())
    # TA cost drivers for a test run only on rects with more than one tile
    sel = ok & (n > 1)
    acc["G"] += n.size
    acc["G_sel"] += int(sel.sum())
    acc["rows_sel"] += int(h[sel].sum())
    acc["pairs_sel"] += int(n[sel].sum())
    tid = ty * txn + tx
    for k in ("dev", "pc", "row", "pc+row"):
        cnt = np.bincount(tid[arms[k]], minlength=txn * tyn)
        acc["maxtile_" + k] = max(acc.get("maxtile_" + k, 0), int(cnt.max()))


def main():
    views = sys.argv[1].split(",") if len(sys.argv) > 1 else ["hero"]
    ply = sys.argv[2] if len(sys.argv) > 2 else "scenes/bicycle.ply"
    if views == ["all"]:
        views = json.load(open(CAM))["bicycle"]["order"]
    from collections import defaultdict
    acc = defaultdict(int)
    for v in views:
        one(v, ply, acc)
    P = acc["P"]
    print(f"TOTAL {len(views)} views: records {P} dead {acc['dead']} ({acc['dead'] / P:.4f})")
    for k in ARMS:
        print(f"  {k:5s}: removes {acc[k]} ({acc[k] / P:.4f} of records, "
              f"{acc[k] / acc['dead']:.3f} of dead), live lost {acc[k + '_lost']}")
    dd = acc["dd"]
    print(f"  dead left by dev {dd}: in w,h>=2 rects {acc['dd_big'] / dd:.3f}, 1-wide strips "
          f"{acc['dd_strip'] / dd:.3f}, not-ok (3 sigma kept) {acc['dd_notok'] / dd:.3f}; "
          f"inside the keep ellipse's bbox (corners) {acc['dd_corner'] / dd:.3f}")
    print(f"  dead left by pc {acc['ddp']}: 1-wide strips {acc['ddp_strip'] / acc['ddp']:.3f}, "
          f"not-ok {acc['ddp_notok'] / acc['ddp']:.3f}")
    print(f"  TA per-row test on ok rects > 1 tile: {acc['G_sel']} of {acc['G']} gaussians "
          f"({acc['G_sel'] / acc['G']:.3f}), {acc['rows_sel']} rows, {acc['pairs_sel']} pairs "
          f"({acc['pairs_sel'] / P:.3f}); max tile dev {acc['maxtile_dev']} pc {acc['maxtile_pc']} "
          f"row {acc['maxtile_row']} pc+row {acc['maxtile_pc+row']}")


if __name__ == "__main__":
    main()
