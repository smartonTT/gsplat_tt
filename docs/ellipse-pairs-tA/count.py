"""Task #533 (T-A): how many tile pairs an exact ellipse-tile test removes beyond PRECULL=2.

Host float64 model (docs/precull-t140/estimate.py projection, docs/precull-t157/analyze.py
rect rules), not the device. Per view it expands every visible gaussian's 3-sigma tile rect,
applies the current PRECULL=2 pixel-centre rect (pc), then these exact tests on pc's pairs:

  exact     keep iff max alpha over the tile's pixel-centre box >= floor (1/255):
            min m2 over the box <= 2 ln(op / floor). All gaussians, no margin.
  exact_m   the same with the band cull's +0.05 margin (m2 <= t + 0.05). Keeps every
            record the device blend can draw: the safe image-identical rule.
  row_dev   the device-shaped rule of t157's "row" arm: pfwc's t (log2 bound + 0.25
            margin) and ill-conditioned / huge gaussians (not ok) keep all pc tiles.

The box is the continuous [32k + 0.5, 32k + 31.5] square, a superset of the 32 x 32 pixel
centres, so every rule is conservative. A pair is a 32-B record that goes through k2, sort
and mat+blend, so pairs == records entering blend.
Usage (repo root): python3 docs/ellipse-pairs-tA/count.py [views|all] [ply]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D / "precull-t140"))
sys.path.insert(0, str(D / "precull-t157"))
from estimate import CAM, TILE, box_min_m2, cells, project  # noqa: E402
from analyze import SL, t_dev  # noqa: E402

COND = 64.0
ARMS = ("pc", "exact", "exact_m", "row_dev")


def one(view, ply):
    (a, b, c, mx, my, op, rx, ry), W, H, floor = project(view, ply)
    txn, tyn = W // TILE, H // TILE
    det = a * c - b * b
    td = t_dev(op, floor)
    ok = (COND * det >= a * c) & (np.maximum(rx, ry) <= 4096)
    # PRECULL=2 rect (pfwc PRECULL_PC): r' = min(r, max(sqrt(t cov) - 3/8, 1/8)) when ok
    e_x = np.where(ok, np.minimum(rx, np.maximum(np.sqrt(td * a) - 0.5 + SL, SL)), rx)
    e_y = np.where(ok, np.minimum(ry, np.maximum(np.sqrt(td * c) - 0.5 + SL, SL)), ry)
    x0, x1 = cells(mx, e_x, txn)
    y0, y1 = cells(my, e_y, tyn)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    n = w * h
    P = int(n.sum())
    g = np.repeat(np.arange(n.size), n)
    loc = np.arange(P) - np.repeat(np.cumsum(n) - n, n)
    tx = x0[g] + loc % w[g]
    ty = y0[g] + loc // w[g]
    del loc
    ia, ib, ic = c / det, -b / det, a / det
    u_lo = tx * TILE + 0.5 - mx[g]
    v_lo = ty * TILE + 0.5 - my[g]
    m2 = box_min_m2(ia[g], ib[g], ic[g], u_lo, u_lo + 31.0, v_lo, v_lo + 31.0)
    t = 2 * np.log(op / floor)
    keep = {
        "pc": np.ones(P, bool),
        "exact": m2 <= t[g],
        "exact_m": m2 <= t[g] + 0.05,
        "row_dev": (m2 <= td[g]) | ~ok[g],
    }
    r = {"view": view, "M": int(n.size), "P_pc": P}
    for k in ARMS[1:]:
        r["P_" + k] = int(keep[k].sum())
    # where the removed pairs sit (exact_m): multi-tile gaussians, rows touched
    rem = ~keep["exact_m"]
    r["rm_notok"] = int((rem & ~ok[g]).sum())
    r["G_multi"] = int((n > 1).sum())
    r["pairs_multi"] = int(n[n > 1].sum())
    r["rows_multi"] = int(h[n > 1].sum())
    tid = ty * txn + tx
    r["maxtile_pc"] = int(np.bincount(tid, minlength=txn * tyn).max())
    r["maxtile_exact_m"] = int(np.bincount(tid[keep["exact_m"]], minlength=txn * tyn).max())
    return r


def main():
    views = sys.argv[1].split(",") if len(sys.argv) > 1 else ["hero"]
    ply = sys.argv[2] if len(sys.argv) > 2 else "scenes/bicycle.ply"
    if views == ["all"]:
        views = json.load(open(CAM))["bicycle"]["order"]
    for v in views:
        r = one(v, ply)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
