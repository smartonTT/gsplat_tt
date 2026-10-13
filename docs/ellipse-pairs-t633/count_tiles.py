"""Task #633 (ledger U8): CPU pair count for 16 x 16 vs 32 x 32 screen tiles.

Host float64 model, not the device: docs/ellipse-pairs-tA/count.py (#533) with the tile
size as a parameter (estimate.TILE is read at call time by cells()). Per view and tile size:
  P_pc       pairs under the current PRECULL=2 pixel-centre rect (= records through k2/sort/mat)
  P_exact_m  pairs left after #533's exact ellipse-tile test (+0.05 band margin, image-identical)
  PX_pc      pair-pixels = P_pc * TILE^2 (blend's record x pixel work before microblock culls)
  maxtile    largest tile's pair count (load balance)
Usage (repo root): python3 docs/ellipse-pairs-t633/count_tiles.py [views|all] [tiles=32,16] [ply]
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(D / "precull-t140"))
sys.path.insert(0, str(D / "precull-t157"))
import estimate  # noqa: E402
from estimate import CAM, box_min_m2, cells, project  # noqa: E402
from analyze import SL, t_dev  # noqa: E402

COND = 64.0


def one(view, proj, tile):
    estimate.TILE = tile
    (a, b, c, mx, my, op, rx, ry), W, H, floor = proj
    txn, tyn = W // tile, H // tile
    det = a * c - b * b
    td = t_dev(op, floor)
    ok = (COND * det >= a * c) & (np.maximum(rx, ry) <= 4096)
    e_x = np.where(ok, np.minimum(rx, np.maximum(np.sqrt(td * a) - 0.5 + SL, SL)), rx)
    e_y = np.where(ok, np.minimum(ry, np.maximum(np.sqrt(td * c) - 0.5 + SL, SL)), ry)
    x0, x1 = cells(mx, e_x, txn)
    y0, y1 = cells(my, e_y, tyn)
    w = x1 - x0 + 1
    n = w * (y1 - y0 + 1)
    P = int(n.sum())
    g = np.repeat(np.arange(n.size), n)
    loc = np.arange(P) - np.repeat(np.cumsum(n) - n, n)
    tx = x0[g] + loc % w[g]
    ty = y0[g] + loc // w[g]
    del loc
    ia, ib, ic = c / det, -b / det, a / det
    u_lo = tx * tile + 0.5 - mx[g]
    v_lo = ty * tile + 0.5 - my[g]
    m2 = box_min_m2(ia[g], ib[g], ic[g], u_lo, u_lo + tile - 1.0, v_lo, v_lo + tile - 1.0)
    keep = m2 <= 2 * np.log(op / floor)[g] + 0.05
    tid = ty * txn + tx
    return {"view": view, "tile": tile, "M": int(n.size), "P_pc": P, "P_exact_m": int(keep.sum()),
            "PX_pc": P * tile * tile, "PX_exact_m": int(keep.sum()) * tile * tile,
            "maxtile_pc": int(np.bincount(tid, minlength=txn * tyn).max()), "ntiles": txn * tyn}


def main():
    views = sys.argv[1].split(",") if len(sys.argv) > 1 else ["hero"]
    tiles = [int(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "32,16").split(",")]
    ply = sys.argv[3] if len(sys.argv) > 3 else "scenes/bicycle.ply"
    if views == ["all"]:
        views = json.load(open(CAM))["bicycle"]["order"]
    for v in views:
        proj = project(v, ply)
        for t in tiles:
            print(json.dumps(one(v, proj, t)), flush=True)


if __name__ == "__main__":
    main()
