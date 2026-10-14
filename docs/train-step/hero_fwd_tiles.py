"""#798 (#795 step b): the hero forward gate per 32x32 tile. The device forward image (loop_box.sh
HERO=1 keeps image.f32 and fwd/tiles.u32) against the numpy model of the device math
(hero_model.py's emulate_fast image), split by the tile's record count: tiles over SUB_RECS (8192)
records run through the CONTINUE sub-chunk path (fwd_reader.cpp) that the 64x64 crops never reach.
  python3 docs/train-step/hero_fwd_tiles.py <image.f32> <tiles.u32> <hero_model.npz> [bar]"""

import json
import sys

import numpy as np

TILE, SUB_RECS = 32, 8192


def tile_report(img: np.ndarray, model: np.ndarray, counts: np.ndarray, bar: float = 1e-4) -> dict:
    H, W = img.shape[:2]
    tx, ty = -(-W // TILE), -(-H // TILE)
    err = np.zeros((ty * TILE, tx * TILE), np.float64)
    err[:H, :W] = np.abs(img.astype(np.float64) - model).max(axis=-1)
    te = err.reshape(ty, TILE, tx, TILE).max(axis=(1, 3)).reshape(-1)
    counts = np.asarray(counts[: tx * ty], np.int64)
    sub = (counts + SUB_RECS - 1) // SUB_RECS
    bad = te > bar
    rep = {"bar": bar, "n_tiles": int(te.size), "max_abs": float(te.max()), "bad_tiles": int(bad.sum()),
           "pass": bool(not bad.any())}
    for name, sel in (("le_8192", counts <= SUB_RECS), ("gt_8192", counts > SUB_RECS)):
        rep[name] = {"tiles": int(sel.sum()), "bad": int((bad & sel).sum()),
                     "max_abs": float(te[sel].max()) if sel.any() else 0.0}
    rep["bad_by_subchunks"] = {str(int(k)): [int((bad & (sub == k)).sum()), int((sub == k).sum())]
                               for k in np.unique(sub)}
    worst = np.argsort(-te)[:12]
    rep["worst"] = [{"tile": int(t), "tx": int(t % tx), "ty": int(t // tx), "recs": int(counts[t]),
                     "max_abs": float(te[t])} for t in worst]
    return rep


def main(argv) -> int:
    m = np.load(argv[3])
    model = np.asarray(m["image"], np.float64)
    H, W = model.shape[:2]
    img = np.fromfile(argv[1], np.float32).reshape(H, W, 3)
    counts = np.fromfile(argv[2], np.uint32).reshape(-1, 16)[:, 2]
    print(json.dumps(tile_report(img, model, counts, float(argv[4]) if len(argv) > 4 else 1e-4)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
