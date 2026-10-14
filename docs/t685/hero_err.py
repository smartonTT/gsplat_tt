"""#707: where does the hero device backward differ from the oracle? Per-slot error grouped by
tile pair count and by term, top tiles by error. Usage: hero_err.py hero.npz slots.f32 out.json
[ref.npy]  (ref.npy is reused if present, else written: ref_slots_fast takes ~25 min)."""
import json
import os
import sys

import numpy as np

from gsplat.blend_backward_dev import (TILE, _sweep_tiles, load_dump, ref_slots_fast,
                                       tile_rects_vec)

f32 = np.float32
NAMES = ["dmx", "dmy", "dA", "dB", "dC", "dop", "dr", "dg", "db"]


def main(npz, slots_path, out, ref_path=None):
    d, cfg = load_dump(npz)
    dev = np.fromfile(slots_path, f32).reshape(-1, 16)[:, :9].astype(np.float64)
    if ref_path and os.path.exists(ref_path):
        ref = np.load(ref_path)
    else:
        ref = ref_slots_fast(d, cfg)
        if ref_path:
            np.save(ref_path, ref)
    ref = ref.astype(np.float64)
    H, W = d["image"].shape[:2]
    rects = tile_rects_vec(d["means2d"], d["valid"], d["conic"], H, W, TILE)
    pl, utid, start, cnt, *_ = _sweep_tiles(np.asarray(d["means2d"], f32), np.asarray(d["conic"], f32),
                                            rects, d["order"], H, W)
    slot = pl["P"][pl["g"]] + pl["j"]                     # slot of each list entry
    tcount = np.zeros(dev.shape[0], np.int64)
    ttile = np.full(dev.shape[0], -1, np.int64)
    rank = np.zeros(dev.shape[0], np.int64)
    for t, s0, c in zip(utid, start, cnt):
        tcount[slot[s0:s0 + c]] = c
        ttile[slot[s0:s0 + c]] = t
        rank[slot[s0:s0 + c]] = np.arange(c)
    diff = dev - ref
    scale = np.sqrt((ref ** 2).sum(0) / max(1, ref.shape[0])) + 1e-30   # per-term rms
    e = np.abs(diff) / scale                                             # error in units of rms
    bad = (e > 1e-2).any(1)
    rep = {"n_slots": int(dev.shape[0]), "n_listed": int(slot.size), "bad_frac": float(bad.mean()),
           "nonfinite_dev": int((~np.isfinite(dev)).any(1).sum()),
           "dev_zero_ref_nonzero": int(((dev == 0).all(1) & (ref != 0).any(1)).sum()),
           "by_tile_count": [], "by_rank": [], "top_tiles": []}
    for lo, hi in [(0, 1024), (1024, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 1 << 30)]:
        m = (tcount >= lo) & (tcount < hi)
        if m.any():
            rep["by_tile_count"].append({"cnt": [lo, hi], "slots": int(m.sum()), "bad_frac": float(bad[m].mean()),
                                         "err_sq_share": float((diff[m] ** 2).sum() / max((diff ** 2).sum(), 1e-300))})
    for lo, hi in [(0, 1024), (1024, 4096), (4096, 8192), (8192, 16384), (16384, 1 << 30)]:
        m = (rank >= lo) & (rank < hi)
        if m.any():
            rep["by_rank"].append({"rank": [lo, hi], "slots": int(m.sum()), "bad_frac": float(bad[m].mean())})
    tb = np.bincount(ttile[ttile >= 0], weights=bad[ttile >= 0], minlength=int(utid.max()) + 1)
    for t in np.argsort(-tb)[:15]:
        rep["top_tiles"].append({"tile": int(t), "bad": int(tb[t]), "cnt": int(cnt[utid == t][0]) if (utid == t).any() else 0})
    m = rank < 1024
    rep["rel_err_rank_lt_1024"] = float(np.linalg.norm(diff[m]) / max(np.linalg.norm(ref[m]), 1e-300))
    first_bad = [int(rank[ttile == t][bad[ttile == t]].min()) for t in utid[:20] if bad[ttile == t].any()]
    rep["first_bad_rank_busiest_tiles"] = first_bad
    rep["worst_term_slot"] = {n: float(np.abs(diff[:, k]).max()) for k, n in enumerate(NAMES)}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep)[:3000])


if __name__ == "__main__":
    main(*sys.argv[1:])
