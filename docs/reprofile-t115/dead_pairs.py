#!/usr/bin/env python3
"""t115: share of slab records (gaussian, tile pairs) whose device microblock mask is 0.

Input: a GSPLAT_TT_DUMP_CULL directory (first frame = hero warmup). A record with
mask 0 has no microblock with a pixel above the floor, so it flows through
tile_assign, the sort and the materialize for nothing. Also prints the mask
popcount histogram and the share of dead records in big (> 8192) tiles.
Usage: dead_pairs.py DUMP_DIR
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "opt" / "golden-check"))
from cull_mask_check import gather, load  # noqa: E402

meta, u = load(sys.argv[1])
recs, tid, rank, tiles_x = gather(meta, u)
mask = recs[:, 3]
n = len(mask)
dead = mask == 0
pc = np.array([bin(int(m)).count("1") for m in mask[:2000000]]) if n else np.zeros(0)
print(f"records {n} dead(mask=0) {int(dead.sum())} share {dead.mean():.4f} floor {meta.get('floor')}")
cnt = np.bincount(tid, minlength=int(meta["num_tiles"]))
big = cnt[tid] > 8192
print(f"records in tiles > 8192: {int(big.sum())} ({big.mean():.4f}); dead there {dead[big].mean() if big.any() else 0:.4f}")
print(f"tiles with records {int((cnt > 0).sum())} max/tile {cnt.max()} mean/tile(nonempty) {cnt[cnt > 0].mean():.1f}")
h = np.bincount(pc, minlength=33)
print("mask popcount histogram (0..32): " + " ".join(str(int(x)) for x in h))
print(f"mean microblocks per live record {pc[pc > 0].mean():.2f}")
