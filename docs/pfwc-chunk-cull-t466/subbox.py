"""Task #466: how much of the 'empty tile' bound a tighter test recovers (host, no device).
A tile is culled only when every one of its sub-boxes (S consecutive gaussians of the tile,
each with its own box and rho_max) passes the chunk_cull.h test. Views: every 5th.
Usage (repo root): python3 docs/pfwc-chunk-cull-t466/subbox.py [ply]
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cullshare as cs  # noqa: E402

SUBS = (1024, 256, 64)


def main():
    cam = json.load(open(cs.CAM))["bicycle"]
    ply = sys.argv[1] if len(sys.argv) > 1 else cam["ply"]
    W, H = cam["image_size"]
    f = 0.5 * max(W, H) / math.tan(0.5 * math.radians(cam["fov_deg"]))
    means, scales, q, op = cs.load_ply(ply)
    rho = 3.0 * np.sqrt((scales ** 2).sum(1))
    orders = {"morton": cs.morton_order(means), "core": cs.core_local_order(means, 120)[0]}
    print("tile cull share with sub-boxes of S gaussians (1024 = today's chunk_cull.h test)")
    print("view     " + " ".join(f"{k}:S{s:<5}" for k in orders for s in SUBS))
    for v in cam["order"][::5]:
        w2c = np.linalg.inv(np.array(cam["views"][v]["c2w"], np.float64))
        row = []
        for k, o in orders.items():
            for s in SUBS:
                sub = cs.culled(*cs.tables(means, rho, o, C=s), w2c, f, W, H)
                per = cs.TILE // s
                sub = np.concatenate([sub, np.repeat(sub[-1:], (-len(sub)) % per)])  # last partial tile
                row.append(sub.reshape(-1, per).all(1).mean())
        print(f"{v:8s} " + " ".join(f"{x:13.3f}" for x in row), flush=True)


if __name__ == "__main__":
    main()
