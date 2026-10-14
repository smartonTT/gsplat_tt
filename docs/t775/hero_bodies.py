"""#775: per-tile pair and body counts of the T1 blend backward on the bicycle hero view.

    PYTHONPATH=. python3 docs/t775/hero_bodies.py tmp/t775/hero_bodies.npz

Same projection and Gaussian set as docs/t685/hero.py (#707), but no forward: the T1 vector
masks (bodies) depend only on the geometry and opacity, so blend_backward_dev.prep() runs with
zero image and dL/dimage. Writes tiles (n_tiles, 4) and the per-pair vector masks (u32) to the
npz under tmp/ (not committed). A body is one visited SFPU vector of one pair (popcount of the mask).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "t685"))
from hero import ROOT, load_ply, project, scenes_dir  # noqa: E402

from gsplat.blend_backward_dev import prep  # noqa: E402
from gsplat.blend_backward_ref import BlendCfg  # noqa: E402


def main(argv=None) -> int:
    out = (argv or sys.argv[1:])[0]
    t0 = time.time()
    cams = json.load(open(os.path.join(ROOT, "benchmarks/cameras_test.json")))["bicycle"]
    W, H = cams["image_size"]
    ply, cm = load_ply(os.path.join(scenes_dir(), "bicycle.ply"))
    op_all = 1.0 / (1.0 + np.exp(-ply[:, cm["opacity"]].astype(np.float64)))
    keep = np.flatnonzero(op_all >= 1.0 / 255.0)
    m2, conic, col, tz, vis = project(ply[keep], cm, cams["views"]["_DSC8679"]["c2w"],
                                      cams["intrinsics"], H, W)
    del ply
    sel = np.flatnonzero(vis)
    m2, conic, col, tz = m2[sel], conic[sel], col[sel], tz[sel]
    op = op_all[keep][sel]
    order = np.argsort(tz, kind="stable").astype(np.int64)
    f = np.float32
    m2, conic, op, col = m2.astype(f), conic.astype(f), op.astype(f), col.astype(f)
    t1 = time.time()
    z = np.zeros((H, W, 3), f)
    p = prep(m2, conic, op, col, order, np.ones(len(sel), bool), z, z, BlendCfg())
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, tiles=p["tiles"][:, :3], masks=p["recs"][:, 12], n_visible=len(sel), H=H, W=W)
    print(json.dumps({"n_visible": len(sel), "n_pairs": int(p["recs"].shape[0]),
                      "n_tiles": int(p["tiles"].shape[0]), "max_tile_pairs": int(p["tiles"][:, 2].max()),
                      "project_s": round(t1 - t0, 1), "prep_s": round(time.time() - t1, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
