"""#801: hero-view T2 row counts (pairs per visible Gaussian, 3-sigma 32x32-tile rect, no
jclamp: the #799 pair_model "today" row) -> light planes with rows in order vs k-grouped.

Mac only, no device. Writes the counts (row order = bicycle.ply order of the kept, visible
Gaussians) to tests/fixtures/t801/hero_model_counts.npz for tests/unit/test_t2_split.py.
  PYTHONPATH=$PWD python3 docs/t801/kgroup_model.py   (needs scenes/bicycle.ply)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(ROOT, "docs/t685"))
sys.path.insert(0, ROOT)
try:
    import gsplat.train_oracle  # noqa: F401  (needs torch)
except ImportError:   # Mac venv without torch: hero.py only needs the SH constants (as #799 pair_model)
    import types
    _m = types.ModuleType("gsplat.train_oracle")
    _m.SH_C0, _m.SH_C1 = 0.28209479177387814, 0.4886025119029199
    _m.SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    _m.SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
    sys.modules["gsplat.train_oracle"] = _m
import hero  # noqa: E402
from gsplat import t2_split  # noqa: E402

TILE, AMIN = 32, 1.0 / 255.0


def hero_counts() -> np.ndarray:
    cams = json.load(open(os.path.join(ROOT, "benchmarks/cameras_test.json")))["bicycle"]
    W, H = cams["image_size"]
    ply, cm = hero.load_ply(os.path.join(hero.scenes_dir(), "bicycle.ply"))
    op = 1.0 / (1.0 + np.exp(-ply[:, cm["opacity"]].astype(np.float64)))
    keep = np.flatnonzero(op >= AMIN)
    m2, con, _, _, vis = hero.project(ply[keep], cm, cams["views"]["_DSC8679"]["c2w"], cams["intrinsics"],
                                      H, W, jclamp=None)
    sel = np.flatnonzero(vis)
    m2, con = m2[sel], con[sel]
    det = 1.0 / (con[:, 0] * con[:, 2] - con[:, 1] ** 2)
    rx, ry = np.ceil(3 * np.sqrt(con[:, 2] * det)), np.ceil(3 * np.sqrt(con[:, 0] * det))
    tx, ty = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    x0 = np.clip(np.floor((m2[:, 0] - rx) / TILE), 0, tx - 1).astype(np.int64)
    x1 = np.clip(np.floor((m2[:, 0] + rx) / TILE), 0, tx - 1).astype(np.int64)
    y0 = np.clip(np.floor((m2[:, 1] - ry) / TILE), 0, ty - 1).astype(np.int64)
    y1 = np.clip(np.floor((m2[:, 1] + ry) / TILE), 0, ty - 1).astype(np.int64)
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def main() -> int:
    k = hero_counts()
    k = k[k > 0]
    out = os.path.join(ROOT, "tests/fixtures/t801/hero_model_counts.npz")
    np.savez_compressed(out, counts=k.astype(np.uint16))
    res = {"rows": int(k.size), "slots": int(k.sum()), "max_k": int(k.max())}
    for kcap in (16, 8):
        res[f"kcap{kcap}"] = {"in_order": t2_split.light_planes(k, kcap),
                              "k_grouped": t2_split.light_planes(k, kcap, kgroup=True)}
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
