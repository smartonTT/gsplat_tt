"""#803: hero training pair counts with the exact per-pair alpha cull (GSPLAT_TT_TRAIN_PAIR_CULL=1).

Mac only, no device. Bicycle hero view (_DSC8679, 1237x822, bicycle.ply), projected by
docs/t685/hero.py with and without the 3DGS Jacobian clamp (1.3). Counts the (Gaussian, 32x32
tile) pairs of the trainer's 3-sigma tile rectangles (blend_backward_dev.tile_rects_vec), the
pairs gsplat.blend_backward_dev.pair_alive keeps (the code the training prep runs), and, for
comparison, the opacity-aware rectangle (pair_model.py's lever-4 variant, counted only).
Rows with opacity < 1/255 are left out first, as #799's pair_model.py.
Usage: PYTHONPATH=$PWD python3 docs/training-plan/t803/hero_pairs.py  -> hero_pairs.json here
"""
import json, os, sys, types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "docs/t685"))
try:
    import gsplat.train_oracle  # noqa: F401  (needs torch)
except ImportError:   # Mac venv without torch: hero.py only needs the SH constants
    _m = types.ModuleType("gsplat.train_oracle")
    _m.SH_C0, _m.SH_C1 = 0.28209479177387814, 0.4886025119029199
    _m.SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    _m.SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
    sys.modules["gsplat.train_oracle"] = _m
import hero  # noqa: E402
from gsplat import blend_backward_dev as dev  # noqa: E402

AMIN = 1.0 / 255.0


def run(jclamp):
    cams = json.load(open(os.path.join(ROOT, "benchmarks/cameras_test.json")))["bicycle"]
    W, H = cams["image_size"]
    ply, cm = hero.load_ply(os.path.join(hero.scenes_dir(), "bicycle.ply"))
    op = 1.0 / (1.0 + np.exp(-ply[:, cm["opacity"]].astype(np.float64)))
    keep = np.flatnonzero(op >= AMIN)
    m2, con, _, _, vis = hero.project(ply[keep], cm, cams["views"]["_DSC8679"]["c2w"], cams["intrinsics"],
                                      H, W, jclamp=jclamp)
    sel = np.flatnonzero(vis)
    m2, con, op = m2[sel].astype(np.float32), con[sel].astype(np.float32), op[keep][sel].astype(np.float32)
    valid = np.ones(len(sel), bool)
    cfg = dev.BlendCfg()
    tx = (W + dev.TILE - 1) // dev.TILE
    rects = dev.tile_rects_vec(m2, valid, con, H, W)
    order = np.arange(len(sel))
    full = dev.pair_lists(rects, order, tx)
    cut = dev.pair_lists(rects, order, tx, dev.pair_alive(m2, con, op, cfg, H, W))
    # opacity-aware rectangle: extent where min(op, 0.99) exp(-r^2/2) >= 1/255, capped at 3 sigma
    cov = dev.cov_diag_from_conic(con.astype(np.float64))
    kc = np.minimum(3.0, np.sqrt(2.0 * np.log(np.maximum(255.0 * np.minimum(op, 0.99), 1.0))))
    cov = np.maximum(cov, 0.0)   # as tile_rects_vec
    rx, ry = np.ceil(kc * np.sqrt(cov[:, 0])), np.ceil(kc * np.sqrt(cov[:, 1]))
    cl = lambda v, hi: np.clip(np.trunc(v), 0, hi)  # noqa: E731
    ok = ((cl((m2[:, 0] + rx) / 32, tx - 1) - cl((m2[:, 0] - rx) / 32, tx - 1) + 1)
          * (cl((m2[:, 1] + ry) / 32, (H + 31) // 32 - 1) - cl((m2[:, 1] - ry) / 32, (H + 31) // 32 - 1) + 1))
    n_rect, n_cut = int(full["counts"].sum()), int(cut["counts"].sum())
    tp = np.bincount(full["tid"]); tc = np.bincount(cut["tid"], minlength=tp.size)
    return {"jclamp": jclamp, "visible_rows": int(len(sel)), "pairs_rect_3sigma": n_rect,
            "pairs_exact_cull": n_cut, "dropped_frac": 1.0 - n_cut / n_rect,
            "pairs_opacity_rect": int(ok[rects[:, 1] >= rects[:, 0]].sum()),
            "rows_with_pairs_rect_cut": [int((full["counts"] > 0).sum()), int((cut["counts"] > 0).sum())],
            "max_k_rect_cut": [int(full["counts"].max()), int(cut["counts"].max())],
            "tile_pairs_max_mean_rect": [int(tp.max()), float(tp.mean())],
            "tile_pairs_max_mean_cut": [int(tc.max()), float(tc.mean())]}


if __name__ == "__main__":
    out = {"off": run(None), "on": run(1.3)}
    json.dump(out, open(os.path.join(HERE, "hero_pairs.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
