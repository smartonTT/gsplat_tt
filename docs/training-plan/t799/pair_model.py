"""#799: hero pair counts with/without the 3DGS Jacobian clamp, and how many pairs can contribute.

Mac only, no device. For the bicycle hero view (_DSC8679, 1237x822, bicycle.ply) it enumerates the
(Gaussian, 32x32 tile) pairs the trainer uses today (3-sigma axis-aligned rectangles, as
docs/t685/hero.py and the device T1/T2 slot lists) and, per pair, the largest alpha the Gaussian
reaches anywhere in the tile rectangle (exact minimum of the conic over the rectangle). Pairs whose
largest alpha is under 1/255 are skipped by every pixel of the blend (3DGS alpha floor), so dropping
them at tile assignment is exact.
Usage: python3 docs/training-plan/t799/pair_model.py <hero.py dir>   (needs scenes/bicycle.ply)
Writes docs/training-plan/t799/pair_model.json.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../../t685"))
try:
    import gsplat.train_oracle  # noqa: F401  (needs torch)
except ImportError:   # Mac venv without torch: hero.py only needs the SH constants
    import types
    _m = types.ModuleType("gsplat.train_oracle")
    _m.SH_C0, _m.SH_C1 = 0.28209479177387814, 0.4886025119029199
    _m.SH_C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
    _m.SH_C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
    sys.modules["gsplat.train_oracle"] = _m
import hero  # noqa: E402

ROOT = "/Users/smarton/dev/gsplat_tt"
TILE, G, CORES, KCAP = 32, 64, 110, 16
AMIN = 1.0 / 255.0


def edge_min(A, B, C, d_fixed, lo, hi, fixed_is_x):
    """min over t in [lo, hi] of the quadratic with one coordinate fixed."""
    if fixed_is_x:   # q = A dx^2 + 2B dx dy + C dy^2, dx fixed, minimise over dy
        t = np.clip(-B * d_fixed / C, lo, hi)
        return A * d_fixed ** 2 + 2 * B * d_fixed * t + C * t ** 2
    t = np.clip(-B * d_fixed / A, lo, hi)
    return C * d_fixed ** 2 + 2 * B * d_fixed * t + A * t ** 2


def run(jclamp):
    cams = json.load(open(os.environ.get("CAMS", os.path.join(ROOT, "benchmarks/cameras_test.json"))))["bicycle"]
    W, H = cams["image_size"]
    ply, cm = hero.load_ply(os.path.join(hero.scenes_dir(), "bicycle.ply"))
    op = 1.0 / (1.0 + np.exp(-ply[:, cm["opacity"]].astype(np.float64)))
    keep = np.flatnonzero(op >= AMIN)
    m2, con, _, tz, vis = hero.project(ply[keep], cm, cams["views"]["_DSC8679"]["c2w"], cams["intrinsics"],
                                       H, W, jclamp=jclamp)
    sel = np.flatnonzero(vis)
    m2, con, op = m2[sel], con[sel], op[keep][sel]
    det = 1.0 / (con[:, 0] * con[:, 2] - con[:, 1] ** 2)
    rx, ry = np.ceil(3 * np.sqrt(con[:, 2] * det)), np.ceil(3 * np.sqrt(con[:, 0] * det))
    tx, ty = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    x0 = np.clip(np.floor((m2[:, 0] - rx) / TILE), 0, tx - 1).astype(np.int64)
    x1 = np.clip(np.floor((m2[:, 0] + rx) / TILE), 0, tx - 1).astype(np.int64)
    y0 = np.clip(np.floor((m2[:, 1] - ry) / TILE), 0, ty - 1).astype(np.int64)
    y1 = np.clip(np.floor((m2[:, 1] + ry) / TILE), 0, ty - 1).astype(np.int64)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    k = nx * ny
    # enumerate pairs
    g = np.repeat(np.arange(len(k)), k)
    off = np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k)
    ptx, pty = x0[g] + off % nx[g], y0[g] + off // nx[g]
    A, B, C = con[g, 0], con[g, 1], con[g, 2]
    lx, hx = ptx * TILE + 0.5 - m2[g, 0], np.minimum(ptx * TILE + TILE, W) - 0.5 - m2[g, 0]
    ly, hy = pty * TILE + 0.5 - m2[g, 1], np.minimum(pty * TILE + TILE, H) - 0.5 - m2[g, 1]
    inside = (lx <= 0) & (hx >= 0) & (ly <= 0) & (hy >= 0)
    q = np.minimum.reduce([edge_min(A, B, C, lx, ly, hy, True), edge_min(A, B, C, hx, ly, hy, True),
                           edge_min(A, B, C, ly, lx, hx, False), edge_min(A, B, C, hy, lx, hx, False)])
    q = np.where(inside, 0.0, q)
    amax = np.minimum(op[g], 0.99) * np.exp(-0.5 * q)
    live = amax >= AMIN
    kt = np.bincount(g[live], minlength=len(k))
    tile = pty * tx + ptx
    tp, tpt = np.bincount(tile, minlength=tx * ty), np.bincount(tile[live], minlength=tx * ty)

    def kstats(kk):
        kk = kk[kk > 0]
        n = len(kk); ng = (n + G - 1) // G
        kp = np.zeros(ng * G, np.int64); kp[:n] = kk
        planes_capped = np.minimum(kp, KCAP).reshape(ng, G).max(1).sum()
        ks = np.sort(np.minimum(kp, KCAP))[::-1]   # k-bucketed: group rows of equal capped k
        planes_bucketed = ks.reshape(ng, G).max(1).sum()
        heavy = kk[kk > KCAP]
        return {"rows": int(n), "slots": int(kk.sum()), "max_k": int(kk.max()),
                "rows_k_gt_16": int((kk > 16).sum()), "rows_k_gt_64": int((kk > 64).sum()),
                "rows_k_gt_256": int((kk > 256).sum()), "heavy_slots_k_gt_16": int(heavy.sum()),
                "light_planes_kcap16": int(planes_capped),
                "light_planes_kcap16_bucketed": int(planes_bucketed),
                "p50_p99_p999": [float(np.percentile(kk, p)) for p in (50, 99, 99.9)]}

    # opacity-aware rectangle (3DGS/gsplat-style extent where op*exp(-r^2/2) >= 1/255, capped at 3 sigma)
    kc = np.minimum(3.0, np.sqrt(2.0 * np.log(np.maximum(255.0 * np.minimum(op, 0.99), 1.0))))
    orx, ory = np.ceil(kc * np.sqrt(con[:, 2] * det)), np.ceil(kc * np.sqrt(con[:, 0] * det))
    ok = ((np.clip(np.floor((m2[:, 0] + orx) / TILE), 0, tx - 1) - np.clip(np.floor((m2[:, 0] - orx) / TILE), 0, tx - 1) + 1)
          * (np.clip(np.floor((m2[:, 1] + ory) / TILE), 0, ty - 1) - np.clip(np.floor((m2[:, 1] - ory) / TILE), 0, ty - 1) + 1)
          ).astype(np.int64)
    return {"jclamp": jclamp, "visible_rows": int(len(k)), "opacity_rect": kstats(ok),
            "rect": kstats(k), "tight_alpha": kstats(kt),
            "pairs_rect": int(k.sum()), "pairs_tight": int(live.sum()),
            "tight_frac": float(live.mean()),
            "tile_pairs_rect_max_mean": [int(tp.max()), float(tp.mean())],
            "tile_pairs_tight_max_mean": [int(tpt.max()), float(tpt.mean())]}


if __name__ == "__main__":
    out = {"off": run(None), "on": run(1.3)}
    json.dump(out, open(os.path.join(HERE, "pair_model.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
