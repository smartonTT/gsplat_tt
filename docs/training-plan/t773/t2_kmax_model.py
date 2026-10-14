"""T2 rank-plane padding model on the bicycle hero view (#773, Mac only, no device).

The t697/t750 T2 gather (render/train/kernels/t2g_reader.cpp:69-71) pads each 64-row group to
the group's largest slot count kmax. This counts, for the hero view, the slots T1 v2 writes
(3-sigma tile rectangles, as docs/t685/hero.py), the rank planes the gather walks (sum of kmax
over groups) and how that work splits over 110 cores (contiguous group ranges, as the bench).
Usage: python3 docs/training-plan/t773/t2_kmax_model.py  (needs scenes/bicycle.ply)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, os.path.join(ROOT, "docs/t685"))
import hero  # noqa: E402

G, CORES, TILE = 64, 110, 32


def main():
    cams = json.load(open(os.path.join(ROOT, "benchmarks/cameras_test.json")))["bicycle"]
    W, H = cams["image_size"]
    ply, cm = hero.load_ply(os.path.join(hero.scenes_dir(), "bicycle.ply"))
    op = 1.0 / (1.0 + np.exp(-ply[:, cm["opacity"]].astype(np.float64)))
    keep = np.flatnonzero(op >= 1.0 / 255.0)
    rows = ply[keep]
    m2, conic, col, tz, vis = hero.project(rows, cm, cams["views"]["_DSC8679"]["c2w"], cams["intrinsics"], H, W)
    sel = np.flatnonzero(vis)
    # radii again (project() does not return them)
    det = 1.0 / (conic[:, 0] * conic[:, 2] - conic[:, 1] ** 2)
    a, c = conic[:, 2] * det, conic[:, 0] * det
    rx, ry = np.ceil(3 * np.sqrt(np.maximum(a, 0))), np.ceil(3 * np.sqrt(np.maximum(c, 0)))
    mx, my, rx, ry = m2[sel, 0], m2[sel, 1], rx[sel], ry[sel]
    tx, ty = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    x0 = np.clip(np.floor((mx - rx) / TILE), 0, tx - 1); x1 = np.clip(np.floor((mx + rx) / TILE), 0, tx - 1)
    y0 = np.clip(np.floor((my - ry) / TILE), 0, ty - 1); y1 = np.clip(np.floor((my + ry) / TILE), 0, ty - 1)
    k = ((x1 - x0 + 1) * (y1 - y0 + 1)).astype(np.int64)
    n = len(k)
    ng = (n + G - 1) // G
    kp = np.zeros(ng * G, np.int64); kp[:n] = k
    kg = kp.reshape(ng, G)
    kmax = kg.max(1)
    planes = int(kmax.sum())
    # contiguous equal group ranges per core (bench split), and an ideal balance
    per_core = [int(x.sum()) for x in np.array_split(kmax, CORES)]
    q = np.quantile(k, [0.5, 0.9, 0.99, 0.999, 1.0])
    heavy = {t: int((k > t).sum()) for t in (16, 64, 256)}
    out = {"n_visible": n, "n_slots": int(k.sum()), "n_groups": ng, "rank_planes_sum_kmax": planes,
           "padding_factor_planes_over_meanK": round(planes * G / max(1, k.sum()), 2),
           "k_quantiles_50_90_99_999_max": [float(x) for x in q],
           "rows_with_k_gt": heavy,
           "groups_with_kmax_gt_64": int((kmax > 64).sum()),
           "planes_in_groups_kmax_gt_64": int(kmax[kmax > 64].sum()),
           "busiest_core_planes_contig": max(per_core), "mean_core_planes": round(planes / CORES, 1),
           "planes_if_rows_capped_at_16_plus_flat_tail": int(np.minimum(kmax, 16).sum())}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "t2_kmax_model.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
