"""T2 per-core rank-plane counts on the bicycle hero view (#778, Mac only, no device).

Same per-row slot count k as docs/training-plan/t773/t2_kmax_model.py (#773: 3-sigma tile
rectangles, groups of 64 visible rows in sel order, plane count per group = kmax), but split
over cores the way render/train/train_step.cpp:276-282 does it: blocks of 16 groups, block b on
core b % n_cores (strided), not contiguous ranges. Writes t2_core_planes.json next to this file.
Usage: python3 docs/t778/t2_core_planes.py  (needs scenes/bicycle.ply)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../.."))
sys.path[:0] = [ROOT, os.path.join(ROOT, "docs/t685")]
import types  # noqa: E402
# hero.py imports SH constants from gsplat.train_oracle, which needs torch (not in the project
# venv). Slot counts do not use colour, so stand-in constants are enough here.
_tor = types.ModuleType("gsplat.train_oracle")
_tor.SH_C0, _tor.SH_C1, _tor.SH_C2, _tor.SH_C3 = 0.28209479177387814, 0.4886025119029199, [0.0] * 5, [0.0] * 7
sys.modules["gsplat.train_oracle"] = _tor
import hero  # noqa: E402

G, GPB, CORES, TILE = 64, 16, 110, 32


def stats(per_core):
    pc = np.asarray(per_core)
    top = np.argsort(-pc)[:5]
    return {"busiest": int(pc.max()), "mean": round(float(pc.mean()), 1), "min": int(pc.min()),
            "mean_over_busiest": round(float(pc.mean() / pc.max()), 3),
            "top5_cores": [[int(i), int(pc[i])] for i in top],
            "quantiles_10_50_90": [float(x) for x in np.quantile(pc, [0.1, 0.5, 0.9])]}


def main():
    cams = json.load(open(os.path.join(ROOT, "benchmarks/cameras_test.json")))["bicycle"]
    W, H = cams["image_size"]
    ply, cm = hero.load_ply(os.path.join(hero.scenes_dir(), "bicycle.ply"))
    op = 1.0 / (1.0 + np.exp(-ply[:, cm["opacity"]].astype(np.float64)))
    rows = ply[np.flatnonzero(op >= 1.0 / 255.0)]
    m2, conic, col, tz, vis = hero.project(rows, cm, cams["views"]["_DSC8679"]["c2w"], cams["intrinsics"], H, W)
    sel = np.flatnonzero(vis)
    det = 1.0 / (conic[:, 0] * conic[:, 2] - conic[:, 1] ** 2)
    a, c = conic[:, 2] * det, conic[:, 0] * det
    rx, ry = np.ceil(3 * np.sqrt(np.maximum(a, 0))), np.ceil(3 * np.sqrt(np.maximum(c, 0)))
    mx, my, rx, ry = m2[sel, 0], m2[sel, 1], rx[sel], ry[sel]
    tx, ty = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    x0 = np.clip(np.floor((mx - rx) / TILE), 0, tx - 1); x1 = np.clip(np.floor((mx + rx) / TILE), 0, tx - 1)
    y0 = np.clip(np.floor((my - ry) / TILE), 0, ty - 1); y1 = np.clip(np.floor((my + ry) / TILE), 0, ty - 1)
    k = ((x1 - x0 + 1) * (y1 - y0 + 1)).astype(np.int64)
    n = len(k)
    ng = -(-n // (G * GPB)) * GPB                       # t2_dev.py:92, whole blocks
    kp = np.zeros(ng * G, np.int64); kp[:n] = k
    kmax = kp.reshape(ng, G).max(1)
    nb = ng // GPB
    blk = kmax.reshape(nb, GPB).sum(1)
    strided = np.bincount(np.arange(nb) % CORES, weights=blk, minlength=CORES).astype(np.int64)
    contig = np.array([int(x.sum()) for x in np.array_split(kmax, CORES)])
    # strided split with every row capped at k=16 (L1's rank-plane part only)
    blk16 = np.minimum(kmax, 16).reshape(nb, GPB).sum(1)
    strided16 = np.bincount(np.arange(nb) % CORES, weights=blk16, minlength=CORES).astype(np.int64)
    out = {"n_visible": n, "n_groups": int(ng), "n_blocks": int(nb), "planes": int(kmax.sum()),
           "device_split_strided_blocks": stats(strided),
           "t773_model_contiguous": stats(contig),
           "strided_capped_k16": stats(strided16),
           "heaviest_blocks": [[int(b), int(blk[b]), int(b % CORES)] for b in np.argsort(-blk)[:8]],
           "per_core_strided": strided.tolist()}
    print(json.dumps({k: v for k, v in out.items() if k != "per_core_strided"}, indent=1))
    json.dump(out, open(os.path.join(HERE, "t2_core_planes.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
