"""T0 oracle check of the device training forward (task #716): blend-only harness inputs,
a numpy model of the device math, and the comparison with the oracle image / final_T.

render/run.py cannot run the T0 scene from the oracle's 2D inputs (device projection, k_cap
bounds), so render/train/blend_fwd_bench feeds them straight to the production blend compute
kernel (render/kernels/compute/alpha_blend_compute_mb.cpp, BLEND_TRAIN=1, the host's train-mode
defines) and reads back its fp32 R/G/B/T train buffer (one 16 KB page per screen tile).

Lossless: every valid Gaussian goes to every tile in the oracle's front-to-back order with all
32 microblocks in its mask, so the only cull is the oracle's own alpha < alpha_min rule (no
3-sigma tile rect, no microblock cull, no T early-out). Opacity and colours travel as UNORM16
(the production record format; |error| <= 2^-17); the conic and mean are fp32.

File layout (little endian):
  tiles.u32  16 words per tile (every screen tile, row-major): tile_id, rec_page, num_g, 0...
  recs.u32   PACK2 records, 2 per 64 B page, each tile's list starting on a page:
             A' = -A/2, B' = -B, C' = -C/2, mask (0xffffffff), mx, my (tile-local fp32),
             op16 | r16 << 16, g16 | b16 << 16
  ramps.f32  2 tilized fp32 tiles: x = c + 0.5, y = r + 0.5
  meta.json  n_tiles, n_rec_pages, H, W
The bench writes train.f32: per tile 4 tilized fp32 tiles R, G, B, T (T < 0: the pixel
stopped on the t_min rule; |T| is the final T).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from gsplat.blend_backward_dev import TILE, cfg_from_dump, f32, fbits, tile_index, tilize

REC_WORDS = 8
FULL_MASK = 0xFFFFFFFF


def unorm16(v: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(np.asarray(v, np.float64), 0.0, 1.0) * 65535.0).astype(np.uint32)


def unorm16_decode(q: np.ndarray) -> np.ndarray:
    """Device decode: fl(float(q) * fl(1/65535)) (BLEND_SFPU_UNORM=1)."""
    return (np.asarray(q, np.uint32).astype(f32) * f32(1.0 / 65535.0)).astype(f32)


def untilize(t: np.ndarray) -> np.ndarray:
    r, c = np.meshgrid(np.arange(TILE), np.arange(TILE), indexing="ij")
    return np.asarray(t).reshape(1024)[tile_index(r, c)]


def prep(means2d, conic, opacity, colors, order, valid, H: int, W: int) -> dict:
    m2 = np.asarray(means2d, f32)
    con = np.asarray(conic, f32)
    op16 = unorm16(opacity)
    col16 = unorm16(colors)
    sel = [int(g) for g in np.asarray(order, np.int64) if bool(valid[g])]
    tiles_x, tiles_y = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    rr, cc = np.meshgrid(np.arange(TILE), np.arange(TILE), indexing="ij")
    ramps = np.stack([tilize(cc.astype(f32) + f32(0.5)), tilize(rr.astype(f32) + f32(0.5))])
    tiles, recs = [], []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            page = len(recs) // 2
            tiles.append([ty * tiles_x + tx, page, len(sel), 0] + [0] * 12)
            for g in sel:
                A, B, C = con[g]
                recs.append([fbits(f32(-0.5) * A), fbits(-B), fbits(f32(-0.5) * C), FULL_MASK,
                             fbits(m2[g, 0] - f32(tx * TILE)), fbits(m2[g, 1] - f32(ty * TILE)),
                             int(op16[g] | (col16[g, 0] << 16)), int(col16[g, 1] | (col16[g, 2] << 16))])
            if len(recs) % 2:
                recs.append([0] * REC_WORDS)  # pad: the next tile starts on a page
    return {"tiles": np.asarray(tiles, np.uint32).reshape(-1, 16),
            "recs": np.asarray(recs, np.uint32).reshape(-1, REC_WORDS), "ramps": ramps,
            "H": H, "W": W}


def write(p: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    p["tiles"].tofile(os.path.join(out, "tiles.u32"))
    p["recs"].tofile(os.path.join(out, "recs.u32"))
    p["ramps"].astype(f32).tofile(os.path.join(out, "ramps.f32"))
    meta = {"n_tiles": len(p["tiles"]), "n_rec_pages": max(1, len(p["recs"]) // 2), "H": p["H"], "W": p["W"]}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def emulate(p: dict, cfg) -> np.ndarray:
    """numpy fp32 model of BLEND_TRAIN's blend_train_gaussian_math (np.exp for the exp):
    the train buffer the device should write, (n_tiles, 4, 1024)."""
    out = np.zeros((len(p["tiles"]), 4, 1024), f32)
    x, y = p["ramps"]
    amax, amin, tmin = f32(cfg.alpha_max), f32(cfg.alpha_min), f32(cfg.t_min)
    for i, (_, page, n) in enumerate(p["tiles"][:, :3]):
        R, G, Bc = (np.zeros(1024, f32) for _ in range(3))
        T = np.ones(1024, f32)
        for w in p["recs"][2 * page:2 * page + n]:
            A, B, C, _, mx, my = (np.uint32(v).view(f32) for v in w[:6])
            op, cr = unorm16_decode(w[6] & 0xFFFF), unorm16_decode(w[6] >> 16)
            cg, cb = unorm16_decode(w[7] & 0xFFFF), unorm16_decode(w[7] >> 16)
            dx, dy = x - mx, y - my
            pw = A * (dx * dx) + B * (dx * dy) + C * (dy * dy)
            pw = np.clip(pw, f32(-80.0), f32(0.0))
            al = np.minimum(op * np.exp(pw).astype(f32), amax)
            tn = T * (f32(1.0) - al)
            live = (al >= amin) & (T >= 0)
            acc = live & (tn >= tmin)
            at = al * T
            R = np.where(acc, R + at * cr, R)
            G = np.where(acc, G + at * cg, G)
            Bc = np.where(acc, Bc + at * cb, Bc)
            T = np.where(acc, tn, np.where(live, -np.abs(T), T)).astype(f32)
        out[i] = np.stack([R, G, Bc, T])
    return out


def to_image(train: np.ndarray, tiles: np.ndarray, H: int, W: int, background) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(image (H,W,3) = C + |T| bg, final_T (H,W) = |T|, stopped (H,W) bool) from the train pages."""
    tiles_x = (W + TILE - 1) // TILE
    img = np.zeros((H, W, 3), np.float64)
    fT = np.zeros((H, W), np.float64)
    stop = np.zeros((H, W), bool)
    for i, tid in enumerate(tiles[:, 0]):
        ty, tx = divmod(int(tid), tiles_x)
        oy, ox = ty * TILE, tx * TILE
        h, w = min(TILE, H - oy), min(TILE, W - ox)
        pl = [untilize(train[i, k])[:h, :w] for k in range(4)]
        T = pl[3].astype(np.float64)
        fT[oy:oy + h, ox:ox + w] = np.abs(T)
        stop[oy:oy + h, ox:ox + w] = np.signbit(pl[3])
        for k in range(3):
            img[oy:oy + h, ox:ox + w, k] = pl[k] + np.abs(T) * background[k]
    return img, fT, stop


def compare(d, img: np.ndarray, fT: np.ndarray, bar: float) -> dict:
    ei = np.abs(img - d["image"])
    eT = np.abs(fT - d["final_T"])
    rep = {"max_abs_image": float(ei.max()), "max_abs_image_rgb": [float(ei[..., k].max()) for k in range(3)],
           "max_abs_final_T": float(eT.max()), "bar": bar}
    rep["pass"] = bool(rep["max_abs_image"] <= bar and rep["max_abs_final_T"] <= bar)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("npz")
    a.add_argument("out")
    a.add_argument("--emulate", action="store_true", help="also compare the numpy model with the oracle")
    c = sub.add_parser("check")
    c.add_argument("npz")
    c.add_argument("prep_dir", help="prep dir holding the device train.f32")
    c.add_argument("--bar", type=float, default=1e-4, help="max abs error for image and final_T")
    c.add_argument("--image-out", help="write the device image (H,W,3 fp32) here for blend_backward_dev")
    args = ap.parse_args(argv)
    d = np.load(args.npz, allow_pickle=False)
    cfg = cfg_from_dump(d)
    H, W = d["image"].shape[:2]
    p = prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"], H, W)
    if args.cmd == "prep":
        write(p, args.out)
        print(json.dumps({"n_tiles": len(p["tiles"]), "n_recs": len(p["recs"])}))
        if args.emulate:
            img, fT, _ = to_image(emulate(p, cfg), p["tiles"], H, W, cfg.background)
            print(json.dumps({"emulate_vs_oracle": compare(d, img, fT, 1e-4)}))
        return 0
    train = np.fromfile(os.path.join(args.prep_dir, "train.f32"), f32).reshape(-1, 4, 1024)
    img, fT, stop = to_image(train, p["tiles"], H, W, cfg.background)
    rep = compare(d, img, fT, args.bar)
    model = to_image(emulate(p, cfg), p["tiles"], H, W, cfg.background)
    rep["max_abs_image_vs_model"] = float(np.abs(img - model[0]).max())
    rep["max_abs_final_T_vs_model"] = float(np.abs(fT - model[1]).max())
    rep["stopped_px"] = int(stop.sum())
    rep["stopped_px_model"] = int(model[2].sum())
    if args.image_out:
        img.astype(f32).tofile(args.image_out)
    print(json.dumps({"npz": os.path.basename(args.npz), **rep}))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
