"""T1 blend backward on device (task #685): input prep, a numpy model of the device
kernel, and the slot comparison against gsplat.blend_backward_ref.

The device harness (render/train, blend_bwd_bench) reads the files written by
``prep()`` and writes one 64 B page per slot (9 fp32 + pad).

Device decomposition (docs/training-plan/T1-blend-backward.md section 4):
  * one 32x32 tile per pass, pairs in the forward order (ascending ``order``),
  * Dst fp32 planes: 0 T, 1 BH = image.g - C_acc.g, 2-4 dL/dimage, 5-6 x/y ramps,
    7 ACC (vectors 0-8 per-lane term sums, 9.. lane-reduced pair results),
  * per pair only the SFPU vectors (4 Dst rows x 8 same-parity columns of one face)
    in the pair's mask are visited. A vector is skipped only when no pixel in it
    reaches alpha >= alpha_min / 2 (conservative geometric cull, no T involved).
  * dead pixels carry T = 0, so the stop test T (1 - alpha) >= t_min also kills them.

File layout (little endian):
  tiles.u32  16 words per tile with pairs: tile_id, rec_start, rec_count, n_batches
  recs.u32   16 words per pair: A', B', C' (A' = -A/2, B' = -B, C' = -C/2), mx, my
             (tile-local), opacity, r, g, b, A, B, C, vector mask, slot, 0, 0
  planes.f32 5 tilized fp32 tiles per listed tile: T0, BH0, g0, g1, g2
  ramps.f32  2 tilized fp32 tiles: x = c + 0.5, y = r + 0.5
  meta.json  counts and the blend cfg bits
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from gsplat.blend_backward_ref import (BlendCfg, blend_backward, cov_diag_from_conic,
                                       rel_error, tile_rects)

TILE = 32
BATCH = 16         # pairs per ACC pack (v2: 3 result vectors per 8 pairs, vectors 24..29)
REC_WORDS = 16
N_PLANES = 5
f32 = np.float32


def tile_index(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Element index in a packed 32x32 tile (four 16x16 faces, row-major in a face)."""
    return 256 * (2 * (r >> 4) + (c >> 4)) + 16 * (r & 15) + (c & 15)


def tilize(a: np.ndarray) -> np.ndarray:
    out = np.zeros(1024, f32)
    r, c = np.meshgrid(np.arange(TILE), np.arange(TILE), indexing="ij")
    out[tile_index(r, c)] = a
    return out


def vector_lanes() -> np.ndarray:
    """(32, 4, 8) packed-tile element of SFPU vector V, LREG row j, column k."""
    V = np.arange(32)[:, None, None]
    j = np.arange(4)[None, :, None]
    k = np.arange(8)[None, None, :]
    return 64 * (V >> 1) + 16 * j + 2 * k + (V & 1)


def fbits(x) -> int:
    return int(np.asarray(x, f32).view(np.uint32))


def cfg_from_dump(dump) -> BlendCfg:
    d = json.loads(str(dump["cfg_json"]))
    return BlendCfg(d["alpha_max"], d["alpha_min"], d["t_min"], tuple(d["background"]))


def prep(means2d, conic, opacity, colors, order, valid, image, dL_dimage, cfg: BlendCfg) -> dict:
    H, W = image.shape[:2]
    m2 = np.asarray(means2d, f32)
    con = np.asarray(conic, f32)
    op = np.asarray(opacity, f32)
    col = np.asarray(colors, f32)
    rects = tile_rects(np.asarray(means2d, np.float64), conic, valid,
                       cov_diag_from_conic(np.asarray(conic, np.float64)), H, W, TILE)
    counts = np.where(rects[:, 1] >= rects[:, 0],
                      (rects[:, 1] - rects[:, 0] + 1) * (rects[:, 3] - rects[:, 2] + 1), 0)
    P = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    img = np.asarray(image, f32)
    gimg = np.asarray(dL_dimage, f32)
    Ig = (img * gimg).sum(-1, dtype=f32)
    tiles_x, tiles_y = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    lanes = vector_lanes().reshape(32, 32)
    rr, cc = np.meshgrid(np.arange(TILE), np.arange(TILE), indexing="ij")
    xr = tilize(cc.astype(f32) + f32(0.5))
    yr = tilize(rr.astype(f32) + f32(0.5))
    order = np.asarray(order, np.int64)
    x0, x1, y0, y1 = (rects[order, i] for i in range(4))
    tiles, recs, planes = [], [], []
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            sel = order[(x0 <= tx) & (tx <= x1) & (y0 <= ty) & (ty <= y1)]
            if sel.size == 0:
                continue
            ox, oy = tx * TILE, ty * TILE
            h, w = min(TILE, H - oy), min(TILE, W - ox)
            inside = np.zeros((TILE, TILE), bool)
            inside[:h, :w] = True
            pl = np.zeros((N_PLANES, TILE, TILE), f32)
            pl[0][inside] = 1.0
            pl[1, :h, :w] = Ig[oy:oy + h, ox:ox + w]
            for k in range(3):
                pl[2 + k, :h, :w] = gimg[oy:oy + h, ox:ox + w, k]
            in_t = tilize(inside.astype(f32))[lanes] > 0          # (32 vectors, 32 lanes)
            tiles.append([ty * tiles_x + tx, len(recs), sel.size, -(-sel.size // BATCH)])
            planes.append(np.stack([tilize(p) for p in pl]))
            for g in sel:
                A, B, C = (float(v) for v in con[g])
                mx, my = m2[g, 0] - f32(ox), m2[g, 1] - f32(oy)
                u = xr[lanes].astype(np.float64) - float(mx)
                v = yr[lanes].astype(np.float64) - float(my)
                pw = -0.5 * (A * u * u + 2.0 * B * u * v + C * v * v)
                a = float(op[g]) * np.exp(np.minimum(pw, 0.0))
                hit = ((a >= 0.5 * cfg.alpha_min) & in_t).any(1)
                mask = int(sum(1 << V for V in np.nonzero(hit)[0]))
                rx0, rx1, ry0, ry1 = rects[g]
                j = (ty - ry0) * (rx1 - rx0 + 1) + (tx - rx0)
                cA, cB, cC = con[g]
                rec = [fbits(f32(-0.5) * cA), fbits(-cB), fbits(f32(-0.5) * cC), fbits(mx), fbits(my),
                       fbits(op[g]), fbits(col[g, 0]), fbits(col[g, 1]), fbits(col[g, 2]),
                       fbits(cA), fbits(cB), fbits(cC), mask, int(P[g] + j), 0, 0]
                recs.append(rec)
    return {"tiles": np.asarray(tiles, np.uint32).reshape(-1, 4),
            "recs": np.asarray(recs, np.uint32).reshape(-1, REC_WORDS),
            "planes": np.asarray(planes, f32).reshape(-1, N_PLANES, 1024),
            "ramps": np.stack([xr, yr]), "n_slots": int(counts.sum()), "P": P, "counts": counts,
            "cfg": cfg, "H": H, "W": W}


def write(p: dict, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    t = np.zeros((len(p["tiles"]), 16), np.uint32)
    t[:, :4] = p["tiles"]
    t.tofile(os.path.join(out, "tiles.u32"))
    p["recs"].astype(np.uint32).tofile(os.path.join(out, "recs.u32"))
    p["planes"].astype(f32).tofile(os.path.join(out, "planes.f32"))
    p["ramps"].astype(f32).tofile(os.path.join(out, "ramps.f32"))
    cfg = p["cfg"]
    meta = {"n_tiles": len(p["tiles"]), "n_recs": len(p["recs"]), "n_slots": p["n_slots"],
            "H": p["H"], "W": p["W"], "alpha_max_bits": fbits(cfg.alpha_max),
            "alpha_min_bits": fbits(cfg.alpha_min), "t_min_bits": fbits(cfg.t_min),
            "n_vectors": int(sum(bin(int(m)).count("1") for m in p["recs"][:, 12]))}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def emulate(p: dict, exp=np.exp) -> np.ndarray:
    """numpy model of the device kernel (same planes, masks, lane math order). `exp` is the
    kernel's exp (fp32-accurate on device; exp21f() models the forward's bf16-grade one)."""
    cfg = p["cfg"]
    lanes = vector_lanes().reshape(32, 32)
    xr, yr = p["ramps"][0][lanes], p["ramps"][1][lanes]
    slots = np.zeros((p["n_slots"], 9), f32)
    amax, amin, tmin, one = f32(cfg.alpha_max), f32(cfg.alpha_min), f32(cfg.t_min), f32(1.0)
    for (tid, start, cnt, _), pl in zip(p["tiles"], p["planes"]):
        T, BH, g0, g1, g2 = (pl[k][lanes].copy() for k in range(N_PLANES))
        for rec in p["recs"][start:start + cnt]:
            fr = rec[:12].view(f32)
            Af, Bf, Cf, mx, my, op, cr, cgc, cb, A, B, C = fr
            acc = np.zeros((9, 32, 32), f32)
            for V in range(32):
                if not (int(rec[12]) >> V) & 1:
                    continue
                u, v = xr[V] - mx, yr[V] - my
                pw = Af * (u * u) + Bf * (u * v) + Cf * (v * v)
                w = exp(np.minimum(pw, f32(0))).astype(f32)
                al = op * w
                ga = np.where(pw > 0, f32(0), al)
                cl = al > amax
                al = np.where(cl, amax, al)
                w = np.where(cl, f32(0), w)
                ga = np.where(cl, f32(0), ga)
                om = one - al
                Tn = T[V] * om
                live = (al >= amin) & (Tn >= tmin)
                stop = (al >= amin) & ~live
                at = al * T[V]
                cg = g0[V] * cr + g1[V] * cgc + g2[V] * cb
                bh = BH[V] - at * cg
                da = T[V] * cg - bh / om
                G = da * ga
                terms = [da * w, at * g0[V], at * g1[V], at * g2[V],
                         G * u * u, G * u * v, G * v * v, G * u, G * v]
                for k in range(9):
                    acc[k, V] += np.where(live, terms[k], f32(0))
                BH[V] = np.where(live, bh, BH[V])
                T[V] = np.where(live, Tn, np.where(stop, f32(0), T[V]))
            s = acc.sum((1, 2), dtype=f32)
            dw, ag0, ag1, ag2, Suu, Suv, Svv, Su, Sv = s
            slots[int(rec[13])] = [A * Su + B * Sv, B * Su + C * Sv, f32(-0.5) * Suu, -Suv,
                                   f32(-0.5) * Svv, dw, ag0, ag1, ag2]
    return slots


def pack_v2(acc: np.ndarray, Af, Bf, Cf) -> tuple[np.ndarray, np.ndarray]:
    """numpy model of the v2 on-device finish (#704) for the NB pairs of one ACC pack.
    acc: (n, 9, 4, 8) per-lane term sums (LREG row j, column k) of pair i.
    Returns (packed ACC tile (1024,), slots (n, 9)) where slots is what bwd_writer gathers.
    Models blend_bwd_compute.cpp bwd_finish: per-lane finish, SFPTRANSP (reg i row j <- reg j
    row i), add tree, row rotate-add by 1, 2, 4 (SFPSHFT2 SUBVEC_SHFLROR1), column select."""
    acc = np.asarray(acc, f32)
    Af, Bf, Cf = (np.broadcast_to(np.asarray(x, f32), (acc.shape[0],)) for x in (Af, Bf, Cf))
    tile = np.zeros(1024, f32)
    lanes = vector_lanes()                                   # (32, 4, 8)
    for i, a in enumerate(acc):
        t = Af[i] * a[7]
        dmx = -((t + t) + Bf[i] * a[8])
        t = Cf[i] * a[8]
        dmy = -((t + t) + Bf[i] * a[7])
        z = np.zeros_like(a[0])
        groups = [(a[0], a[1], a[2], a[3]),
                  (a[4] * f32(-0.5), -a[5], a[6] * f32(-0.5), dmx), (dmy, z, z, z)]
        V0, q = 24 + 3 * (i // 8), i % 8
        for gk, g in enumerate(groups):
            r = np.stack(g)                                  # (reg, row, col)
            r = r.transpose(1, 0, 2)                         # SFPTRANSP
            s = (r[0] + r[1]) + (r[2] + r[3])
            for sh in (1, 2, 4):
                s = s + np.roll(s, sh, axis=1)
            tile[lanes[V0 + gk][:, q]] = s[:, q]
    o = tile
    lane = lambda V, j, k: 64 * (V >> 1) + 16 * j + 2 * k + (V & 1)  # noqa: E731
    slots = np.zeros((acc.shape[0], 9), f32)
    for k in range(acc.shape[0]):
        V, q = 24 + 3 * (k >> 3), k & 7
        slots[k] = [o[lane(V + 1, 3, q)], o[lane(V + 2, 0, q)], o[lane(V + 1, 0, q)],
                    o[lane(V + 1, 1, q)], o[lane(V + 1, 2, q)], o[lane(V, 0, q)],
                    o[lane(V, 1, q)], o[lane(V, 2, q)], o[lane(V, 3, q)]]
    return tile, slots


def exp21f(x: np.ndarray) -> np.ndarray:
    """numpy model of _sfpu_exp_21f_bf16_ for x <= 0 (degree-2 2^f, ~1.7e-3 relative error)."""
    z = np.floor(np.maximum(np.asarray(x, f32) * f32(1.442695) + f32(127), f32(0)).astype(np.float64) * 2**23)
    z = z.astype(np.int64)
    fr = (z & 0x7FFFFF).astype(np.float64)
    poly = 1.0017248 + fr * (7.839635491371155e-08 + fr * 4.791750143340323e-15)
    return (poly * np.exp2((z >> 23) - 127.0)).astype(f32)


def load_dump(path: str):
    d = np.load(path, allow_pickle=False)
    return d, cfg_from_dump(d)


def ref_slots(d, cfg) -> np.ndarray:
    return blend_backward(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"],
                          d["valid"], d["image"], d["dL_dimage"], cfg)["slots"]


def compare_slots(dev: np.ndarray, ref: np.ndarray) -> dict:
    rep = {"rel_err": rel_error(dev.astype(np.float64), ref.astype(np.float64))}
    names = ["dmx", "dmy", "dA", "dB", "dC", "dop", "dr", "dg", "db"]
    rep["per_term"] = {n: rel_error(dev[:, k].astype(np.float64), ref[:, k].astype(np.float64))
                       for k, n in enumerate(names)}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prep")
    a.add_argument("npz")
    a.add_argument("out")
    a.add_argument("--emulate", action="store_true")
    a.add_argument("--image", help="fp32 (H, W, 3) forward image to use in place of the oracle's "
                   "(#716: the device forward from blend_forward_dev check --image-out)")
    c = sub.add_parser("check")
    c.add_argument("npz")
    c.add_argument("slots", help="device slots.f32 (16 floats per slot)")
    c.add_argument("--bar", type=float, default=1e-2)
    args = ap.parse_args(argv)
    d, cfg = load_dump(args.npz)
    if args.cmd == "prep":
        image = d["image"]
        if args.image:
            image = np.fromfile(args.image, f32).reshape(image.shape)
        p = prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                 image, d["dL_dimage"], cfg)
        write(p, args.out)
        print(json.dumps({"n_tiles": len(p["tiles"]), "n_recs": len(p["recs"]), "n_slots": p["n_slots"]}))
        if args.emulate:
            print(json.dumps({"emulate_vs_ref": compare_slots(emulate(p), ref_slots(d, cfg))}))
        return 0
    dev = np.fromfile(args.slots, f32).reshape(-1, 16)[:, :9]
    rep = compare_slots(dev, ref_slots(d, cfg))
    rep["pass"] = bool(rep["rel_err"] <= args.bar)
    print(json.dumps({"npz": os.path.basename(args.npz), **rep}))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
