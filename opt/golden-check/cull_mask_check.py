#!/usr/bin/env python3
"""Compare the device microblock keep mask with an exact fp64 box-min test.

Input: a GSPLAT_TT_DUMP_CULL directory (raw u32 dumps of sort_tile_ranges,
blend_subchunk_meta, sort_subchunk_dir, sort_subchunk_payload + meta.txt).
Each slab record (32 B) carries the pre-folded conic {A,B,C} (w0..2), the
device mask (w3, bit m = microblock m: x-block m&3 of 8 px, y-block m>>2 of
4 px), the tile-local mean (w4, w5) and UNORM16 opacity (w6 & 0xffff).

For every (gaussian, tile, microblock) we compute, in fp64:
  m2_min = min over the box [ox, ox+8] x [oy, oy+4] (tile-local px) of
           ci_a*u^2 + 2*ci_b*u*v + ci_c*v^2, u = x - mx, v = y - my
  keep   = op * exp(-0.5 * m2_min) >= floor
and the peak contribution at the 32 pixel centres (c+0.5, r+0.5).
A false cull is device bit 0 while fp64 keep is 1.

Usage: cull_mask_check.py DUMP_DIR [--px X --py Y] [--out report.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

MB_FIT = 8192
BOX_W, BOX_H = 7.0, 3.0  # pixel-centre extent of an 8x4 microblock (task #44)
THR_MARGIN = 0.05  # m2 slack added to thr on device (kThrMargin, task #44)
SLAB_PAGE_WORDS = 2048 // 4  # sort_subchunk_payload interleave page (64 recs)


def load(d):
    d = Path(d)
    meta = dict(l.split() for l in (d / "meta.txt").read_text().splitlines() if l.strip())
    u = {n: np.fromfile(d / f"{n}.u32", dtype=np.uint32) for n in
         ("sort_tile_ranges", "blend_subchunk_meta", "sort_subchunk_dir",
          "sort_subchunk_payload")}
    return meta, u


def gather(meta, u):
    ntiles = int(meta["num_tiles"])
    tiles_x = int(meta["tiles_x"])
    rng, scm, sdir, pay = (u["sort_tile_ranges"], u["blend_subchunk_meta"],
                           u["sort_subchunk_dir"], u["sort_subchunk_payload"])
    recs, tid, rank = [], [], []
    for t in range(ntiles):
        L = int(rng[2 * t + 1]) - int(rng[2 * t])
        if L <= 0:
            continue
        dir_base, nsc = int(scm[2 * t]), max(1, int(scm[2 * t + 1]))
        for sc in range(nsc):
            off = sc * MB_FIT
            Ls = min(MB_FIT, L - off)
            if Ls <= 0:
                continue
            page = int(sdir[(dir_base + sc) * 4])
            w = pay[page * SLAB_PAGE_WORDS: page * SLAB_PAGE_WORDS + Ls * 8].reshape(Ls, 8)
            recs.append(w)
            tid.append(np.full(Ls, t, np.int32))
            rank.append(np.arange(off, off + Ls, dtype=np.int32))
    recs = np.concatenate(recs)
    return recs, np.concatenate(tid), np.concatenate(rank), tiles_x


def boxmin_m2(ca, cb, cc, ulo, uhi, vlo, vhi, dtype=np.float64):
    """Exact box-constrained min of a PSD quadratic (two nearest-edge candidates).

    Vectorised; all args broadcast. Mirrors the device algorithm but in `dtype`.
    """
    z = dtype(0)
    uc = np.clip(z, ulo, uhi)
    vc = np.clip(z, vlo, vhi)
    vs = np.clip(-cb * uc / cc, vlo, vhi)
    us = np.clip(-cb * vc / ca, ulo, uhi)
    qv = ca * uc * uc + dtype(2) * cb * uc * vs + cc * vs * vs
    qh = ca * us * us + dtype(2) * cb * us * vc + cc * vc * vc
    return np.minimum(qv, qh)


def band_keep_f32(A, B, C, mx, my, q, floor, margin=THR_MARGIN, log_under=0.0):
    """fp32 model of the band-extent microblock cull (task #59).

    Mirrors microblock_cull_compute.cpp: one SFPU lane per gaussian, the 8
    microblock rows ("bands") of a tile evaluated in turn. For band j (pixel-
    centre rows v in [4j+0.5, 4j+3.5] - my) the ellipse m2 <= t has an exact
    u-extent [R*vl - S*sqrt(dl), R*vr + S*sqrt(dr)], where vr/vl clamp the
    ellipse's right/left-most point into the band (the extent is concave /
    convex in v). Column k is kept iff that extent meets its pixel-centre span
    [8k+0.5, 8k+7.5] - mx; the comparison is done on squares (no sqrt). This is
    the same "ellipse meets the pixel-centre box" test as boxmin_m2, just
    organised by rows. Inputs are the slab record fields (A,B,C pre-folded
    conic, tile-local mean, UNORM16 opacity q). Returns bool [n, 32], bit m =
    microblock m (x-block m&3, y-block m>>2).
    """
    f = np.float32
    A, B, C, mx, my = (np.asarray(x, dtype=f) for x in (A, B, C, mx, my))
    ci_a = f(-2) * A
    ci_c = f(-2) * C
    det = ci_a * ci_c - B * B  # ci_b = -B
    op = np.asarray(q, dtype=f) * f(1.0 / 65535.0)
    ratio = op * f(1.0 / floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (f(2) * (np.log(ratio.astype(np.float64)) - log_under).astype(f) + f(margin)).astype(f)
        P = t * ci_a
        nQ = -det
        S = f(1) / ci_a
        S2 = S * S
        R = B * S
        nR = -R
        vR = B * np.sqrt(np.maximum(t, f(0)) * (f(1) / (det * ci_c)))
        nvR = -vR
        vb = f(0.5) - my
        nc0 = mx - f(0.5)
        c07 = f(7.5) - mx
        keep = np.zeros((len(A), 32), dtype=bool)
        for j in range(8):
            v0 = vb + f(4 * j)
            v1 = vb + f(4 * j + 3)
            vc = np.minimum(np.maximum(f(0), v0), v1)
            dc = nQ * (vc * vc) + P
            vr = np.minimum(np.maximum(v0, vR), v1)
            vl = np.minimum(np.maximum(v0, nvR), v1)
            Dr = S2 * (nQ * (vr * vr) + P)
            Dl = S2 * (nQ * (vl * vl) + P)
            Dr = np.where(dc < 0, f(-1e30), Dr)
            Dl = np.where(dc < 0, f(-1e30), Dl)
            nwr0 = R * vr + nc0
            nzl0 = nR * vl + c07
            for k in range(4):
                nw = nwr0 - f(8 * k)
                nz = nzl0 + f(8 * k)
                tr = nw * np.abs(nw) + Dr
                tl = nz * np.abs(nz) + Dl
                keep[:, 4 * j + k] = np.minimum(tr, tl) >= 0
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--px", type=float, default=600)
    ap.add_argument("--py", type=float, default=459)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    meta, u = load(args.dump)
    floor = float(meta["floor"])
    recs, tid, rank, tiles_x = gather(meta, u)
    f32 = recs.view(np.float32)
    A, B, C = (f32[:, i].astype(np.float64) for i in range(3))
    mask = recs[:, 3]
    mx, my = f32[:, 4].astype(np.float64), f32[:, 5].astype(np.float64)
    op = (recs[:, 6] & 0xFFFF).astype(np.float64) / 65535.0
    ca, cb, cc = -2 * A, -B, -2 * C
    n = len(recs)
    print(f"pairs={n} floor={floor:.9g} tiles_x={tiles_x}")

    m = np.arange(32)
    # Task #44: the device box is the pixel-centre box of each 8x4 microblock,
    # [x0+0.5, x0+7.5] x [y0+0.5, y0+3.5] (BOX_W x BOX_H), not the continuous one.
    ox = ((m & 3) * 8 + 0.5).astype(np.float64)
    oy = ((m >> 2) * 4 + 0.5).astype(np.float64)
    dev = ((mask[:, None] >> m[None, :].astype(np.uint32)) & 1).astype(bool)

    thr = 2 * np.log(np.maximum(op, 1e-300) / floor) + THR_MARGIN
    res = {}
    CH = 200000
    fc_list, fk_list = [], []
    peak_lost = np.zeros(n)
    for s in range(0, n, CH):
        e = min(n, s + CH)
        sl = slice(s, e)
        ulo = ox[None, :] - mx[sl, None]
        vlo = oy[None, :] - my[sl, None]
        m2 = boxmin_m2(ca[sl, None], cb[sl, None], cc[sl, None], ulo, ulo + BOX_W, vlo, vlo + BOX_H)
        # fp32 emulation of the device math
        f = np.float32
        m2f = boxmin_m2(ca[sl, None].astype(f), cb[sl, None].astype(f), cc[sl, None].astype(f),
                        ulo.astype(f), (ulo + BOX_W).astype(f), vlo.astype(f), (vlo + BOX_H).astype(f),
                        dtype=f)
        fk_list.append(m2f.astype(np.float64) <= thr[sl, None])
        fc_list.append(m2 <= thr[sl, None])
    def m2_at(i, mb):
        ulo = ox[mb] - mx[i]
        vlo = oy[mb] - my[i]
        return float(boxmin_m2(ca[i], cb[i], cc[i], ulo, ulo + BOX_W, vlo, vlo + BOX_H))

    keep_exact = np.concatenate(fc_list)
    keep_f32 = np.concatenate(fk_list)

    # peak pixel-centre contribution per (pair, microblock)
    cxs = np.arange(8) + 0.5
    cys = np.arange(4) + 0.5
    false_cull = keep_exact & ~dev
    false_keep = dev & ~keep_exact
    fi, fm = np.nonzero(false_cull)
    peak = np.zeros(len(fi))
    for k in range(0, len(fi), CH):
        ii, mm = fi[k:k + CH], fm[k:k + CH]
        du = (ox[mm, None, None] + cxs[None, None, :]) - mx[ii, None, None]
        dv = (oy[mm, None, None] + cys[None, :, None]) - my[ii, None, None]
        q = ca[ii, None, None] * du * du + 2 * cb[ii, None, None] * du * dv + cc[ii, None, None] * dv * dv
        peak[k:k + CH] = (op[ii] * np.exp(-0.5 * q.min(axis=(1, 2))))
    res.update(
        pairs=int(n), pair_mb=int(n * 32),
        dev_keep=int(dev.sum()), exact_keep=int(keep_exact.sum()),
        f32_emul_keep=int(keep_f32.sum()),
        false_cull=int(false_cull.sum()), false_keep=int(false_keep.sum()),
        f32_emul_vs_exact_diff=int((keep_f32 != keep_exact).sum()),
        dev_vs_f32_emul_diff=int((keep_f32 != dev).sum()),
        false_cull_peak_alpha_max=float(peak.max()) if len(peak) else 0.0,
        false_cull_peak_alpha_ge_1_over_64=int((peak >= 1 / 64).sum()),
        false_cull_peak_alpha_ge_floor=int((peak >= floor).sum()),
    )
    # focus pixel: which pairs cover it and what the device did
    tx, ty = int(args.px) // 32, int(args.py) // 32
    t = ty * tiles_x + tx
    lx, ly = args.px - tx * 32, args.py - ty * 32
    mb = int(ly // 4) * 4 + int(lx // 8)
    sel = np.nonzero(tid == t)[0]
    du = lx + 0.5 - mx[sel]
    dv = ly + 0.5 - my[sel]
    a_px = op[sel] * np.exp(-0.5 * (ca[sel] * du * du + 2 * cb[sel] * du * dv + cc[sel] * dv * dv))
    order = np.argsort(-a_px)[:10]
    focus = []
    for j in order:
        i = sel[j]
        focus.append(dict(rank=int(rank[i]), alpha_at_px=float(a_px[j]), op=float(op[i]),
                          ci=[float(ca[i]), float(cb[i]), float(cc[i])],
                          mean_local=[float(mx[i]), float(my[i])],
                          dev_bit=bool(dev[i, mb]), exact_keep=bool(keep_exact[i, mb]),
                          f32_keep=bool(keep_f32[i, mb]),
                          m2_exact=m2_at(i, mb), thr=float(thr[i])))
    res["focus"] = dict(px=args.px, py=args.py, tile=t, mb=mb, top_pairs=focus)
    # worst false culls
    worst = np.argsort(-peak)[:10]
    res["worst_false_culls"] = [
        dict(tile=int(tid[fi[k]]), mb=int(fm[k]), rank=int(rank[fi[k]]), peak_alpha=float(peak[k]),
             op=float(op[fi[k]]), ci=[float(ca[fi[k]]), float(cb[fi[k]]), float(cc[fi[k]])],
             mean_local=[float(mx[fi[k]]), float(my[fi[k]])],
             m2_exact=m2_at(fi[k], fm[k]), thr=float(thr[fi[k]]))
        for k in worst]
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
