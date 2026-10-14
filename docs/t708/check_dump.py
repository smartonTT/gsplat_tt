#!/usr/bin/env python3
"""t708: sanity check of the GSPLAT_TT_BLEND_TRAIN fp32 train buffer (blend_train.h).

  python3 docs/t708/check_dump.py <train-v0.bin> <view.png>

The buffer is one 16 KB page per screen tile (page ty * tiles_x + tx), 4 fp32 32x32
tiles R, G, B, T, element r*32 + c holding pixel (i, j) of the screen tile with the
microblock permutation of img_pack_u8.h. Checks: page count = screen tiles, finite values,
T in [-1, 1], per-pixel |u8 - trunc(clip(fp32) * 255)| against the device u8 PNG
(the PNG goes through bf16, so up to 1 LSB is expected), and the T < 0 count
(pixels whose blend stopped on the 1e-4 rule).
"""
import json
import sys

import numpy as np
from PIL import Image

TILE = 32
PAGE_F32 = 4 * TILE * TILE


def perm():
    k = np.arange(TILE * TILE)
    r, c = k >> 5, k & 31
    i = (r >> 2) * 4 + (r & 1) * 2 + (c >> 4)
    j = ((r >> 1) & 1) * 16 + (c & 1) * 8 + ((c >> 1) & 7)
    return i, j


def unpack(buf, H, W):
    tx, ty = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    pages = buf.size // PAGE_F32
    assert buf.size % PAGE_F32 == 0, buf.size
    i, j = perm()
    assert len(set(zip(i.tolist(), j.tolist()))) == TILE * TILE
    out = np.zeros((ty * TILE, tx * TILE, 4), np.float32)
    p = buf.reshape(pages, 4, TILE * TILE)
    for t in range(min(pages, tx * ty)):
        y, x = t // tx, t % tx
        for ch in range(4):
            out[y * TILE + i, x * TILE + j, ch] = p[t, ch]
    return out[:H, :W], pages, tx * ty


def main():
    bin_path, png_path = sys.argv[1], sys.argv[2]
    img = np.asarray(Image.open(png_path).convert("RGB")).astype(np.int32)
    H, W = img.shape[:2]
    buf = np.fromfile(bin_path, np.float32)
    f, pages, want = unpack(buf, H, W)
    rgb, T = f[..., :3], f[..., 3]
    u8 = np.floor(np.clip(rgb, 0, 1) * 255.0).astype(np.int32)
    d = np.abs(u8 - img)
    res = {
        "png": png_path, "H": H, "W": W, "bytes": int(buf.size * 4), "pages": int(pages),
        "screen_tiles": int(want), "finite": bool(np.isfinite(f).all()),
        "rgb_min": float(rgb.min()), "rgb_max": float(rgb.max()),
        "T_min": float(T.min()), "T_max": float(T.max()),
        "absT_in_0_1": bool(((np.abs(T) >= 0) & (np.abs(T) <= 1)).all()),
        "max_lsb_vs_png": int(d.max()), "px_over_1lsb": int((d.max(-1) > 1).sum()),
        "px_eq_png": float((d.max(-1) == 0).mean()),
        "T_neg_px": int((T < 0).sum()), "T_neg_frac": float((T < 0).mean()),
        "T_zero_px": int((T == 0).sum()),
        "mean_rgb": [float(x) for x in rgb.reshape(-1, 3).mean(0)],
    }
    print(json.dumps(res, indent=1))
    ok = res["finite"] and res["absT_in_0_1"] and res["max_lsb_vs_png"] <= 1 and pages == want
    print("DUMP_CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
