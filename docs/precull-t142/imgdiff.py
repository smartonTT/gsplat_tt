# t142: per-view pixel diff of two PNG dump dirs (PSNR, differing pixels, max |d|, 16x16 tiles touched).
#   python3 imgdiff.py <ref_dir> <test_dir>
import sys, numpy as np
from PIL import Image
A, B = sys.argv[1], sys.argv[2]
import os
for f in sorted(os.listdir(B)):
    a = np.asarray(Image.open(f"{A}/{f}")).astype(np.int32); b = np.asarray(Image.open(f"{B}/{f}")).astype(np.int32)
    d = np.abs(a-b)[..., :3]; px = d.max(-1)
    mse = (d**2).mean(); psnr = 10*np.log10(255**2/mse) if mse else float('inf')
    ys, xs = np.nonzero(px)
    tiles = set(zip(ys//16, xs//16))
    hist = np.bincount(px[px>0].ravel(), minlength=4)[:8]
    print(f"{f}: psnr={psnr:.2f} diff_px={len(ys)} max={d.max()} tiles={len(tiles)} hist(1..)={hist[1:].tolist()}")
    if tiles and len(tiles) < 15: print("   tiles(ty,tx):", sorted(tiles))
