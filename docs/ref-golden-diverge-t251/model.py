"""Task #251: CPU emulation of the hero blend, to attribute device-vs-reference error.

Inputs: tests/fixtures/hero/project_outputs.npz + blend_inputs.npz (projected hero
gaussians; projection is shared by both paths). Each variant toggles one device
approximation. Output: 8-bit PSNR of every variant vs benchmarks/reference_v2/hero.png.
Run: python3 docs/ref-golden-diverge-t251/model.py [outdir]
"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image

torch.set_num_threads(14)
FX = 'tests/fixtures/hero/'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/t251'
H = W = 1024; TS = 32; NT = 32
F16K, F255 = 1.0 / 16384, 1.0 / 255

# name: (tile_floor, pixel_floor, T rule, T eps, unorm16)
V = {
    'ref':          (F16K, 0.0,  'ref', None, 0),
    'ref+floor255': (F255, F255, 'ref', None, 0),
    'ref+maskonly': (F255, 0.0,  'ref', None, 0),
    'ref+Tdev':     (F16K, 0.0,  'dev', 1/256, 0),
    'ref+u16':      (F16K, 0.0,  'ref', None, 1),
    'dev_pre156':   (F16K, 0.0,  'dev', 1/256, 1),
    'dev':          (F255, F255, 'dev', 1/256, 1),
    'dev-Tdev':     (F255, F255, 'ref', None, 1),
    'dev-u16':      (F255, F255, 'dev', 1/256, 0),
    'dev_floor1024':(1/1024, 1/1024, 'dev', 1/256, 1),
    'dev_floor4096':(1/4096, 1/4096, 'dev', 1/256, 1),
    'dev_Teps1024': (F255, F255, 'dev', 1/1024, 1),
    'dev_Teps1e-4': (F255, F255, 'dev', 1e-4, 1),
    'pre156_Teps1e-4': (F16K, 0.0, 'dev', 1e-4, 1),
}

# argv[2]: hero-view inputs from dump_inputs.py. The tests/fixtures/hero npz files are a
# different view (their blend_output.npy is 12.3 dB vs reference_v2), so do not use them.
b = np.load(sys.argv[2])
m2 = b['means_2d'].astype(np.float32); cov = b['covs_2d'].astype(np.float32)
op = b['opacities'].astype(np.float32).reshape(-1); col = b['colors'].astype(np.float32)
dep = b['depths'].astype(np.float32); rad = b['radii'].astype(np.float32)
if rad.ndim == 1: rad = np.stack([rad, rad], 1)
ONLY = [x for x in os.environ.get('T251_VARIANTS', '').split(',') if x]
if ONLY: V = {k: V[k] for k in ONLY}
TILES = [int(x) for x in os.environ.get('T251_TILES', '').split(',') if x]
a, bb, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
det = np.maximum(a * c - bb * bb, np.float32(1e-6))
ia, ib, ic = c / det, -bb / det, a / det
q16 = lambda x: (np.rint(np.clip(x, 0, 1) * 65535) / 65535).astype(np.float32)
op16, col16 = q16(op), q16(col)

# bbox tile candidates (superset of every floor's tile list), sorted by depth then id
x0 = np.clip(np.floor((m2[:, 0] - rad[:, 0]) / TS), 0, NT - 1).astype(int)
x1 = np.clip(np.floor((m2[:, 0] + rad[:, 0]) / TS), 0, NT - 1).astype(int)
y0 = np.clip(np.floor((m2[:, 1] - rad[:, 1]) / TS), 0, NT - 1).astype(int)
y1 = np.clip(np.floor((m2[:, 1] + rad[:, 1]) / TS), 0, NT - 1).astype(int)
gids, tids = [], []
for g in range(len(m2)) if False else []:
    pass
nw, nh = x1 - x0 + 1, y1 - y0 + 1
cnt = nw * nh
gid = np.repeat(np.arange(len(m2)), cnt)
k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
tid = (np.repeat(y0, cnt) + k // np.repeat(nw, cnt)) * NT + np.repeat(x0, cnt) + k % np.repeat(nw, cnt)
order = np.lexsort((gid, dep[gid], tid))
gid, tid = gid[order], tid[order]
starts = np.searchsorted(tid, np.arange(NT * NT + 1))
print('pairs', len(gid), 'max/tile', np.diff(starts).max(), flush=True)

yy, xx = np.meshgrid(np.arange(TS), np.arange(TS), indexing='ij')
mb_of_pix = torch.from_numpy(((yy // 4) * 4 + xx // 8).reshape(-1))
img = {n: np.zeros((H, W, 3), np.float32) for n in V}
nrec = {n: 0 for n in V}
T0 = time.time()
for t in range(NT * NT):
    s, e = starts[t], starts[t + 1]
    if s == e or (TILES and t not in TILES):
        continue
    g = gid[s:e]; ty, tx = divmod(t, NT)
    px = torch.from_numpy((tx * TS + xx + 0.5).reshape(-1).astype(np.float32))
    py = torch.from_numpy((ty * TS + yy + 0.5).reshape(-1).astype(np.float32))
    dx = px[None] - torch.from_numpy(m2[g, 0])[:, None]
    dy = py[None] - torch.from_numpy(m2[g, 1])[:, None]
    pw = -0.5 * (torch.from_numpy(ia[g])[:, None] * dx * dx + 2.0 * torch.from_numpy(ib[g])[:, None] * dx * dy
                 + torch.from_numpy(ic[g])[:, None] * dy * dy)
    gw = torch.exp(torch.clamp(pw, max=0.0)); del dx, dy, pw
    for n, (ft, pf, trule, teps, u16) in V.items():
        o = torch.from_numpy(op16[g] if u16 else op[g])
        cc = torch.from_numpy(col16[g] if u16 else col[g])
        raw = o[:, None] * gw
        keep = raw.max(1).values >= ft            # tile-assign cull (pixel-centre max)
        raw, cc = raw[keep], cc[keep]
        nrec[n] += int(keep.sum())
        al = torch.clamp(raw, max=0.99)
        if pf > 0:
            al = torch.where(al < pf, torch.zeros_like(al), al)
        Ta = torch.cumprod(1.0 - al, 0)                  # T after each record
        Tb = torch.cat([torch.ones_like(Ta[:1]), Ta[:-1]], 0)
        w = al * Tb
        if trule == 'ref':
            tmax = Ta.max(1).values
            hit = torch.nonzero(tmax < 1e-4)
            if len(hit):
                w[int(hit[0]) + 1:] = 0
        else:
            G = w.shape[0]
            alive_until = torch.full((32,), G, dtype=torch.long)
            for kk in range(512, G, 512):
                tb = Tb[kk].to(torch.bfloat16).float()
                mbmax = torch.zeros(32).scatter_reduce(0, mb_of_pix, tb, 'amax')
                dead = (mbmax < teps) & (alive_until == G)
                alive_until[dead] = kk
                if bool((alive_until < G).all()):
                    break
            rows = torch.arange(G)[:, None]
            w = w * (rows < alive_until[mb_of_pix][None]).float()
        rgb = (w.T @ cc).numpy().reshape(TS, TS, 3)
        img[n][ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS] = rgb
    if t % 128 == 0:
        print('tile', t, f'{time.time() - T0:.0f}s', flush=True)

import os
os.makedirs(OUT, exist_ok=True)
ref = np.asarray(Image.open('benchmarks/reference_v2/hero.png').convert('RGB')).astype(np.float64)
gold = np.asarray(Image.open('tests/fixtures/hero/hero_golden_8bit.png').convert('RGB')).astype(np.float64)
gold_pre = np.asarray(Image.open('tests/fixtures/hero/archive/hero_golden_8bit_floor16384_pre-iter156.png').convert('RGB')).astype(np.float64)
def psnr(x, y):
    mse = ((x - y) ** 2).mean(); return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
def u8_f32(x): return (np.clip(x, 0, 1) * 255).astype(np.uint8).astype(np.float64)
def u8_bf16(x):
    xb = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    return (np.clip(xb, 0, 1) * 255).astype(np.uint8).astype(np.float64)
res = {}
if TILES:
    msk = np.zeros((H, W), bool)
    for t in TILES:
        ty, tx = divmod(t, NT); msk[ty*TS:(ty+1)*TS, tx*TS:(tx+1)*TS] = True
    ref, gold, gold_pre = ref[msk], gold[msk], gold_pre[msk]
    img = {n: img[n][msk] for n in img}
for n in V:
    for oq, f in (('f32', u8_f32), ('bf16', u8_bf16)):
        im = f(img[n])
        res[f'{n}|{oq}'] = dict(vs_ref=round(psnr(im, ref), 2), vs_golden=round(psnr(im, gold), 2),
                                vs_golden_pre156=round(psnr(im, gold_pre), 2),
                                mean_diff_vs_ref=round(float((im - ref).mean()), 3), records=nrec[n])
        print(n, oq, res[f'{n}|{oq}'], flush=True)
np.savez_compressed(f'{OUT}/model_images.npz', **img)
json.dump(res, open(f'{OUT}/model_psnr.json', 'w'), indent=1)
