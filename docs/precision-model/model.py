"""Task #580 (L5/L6): CPU PSNR + work model of blend precision variants. No device.

Extends the t251/t260 blend replay (docs/floor-ab-t260/t251/model.py) with:
- exp modes: 'exact' (torch.exp), '21f' (bit-level emulation of the device
  blend_exp21f: xlog2 = x/ln2 + 127, z = floor(xlog2 * 2^23), degree-2 poly on the
  23-bit fraction, setexp), '1' (degree-1 Moroz-style: one MAD on the fraction,
  minimax coefficients with the input shift folded into the 127 constant).
- T_EPS, T period (device periodic T readback, bf16, per 4x8 microblock), tile/pixel
  floor per variant.
- work counts per variant: scanned records (tile list), live records, microblock
  evaluations (record x microblock with mask bit set and microblock still live),
  T readbacks. Saving model in summarize.py (t205 cycle units).

Run: python3 model.py <outdir> <inputs.npz> (inputs from t260: model_in_{cpu,dev,cpu_devset}.npz)
Env: PM_VARIANTS=a,b (subset), PM_TILES=1,2 (tile subset), PM_THREADS=n.
"""
import json, os, sys, time
import numpy as np, torch
from PIL import Image

torch.set_num_threads(int(os.environ.get('PM_THREADS', '14')))
OUT, INP = sys.argv[1], sys.argv[2]
H = W = 1024; TS = 32; NT = 32
F255 = 1.0 / 255

# degree-1 coefficients: Q(u) = 0.97030 + 0.97000 u ~ 2^u on [0, 1), minimax rel err 2.99 %.
# setexp needs the poly in [1, 2): scale by k = 1.0307 (P(0) = 1.00009, P(1) = 1.99987)
# and shift the input by log2(k), i.e. the "+127" constant becomes 127 - log2(k).
# '1ls': least-squares fit (mean rel err ~0, max 3.9 %): Q(u) = 0.96060571 + 0.96060331 u, k = 1.0410105.
def _e1(q0, q1, k):
    return np.float32(q0 * k), np.float32(q1 * k / 2 ** 23), np.float32(127.0 - np.log2(k))
E1 = {'1': _e1(0.97030, 0.97000, 1.0307), '1ls': _e1(0.96060571, 0.96060331, 1.0410105)}

# name: (tile_floor, pixel_floor, T eps, T period (0 = no check), exp mode)
BASE = (F255, F255, 1 / 256, 512, '21f')
def var(**kw):
    t = dict(zip(('ft', 'pf', 'teps', 'per', 'exp'), BASE)); t.update(kw)
    return (t['ft'], t['pf'], t['teps'], t['per'], t['exp'])
V = {
    'base':        var(),
    'base_exact':  var(exp='exact'),
    'base_nochk':  var(per=0),
    'exp1':        var(exp='1'),
    'exp1ls':      var(exp='1ls'),
    'teps128':     var(teps=1 / 128),
    'per1024':     var(per=1024),
    'per2048':     var(per=2048),
    'floor2':      var(ft=2 / 255, pf=2 / 255),
    'teps128_per2048_floor2': var(teps=1 / 128, per=2048, ft=2 / 255, pf=2 / 255),
    'all':         var(exp='1', teps=1 / 128, per=2048, ft=2 / 255, pf=2 / 255),
}
ONLY = [x for x in os.environ.get('PM_VARIANTS', '').split(',') if x]
if ONLY: V = {k: V[k] for k in ONLY}
TILES = [int(x) for x in os.environ.get('PM_TILES', '').split(',') if x]
EXPS = sorted({v[4] for v in V.values()} | {'exact'})


def exp21f(pw):
    """Device blend_exp21f (fp32), input pw <= 0."""
    xl = pw * np.float32(1.4426950216293334961) + np.float32(127.0)
    return _setexp_poly(xl, lambda f: (f * np.float32(4.791750143340323e-15) + np.float32(7.839635491371155e-08)) * f
                        + np.float32(1.0017248))


def exp1(pw, mode='1'):
    c0, c1, bias = E1[mode]
    xl = pw * np.float32(1.4426950216293334961) + bias
    return _setexp_poly(xl, lambda f: f * c1 + c0)


def _setexp_poly(xl, poly):
    xl = torch.clamp(xl, min=0.0)
    z = torch.floor(xl.double() * 2 ** 23).to(torch.int32)        # _float_to_int32_for_exp_21f_
    e = z >> 23                                                    # exexp (no debias)
    f = (z & 0x7FFFFF).float()                                     # exman -> int32_to_float
    p = poly(f).float()
    bits = (p.view(torch.int32) & 0x807FFFFF) | (e << 23)          # setexp
    r = bits.view(torch.float32)
    return torch.where(e > 0, r, torch.zeros_like(r))


b = np.load(INP)
m2 = b['means_2d'].astype(np.float32); cov = b['covs_2d'].astype(np.float32)
op = b['opacities'].astype(np.float32).reshape(-1); col = b['colors'].astype(np.float32)
dep = b['depths'].astype(np.float32); rad = b['radii'].astype(np.float32)
if rad.ndim == 1: rad = np.stack([rad, rad], 1)
a, bb, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
det = np.maximum(a * c - bb * bb, np.float32(1e-6))
ia, ib, ic = c / det, -bb / det, a / det
q16 = lambda x: (np.rint(np.clip(x, 0, 1) * 65535) / 65535).astype(np.float32)
op16, col16 = q16(op), q16(col)                                    # BLEND_SFPU_UNORM

x0 = np.clip(np.floor((m2[:, 0] - rad[:, 0]) / TS), 0, NT - 1).astype(int)
x1 = np.clip(np.floor((m2[:, 0] + rad[:, 0]) / TS), 0, NT - 1).astype(int)
y0 = np.clip(np.floor((m2[:, 1] - rad[:, 1]) / TS), 0, NT - 1).astype(int)
y1 = np.clip(np.floor((m2[:, 1] + rad[:, 1]) / TS), 0, NT - 1).astype(int)
nw, nh = x1 - x0 + 1, y1 - y0 + 1
cnt = nw * nh
gid = np.repeat(np.arange(len(m2)), cnt)
k = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
tid = (np.repeat(y0, cnt) + k // np.repeat(nw, cnt)) * NT + np.repeat(x0, cnt) + k % np.repeat(nw, cnt)
order = np.lexsort((gid, dep[gid], tid))
gid, tid = gid[order], tid[order]
starts = np.searchsorted(tid, np.arange(NT * NT + 1))
print('rows', len(m2), 'pairs', len(gid), 'max/tile', np.diff(starts).max(), flush=True)

yy, xx = np.meshgrid(np.arange(TS), np.arange(TS), indexing='ij')
mb_of_pix = torch.from_numpy(((yy // 4) * 4 + xx // 8).reshape(-1))     # 4x8 microblocks
img = {n: np.zeros((H, W, 3), np.float32) for n in V}
CNT = ('scanned', 'live_recs', 'mb_evals', 'mb_masked', 'readbacks')
work = {n: dict.fromkeys(CNT, 0) for n in V}
o16 = torch.from_numpy(op16); c16 = torch.from_numpy(col16)
T0 = time.time()
for t in range(NT * NT):
    s, e = starts[t], starts[t + 1]
    if s == e or (TILES and t not in TILES):
        continue
    g = torch.from_numpy(gid[s:e]); ty, tx = divmod(t, NT)
    px = torch.from_numpy((tx * TS + xx + 0.5).reshape(-1).astype(np.float32))
    py = torch.from_numpy((ty * TS + yy + 0.5).reshape(-1).astype(np.float32))
    dx = px[None] - torch.from_numpy(m2[g, 0])[:, None]
    dy = py[None] - torch.from_numpy(m2[g, 1])[:, None]
    pw = -0.5 * (torch.from_numpy(ia[g])[:, None] * dx * dx + 2.0 * torch.from_numpy(ib[g])[:, None] * dx * dy
                 + torch.from_numpy(ic[g])[:, None] * dy * dy)
    del dx, dy
    pw = torch.clamp(pw, max=0.0)
    gw = {'exact': torch.exp(pw)}
    if '21f' in EXPS: gw['21f'] = exp21f(pw)
    for em in ('1', '1ls'):
        if em in EXPS: gw[em] = exp1(pw, em)
    del pw
    o, cc_all = o16[g], c16[g]
    raw_x = o[:, None] * gw['exact']                                    # cull math (exact)
    rmax = raw_x.max(1).values
    mbmax_x = torch.zeros(len(g), 32).scatter_reduce(1, mb_of_pix[None].expand(len(g), -1), raw_x, 'amax')
    del raw_x
    for n, (ft, pf, teps, per, em) in V.items():
        keep = rmax >= ft                                               # tile list (K2)
        kidx = torch.nonzero(keep).reshape(-1)
        G = len(kidx)
        wk = work[n]; wk['scanned'] += G
        if G == 0:
            continue
        mmask = mbmax_x[kidx] >= ft                                     # microblock cull mask
        wk['mb_masked'] += int(mmask.sum())
        al = torch.clamp(o[kidx, None] * gw[em][kidx], max=0.99)
        al = torch.where(al < pf, torch.zeros_like(al), al)            # pixel floor
        Ta = torch.cumprod(1.0 - al, 0)
        Tb = torch.cat([torch.ones_like(Ta[:1]), Ta[:-1]], 0)
        w = al * Tb
        del Ta, al
        alive_until = torch.full((32,), G, dtype=torch.long)
        if per:
            for kk in range(per, G, per):
                wk['readbacks'] += 1
                tb = Tb[kk].to(torch.bfloat16).float()
                mbm = torch.zeros(32).scatter_reduce(0, mb_of_pix, tb, 'amax')
                dead = (mbm < teps) & (alive_until == G)
                alive_until[dead] = kk
                if bool((alive_until < G).all()):
                    break
            w = w * (torch.arange(G)[:, None] < alive_until[mb_of_pix][None]).float()
        ev = mmask & (torch.arange(G)[:, None] < alive_until[None])
        wk['mb_evals'] += int(ev.sum()); wk['live_recs'] += int(ev.any(1).sum())
        del Tb
        rgb = (w.T @ cc_all[kidx]).numpy().reshape(TS, TS, 3)
        img[n][ty * TS:(ty + 1) * TS, tx * TS:(tx + 1) * TS] = rgb
    if t % 128 == 0:
        print('tile', t, f'{time.time() - T0:.0f}s', flush=True)

os.makedirs(OUT, exist_ok=True)
ref = np.asarray(Image.open('benchmarks/reference_v2/hero.png').convert('RGB')).astype(np.float64)
gold = np.asarray(Image.open('tests/fixtures/hero/hero_golden_8bit.png').convert('RGB')).astype(np.float64)
def psnr(x, y):
    mse = ((x - y) ** 2).mean(); return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8).astype(np.float64)
if TILES:
    msk = np.zeros((H, W), bool)
    for t in TILES:
        ty, tx = divmod(t, NT); msk[ty*TS:(ty+1)*TS, tx*TS:(tx+1)*TS] = True
    ref, gold = ref[msk], gold[msk]
    img = {n: img[n][msk] for n in img}
res = {}
for n in V:
    im = u8(img[n])
    res[n] = dict(vs_ref=round(psnr(im, ref), 3), vs_golden=round(psnr(im, gold), 3),
                  mean_diff_vs_ref=round(float((im - ref).mean()), 3), **work[n])
    if 'base' in img and n != 'base':
        res[n]['vs_base'] = round(psnr(im, u8(img['base'])), 3)
    print(n, res[n], flush=True)
np.savez_compressed(f'{OUT}/model_images.npz', **img)
json.dump(dict(inputs=INP, variants={n: list(map(str, v)) for n, v in V.items()}, results=res),
          open(f'{OUT}/model_psnr.json', 'w'), indent=1)
