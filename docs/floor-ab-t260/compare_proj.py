"""t260: compare device pfwc outputs (GSPLAT_TT_DUMP_PROJ -> extract_dump.py -> proj_dev.npz)
with the cpu_cpp project of the hero view, and write model.py inputs from both.
Run (worktree root): PYTHONDONTWRITEBYTECODE=1 ~/dev/gstt2/.venv/bin/python \
    docs/floor-ab-t260/compare_proj.py <gstt2 root> <proj_dev.npz> <outdir>
Device rows are mapped to source ids by segment (core c owns source tiles c, c+C, ...;
its rows are [SeqMap.first(c)*1024, +m_c)) and the exact fp32 (opacity, r, g, b) key."""
import json, math, sys, os
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from backends import get_backend
from gsplat.loading_gaussians import load_ply
from gsplat.utils import c2w_to_w2c
OUT = sys.argv[3]; os.makedirs(OUT, exist_ok=True)

cam = json.load(open('benchmarks/cameras_v2.json'))['bicycle']
W = H = 1024
f = 0.5 * W / math.tan(0.5 * math.radians(cam['fov_deg']))
K = torch.tensor([[f, 0, W * .5], [0, f, H * .5], [0, 0, 1]], dtype=torch.float32)
E = c2w_to_w2c(torch.tensor(np.asarray(cam['views']['hero']['c2w'], np.float32)))
g = load_ply('/Users/smarton/dev/gsplat_tt/scenes/bicycle.ply')
be = get_backend('cpu_cpp')
m2, cov, dep, rad, valid = be.project(g.means, g.scales, g.rotations, E, K, H, W,
                                      opacities=g.opacities, sub_timings={})
N = len(g.opacities)
v = valid.bool().numpy()
full = len(m2) == N
def fullN(x, shape):
    x = np.asarray(x, np.float32)
    if full: return x
    y = np.zeros((N,) + shape, np.float32); y[v] = x; return y
m2, cov, dep = fullN(m2, (2,)), fullN(cov, (2, 2)), fullN(dep, ())
rad = np.asarray(rad, np.float32); rad = fullN(rad, rad.shape[1:])
op = g.opacities.numpy().astype(np.float32).reshape(-1); col = g.colors.numpy().astype(np.float32)
print('N', N, 'cpu visible', int(v.sum()), flush=True)

d = np.load(sys.argv[2])
sid, F, aabb, cnt, M = d['sid'], d['f'], d['aabb'], d['counts'], int(d['M'])
ntile = (N + 1023) // 1024
C = int((cnt[:, 0] > 0).sum())
base, rem = ntile // C, ntile % C
first = lambda c: c * base + (c if c < rem else rem)
count = lambda c: base + (1 if c < rem else 0)
src = np.full(len(sid), -1, np.int64); keep = np.zeros(len(sid), bool); amb = np.zeros(len(sid), bool)
key = lambda o, cc: np.stack([o.view(np.uint32), cc[:, 0].view(np.uint32), cc[:, 1].view(np.uint32),
                              cc[:, 2].view(np.uint32)], 1)
nambig = nmiss = 0
for c in range(C):
    lo = first(c) * 1024; hi = lo + int(cnt[c, 0])
    r = np.nonzero((sid >= lo) & (sid < hi))[0]
    keep[r] = True
    ids = np.concatenate([np.arange(t * 1024, min(N, t * 1024 + 1024)) for t in range(c, ntile, C)])
    kk = {}
    for i, k in zip(ids, map(bytes, key(op[ids], col[ids]))):
        kk.setdefault(k, []).append(i)
    for j, k in zip(r, map(bytes, key(F[r, 5].copy(), F[r, 6:9].copy()))):
        cand = kk.get(k)
        if not cand: nmiss += 1; continue
        if len(cand) > 1: nambig += 1; amb[j] = True
        src[j] = cand.pop(0)   # stream order is source order, so take the first unused
res = dict(M=M, rows_in_segments=int(keep.sum()), heuristic_rows=len(sid), unmatched=nmiss, ambiguous=nambig, pfwc_cores=C)
sel = keep & (src >= 0)
if os.environ.get('T260_UNAMBIG'): sel &= ~amb
s, Fd = src[sel], F[sel]
dv = np.zeros(N, bool); dv[s] = True
res['dev_visible'] = int(dv.sum()); res['cpu_visible'] = int(v.sum())
res['dev_only'] = int((dv & ~v).sum()); res['cpu_only'] = int((v & ~dv).sum())
both = v[s]; s2, F2 = s[both], Fd[both]

# a,b,c: conic or cov2d?
ca, cb, cc_ = cov[s2, 0, 0], cov[s2, 0, 1], cov[s2, 1, 1]
det = ca * cc_ - cb * cb
ia, ib, ic = cc_ / det, -cb / det, ca / det
def err(dev, ref, name):
    ae = np.abs(dev.astype(np.float64) - ref); re = ae / np.maximum(np.abs(ref), 1e-12)
    res[name] = dict(max_abs=float(ae.max()), mean_abs=float(ae.mean()), max_rel=float(re.max()),
                     mean_rel=float(re.mean()), p99_rel=float(np.quantile(re, .99)),
                     p99_abs=float(np.quantile(ae, .99)), p999_abs=float(np.quantile(ae, .999)), n_abs_gt_0p5=int((ae > .5).sum()) if ae.ndim == 1 else None)
r_cov = np.median(np.abs(F2[:, 0] - ca) / np.abs(ca)); r_con = np.median(np.abs(F2[:, 0] - ia) / np.abs(ia))
for k in ('a', 'b', 'c'):  # half-b convention check
    pass
is_conic = r_con < r_cov
res['abc_is'] = 'conic' if is_conic else 'cov2d'
if is_conic:  # median ratios tell the stored scale: the device stores (-a/2, -b, -c/2)
    res['abc_ratio_vs_cpu'] = [float(np.nanmedian(F2[:, 0] / ia)), float(np.nanmedian(F2[:, 1] / np.where(ib == 0, np.nan, ib))),
                               float(np.nanmedian(F2[:, 2] / ic))]
    F2 = F2.copy(); F2[:, 0] *= -2.0; F2[:, 1] *= -1.0; F2[:, 2] *= -2.0
err(F2[:, 3], m2[s2, 0], 'mean_x'); err(F2[:, 4], m2[s2, 1], 'mean_y')
if is_conic:
    err(F2[:, 0], ia, 'conic_a'); err(F2[:, 1], ib, 'conic_b'); err(F2[:, 2], ic, 'conic_c')
else:
    err(F2[:, 0], ca, 'cov_a'); err(F2[:, 1], cb, 'cov_b'); err(F2[:, 2], cc_, 'cov_c')
err(F2[:, 9], dep[s2], 'depth')
err(F2[:, 5], op[s2], 'opacity'); err(F2[:, 6:9], col[s2], 'color')

# depth order swaps within tiles: pairs from the CPU radius bbox (same as model.py), ranked by
# device vs CPU depth; count adjacent inversions after a stable sort by CPU (depth, id).
TS, NT = 32, 32
rr = rad[s2] if rad.ndim == 2 else np.stack([rad[s2]] * 2, 1)
mx, my = m2[s2, 0], m2[s2, 1]
x0 = np.clip(np.floor((mx - rr[:, 0]) / TS), 0, NT - 1).astype(int); x1 = np.clip(np.floor((mx + rr[:, 0]) / TS), 0, NT - 1).astype(int)
y0 = np.clip(np.floor((my - rr[:, 1]) / TS), 0, NT - 1).astype(int); y1 = np.clip(np.floor((my + rr[:, 1]) / TS), 0, NT - 1).astype(int)
nw, nh = x1 - x0 + 1, y1 - y0 + 1; cntp = nw * nh
gi = np.repeat(np.arange(len(s2)), cntp)
k = np.arange(cntp.sum()) - np.repeat(np.cumsum(cntp) - cntp, cntp)
tid = (np.repeat(y0, cntp) + k // np.repeat(nw, cntp)) * NT + np.repeat(x0, cntp) + k % np.repeat(nw, cntp)
dc, dd = dep[s2], F2[:, 9]
oc = np.lexsort((s2[gi], dc[gi], tid)); od = np.lexsort((s2[gi], dd[gi], tid))
res['pairs'] = int(len(gi)); res['pairs_order_differs'] = int((oc != od).sum())
# exact count of inverted pairs per tile is O(n^2); count adjacent swaps in CPU order instead
g_c = gi[oc]; t_c = tid[oc]
same = t_c[1:] == t_c[:-1]
inv = same & (dd[g_c[1:]] < dd[g_c[:-1]])
res['adjacent_depth_swaps'] = int(inv.sum()); res['tiles_with_swap'] = int(len(np.unique(t_c[1:][inv])))
# device aabb tile rect vs CPU bbox rect
ab = aabb[sel][both]
amin_x, amin_y, aw = ab & 0x3FF, (ab >> 10) & 0x3FF, ((ab >> 20) & 0x1FF) + 1
res['aabb_minx_eq'] = float((amin_x == x0).mean()); res['aabb_miny_eq'] = float((amin_y == y0).mean())
res['aabb_w_eq'] = float((aw == nw).mean())
rows_worst = np.argsort(-np.abs(F2[:, 3] - m2[s2, 0]))[:5]
res['worst_mean_x'] = [dict(src=int(s2[i]), dev=[float(x) for x in F2[i, [3, 4, 9]]], cpu=[float(m2[s2[i], 0]), float(m2[s2[i], 1]), float(dep[s2[i]])]) for i in rows_worst]
json.dump(res, open(f'{OUT}/proj_compare.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

# model.py inputs: device values (cov from conic if needed), CPU radii, same row set and order.
if is_conic:
    da, db, dc2 = F2[:, 0].astype(np.float64), F2[:, 1].astype(np.float64), F2[:, 2].astype(np.float64)
    dt = da * dc2 - db * db
    dcov = np.stack([np.stack([dc2 / dt, -db / dt], 1), np.stack([-db / dt, da / dt], 1)], 1)
else:
    dcov = np.stack([np.stack([F2[:, 0], F2[:, 1]], 1), np.stack([F2[:, 1], F2[:, 2]], 1)], 1)
o = np.argsort(s2)
np.savez(f'{OUT}/model_in_dev.npz', means_2d=F2[o, 3:5], covs_2d=dcov[o].astype(np.float32), depths=F2[o, 9],
         radii=rr[o], opacities=op[s2][o], colors=col[s2][o])
np.savez(f'{OUT}/model_in_cpu.npz', means_2d=m2[v], covs_2d=cov[v], depths=dep[v],
         radii=(rad[v] if rad.ndim == 2 else rad[v]), opacities=op[v], colors=col[v])
np.savez(f'{OUT}/model_in_cpu_devset.npz', means_2d=m2[s2][o], covs_2d=cov[s2][o], depths=dep[s2][o],
         radii=rr[o], opacities=op[s2][o], colors=col[s2][o])
