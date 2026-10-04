#!/usr/bin/env python3
"""t183: extract per-view, per-core blend timelines from a stitched dev30.csv.
Writes blend.pkl: list of views; each view = dict core -> {k0,k1 (BRISC/NCRISC/TRISC kernel
span), tr_end (max TRISC tile_blend_sfpu end), bulk [(s,e)] rd_l1_bulk zones} in us rel. to
the view's first blend NCRISC zone start."""
import csv, sys, pickle, collections
F = 1350.0  # MHz -> cycles per us
path = sys.argv[1]
f = open(path); next(f); r = csv.reader(f); next(r)
ev = collections.defaultdict(list)  # (core,risc) -> [(t,zone,type)]
for row in r:
    if len(row) < 12: continue
    ev[((int(row[1]), int(row[2])), row[3])].append((int(row[5]), row[10], row[11]))
# Pair zones per (core,risc), keeping order.
zones = collections.defaultdict(list)  # (core,risc) -> [(zone,s,e)]
for k, lst in ev.items():
    lst.sort()
    st = collections.defaultdict(list)
    for t, z, ty in lst:
        if ty == 'ZONE_START': st[z].append(t)
        elif st[z]: zones[k].append((z, st[z].pop(), t))
# Blend instances: NCRISC tile_blend_load zones; TRISC tile_blend_sfpu; BRISC kernel enclosing.
loads = []
for (c, risc), zl in zones.items():
    if risc == 'NCRISC':
        for z, s, e in zl:
            if z == 'tile_blend_load': loads.append((s, e, c))
loads.sort()
# split into views: a gap > 2 ms between successive load starts starts a new view
views = []; cur = []
for s, e, c in loads:
    if cur and s - cur[-1][0] > 2000 * F: views.append(cur); cur = []
    cur.append((s, e, c))
views.append(cur)
print('views', len(views), 'cores/view', sorted(set(len(v) for v in views)))
out = []
for v in views:
    t0 = min(s for s, e, c in v); t1 = max(e for s, e, c in v)
    lo, hi = t0 - 50 * F, t1 + 3000 * F
    d = {}
    for s, e, c in v:
        rec = {'ld': ((s - t0) / F, (e - t0) / F)}
        tr = [ze for (z, zs, ze) in sum((zones[(c, t)] for t in ('TRISC_0', 'TRISC_1', 'TRISC_2')), [])
              if z == 'tile_blend_sfpu' and lo <= zs <= hi]
        trs = [zs for (z, zs, ze) in sum((zones[(c, t)] for t in ('TRISC_0', 'TRISC_1', 'TRISC_2')), [])
               if z == 'tile_blend_sfpu' and lo <= zs <= hi]
        rec['tr'] = ((min(trs) - t0) / F, (max(tr) - t0) / F) if tr else None
        bk = [(z, zs, ze) for (z, zs, ze) in zones[(c, 'BRISC')] if z == 'BRISC-KERNEL' and lo <= zs <= hi]
        rec['br'] = ((bk[0][1] - t0) / F, (bk[0][2] - t0) / F) if bk else None
        rec['bulk'] = sorted(((zs - t0) / F, (ze - t0) / F) for (z, zs, ze) in zones[(c, 'NCRISC')]
                             if z == 'rd_l1_bulk' and s <= zs <= e)
        d[c] = rec
    out.append(d)
pickle.dump(out, open(sys.argv[2], 'wb'))
