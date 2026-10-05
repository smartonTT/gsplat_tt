#!/usr/bin/env python3
"""Per-item cost of the one-launch materialize from a GSPLAT_TT_MATCULL_PROF=1
device profile (t121). An item starts at its mat_meta zone; items with a
mat_ol_keys zone are big-tile select parts, the rest whole-tile items.
  mat_items.py profile_log_device.csv
Prints per item kind: count/view, mean and max item time, mean of each fine
zone, and per run the busiest mover with its share of select-part time."""
import csv, sys
from collections import defaultdict

FREQ = 1350.0  # MHz -> cycles per us
rows = defaultdict(list)  # (x, y, risc) -> [(t, name, start?, run)]
with open(sys.argv[1]) as f:
    next(f)
    rd = csv.reader(f)
    hdr = [h.strip() for h in next(rd)]
    ix = {h: i for i, h in enumerate(hdr)}
    for r in rd:
        risc = r[ix['RISC processor type']].strip()
        if risc not in ('BRISC', 'NCRISC'):
            continue
        name = r[ix['zone name']].strip()
        if not (name.startswith('mat_') or name == 'sort_subchunk_mat'):
            continue
        rows[(r[1], r[2], risc)].append(
            (int(r[ix['time[cycles since reset]']]), name,
             r[ix['type']].strip() == 'ZONE_START', r[ix['run host ID']].strip()))

kinds = {'sel': [], 'whole': []}
per_run_mover = defaultdict(lambda: [0.0, 0.0])  # (run, mover) -> [total us, select us]
for key, ev in rows.items():
    ev.sort()
    open_t = {}
    item = None  # dict of zone -> us, plus start/end
    frame = -1   # i-th sort_subchunk_mat instance on this mover = frame i
    def close(end_t):
        if item is None:
            return
        item['_us'] = (end_t - item['_t0']) / FREQ
        k = 'sel' if 'mat_ol_keys' in item else 'whole'
        kinds[k].append(item)
        pm = per_run_mover[(item['_run'], key)]
        pm[0] += item['_us']
        if k == 'sel':
            pm[1] += item['_us']
    for t, name, st, rn in ev:
        if st:
            open_t[name] = t
            if name == 'sort_subchunk_mat':
                frame += 1
            # an item's 3 mat_meta reads come first; the next item starts at
            # the first mat_meta after any other zone
            if name == 'mat_meta' and (item is None or item.get('_work')):
                close(t)
                item = {'_t0': t, '_run': frame}
            continue
        t0 = open_t.pop(name, None)
        if t0 is None:
            continue
        if name == 'sort_subchunk_mat':
            close(t)
            item = None
            continue
        if item is None:
            continue
        if name != 'mat_meta':
            item['_work'] = True
        item[name] = item.get(name, 0.0) + (t - t0) / FREQ

runs = sorted({rn for (rn, _) in per_run_mover})
nv = max(len(runs), 1)
for k, its in kinds.items():
    if not its:
        continue
    zs = sorted({z for it in its for z in it if not z.startswith('_')})
    us = [it['_us'] for it in its]
    print(f"{k:5s} items/run={len(its)/nv:7.1f} mean_us={sum(us)/len(us):8.1f} "
          f"max_us={max(us):8.1f} total_ms/run={sum(us)/nv/1000:7.2f}")
    for z in zs:
        v = [it.get(z, 0.0) for it in its]
        print(f"      {z:14s} mean_us={sum(v)/len(v):8.1f} max_us={max(v):8.1f}")
print("per frame: busiest mover (us), its select-part us, mean mover us")
for rn in runs:
    m = {mv: v for (r, mv), v in per_run_mover.items() if r == rn}
    if not m:
        continue
    mv, (tot, sel) = max(m.items(), key=lambda kv: kv[1][0])
    mean = sum(v[0] for v in m.values()) / len(m)
    by = {}
    for rk in ('NCRISC', 'BRISC'):
        v = [x[0] for k2, x in m.items() if k2[2] == rk]
        if v:
            by[rk] = f"{rk} mean {sum(v)/len(v):7.1f} max {max(v):7.1f}"
    print(f"  frame {rn}: busiest {tot:8.1f} (select {sel:7.1f}, {mv[2]} core {mv[0]},{mv[1]}) "
          f"mean {mean:8.1f} | " + " | ".join(by.values()))
