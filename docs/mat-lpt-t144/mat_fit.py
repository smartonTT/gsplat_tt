#!/usr/bin/env python3
"""Task #144: fit the one-launch materialize item cost per kind and per mover
from a GSPLAT_TT_MATCULL_PROF=1 device profile plus the host worklist dump
(GSPLAT_TT_MAT_DUMP), then replay the LPT per frame with the legacy and the
fitted costs and report the predicted busiest mover.
  mat_fit.py profile_log_device.csv[.gz] dump.txt[.gz] [--items items.csv]
Items are split like docs/lever-a-t121/mat_items.py (an item starts at its
first mat_meta zone). Core (x, y) -> logical core by rank of the distinct x / y
(harvested columns are skipped), slot 2c = NCRISC, 2c+1 = BRISC."""
import csv, gzip, sys
from collections import defaultdict
import numpy as np

FREQ = 1350.0
OV_CAP = None  # read from dump? whole vs big decided by sc/cnt below
BUCKET_FIT = 4096
M0_CAP = 6144


def opn(p):
    return gzip.open(p, 'rt') if p.endswith('.gz') else open(p)


def load_prof(path):
    rows = defaultdict(list)
    with opn(path) as f:
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
            rows[(int(r[1]), int(r[2]), risc)].append(
                (int(r[ix['time[cycles since reset]']]), name, r[ix['type']].strip() == 'ZONE_START'))
    # per mover: list of frames, each {'items': [us...], 'span': us, 'zones': [...]}
    out = {}
    for key, ev in rows.items():
        ev.sort()
        frames, cur, item, open_t = [], None, None, {}
        def close(t):
            nonlocal item
            if item is not None and cur is not None:
                item['us'] = (t - item['t0']) / FREQ
                cur['items'].append(item)
            item = None
        for t, name, st in ev:
            if st:
                open_t[name] = t
                if name == 'sort_subchunk_mat':
                    cur = {'items': [], 't0': t}
                    frames.append(cur)
                if name == 'mat_meta' and (item is None or item.get('work')):
                    close(t)
                    item = {'t0': t}
                continue
            t0 = open_t.pop(name, None)
            if t0 is None:
                continue
            if name == 'sort_subchunk_mat':
                close(t)
                cur['span'] = (t - cur['t0']) / FREQ
                continue
            if item is not None and name != 'mat_meta':
                item['work'] = True
                item[name] = item.get(name, 0.0) + (t - t0) / FREQ
        out[key] = frames
    return out


def load_dump(path):
    frames = []
    with opn(path) as f:
        for ln in f:
            p = ln.split()
            if p[0] == 'F':
                frames.append({})
            elif p[0] == 'S':
                its = []
                for tok in p[3:]:
                    t, w, c = (int(x) for x in tok.split(':'))
                    its.append((t, w, c))
                frames[-1][int(p[1])] = its
    return frames


def kind_of(w, cnt, ov_cap):
    return 'w' if cnt <= ov_cap else 'b'


def main():
    prof = load_prof(sys.argv[1])
    dump = load_dump(sys.argv[2])
    xs = sorted({k[0] for k in prof}); ys = sorted({k[1] for k in prof})
    gx = len(xs)
    slot_of = {k: 2 * (ys.index(k[1]) * gx + xs.index(k[0])) + (0 if k[2] == 'NCRISC' else 1)
               for k in prof}
    nf = min(len(dump), min(len(v) for v in prof.values()))
    # whole vs big: a tile with several items (one per subchunk) is big; cap
    # from the largest whole-tile count is not needed.
    recs, bad = [], 0
    spans = defaultdict(dict)
    for key, frames in prof.items():
        s = slot_of[key]
        for fi in range(nf):
            its = frames[fi]['items']; d = dump[fi].get(s, [])
            spans[fi][s] = frames[fi].get('span', 0.0)
            if len(its) != len(d):
                bad += 1
                continue
            for it, (t, w, c) in zip(its, d):
                ntile = sum(1 for (t2, _, _) in d if t2 == t)
                recs.append(dict(f=fi, slot=s, risc=s & 1, tile=t, w=w, cnt=c, us=it['us'],
                                 sort=it.get('mat_ol_sort', 0.0), perm=it.get('mat_ol_perm', 0.0),
                                 cull=it.get('mat_cull', 0.0), rd=it.get('mat_ol_rd', 0.0),
                                 keys=it.get('mat_ol_keys', 0.0), gather=it.get('mat_ol_gather', 0.0)))
    # big items: tile appears with w != 0 somewhere, or has mat_ol_keys
    bigt = {(r['f'], r['tile']) for r in recs if r['w'] != 0 or r['keys'] > 0}
    for r in recs:
        r['k'] = 1 if (r['f'], r['tile']) in bigt else 0
    print(f"frames={nf} movers={len(prof)} grid={gx}x{len(ys)} items={len(recs)} "
          f"mover-frames with item-count mismatch={bad}")
    if len(sys.argv) > 4 and sys.argv[3] == '--items':
        with open(sys.argv[4], 'w') as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    fit = {}
    for k in (0, 1):
        for rr in (0, 1):
            sel = [r for r in recs if r['k'] == k and r['risc'] == rr]
            if len(sel) < 5:
                continue
            x = np.array([r['cnt'] for r in sel], float); y = np.array([r['us'] for r in sel])
            A = np.vstack([np.ones_like(x), x]).T
            (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
            res = y - (a + b * x)
            fit[(k, rr)] = (a, b)
            print(f"kind={'whole' if k == 0 else 'big'} risc={'NCRISC' if rr == 0 else 'BRISC'} n={len(sel)} "
                  f"a={a:.2f}us b={b * 1000:.2f}ns/rec rms={np.sqrt((res ** 2).mean()):.1f}us "
                  f"mean_us={y.mean():.1f} cnt_range={int(x.min())}-{int(x.max())}")
            # residual by count bin
            edges = [0, 256, 512, 1024, 2048, 4096, 6144, 8192, 1 << 20]
            row = []
            for lo, hi in zip(edges, edges[1:]):
                m = (x >= lo) & (x < hi)
                if m.sum():
                    row.append(f"[{lo},{hi}) n={m.sum()} res={res[m].mean():+.1f}")
            print("    " + " | ".join(row))
    # per-mover overhead outside items
    ov = []
    for key, frames in prof.items():
        for fi in range(nf):
            fr = frames[fi]
            if 'span' in fr and fr['items']:
                ov.append(fr['span'] - sum(i['us'] for i in fr['items']))
    print(f"span - sum(items) per mover-frame: mean {np.mean(ov):.1f} max {np.max(ov):.1f} us")
    # measured busiest vs mean per frame, and per-RISC
    print("frame  meas_max  meas_mean  NC_mean  BR_mean")
    for fi in range(nf):
        v = spans[fi]
        nc = [u for s, u in v.items() if s % 2 == 0]; br = [u for s, u in v.items() if s % 2 == 1]
        print(f"{fi:5d} {max(v.values()):9.1f} {np.mean(list(v.values())):10.1f} {np.mean(nc):8.1f} {np.mean(br):8.1f}")
    return recs, fit, spans


if __name__ == '__main__':
    main()
