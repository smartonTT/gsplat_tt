#!/usr/bin/env python3
"""t183 step 1: per-view spread of blend core end times (TRISC tile_blend_sfpu end) and
the rd_l1_bulk subchunk timeline of the late cores."""
import pickle, sys, statistics as S
views = pickle.load(open(sys.argv[1], 'rb'))
rows = []
for i, d in enumerate(views):
    ends = sorted(r['tr'][1] for r in d.values())
    bend = max(r['br'][1] for r in d.values())
    starts = [r['tr'][0] for r in d.values()]
    mean = S.mean(ends); med = S.median(ends)
    nb = sum(len(r['bulk']) for r in d.values())
    rows.append((i, max(starts), ends[0], med, mean, ends[-1], bend, ends[-1] - mean, ends[-1] - med,
                 ends[-1] - ends[-11], nb))
print('view tr_start_max min_end med_end mean_end max_end brisc_end tail(max-mean) max-med max-p90 n_bulk')
for x in rows: print('%2d ' % x[0] + ' '.join('%8.1f' % v for v in x[1:-1]) + ' %5d' % x[-1])
for j, name in [(7, 'max-mean'), (8, 'max-med'), (9, 'max-p90'), (5, 'max_end'), (4, 'mean_end')]:
    v = [x[j] for x in rows]
    print('%-9s mean %.1f median %.1f min %.1f max %.1f us' % (name, S.mean(v), S.median(v), min(v), max(v)))
