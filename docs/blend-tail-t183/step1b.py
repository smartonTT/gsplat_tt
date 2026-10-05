#!/usr/bin/env python3
"""t183 step 1b: for each view, the latest core: when it made its last claim (start of its
last rd_l1_bulk zone), what was still queued on it then, and the global last-claim time."""
import pickle, sys, statistics as S
views = pickle.load(open(sys.argv[1], 'rb'))
agg = []
print('view  max_end  mean_end last_claim(all) | late core: last_claim  end-last_claim  fin(N-2)  cost(N-1..N)  n_zones')
for i, d in enumerate(views):
    ends = {c: r['tr'][1] for c, r in d.items()}
    mean = S.mean(ends.values())
    lastclaim = max(r['bulk'][-1][0] for r in d.values() if r['bulk'])
    c = max(ends, key=ends.get); b = d[c]['bulk']
    finN2 = b[-1][1]  # e_N ~ compute finished subchunk N-2 (+ read of N)
    agg.append((ends[c] - mean, ends[c] - b[-1][0], ends[c] - finN2, ends[c] - lastclaim))
    print('%2d %8.1f %8.1f %8.1f | %8.1f %8.1f %8.1f %8.1f %3d' % (i, ends[c], mean, lastclaim, b[-1][0],
          ends[c] - b[-1][0], finN2, ends[c] - finN2, len(b)))
for j, n in enumerate(['tail max-mean', 'late core end - its last claim', 'late core last-2 subchunk cost',
                       'late core end - global last claim']):
    v = [a[j] for a in agg]; print('%-34s mean %.0f  median %.0f  min %.0f  max %.0f us' % (n, S.mean(v), S.median(v), min(v), max(v)))
