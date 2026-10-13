#!/usr/bin/env python3
"""t614: per-view host phases from an HTL file (median over views after SKIP=60).
  phases.py <htl-*.txt.gz> ...
A = view start -> gather result (host waits for gather(N+1) on device)
B = gather result -> pfwc(N+2) enqueue end (TA, sort host+device, spans, pfwc rtargs)
C = pfwc enqueue end -> mat+blend(N+1) enqueue end (fold hook rest, fold rtargs)
D = mat+blend enqueue end -> view end (host blocked until blend(N+1) output is back)
G = view end -> next view start (python between views)"""
import gzip, statistics, sys

SKIP = 60
for path in sys.argv[1:]:
    marks = sorted((int(a), b, c.strip()) for a, b, c in
                   (l.split(None, 2) for l in gzip.open(path, "rt")))
    views, cur = [], None
    for ns, k, t in marks:
        if t == "view_total" and k == "B":
            cur = {"vb": ns}; views.append(cur)
        elif cur is None:
            continue
        elif k == "E" and t in ("project_gather_result", "project_pfwc_enqueue", "fz_enqueue", "view_total"):
            cur[t] = ns
    rows = []
    for v, nxt in zip(views[SKIP:], views[SKIP + 1:]):
        try:
            rows.append(((v["project_gather_result"] - v["vb"]), (v["project_pfwc_enqueue"] - v["project_gather_result"]),
                         (v["fz_enqueue"] - v["project_pfwc_enqueue"]), (v["view_total"] - v["fz_enqueue"]),
                         (nxt["vb"] - v["view_total"])))
        except KeyError:
            pass
    m = [statistics.median(c) / 1e6 for c in zip(*rows)]
    print(f"{path.split('/')[-1]:24s} n={len(rows)} A={m[0]:.3f} B={m[1]:.3f} C={m[2]:.3f} D={m[3]:.3f} G={m[4]:.3f} "
          f"sum={sum(m):.3f}")
