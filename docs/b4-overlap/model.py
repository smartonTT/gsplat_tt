"""B4 overlap sizing (task #591): disjoint-core pipeline (pfwc on group A, sort+mat+blend on B).

Inputs are existing numbers, not new measurements:
  base b2b 8.454 ms/view (best-iter-227, bh-33 p100, 110 cores);
  boundary costs from the t224 profile (docs/structural-plan-2/PLAN.md):
  end spread pfwc 0.105, sort 0.092, mat+blend 0.146, gaps 0.040;
  stage shares from the same table (pfwc 1.62 of 8.93 profiled ms incl. boundaries).
Assumes perfect linear core scaling and no sub-device sync cost (both optimistic).
"""
CORES = 110
BASE = 8.454
SPREAD = {"pfwc": 0.105, "sort": 0.092, "blend": 0.146}
GAPS = 0.040
bound = sum(SPREAD.values()) + GAPS
work = (BASE - bound) * CORES                      # core-ms per view, all stages
pfwc_share = (1.62 - SPREAD["pfwc"]) / (8.93 - bound)
wa, wb = work * pfwc_share, work * (1 - pfwc_share)
sb = SPREAD["sort"] + SPREAD["blend"] + GAPS * 2 / 3  # B keeps two boundaries

def period(a, jitter=0.0):
    """Mean period over views whose B work varies +-jitter (two-point model)."""
    pa = wa / a + SPREAD["pfwc"]
    pbs = [(wb * (1 + s * jitter)) / (CORES - a) + sb for s in (-1, 1)]
    return sum(max(pa, pb) for pb in pbs) / len(pbs)

print(f"boundary ceiling {bound:.3f} ms; work {work:.0f} core-ms; pfwc share {pfwc_share:.3f}")
for jit in (0.0, 0.02, 0.05):
    best = min(range(10, 40), key=lambda a: period(a, jit))
    p = period(best, jit)
    print(f"B jitter +-{jit:.0%}: best a={best} cores, period {p:.3f}, gain {BASE - p:+.3f} ms/view")
