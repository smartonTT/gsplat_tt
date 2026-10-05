#!/usr/bin/env python3
"""t185: model dropping mask-0 records in mat before blend (no device).

Inputs are measured numbers already on disk:
  - dead (mask-0) record counts: docs/blend-fixed-t172/out/tc_split.txt (PRECULL=2 tip,
    views 0/1) and docs/blend-waste-t148/out/summary.txt (30 views + hero, PRECULL=1)
  - blend MATH marginal cost per record: t172 fits (0.0115 / 0.0161 us); t148 assumed 44 cyc
  - mat: largest whole tile 14-16k records on one NCRISC sets the window; busiest-vs-mean
    gap 0.618 ms traced; cull+write 0.059 us/own record (docs/mat-split-sort-model/RESULT.md)
  - traced->untraced ratio for mat ~0.5 (t168)
"""
CORES, MOVERS = 110, 220
views = {"t172 v0": (2895701, 2345549), "t172 v1": (2090634, 1693003),
         "t148 30v": (2954079, 2465815)}
blend_us_per_dead = {"t172 fit (cyc-trb-stage)": 0.0115, "t172 fit (full)": 0.0161,
                     "t148 assumed 44 cyc": 44 / 1350}
comp_us_per_rec = (0.010, 0.020, 0.030)  # extra mover copy per record (7 ld + 7 st, ~81% kept)
big_tile = (14000, 16000)
MAT_UNTRACED = 0.5
SLACK_TRACED = 0.618
reader_count_read_ms = 0.015  # blend reader: one small count read per subchunk, if not hidden

print("blend MATH saving, ms/view (= per-core MATH wall cut):")
for vn, (rec, live) in views.items():
    dead = rec - live
    row = ", ".join(f"{k}: {dead / CORES * c / 1000:.3f}" for k, c in blend_us_per_dead.items())
    print(f"  {vn}: dead {dead} ({dead / rec:.1%})  {row}")

rec = 2.9e6
print("mat cost of compaction:")
for c in comp_us_per_rec:
    crit = [n * c / 1000 for n in big_tile]
    mean = rec * c / MOVERS / 1000
    print(f"  {c:.3f} us/rec: critical item +{crit[0]:.2f}..{crit[1]:.2f} ms traced "
          f"(+{crit[0] * MAT_UNTRACED:.2f}..{crit[1] * MAT_UNTRACED:.2f} untraced); "
          f"mean mover +{mean:.2f} ms vs slack {SLACK_TRACED} traced")

lo = (2090634 - 1693003) / CORES * 0.0115 / 1000 - reader_count_read_ms
hi = (2895701 - 2345549) / CORES * (44 / 1350) / 1000
print(f"net, compaction skipped on the critical item: {lo:+.3f} .. {hi:+.3f} ms/view (gate 0.3)")
c = 0.020
print(f"net, compaction on every item (c={c}): "
      f"{lo - big_tile[1] * c / 1000 * MAT_UNTRACED:+.3f} .. {hi - big_tile[0] * 0.010 / 1000 * MAT_UNTRACED:+.3f} ms/view")
