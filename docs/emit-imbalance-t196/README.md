# Task #196: sort_ol emit mover imbalance

Tip b92a28f (#181 + #188). The t194 Tracy split showed the emit makespan at
2.862 ms/view, with a mover median of 2.53 ms and a max of 3.62 ms.

## Step 1: which movers are slow, and why (no device)

Data: the t194 30-view tip trace (`yyzo-bh-07:/localdev/smarton/gstt2-t194/opt/profiler/t194-tip/dev30.csv`).
`movers.py` reads the `sort_ol_emit` zones (13200 rows: 220 movers x 30 launches; launch 0 skipped).
It joins them with the #174 page ranges, recomputed from the logged P with the host's translated-x lookup.

```
python3 docs/emit-imbalance-t196/movers.py tmp/t196/sortol.csv.gz tmp/t196/capture.log [--per-view] [--per-mover]
```

- **The 3.62 ms max is one view.** View 10 has P = 3.18M pairs. The mean per-view numbers are:
  - makespan 2861.8 us
  - median mover 2468.8 us
  - max mover 2809.2 us
  - start skew about 53 us

  The imbalance (makespan minus median) is about 0.39 ms/view.
- **It is a fixed per-mover rate, not content.** The mean rate is 4.85 pairs/us (about 278 cycles/pair).
  Rate relative to the mean:
  - BRISC (NOC0), row y=2: 0.67-0.88
  - BRISC, row y=3: 0.85-0.99
  - BRISC, rows 4-11: about 1.00-1.04
  - NCRISC (NOC1), physical columns x=14,15: 0.90-0.98
  - other NCRISC: about 0.98-1.09

  The same movers end last in most views: (14,3,N), (14,5,N), (14,6,N), (3,2,B), (4,2,B), (5,2,B), (15,6,N), (12,2,B).
- **The extra time is the blendrec read issue.** The t166/t174 EMIT_PROF split (`docs/precull-fastemit-t166/out/t166-p2f/emit_parts.txt`) shows:
  - `ep_brec` (scan the batch, issue one 64 B blendrec read per distinct gaussian) costs 0.547 ms of a 2.499 ms mean mover.
  - On the busiest mover it costs 1.368 ms.
  - The slow BRISC movers (y=2,3) spend 0.95-1.30 ms there, and the slow NCRISC movers (x=14,15) 0.72-0.87 ms.
  - Each mover issues about 7.7k such reads per view (12.0k records, 94 batches of 128 pairs).

  This is the same request-bound pattern as the #181 window fill: 64 B page reads ran at 27 GB/s, against 140 GB/s for per-bank runs, and BRISC/NOC0 was hurt most.
  The hot spots are the most-upstream requesters on each NoC (NOC0 rows y=2,3; NOC1 columns x=14,15).
  Their read requests wait longest to be injected.
- **Rebalancing the page split cannot fix much.**
  - The ceiling is the makespan-to-median gap, about 0.39 ms.
  - The v2 per-mover speed table (#177) realized only -0.2 ms and is shelved (#187).
  - The cause is per-request NoC cost, so the fix is fewer requests.

## Step 2: fix (bulk blendrec reads)

Under the K2 fold, pairs are gaussian-major, so g is nondecreasing, and keep is all ones.
A batch's gaussians are a near-dense run [ga, gb] of storage indices.
pfwc segments are dense; gaps come only from zero-pair gaussians and segment ends.
An interleaved buffer puts page p + nb right after page p in the same DRAM bank.

When gb - ga + 1 fits a ring half, `issue_brec` reads the run as at most nb NoC reads, one per bank, instead of one 64 B read per distinct g.
Page ga + j lands at slot (j % nb) * sf + j / nb.
The fast loop tracks (j % nb, j / nb) from the last g, so it needs no division and no table.
If the run does not fit, the batch falls back to per-g reads.
A g outside the run (pairs not g-sorted) gets one blocking read, so the output never depends on the ordering assumption.

- Knob: `GSPLAT_TT_OL_BREC_BULK=0` is the kill switch (default on).
- Ring half: 256 pages when bulk is on, so +16 KB of L1 per mover.
- EMIT_PROF counters `ep_nbk` (bulk batches) and `ep_bpg` (bulk pages read).

Result: see below.
