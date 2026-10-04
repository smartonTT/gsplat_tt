# t165: sort_ol_emit lever 2, pack records on the idle TRISCs

Board: yyzo-bh-07 p100a. Base: iter-184 tip (220ba33 + t167 docs, 18.59 ms/view).
Kill switch / opt-in: `GSPLAT_TT_OL_EMIT_TPACK=1` (off by default).

## Design
The movers (BRISC, NCRISC) keep the plane scan, slot assignment, blendrec read
issue and run writes. Per segment (<= 128 records) they publish one word per
record (ring index, blendrec page slot) in a per-mover mailbox CB (CB 14,
2176 B). Each TRISC packs a third of the segment's 32 B records into the ring
(L1 only, no LLK), then writes a done word. The mover scans the next segment
while the TRISCs pack, and writes the previous segment's completed runs once
all 3 done words match. Protocol: `render/kernels/dataflow/sort_ol_tpack.h`.

## Upper bound (from t160's emit_parts, no TPACK)
pack/key build 1.815 ms of the 2.61 ms emit zone per mover. TPACK can only
remove the part of that which is not the scan itself: the mover still reads
the pair planes, does the cursor read-modify-write and writes an item word
per record. A gain of >= 0.3 ms/view needs the per-record pack (8 stores,
2 fp32 subtracts, page loads) to be well over half of the 263 cycles/record.

## Results
Smoke (30 views, TPACK=1): md5-identical to md5-r82new.txt, sort 4.361 ms
vs ~4.40 at the tip (t160), i.e. no clear gain.

Tracy with `GSPLAT_TT_OL_EMIT_PROF=1` and TPACK=1: the render hung right
after the one-launch sort program was built (capture killed by the 450 s
timeout, `out/capture.log`); no device CSV, so no emit-part table. The
untraced runs do not hang. Not debugged (see Decision).

Swapped paired A/B, untraced, 4 rounds (base/tp, tp/base, base/tp, tp/base)
in one p100 lock, code 9ececc0, `ab.sh` -> `out/run-r*-*.log`,
`out/ab-table.txt`. All 8 runs md5-identical to md5-r82new.txt (30 views).

| ms/view          | base mean | TPACK mean | paired delta (r1..r4)        | mean   |
|------------------|-----------|------------|------------------------------|--------|
| frame            | 18.592    | 18.648     | +0.137 -0.128 +0.016 +0.200  | +0.056 |
| sort             | 4.388     | 4.442      | +0.038 -0.033 +0.091 +0.119  | +0.054 |
| sort_bin_emit    | 4.001     | 4.024      | +0.017 +0.014 +0.014 +0.046  | +0.023 |
| blend            | 9.285     | 9.291      | +0.041 -0.025 -0.020 +0.027  | +0.006 |

TPACK is slightly slower: emit is worse in all 4 rounds (+0.02 ms), and the
frame delta is noise around +0.06 ms.

## Why it does not pay
The movers keep the plane scan, the cursor read-modify-write and the blendrec
read issue, and now also write one mailbox word per record and wait on the
3 TRISC done words per segment. What moved to the TRISCs (the 32 B record
stores and 2 fp32 subtracts) is a small slice of the 263 cycles/record; the
mailbox write plus the per-segment handshake costs about the same. The emit
zone is bound by the scan and the per-record cursor/slot work on the
movers, not by the pack stores.

## Decision
Not landed (gate 0.3 ms/view; measured +0.056). Code stays on branch
`ttp/t165-sort-ol-emit-lever-2-pack-records-on-idl`, opt-in
`GSPLAT_TT_OL_EMIT_TPACK=1`. The Tracy hang under TPACK=1 + EMIT_PROF=1 is
not worth debugging for a lever that shows no untraced gain. iters.jsonl row
186 (reject) on this branch.
