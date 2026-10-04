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
