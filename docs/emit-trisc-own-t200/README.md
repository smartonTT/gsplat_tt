# t200: emit pack on the idle TRISCs after t196 (model, no device)

Question: after t196 (bulk blendrec reads, 7a23beb) the sort_ol emit is ~90% per-record pack
loop. Does that change #165's rejection of TRISC packing? Verdict needed: BUILD only if the
credible net is >= 0.3 ms/view untraced.

**Verdict: BUILD, but a different design from #165, with an early kill gate.**
Nominal saving 0.63 ms/view (range 0.10-0.75). It falls below 0.3 only if a TRISC needs
>= 1.25x the mover's cycles for the same per-record work. Gate step 1 measures exactly that.

## What t196 changed and what it did not

Data: `docs/emit-imbalance-t196/out/tracy-{on,off}-emit_parts.txt` (10 views, traced, mean mover).

| | t160 (#165's base) | t196 off | t196 on (tip) |
|---|---|---|---|
| emit zone, ms | 2.610 | 2.522 | 2.162 |
| issue_brec scan + read issue, ms | 0.507 | 0.477 | **0.037** |
| pack loop (proc - wfl - wiss), ms | 1.815 | 1.807 | 1.901 |
| records / mover / view | 13385 | 11888 | 11888 |
| pack cycles / record | 183 | 205 | 216 |

- t196 removed the brec scan, which sat outside the pack loop. The pack loop's own work per
  record is unchanged: 3 plane loads, cursor read-modify-write in L1, g-change check, 2
  integer subtracts, 8 word stores, run-write checks. (+17 cyc/rec came from #187's cursor
  move to L1, +11 from t196's bulk page stepping.)
- So #165's lesson still holds for #165's design. There the movers kept the plane loads and
  the cursor read-modify-write, and added a mailbox word per record. Only the 8 stores and 2
  subtracts moved to the TRISCs. Those are a small share of the 216 cycles; the mailbox and
  handshake cost the same. Re-running #165 on the t196 tip would again give ~0.
- What t196 does change: the scan is gone and pack is 88% of the zone. If the **whole**
  per-record loop (cursor included) moves off the movers, there is nothing left on the
  movers' critical path but NoC issue.

## Design that can pay: tile-owned cursors on the TRISCs

The expensive per-record step is the cursor read-modify-write. It cannot be split across RISCs
by record ranges without a prefix pass, and that pass costs another read-modify-write per
record. A per-sub-batch cursor prefix (each TRISC histograms its sub-batch, then adds the
earlier sub-batches' counts) doubles the cursor work. Splitting by **tile** avoids it:

- **Ownership:** TRISC i owns tiles with `t % 3 == i`, for both movers' streams. The owner is
  the only RISC that reads or writes those tiles' cursors. It sees each tile's records in
  stream order, so every record gets the same cursor `c`, the same slot and the same 32 B as
  now. Output is byte-identical (md5 46a725ab) by construction; over-capacity drops use the
  same `c >= cap` test.
- **Movers (BRISC, NCRISC):** per batch, after the read barrier, build 3 owner lists (8-bit
  record index j into the batch's planes; `t % 3` via a 1 KB byte table) and publish a
  "batch k ready" word. They keep everything that touches the NoC: pair and bulk brec reads,
  run writes, the final drain over the cursors (read from the shared L1 cursor array), the
  write barrier.
- **TRISCs (UNPACK, MATH, PACK; idle during sort_ol):** for each stream and batch, walk their
  list. Per record: load g, t, step the bulk brec pointer (same `(j % nb, j / nb)` stepping
  as t196), cursor read-modify-write, `sub_int32` x/y, 8 stores into the ring. On
  `ri == R-1`, push `(t, c)` to a single-producer queue for that stream's mover. At the end
  of the batch, write a done word.
- **Ring reuse:** the mover issues each queued run's write, calls `noc_async_writes_flushed`,
  then publishes a flushed-sequence count per queue. At `ri == 0` the owner checks that the
  tile's last run sequence is flushed (one compare per 8 records). Same-tile records are
  ~100+ records apart even on hot tiles (11.9k records over ~1024 tiles per mover), and
  t196's `ep_wfl` is only 0.018 ms, so this should rarely wait.
- **Buffering:** a mover's pair pages fit in its 1024-page window (93 batches x 8 pages =
  744), so only the brec ring limits TRISC skew. Use 3-4 brec slots instead of 2 halves
  (+16-32 KB per mover; current CB use is 2 x 521 KB + 22 KB of ~1.4 MB). That lets the
  3 TRISCs drift 2-3 batches apart, so per-batch Poisson imbalance averages out.
- **Reuse from #165** (branch `ttp/t165-sort-ol-emit-lever-2-pack-records-on-idl`, 9ececc0):
  the TRISC kernel hook, the mailbox CB plumbing, and the fix where PACK gets the L1
  addresses from UNPACK. Note: #165's TPACK hung under Tracy + EMIT_PROF and was never
  debugged; the new protocol must be checked traced before any A/B.
- Knob: `GSPLAT_TT_OL_EMIT_TOWN=1` opt-in until it passes the gate.

## Model

`model.py` -> `model.txt`. Per core: 2 x 11888 records split over 3 TRISCs, each at
`alpha * 216 + 10` cycles (10 = list load). Also 300 cycles per batch handshake, 3-8% owner
imbalance and 0.01-0.08 ms flush stalls. The mover side (list build ~45 cyc/rec + NoC issue)
is 0.60 ms, so movers are never the bottleneck. Drain, prologue and barrier are unchanged.

| case | alpha | TRISC ms | emit zone ms | saving ms/view |
|---|---|---|---|---|
| optimistic (owned cursors in TRISC local mem, 16-bit) | 0.92 | 1.31 | 1.41 | 0.75 |
| nominal | 1.00 | 1.44 | 1.54 | **0.63** |
| TRISC 10% slower | 1.10 | 1.61 | 1.70 | 0.46 |
| TRISC 20% slower | 1.20 | 1.78 | 1.87 | 0.29 |
| TRISC 30% slower | 1.30 | 1.96 | 2.06 | 0.10 |

Break-even for 0.3 ms/view: alpha = 1.25. These are traced mean-mover numbers. Untraced
emit savings tracked traced ones in t196 (-0.41 ms traced emit makespan, -0.52 ms untraced
frame). Pooling both movers' records per core also removes the BRISC/NCRISC split, so the
makespan should drop at least as much as the mean.

Why alpha ~1 is credible: the TRISCs are the same RV32 cores at the same clock, and they do
the same L1 loads/stores. Their L1 port is idle during sort_ol, because unpack/pack are not
running. #160 vs #187 shows cursors in local memory are ~17 cyc/rec cheaper than in L1.
With 1/3 of the tiles each, 16-bit per-stream offsets fit in ~1.4 KB of TRISC local memory,
which gives the optimistic row. What is **not** measured: TRISC scalar cycles per record.
No prior run produced them (#165's Tracy capture hung).

Upside left out: the movers have ~0.8 ms of slack. Giving them a share of the tiles
(5 lanes instead of 3) could take the nominal case toward ~0.9 ms. That is a v2 step.

## Build plan with a kill gate

1. **Gate (one device run):** implement ownership + lists + queue, with `GSPLAT_TT_OL_EMIT_PROF`
   counters on TRISC (owned records, cycles in the pack loop, wait cycles on ready/flushed).
   Run 2-view smoke for md5, then Tracy views 0:10. Kill if TRISC cycles/owned record >
   1.25 x 216 (i.e. > 270) or the emit zone does not drop by >= 0.3 ms traced.
2. If the gate passes: 3-round swapped untraced A/B on the bicycle 30 views (same drivers as
   t196), md5 46a725ab for all runs. KEEP at <= -0.3 ms/view.
3. Optional: owned cursors in TRISC local memory, then movers as 4th/5th owners.

Cost/risk: medium-high code (3-RISC + 2-mover protocol, new TRISC kernel path in the sort_ol
program, Tracy hang history). It is the largest lever left in the emit. The pack loop is
1.90 ms of the 2.49 ms sort_ol program, and no single-RISC diet has found more than ~0.6 ms.
