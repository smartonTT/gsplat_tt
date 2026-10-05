# Mat mover op count and TRISC offload model (task #304)

No device runs. This is a model built from the code and from existing profiler data.

## Question

In the fused mat+blend program (iter 206), the movers' time sets the mat phase 1:1. BRISC is busy 2.38 ms and NCRISC 2.48 ms. The mat phase averages 2.513 ms and peaks at 2.989 ms per core. Meanwhile the cull TRISCs sit idle for 2.415 ms per core (`docs/profile-postl1-t297.md`).

How much of the mover work is local L1 compute that the idle TRISCs could do instead?

## Sources

- Code: `render/kernels/dataflow/sort_subchunk_materialize.cpp`, `sort_radix_tile_algo.h` and `render/kernels/compute/mat_cull_compute.cpp`. Defaults: SORT_ONELAUNCH=1, OL_MAT_SELECT=0, FUSE_CULL=1 (depth 2), MATCULL_FOLD=0 (`permute_records`, then `cull_slab`), MATBLEND_FUSE=1.
- Per-item zones: `docs/lever-a-t121/out/t121-nosel-mprof-items.txt`, MATCULL_PROF, 30 views. `mat_cull` is nested inside `mat_ol_perm`, so the pure permute time is perm − cull. With that correction the whole-item zones sum to 549.0 of the 549.3 µs mean, so the attribution closes.
- Totals: the t297 Tracy capture puts `sort_subchunk_mat` at 537 ms/view over 220 movers, about 2.44 ms per mover. That matches t121's 644 ms/view of zone time within the expected drift, so the zone shares carry over. Tracing costs little here because the zones are per item, not per record.
- Clock 1.35 GHz. About 12.35k records per mover, about 266 cycles per record.

## Op count per record (default whole-tile path)

| Class | Step | Ops per record (from code) | Share of mover time | ms per mover (of 2.43) | Cycles per record |
|---|---|---|---|---|---|
| NoC/DRAM | `read_bucket` (2 KB reads) | 32 B NoC read, 1 issue per 64 records | 2.7 % | 0.065 | 7 |
| NoC/DRAM | slab emit, ready flag, 3 meta reads per item | 32 B NoC write; about 6 barriers per item | 0.7 % | 0.017 | 2 |
| NoC/DRAM | big-tile gather (NCRISC only) | 32 B NoC read per id | 1.6 % | 0.040 | 4 |
| L1 compute | `sort_record_ids` (radix, typically 3 passes × 8-9 bits) | about 10 L1 loads, 8 L1 stores, 6 histogram read-modify-writes in local memory, about 3 compares | 52.1 % | 1.27 | 139 |
| L1 compute | big-tile key build | 1 load, 2 stores | 1.6 % | 0.040 | 4 |
| L1 compute | `permute_records` | 7 loads (1 index plus 6 words of the 8) and 7 stores | 12.8 % | 0.31 | 34 |
| L1 compute | `cull_slab`: `fill_coeff_tile` and `patch_batch` | fill: 6 loads, 6 transposed stores; patch: 2 loads, 1 store; one CB round trip per 128 records | 28.1 % | 0.68 | 75 |
| **Total** | | | 99.6 % | 2.43 | 266 |

95 % of the mover time is L1-local compute and only 5 % is NoC/DRAM. The movers are not moving data. They are running scalar loops over records that already sit in L1.

The cull row is mover compute, not time spent waiting on the TRISCs. The TRISC unpacker spends only about 0.5 µs of each 12.9 µs gap between batches, and `mc_uw` shows 2.415 ms of TRISC waiting per core (t297). So the TRISCs wait on the mover, not the other way round.

## Candidates

1. **Fill and patch on the TRISCs (recommended).** The mover hands over one descriptor per slab (address, count) instead of 1-2 coefficient tiles per 128 records. TRISC0 runs `fill_coeff_tile` straight from the slab into its own coefficient tile, unpack and SFPU cull run as today, and TRISC2 packs and then runs `patch_batch` straight into the slab's word3. The mover only waits for a "slab done" flag before it emits the slab.
2. **Radix sort on the TRISCs.** This is the bigger share (52 %), but it is harder:
   - the sort of an item has to finish before that item's permute, so it only overlaps with other items;
   - the 3 KB histograms need TRISC local memory, which is unverified on Blackhole, or L1, which measured about 7 % slower in #26;
   - the sort competes with the per-batch cull on the same TRISCs.

   Revisit this after candidate 1.
3. **Permute on the TRISCs.** It is small (13 %) and needs the bucket and the slab, which the mover reuses for the next item. It only makes sense folded into candidate 1, as TRISC0 doing permute + fill in one pass (the `MATCULL_FOLD` loop). Leave it out of the first build.

## Model for candidate 1

The new per-mover loop:

```
meta_i, rd_i, sort_i                        <- overlaps the TRISC fill/cull/patch of item i-1
wait done_{i-1}, emit_{i-1}
permute_i (bucket -> slab), post job_i
end: wait done_last, emit_last
```

- **Removed from each mover:** 28.1 % of the mover time, 0.68 ms on average.
- **Added to each mover:**
  - about 1 µs of handoff per item, about 5 items per mover;
  - the last item's TRISC time is not hidden: 0.83 × 0.68 / 5 ≈ 0.11 ms. The 0.83 ratio is t202's measured TRISC/mover cycles per record for the same kind of scalar L1 loop.
  - Total added: about 0.12 ms.
- **Hidden in the overlap:** in the steady state the next item's rd + sort (about 55 % of the mover time) is about 2× longer than the TRISC's share of the cull work, so the mover rarely waits.
- **TRISC load per core:** 2 movers × 0.68 × 0.83 ≈ 1.13 ms of the 2.415 ms idle (47 %). Fill runs on TRISC0 and patch on TRISC2, so they pipeline.

| Case | Mat phase now | After | Saving |
|---|---|---|---|
| Mean mover | 2.43 | 2.43 − 0.68 + 0.12 = 1.87 | 0.56 ms |
| Busiest core (mat max) | 2.989 | 2.989 × 0.719 + 0.12 = 2.27 | 0.72 ms |
| Pessimistic (× 0.54, L2's measured/model ratio) | | | 0.30-0.39 ms |

The frame tracks the mat max because blend starts with gap 0 and is balanced within 0.08 ms (t297). **Modeled saving: about 0.55 ms/view (range 0.30-0.72), against the 0.3 ms gate.**

## Comparison with earlier offloads

- **#165 (+0.056 ms, worse).** It moved only the stores and paid a mailbox round trip per element. Candidate 1 moves the whole fill/patch loop and pays one handoff per slab.
- **#274/#291 (L2, K2 on TRISCs, modeled 0.45, measured −0.245).** Each TRISC job waited 0.12-0.29 ms for its DRAM window to land. Here the data is already in L1 when the job is posted, so there is no landing wait.
- **#215 (shelved, ≤ 0.23 ms).** Cursor-level emit and town fixes were below the gate on their own.
- **t200/t202 (−0.96 ms measured vs 0.63 modeled).** It moved the whole per-record loop and gave the TRISCs ownership of the data. Candidate 1 has the same structure. That is the main reason to trust the model over the pessimistic case.

## Interaction with lever A

Lever A (a blend lane during mat) wants the same idle TRISC time and extra NCRISC loads. Candidate 1:

- takes about 1.1 ms of the 2.4 ms TRISC idle on TRISC0 and TRISC2;
- frees about 0.6 ms of each mover;
- shortens the mat window that A fills.

The two levers partly substitute for each other. Candidate 1 is simpler: it reuses the existing cull handshake and needs no partial-blend or ready-flag rework. So build candidate 1 first, then re-check A's gate (TRISC1 idle ≥ 0.6 ms) on the new profile. The separate A model should take this into account.

## Risks

- **Cache coherence.** On Blackhole, the TRISCs must invalidate their data cache and fence before they read the slab the mover wrote, and the mover must see the TRISC's word3 stores before the NoC reads the slab. The existing fences cover the mover side; add an invalidate on the TRISCs.
- **Two movers per core.** Both share one TRISC pipeline. `pick_stream` already arbitrates the two coefficient streams; it becomes arbitration over job descriptors.
- **TRISC0 code size.** The fill loop is small.
- **Multi-subchunk items (> 8192 records).** These still serialize within the item (permute s+1 needs the slab). Few items are that big, but one of them can set the busiest core.

## Recommendation: build

Gate: the modeled ≥ 0.3 ms/view. The model is about 0.55 ms, and even the pessimistic case reaches 0.30.

### Implementation sketch

1. **Mover, `sort_subchunk_materialize.cpp` under a new `MATCULL_TRISC_FILL` define** (env `GSPLAT_TT_MATCULL_TRISC_FILL`; off first for A/B, the default flips if it wins):
   - replace the `cull_slab` call with `post_job(slab, n)`: write {slab, n} into a 1-page job CB (reuse CB_COEFF's slot) and `cb_push_back`;
   - move the emit of item i to just before the permute of item i+1, after `wait_done` (`cb_wait_front` on CB_KEEP, one page per job);
   - at the end, wait for the last job, emit it, then `cull_end_stream`.
2. **TRISC0 (`mat_cull_compute.cpp` UNPACK):**
   - `pick_stream` returns a job;
   - for each 128-record batch, run `fill_coeff_tile(slab, base, nb, own_tile)` into a TRISC-owned coefficient CB (`cb_reserve_back`/`push` within the TRISC), then mailbox the batch count to MATH/PACK.
3. **TRISC2 (PACK):**
   - pack the keep tile as today into a TRISC-private keep CB;
   - run `patch_batch` straight into the slab (word3) from the PACK thread;
   - after the job's last batch, fence and push one "done" page to the mover's CB_KEEP.
4. **Unit and device checks:**
   - md5 906e0435 30/30 (the output must be bit-identical, since the same masks reach the same word3);
   - mat-phase zones under MATCULL_PROF;
   - the `mc_uw` TRISC idle;
   - the standard 30-view A/B under `ttp lock p100`.
