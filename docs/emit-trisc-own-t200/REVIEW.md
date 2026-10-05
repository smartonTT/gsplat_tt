# t214: independent review of t202 (tile-owned TRISC emit pack, GSPLAT_TT_OL_EMIT_TOWN)

Reviewed: 7ab606a..656a1fa on smarton/tt-project-opt (no render/ changes up to 9b1e7ed).
Files: `render/kernels/dataflow/sort_ol_town.h`, `render/kernels/compute/sort_ol_town_compute.cpp`,
the `OL_EMIT_TOWN` paths in `render/kernels/dataflow/sort_bin_onelaunch.cpp`,
`render/host/sort_device.cpp`, `render/host/env_config.h`, `tests/unit/test_sort_ol_town.cpp`.

**Verdict: PASS.** No blocking finding. Device smokes md5-identical (46a725ab), see below.

## Protocol checks (code read)

- **Slot reuse (4 slots, READY/DONE).** Batch k uses list slot and blendrec slot k % 4. The mover
  refills slot (k+1) % 4 only once `min(DONE) >= k + 2 - SLOTS`, i.e. every TRISC finished batch
  k + 1 - 4, the last user of that slot. Lists, desc words and blendrec reads of batch k all
  complete before `fence; READY = k + 1` (read barrier at the top of iteration k). The TRISC
  fences after it sees READY > k and before it reads the list or pages, so stale cache lines
  from batch k - 4 are dropped. Correct.
- **Pair buffers.** `issue_pairs(k + 2)` overwrites pair buffer (k + 2) % 3 = k % 3 only after
  batch k's lists were built; the TRISCs never read pair planes. Correct.
- **Per-tile flushed word (fl).** `fl[t]` starts at `startp[t]` (written before GO). The mover
  sets `fl[t] = last + 1` only after `noc_async_writes_flushed()` for the run, so the ring slots
  are free. A TRISC waits for `fl[t] == c` only at an aligned run start (c % R == 0); an
  unaligned first run uses slots no run has used yet. Because `tile_cap % REC_PAGE_RECS == 0`
  (ring_on) and R divides the page, the last in-cap run always ends on R - 1 and gets a run
  word; partial final runs are drained by the mover after every TRISC is done, as before.
- **Fence before every poll.** All polls fence or `invalidate_l1_cache()` first, except the first
  `fl[t] != c` read in `process()`. That one is benign: fl only grows and the value the TRISC
  waits for is the largest it can take before the TRISC pushes the next run word, so a stale
  read can only be smaller and leads into the fenced loop. Not a bug.
- **Queues.** TRISC: run records + word, fence, then QWP. Mover: invalidate, read QWP, fence,
  read words, flush runs, set fl, fence, then QRD. `qlim` starts at QCAP to match QRD = 0.
- **Deadlock.** Every mover wait (slot wait, tail wait for DONE) calls `service()`, so TRISC waits
  on fl or queue space always make progress; TRISC waits on READY only need batches the mover
  already published; the FIN wait comes after every DONE == nbatch and the final service. No
  cycle found. The model test covers 2..4 slots and a 4-word queue at random interleavings.
- **FIN/GO clear.** Each mover zeroes READY, DONE, QWP, QRD and FIN before `fence; GO2; fence; GO`;
  TRISCs read NB only after seeing both magic words and a fence; they set FIN after their last
  DONE and never read the stream again; the mover clears GO/GO2/FIN after all three FIN words.
  The kernel has no top-level early return between GO and the FIN wait, so GO is always cleared.
  A stale GO pair from another program's L1 data would need two 32-bit magic words at those
  exact offsets; negligible.
- **Cursors.** TRISCs own `cur_lm[t]` (CB_CUR) for their tiles during the emit; the mover does not
  touch it between GO and the drain, and invalidates its cache before the drain reads it.

## Fallbacks

- **`pb * 16 > LIST_MAX`:** host `town` is false: no define, no compute kernel, 2 blendrec halves,
  no mailbox CB: the same binaries as `GSPLAT_TT_OL_EMIT_TOWN=0` (checked by the t202 `rd-off`
  run, md5 46a725ab). The device `static_assert(BATCH_ELEMS <= LIST_MAX)` backs it.
- **Device-side fallback** (define on, but `fast && fold && num_tiles <= 1024 && tile_cap <= 2^22`
  false, e.g. tiles_x not a power of two such as a 960 px viewer, or no K2 fold): H_NB = 0, the
  mover takes the old loop (BK_SLOTS = 4 arrays, halves 0/1 used), TRISCs see NB = 0 and set FIN
  at once. Correct by reading; not exercised by the bicycle 1024 runs (see follow-up).
- **Tiles past tile_cap:** the TRISC increments the cursor and skips (`c >= cap`), never waits or
  queues for those; `ring_drain` clamps to cap. Same drops as the mover loop. kTileCap = 32768,
  bicycle never reaches it; covered by the model test (cap 64/128).
- **Pairs not g-sorted:** `build_lists` returns false on a g outside the bulk run; `issue_town`
  waits for the bulk reads to land, re-reads one page per g change into the same slot and
  rebuilds the lists from the start (per-g offsets match `issue_brec` with scan_g = -1, all
  kept under the fold). Correct; covered by the model test.

## L1 budget

Mailbox BYTES = 256 + 64 + 6144 + 3072 + 4096 = 13632 B; 2 more blendrec slots = 2 x 256 x 64 =
32768 B; +46400 B per mover, as claimed (logged cb_bytes/mover 521280 -> 567680). Two movers +
shared = 1157760 B of 1.5 MiB. None of these sizes depend on resolution (window, ring and
mailbox are fixed; only `shared` grows with num_cores), and tt-metal rejects a CB/L1-buffer
clash at enqueue, so this cannot fail silently.

## Tests

- The model test (`tests/unit/test_sort_ol_town.cpp`) could not be linked on this Mac (the
  Command Line Tools SDK libSystem.tbd lacks the arm64 target; a toolchain problem, not the
  test). Not re-run here; the t202 README reports 300 seeds passing.
- Device: see below.

## Device check

`review-t214/drive.sh 6abbda7` (= 656a1fa render code), own remote tree
/localdev/smarton/gstt2-t214 on yyzo-bh-07 (Blackhole p100a), one `ttp lock p100` for
sync + both runs, 2026-10-05. Logs: `review-t214/out/smoke-{dflt,pb1}.log`.

| run | env | log line | md5 vs md5-r82new (46a725ab) | avg_frame_ms (2 views, smoke only) |
|---|---|---|---|---|
| dflt | defaults | `OL_PB=8 ... OL_EMIT_TOWN=1 cb_bytes/mover=567680` | 2 of 2 | 14.0 |
| pb1 | `GSPLAT_TT_OL_PB=1` | `OL_PB=1 ... OL_EMIT_TOWN=1 cb_bytes/mover=563648` | 2 of 2 | 14.8 |

No hang, no throw. OL_PB=1 runs ~8x more batches per stream, so the 4-slot reuse, fl waits and
queue service run far more often; the output is still byte-identical. The 2-view times are
not an A/B; the t202 3-round 30-view A/B (15.194 -> 14.235 ms/view) was not re-run.

## Follow-up (non-blocking)

- The device-side fallback (define on, `town` false on device: tiles_x not a power of two, or no
  K2 fold) is only checked by reading. One md5 compare of `GSPLAT_TT_OL_EMIT_TOWN=1` vs `=0` at
  a camera set whose tiles_x is not a power of two (e.g. 960 px wide, tiles_x = 30) would cover
  it, and also shows whether the TRISC launch and +46 KB L1 cost anything there.
