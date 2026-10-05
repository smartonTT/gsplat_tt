# t207: pfwc writer split onto NCRISC (build-checked, not run on device)

Lever 3 of `docs/pfwc-breakdown-t197`. In the fused pfwc program (lever B) the BRISC writer
is busy ~2.0 ms/view (classify 0.73 + records 1.27) while the NCRISC reader waits 2.47 of
its 2.60 ms. This change splits the writer's work over both data-movement RISCs, behind
`GSPLAT_TT_PFWC_WRITER_SPLIT=1` (default 0, nothing changes with it off).

Status: built against the syntax stubs and checked by a host model test. **No device
run yet**: md5, timing and the kernel binary size are unmeasured.

## Design

- **Chunk split.** Chunk k of a core is classified and written by BRISC if k is even and by
  NCRISC if k is odd. Each RISC has its own record staging, page staging and color/opacity
  tiles (`writer_pfwc_split.cpp`, one source, `WSPLIT_ROLE` 0 or 1).
- **Compute output sets.** The compute kernel packs odd chunks into a second set of output
  CBs (9..16 → 41..48, 35/36 → 49/50; `OCB()` in `project_pfwc_compute.cpp`), so each writer
  waits on and pops only its own chunks.
- **NCRISC is also the reader.** Its reader is non-blocking (`rd_step`): it reserves all 10
  input CBs, issues the reads, and pushes them on a later poll once
  `ncrisc_noc_reads_flushed()` says they landed. NCRISC polls it in every wait loop, after
  classify, and every 8 mask words of its record loop. Input CBs get a third slot so the
  reader can stay ahead while NCRISC is busy with records.
- **Chunk-to-chunk handoff** (`pfwc_wsplit.h`). Chunk k needs the running visible count
  m_k and pair offset pr_k, and the dep/offs/aabb page it starts in may be shared with
  chunk k−1. BH DRAM writes are whole 16 B words, so a 64 B page must be written by one
  writer, in one piece. Through an L1 mailbox (CB 52, 4 semaphores, 2 slots per direction):
  - PREFIX (m, pr): sent right after classify, so the next chunk's record loop can start
    while this one still runs its own;
  - OPEN: the first m_k % 16 words of the shared page, sent when the chunk ends (only if
    m_k % 16 ≠ 0).
  Chunk k stages its first, shared page separately (the head) and merges in the OPEN words
  at the end, so its record loop never waits on chunk k−1. Each page is written exactly
  once: by the writer that completes it, or by the core's last chunk (padded like the fused
  writer's tail). Records are one page each, so each writer writes only its own; a chunk
  restarts the bank-batched staging at its first record and flushes it at its end.
- **Semaphore protocol.** EMPTY → PREFIX → (OPEN) → EMPTY. The receiver empties the slot,
  so all semaphores end at 0 after every launch. The sender waits for EMPTY before reusing
  a slot (back-pressure of at most 2 chunks per direction).
- **Deadlock-freedom.** Every wait is for something that depends only on earlier chunks,
  and NCRISC polls the reader in each wait, so compute never starves of input while NCRISC
  spins. The host model below checks this under random interleavings.

## Files

| file | change |
|---|---|
| `render/kernels/dataflow/writer_pfwc_split.cpp` | new: split writer (+ reader on NCRISC) |
| `render/kernels/dataflow/pfwc_wsplit.h` | new: page/record staging and mailbox layout, device-free |
| `render/kernels/compute/project_pfwc_compute.cpp` | `OCB()` on the 10 output packs under `PFWC_WSPLIT` |
| `render/host/pfwc_device.cpp` | knob: CBs 41..52, input depth 3, 4 semaphores, kernels and args |
| `render/host/env_config.h` | `pfwc_writer_split()` |
| `render/host/device_state.cpp` | knob on and `GSPLAT_TT_KCFG_EXTRA_KB` unset: +8 KB kernel config buffer |
| `tests/unit/test_pfwc_wsplit.cpp` | new host model test |
| `tests/syntax_stub/*` | stubs + checks for both roles, `EMIT_PUBOC`, `PFWC_STEPCYC`, `PFWC_WSPLIT` |

## Checks run (Mac, no device)

- `tests/syntax_stub/check.sh`: 44/44 ok (7 new: both roles × plain / EMIT_PUBOC /
  EMIT_PUBOC+PFWC_STEPCYC, and the compute kernel with `PFWC_WSPLIT`). The new code is
  also clean under `-Wall -Wextra`.
- `tests/unit/run_cpp.sh tests/unit/test_pfwc_wsplit.cpp`: PASS. It models BRISC, NCRISC
  (with the reader), compute and the CB depths as state machines under a random scheduler,
  using the real `ChunkPages` / `RecStage` code. Inputs: 5660 cores with 0..59 chunks of
  0..1024 visible gaussians (biased to 0, 1-3, 15-17 and multiples of 16), 4 segment bases,
  1..12 DRAM banks. For each, the dep/offs/aabb pages, records and counts page match
  `writer_pfwc_fuse.cpp`'s single writer and are written exactly once; no message is
  overwritten before it is read; all semaphores end EMPTY; no deadlock. Mutating the head
  flush, head merge or tail pad makes it fail.
- `test_pfwc_flush_rec.cpp`, `test_pfwc_fuse.cpp`: PASS (unchanged).

## Risks for the first device run

- **Program size.** NCRISC now carries the writer code. No RISC-V toolchain here, so the
  size is unmeasured. The fused program was ~66-69 KB of the 70656 B kernel config buffer;
  the split adds roughly one writer binary. With the knob on, the device opens with +8 KB
  of kernel config buffer (unless `GSPLAT_TT_KCFG_EXTRA_KB` is set). The t197 runs used
  +10 KB across the whole pipeline without trouble. If it still overflows (TT_FATAL
  "Program size … too large"), rerun with `GSPLAT_TT_KCFG_EXTRA_KB=16`. Profiling builds
  (`PFWC_STEPCYC`, Tracy) need more: try 16-20.
- **L1.** +158 KB per core for the pfwc program (odd output set 80 KB, third input slot
  40 KB, NCRISC staging 33 KB, BRISC staging +4 KB, mailbox 1 KB). That brings it to ~486 KB
  of CBs. There are no persistent L1 buffers, so this fits.
- **NCRISC load.** NCRISC writes half the chunks and also reads all of them. If it ends up
  the slower RISC (the `pfwc_ws` rd and wait counters show this), deal 2:1 or claim chunks
  dynamically.
- **Unequal chunks.** If consecutive chunks differ a lot in cost, a writer waits on the
  other's PREFIX/OPEN (`pfx`/`opn` counters). The head-page staging keeps the OPEN wait off
  the record loop; only the head merge waits.
- `GSPLAT_TT_FUSE_ABL` (t122 ablation) is not supported with the split; the host warns and
  ignores it.

## Expected saving (model from the t197 counters, not measured)

Per view on bicycle, today ~54 chunks per core: compute (TRISC1) ~2.5 ms is the floor, the
writer ~2.0 ms mean and ~2.2 ms on the heaviest core (views 1-3). pfwc makespan ~2.75 ms.

| case | pfwc ms/view | saving vs today |
|---|---:|---:|
| today | ~2.75 | — |
| split alone | ~2.55-2.65 | ~0.1-0.2 (only the heavy-core writer back-pressure, 0.09-0.23 ms in t197; compute stays the floor) |
| SFPU cov2d/conic/radii fusion alone (t197 lever 2) | ~2.2-2.3 | ~0.5-0.6 (capped by the single writer) |
| fusion + split | ~1.3-1.45 | ~1.3-1.45 (writer halves to ~1.1-1.3 per RISC on the heaviest core; compute ~1.0-1.2) |

So the split alone falls under the 0.3 ms/view gate. Its value is removing the writer cap
once compute drops: on top of the fusion it is worth ~0.8-0.9 ms/view. Together that is
~15.2 → ~13.8-13.9 ms/view (~72 FPS) on the tip, if the rest of the pipeline is unchanged.

## Device test plan (follow-up, under `ttp lock p100`)

1. Bicycle 30 views with `GSPLAT_TT_PFWC_WRITER_SPLIT=1`: md5 must equal `46a725ab`
   (`md5-r82new.txt`, all 30 views); look for kernel-config TT_FATALs or hangs first.
2. Paired untraced A/B, knob 0 vs 1, `docs/tip-t199` drivers. Expect ≤ 0.2 ms/view. Keep
   the default at 0 unless it clears the gate.
3. Tracy with `GSPLAT_TT_PFWC_STEPCYC=1 GSPLAT_TT_KCFG_EXTRA_KB=16`: `pfwc_ws` per role
   gives `n wall wait cls pfx rec opn tail rd`. Check the per-RISC writer busy time
   (cls + rec, expect ~1.0-1.1 ms each) and that pfx/opn waits are small.
4. Re-run 2-3 on top of the SFPU fusion when it lands. That is the A/B that decides the
   default.
