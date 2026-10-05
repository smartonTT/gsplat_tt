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

`timeline_model.py` (output in `timeline_model.txt`) replays one core's chunk timeline:
classify 13.4 µs + records 23.3 µs per chunk (CV 0.5), compute emitting one chunk every P µs,
the split's PREFIX chain and OPEN ordering, and 2 µs per odd chunk for NCRISC's reader polls.
P = 46 µs is today's compute; P = 20 µs is compute after the SFPU fusion (t197: ~1.65 ms of
math becomes ~0.2-0.3).

| core | P | single writer | split | saving |
|---|---:|---:|---:|---:|
| mean | 46 µs | 2.524 | 2.522 | 0.002 |
| heavy (records ×1.2) | 46 µs | 2.549 | 2.528 | 0.021 |
| heavy (records ×1.4) | 46 µs | 2.652 | 2.533 | 0.119 |
| mean | 20 µs | 1.997 | 1.158 | 0.839 |
| heavy (records ×1.2) | 20 µs | 2.320 | 1.303 | 1.017 |
| heavy (records ×1.4) | 20 µs | 2.571 | 1.441 | 1.130 |

The slowest core sets the pfwc makespan, so the heavy rows count:

| case | pfwc ms/view | saving vs today |
|---|---:|---:|
| today | ~2.75 | — |
| split alone | ~2.6-2.7 | ~0.02-0.12 (only the heavy cores' writer back-pressure; t197 measured 0.09-0.23 ms of it) |
| SFPU cov2d/conic/radii fusion alone (t197 lever 2) | ~2.2-2.4 | ~0.4-0.55 (capped by the single writer) |
| fusion + split | ~1.3-1.45 | ~1.3-1.45 |

So the split alone falls under the 0.3 ms/view gate. Its value is removing the writer cap
once compute drops: on top of the fusion it is worth ~0.8-1.1 ms/view. Together that is
14.235 → ~12.8-12.9 ms/view (70.2 → ~77-78 FPS) from the iter 194 tip (t202), if the rest
of the pipeline is unchanged (t202 changed sort_ol's emit, not pfwc).
In the model the OPEN ordering costs nothing (a variant that merges the shared page later
gives the same times), so there is no need to decouple it further.

## Device test plan (follow-up, under `ttp lock p100`)

1. Bicycle 30 views with `GSPLAT_TT_PFWC_WRITER_SPLIT=1`: md5 must equal `46a725ab`
   (`md5-r82new.txt`, all 30 views); look for kernel-config TT_FATALs or hangs first.
2. Paired untraced A/B, knob 0 vs 1, `docs/tip-t199` drivers, both arms with
   `GSPLAT_TT_KCFG_EXTRA_KB=8` so only the knob differs. Expect ≤ 0.12 ms/view. Keep the
   default at 0 unless it clears the gate.
3. Tracy with `GSPLAT_TT_PFWC_STEPCYC=1 GSPLAT_TT_KCFG_EXTRA_KB=16`: `pfwc_ws` per role
   gives `n wall wait cls pfx rec opn tail rd`. Check the per-RISC writer busy time
   (cls + rec, expect ~1.0-1.1 ms each) and that pfx/opn waits are small.
4. Re-run 2-3 on top of the SFPU fusion when it lands. That is the A/B that decides the
   default.

## Device results (task #221, 2026-10-05, yyzo-bh-07 p100a, tree 6c0b2e5 = 2edbd71 + t206)

Drivers and raw output: `dev-t221/` (`drive.sh`, `remote_time.sh`, `remote_prof.sh`,
`pc_split.py`, `out/`). Bicycle, 30 views, untraced, `render/run.py --no-ref`.

**Program size.** The split pfwc program is 92496 B. It overflows the kernel config
buffer at the default open (+8 KB, 78848 B) and at `KCFG_EXTRA_KB=16` (87040 B), with
`TT_FATAL: Program size (92496) too large`. `+24 KB` (95232 B) is the smallest that fits,
so every untraced arm ran at `GSPLAT_TT_KCFG_EXTRA_KB=24`. Under Tracy (STEPCYC=1) the
split program is 96448 B and needs `+32 KB`. No hangs.

**md5.** Every arm (base, split, cov, cov+split) gave `46a725ab` (md5-r82new.txt) on all
30 views in every round.

**Split alone, 4 swapped rounds (ms/view):**

| round | base | split | Δ view_total | Δ project |
|---|---:|---:|---:|---:|
| 1 (base first) | 14.188 | 14.179 | -0.009 (split d2h outlier 0.416) | -0.236 |
| 2 (split first) | 14.200 | 13.977 | -0.223 | -0.228 |
| 3 | 14.187 | 13.979 | -0.208 | -0.242 |
| 4 | 14.236 | 13.969 | -0.267 | -0.246 |
| mean | 14.203 | 14.026 | **-0.177** (median -0.216) | **-0.238** |

Below the 0.3 ms/view gate on its own, as the model said, though about twice the
model's 0.02-0.12.

**On top of t206 SFPU cov_cam, 3 rotated rounds (ms/view):**

| round | base | cov | cov+split | cov - base | both - base | both - cov |
|---|---:|---:|---:|---:|---:|---:|
| c1 | 14.283 | 13.979 | 13.584 | -0.304 | -0.699 | -0.395 |
| c2 | 14.195 | 13.990 | 13.578 | -0.205 | -0.617 | -0.412 |
| c3 | 14.337 | 14.000 | 13.735 | -0.337 | -0.602 | -0.265 |
| mean | 14.272 | 13.990 | 13.632 | -0.282 | **-0.639** | **-0.357** |

project stage: base 3.985, cov 3.776, both 3.354 ms/view (both - base -0.631).
The two levers are super-additive: alone -0.24 and -0.21 ms of project, together -0.63.
Each one was hiding the other's saving behind the other bottleneck (split frees the
writer, cov_cam cuts TRISC compute). Combined they clear the gate: 14.27 → 13.63 ms/view
(70.1 → 73.4 FPS) in this paired run.

**Tracy (STEPCYC=1, views 0-4, 110 cores, mean per launch = per view, ms):**

| arm | pfwc wall | writer RISC | wait | cls | rec | cls+rec | pfx | opn | rd | TRISC wall | TRISC wait |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base (k0) | 2.737 | BRISC | 0.701 | 0.730 | 1.270 | 2.000 | — | — | — | 2.69-2.70 | 0.001-0.022 |
| split | 2.609 | BRISC | 1.478 | 0.404 | 0.696 | 1.100 | 0.002 | 0.016 | — | 2.59-2.60 | ≤0.036 |
| split | 2.617 | NCRISC | 1.441 | 0.397 | 0.739 | 1.136 | 0.008 | 0.020 | 1.178 | | |
| cov+split | 2.223 | BRISC | 1.053 | 0.410 | 0.710 | 1.120 | 0.003 | 0.035 | — | 2.20-2.21 | ≤0.036 |
| cov+split | 2.229 | NCRISC | 0.990 | 0.403 | 0.786 | 1.189 | 0.013 | 0.026 | 0.817 | | |

(base NCRISC reader: reserve 2.465 ms, i.e. it waits on the writer the whole time.)

- Per-RISC writer busy time is 1.10-1.19 ms, as predicted (~1.0-1.1). pfx/opn waits are
  small (≤0.035 ms mean; one core's BRISC opn max 0.33 ms).
- NCRISC is the slower writer (rec 0.74-0.79 vs 0.70-0.71 mean, max core 1.04-1.07 vs
  0.88-0.91) because it also runs the reader, but its wall is within 0.01 ms of BRISC.
- With the split, pfwc is TRISC-bound (TRISC wait ~0, both writers wait 1.0-1.5 ms). That
  is why cov_cam only pays off with the split. The next pfwc lever is TRISC compute:
  per-TRISC split under cov+split is xform 0.49/0.21/0.75 and vis+pop 0.35/0.62/0.21 ms
  (T0/T1/T2), so the three threads are already balanced at ~2.2 ms in total.

**Decision.** Both knobs on by default (`GSPLAT_TT_PFWC_WRITER_SPLIT`,
`GSPLAT_TT_PFWC_COVCAM_SFPU`, `=0` turns each off). With the split on and
`GSPLAT_TT_KCFG_EXTRA_KB` unset, the device opens with +24 KB; with only cov_cam on, +8 KB.
Base at +24 KB (14.19-14.34) matches the iter-194 tip (14.235), so the bigger kernel
config buffer costs nothing visible. Profiling the default needs `GSPLAT_TT_KCFG_EXTRA_KB=32`.

**Default verify (f031ba0, 3 rotated rounds, untraced, no KCFG override).** Synced f031ba0 and
ran default / both-off / cov-only per round (`dev-t221/verify.sh`, log `dev-t221/out/t221-verify.log`).
All 9 arms: 30/30 views identical, md5-of-md5s 46a725ab, no hang, no TT_FATAL.

| arm | rv1 | rv2 | rv3 | mean ms/view | project ms/view |
|---|---:|---:|---:|---:|---:|
| default (split+cov) | 13.578 | 13.590 | 13.643 | **13.604** (73.5 FPS) | 3.354 |
| both off | 14.229 | 14.268 | 14.307 | 14.268 | 4.017 |
| cov only | 14.038 | 14.010 | 14.021 | 14.023 | 3.790 |

Default vs both-off -0.664 ms/view, vs cov-only -0.419. The defaults work without any env override.
