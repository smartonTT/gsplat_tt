# t154 — what bounds `sort_ol_emit`

Board: **yyzo-bh-07 p100a** (Blackhole, 1350 MHz), bicycle, 1024x1024, 30 views. Code: c6a3fcd
(iter-180 tip 3372664 + profiling counters). All numbers below are from this board.

## What was added

`GSPLAT_TT_OL_EMIT_PROF=1` (profiling only, off by default) adds `OL_EMIT_PROF` to
`render/kernels/dataflow/sort_bin_onelaunch.cpp`. Each mover (BRISC, NCRISC) then sums wall-clock
cycles per part of the emit loop and records the totals as Tracy timestamped data `ep_*`.
Unset, the host adds no define and the kernel binary is the same as before.
`opt/profiler/emit_parts.py` turns the capture into ms/view.
Scripts: `drive.sh` (Mac, one `ttp lock p100` per device step), `remote_time.sh`, `remote_tracy.sh`.

## Correctness and overhead (untraced, `out/run-r1-*.log`, `out/md5-r1-*.txt`)

| arm | avg_frame_ms | sort stage ms | md5 vs golden (30 views) |
|---|---|---|---|
| default (flag unset) | 19.76 | 5.160 | all identical |
| `GSPLAT_TT_OL_EMIT_PROF=1` | 19.79 | 5.203 | all identical |
| `GSPLAT_TT_OL_EMIT_PROF=0` | 19.68 | 5.144 | all identical |

With the flag on, sort is about 0.04 ms slower. In the Tracy capture below, the busiest-core emit
window is 3.694 ms; t150 measured 3.70 ms without counters.

## Emit split (Tracy, flag on, `out/emit_parts.txt`)

220 movers (110 cores x BRISC+NCRISC). Busiest-core emit = 3.694 ms/view.

| part | mean ms/view | busiest core |
|---|---|---|
| `sort_ol_emit` zone | 3.582 | 3.694 |
| prologue (ring starts, first reads) | 0.019 | 0.029 |
| read barrier in loop (blendrec k, pairs k+1) | 0.007 | 0.008 |
| blendrec scan + read issue (`issue_brec`) | 0.508 | 0.542 |
| pair page read issue | 0.002 | 0.002 |
| **process_batch** | **2.975** | 3.028 |
| -- writes-flushed waits | 0.021 | 0.021 |
| -- run write issue (`flush_run`) | 0.106 | 0.114 |
| -- **record pack / key build (rest)** | **2.848** | 2.893 |
| tail drain | 0.068 | 0.082 |
| final write barrier | 0.001 | 0.001 |
| unattributed | 0.002 | 0.002 |

Per mover per view: 13,944 records, 7,723 blendrec pages (1.8 records per gaussian), 109 batches
(PB=8 pages = 128 pairs each). The whole emit takes 347 cycles per record; the pack loop alone
takes 276.

## What bounds it

The emit is **bound by scalar instruction issue on the mover RISCs**, not by the NoC or by DRAM.
All NoC waits together (read barrier, writes flushed, final barrier) are 0.03 ms. Write issue is
0.11 ms and read issue is in the 0.51 ms scan. 80% of the window (2.85 ms) is the per-record pack
loop, which runs entirely in L1 and on the RISC.

The disassembly of the default BRISC ELF (riscv-tt-elf-objdump -dl on the JIT cache) shows where
the per-record cycles go:
- Each record does 3 volatile L1 loads (keep, gid, tid), a load and a store of the L1 cursor
  `curp[t]` (the read-after-write pattern #26 found expensive), and 8 word stores into the ring.
- The lambdas capture by reference, so `inv_*`, `c_ty`, `tx_shift` and the others live in a
  closure on the stack and are reloaded on every use. Each record-word line compiles to 4 lw + 2 sw.
- `sub_int` (the tile-local mean x) is an out-of-line call on every record, about 30 instructions.
- `pack_invariants` copies `inv_mx`/`inv_my` byte by byte (8 lbu + 8 sb) for every new gaussian.
- `divu`/`remu` are present, but bicycle has tiles_x=32 (a power of two), so that path does not run.
- `issue_brec` scans the same keep/gid planes again that `process_batch` scans one batch later.

The TRISCs run nothing during the sort program. The program has only BRISC and NCRISC zones;
see segment 2 of `out/gaps.txt`.

## Upper bound and levers (estimates, not measured)

- **Hard ceiling:** removing the pack loop entirely saves 2.85 ms/view mean (2.89 ms on the
  busiest core). Sort is serial with blend in the frame (STAGES: project 4.58, sort 5.14,
  blend 9.71 ms), so emit savings go straight into frame time.
- **Lever 1, pack-loop diet (bit-identical):** plain register locals instead of closure
  captures; inline the `sub_int` fast path; copy the record template as words; read the planes as
  words, a batch at a time; keep the per-tile cursor for the batch in local memory; merge the
  blendrec scan into the previous batch's pass. Estimate: 276 -> ~120 cycles per record, about
  1.5 ms/view.
- **Lever 2, pack on the idle TRISCs:** packing only touches L1, so the 3 TRISCs can each pack a
  share of a batch into the ring, and the movers only issue NoC reads and writes. That gives 5
  RISCs instead of 2 for the 2.85 ms. Estimate: about 1.7 ms/view alone. Together with lever 1,
  emit would come down to about 1.2 ms (scan 0.5 + writes 0.1 + pack ~0.5), saving up to
  ~2.4 ms/view.
- Lever 1 is simpler and carries over to lever 2, so it should go first.

## Files

- `out/emit_parts.txt`: per-part table; `out/zones.txt`, `out/gaps.txt`: zone and program
  windows; `out/capture.log`: capture log.
- `out/run-r1-{base,ep,b2}.log`, `out/md5-r1-*.txt`: untraced timing and md5 per arm.
