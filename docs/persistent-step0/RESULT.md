# L4 step 0: persistent per-view program, size feasibility (task #579)

Verdict: **NO-GO. Shelve L4** (persistent sort(N) + pfwc(N+1) + mat+blend(N) with the K2 fold, run as
one launch). The code is 2.4x the kernel config buffer, and the combined BRISC stack has less than the
required 256 B of margin, with or without the t574 stack fix. Both go criteria fail.

No device run. Done 2026-10-12, 21:25-21:35 PDT, on bh-33 (the measurement reservation): read-only
scans of two fresh JIT caches left by t574 (`docs/land-t574`), each holding only one tree's
production kernels:

- `ttmc-t574base` = **best-iter-226** (90268457)
- `ttmc-t574new` = 2c895ad2 (226 + the t574 BRISC stack fix, `muldiv_lo32`; now on smarton/tt-project-opt)

Every ssh call ran under `ttp lock p100` with StrictHostKeyChecking=yes (`rsh.sh`).

## 1. Kernel config (kcfg) size

A program's kcfg size is the sum of the PT_LOAD bytes (code + rodata + .data image) of all its RISC
binaries, plus about 1.3 KB of runtime args, CB config and semaphores. Check against t544: 87,324 B of
ELF + 1,268 B = 88,592 B, the size tt-metal reported. Per-RISC bytes, aligned to 16 B
(`sizes.py out/elfscan-*.jsonl`, raw ELF data in `out/elfscan-*.jsonl`):

**best-iter-226**

| RISC | pfwc | sort_ol | mat+blend + K2 fold | persistent (sum) |
|---|---:|---:|---:|---:|
| BRISC | 25,152 | 22,416 | 29,168 | 76,736 |
| NCRISC | 24,864 | 22,432 | 28,496 | 75,792 |
| TRISC0 | 9,552 | 4,336 | 8,032 | 21,920 |
| TRISC1 | 28,576 | 4,336 | 28,544 | 61,456 |
| TRISC2 | 6,976 | 4,320 | 2,704 | 14,000 |
| **code** | **95,120** | **57,840** | **96,944** | **249,904** |

With the t574 fix, the persistent code total is 246,432 B (sort -2.3 KB, mat+blend -1.2 KB).

| | bytes |
|---|---:|
| kcfg limit today (69 + 34 KB; the size the fused SIB program overflowed, 117,360 B) | 105,472 |
| go line (limit - 4 KB margin) | 101,376 |
| persistent program, code only (226 / with t574 fix) | 249,904 / 246,432 |
| over the limit | **+144,432 / +140,960 (2.37x / 2.34x)** |
| largest single program today (mat+blend + K2), code only | 96,944 |

The limit cannot be raised to fit. Static CBs start right after the kcfg ring. Today both the
onelaunch sort (t565: KCFG +48 KB overflows L1 by 14,208 B; +34 KB leaves about 128 B) and fused
mat+blend (t302: fits up to +34 KB) are at the L1 ceiling. A 250 KB ring would take about 145 KB
more L1 from static CBs that are already full. Merging shared code (dataflow API, TensorAccessor
helpers) cannot close the gap either: fitting at the go line needs a 59 % cut of all code. The two
SFPU math kernels (pfwc and blend TRISC1, 28.5 KB each) already exceed half the budget on their own.

## 2. BRISC stack of the combined chain (`opt/brisc_stack_check.py`, `out/stack-*.jsonl`)

BRISC .data/.bss and the stack share 8 KB of local memory (0xFFB00000-0xFFB02000). Firmware globals
end at 0xFFB00C30. Measured per kernel:

| BRISC kernel | ldm .data/.bss (226) | deepest chain | margin 226 | margin t574 fix |
|---|---|---:|---:|---:|
| writer_pfwc_split (pfwc reader/writer) | 152 B anon .data | 1,184 | 3,728 | 3,728 |
| sort_bin_onelaunch | `__clz_tab` 256 B | 1,184 | 3,632 | 3,888 |
| matblend_brisc (+K2 fold) | `__clz_tab` 256 + `g_rta`/`g_b4a_*` 12 B | **4,688** (K2 `k2k::kernel_main`) | 112 | 368 |

A persistent BRISC kernel holds the globals of all three, while its deepest chain is the K2 one plus
the wrapper frame. That holds only if each stage's main stays `noinline`; if the compiler inlines
them, the frames add up instead.

| | best-iter-226 | with t574 fix |
|---|---:|---:|
| combined ldm globals (`__clz_tab` counted once) | 0xC30 + 256 + 12 + 152 = 0xDD4 -> base 0xDE0 | 0xC30 + 12 + 152 = 0xCD4 -> base 0xCE0 |
| stack available | 4,640 | 4,896 |
| deepest chain (K2) | 4,688 | 4,688 |
| margin before the persistent wrapper's frame and semaphore/claim state | **-48** (overflow) | **+208** |
| margin with a minimal 16 B wrapper frame | -64 | **+192** (< 256 go line) |

So even with the stack fix, the combined BRISC falls 64 B or more below the 256 B go line, before any
persistent-loop state (semaphore addresses, pfwc chunk-claim counters) is added. On the 226 binaries
it overflows outright, the same failure as the B4' crash (t573). NCRISC is not the limit (deepest chain
4,224 B in matblend_ncrisc, combined margin about 540 B).

## 3. Reading

- L4's ceiling was -0.38 ms/view (realistic -0.15 to -0.25; docs/structural-plan-2/PLAN.md). It
  would need a kcfg ring about 2.4x larger than L1 allows today, plus a stack cut in K2's 4.5 KB frame
  (`cnt[1024]` + 2 KB). Each one alone is a project-sized rework, and the gain is below the
  0.4 ms threshold the B4' fusion used.
- What would reopen it: a design where each core runs only one stage (disjoint core groups, so each
  core's code fits). The plan already dropped that, because pfwc and blend are both SFPU-bound and
  splitting moves no core-ms. Or a code overlay that reloads stage text from DRAM between phases,
  which brings back part of the launch cost L4 is meant to remove. Neither is queued.

## Files

- `elfscan.py`: per-ELF PT_LOAD/section/ldm-symbol scan of a JIT cache (piped over ssh, read-only)
- `sizes.py`: the per-RISC tables above
- `brisc_stack_check.py`: copy of opt/brisc_stack_check.py at 9cb7385b, the version that was run
- `rsh.sh`: ssh with strict host-key checking
- `out/`: raw scans (`elfscan-*.jsonl`, `stack-*.jsonl`, `ldm-syms.txt`, `sizes.md`)
