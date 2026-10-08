# Ethernet dispatch on the p150 (task #387, 2026-10-08): not usable on tt-metal 437bc366

Goal: test #383's `GSPLAT_TT_DISPATCH=worker|eth|auto` (branch `ttp/t383-...` @ f70118da, merged here
on top of opt tip 58b47824). ETH dispatch would free the Tensix column that worker dispatch takes,
so the p150 would render on 12x10 = 120 cores instead of 11x10 = 110. #382 predicted ~0.50 ms/view
from that (mat+blend 656.5 core-ms / 120 = 5.47 ms instead of 5.968 ms).

**Result: not kept. The default stays `worker`.** The tt-metal both boxes run (437bc366) cannot
start ETH dispatch on the bh-30 p150b. Both failures happen when the device opens, before any
gsplat kernel runs. There is no 120-core number to report.

## What failed

| attempt | setup | result |
|---|---|---|
| 1 | stock tt-metal, `GSPLAT_TT_DISPATCH=eth`, 2 CQs (default `GSPLAT_TT_MAT_CQ1=1`) | `RuntimeError: No core coordinate found at location: (0, 12, ETH, LOGICAL)` in `L1BankingAllocator::generate_config` |
| 2 | runtime-root overlay (`make_overlay.sh`): ETH dispatch yaml cut to 12 cores, 2 CQs | opens further, then `TT_THROW tt_elffile.cpp:393: cq_dispatch/.../idle_erisc.elf: segment[0] [0x6870,+0x3120) overflows region:0 limit of 0x2ab0 bytes` (two kernels, 0x3120 and 0x3124) |
| 3 (probe) | overlay, 1 CQ (`GSPLAT_TT_MAT_CQ1=0`) | same overflow: `+0x319c` vs limit `0x2ab0` |

- Attempt 1: `core_descriptors/blackhole_140_arch_eth_dispatch.yaml` lists 14 ETH dispatch cores
  (`[0,0]..[0,13]`) for every Blackhole product. The bh-30 card has ETH harvesting mask 0x120
  (`Harvesting masks for Chip 0: Tensix: 0xc0 DRAM: 0x0 ETH: 0x120`), so only 12 ETH cores exist
  and logical ETH core 12 has no coordinate. `core_descriptor.cpp` skips only ETH cores with an
  active link, not harvested ones.
- Attempt 2 works around that without touching the shared tt-metal checkout. tt-metal reads its
  core descriptors from `TT_METAL_RUNTIME_ROOT`, so `make_overlay.sh` builds a symlink copy of the
  viewer's tt-metal in which only that yaml is a real file, with the list cut to `[0,0]..[0,11]`.
  The device then gets past the allocator, but the dispatch kernels built for idle ERISC
  (`cq_dispatch` prefetcher and dispatcher, already built `-Os`, the tt-metal default for non-compute
  kernels) are 12.3-12.4 KB. The idle-ERISC kernel region on Blackhole is 0x2ab0 = 10.9 KB
  (`MEM_IERISC_KERNEL_SIZE = MEM_ERISC_KERNEL_SIZE` in `hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h`).
  That is a tt-metal limit. gsplat cannot fix it from the outside.
- Attempt 3: one command queue makes the kernel slightly larger (0x319c), not smaller, so dropping
  `GSPLAT_TT_MAT_CQ1` does not help either.
- No hangs: every failure is an exception at open. The process exits with rc=1, and the next
  run (and the viewer) opened the device normally with worker dispatch.

## Numbers measured (worker arm, bh-30 p150b, same tree f70118da)

The A/B could not run, so these are only the worker rounds that ran before each failed eth run.
Bicycle, 30 views, untraced, `GSPLAT_PER_VIEW_STAGES=1`:

| run | dispatch line | avg_frame_ms | project | sort | blend | d2h | md5 |
|---|---|---:|---:|---:|---:|---:|---|
| attempt 1 r1-W (stock root) | `dispatch worker (worker), command queues 2, compute grid 11x10` | 10.932 | 2.997 | 0.965 | 6.563 | 0.316 | 906e0435 30/30 |
| attempt 2 r1-W (overlay root) | same | 11.066 | 2.981 | 1.074 | 6.467 | 0.471 | 906e0435 30/30 |

#382's untraced iter-210 baseline on bh-30 was 10.998 ms/view, so the worker path is unchanged by
the merge, and the overlay does not change worker results (identical md5).

Per-core busy on 120 cores (requested in the task update): not measurable, because no 120-core run
was possible. The shortfall against #382's 0.50 ms/view prediction is total. The cause is neither
the emit->mat critical path nor ETH dispatch latency: the ETH dispatch firmware does not fit on
this tt-metal version.

## p100a

`/sys/class/tenstorrent/tenstorrent!0/tt_card_type` reads `p100a` on yyzo-bh-04 (same tt-metal
437bc366), so `auto` resolves to `worker` there (`card_has_eth("p100a")` is false, covered by
`tests/unit/test_dispatch_select.cpp`). The p100a timing A/B (`p100_drive.sh`) was not run: the
default stays `worker` and nothing lands, so no p100a numbers can change.

## Caveat for the t383 code

On the p150 with this tt-metal, `GSPLAT_TT_DISPATCH=auto` resolves to eth and the device open
throws. The code stays off the opt branch, and the default stays `worker`. Do not land `auto` as
the default until a tt-metal fits the dispatch kernels in idle ERISC.

## Viewer downtime on bh-30

The viewer ran under the existing viewer reservation (IRD job 135630). Nothing was reserved,
extended or released, and mutagen-agent and the firmware were not touched. The build ran
at `nice -n 19 ionice -c3 -j 12` while the viewer kept running. The viewer was stopped only for the
three bench windows:

| window | stop | ready again (localhost:8091 = 200) |
|---|---|---|
| attempt 1 | 00:23:44Z | 00:25:10Z |
| attempt 2 | 00:26:40Z | 00:28:06Z |
| probe | 00:29:42Z | 00:30:35Z |

After the last restart: pid 73011, tree 8329be50, selftest median 11.51 ms (86.8 FPS), READY on 8080.

## Files

- `sync_bh30.sh`: own tree `p150bench/tree387` on bh-30 (copy of tree382, synced, nice build).
- `drive_bh30.sh`, `bench387.sh`: one viewer-downtime window. 3 alternating rounds W/E (r3 uses
  `auto`), eth Tracy, and a live-viewer check with eth. `bench387.sh probe` runs the 1-CQ probe only.
- `make_overlay.sh`: the runtime-root overlay with the ETH dispatch list cut to the live ETH count.
- `ana210.py`: #382's analysis, printing core-ms (kept for a future 120-core trace).
- `p100_drive.sh`, `p100_remote_time.sh`: p100a default-vs-auto A/B (not run).
- `out-attempt1/`, `out-attempt2/`, `out-probe/`: logs and md5 lists.
