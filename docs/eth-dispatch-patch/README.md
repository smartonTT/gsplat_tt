# ETH dispatch on the p150: can a local tt-metal patch clear the #387 blockers? (task #390, 2026-10-07)

Read-only study. No device, no ssh. Source: the vendored tt-metal in the project checkout
(`/Users/smarton/dev/gsplat_tt/backends/tt/tt-metal`; its BH `dev_mem_map.h` is byte-identical to
`437bc366`, which is what both boxes run). Line numbers below are from that tree.

## Verdict: yes, feasible in well under a day, low code risk

Neither blocker needs a tt-metal source change or a tt-metal rebuild. Both can be fixed by
replacing **text files that tt-metal reads from `TT_METAL_RUNTIME_ROOT` at run time**, using the
symlink overlay #387 already built (`docs/eth-dispatch-t387/make_overlay.sh` on
`ttp/t387-ethernet-dispatch-120-core-grid-bh-30-p1`). The viewer's tt-metal stays untouched.

Patch size: about 10 more lines in `make_overlay.sh` (the yaml cut already exists), plus merging
#383's `GSPLAT_TT_DISPATCH` code (f70118da, ~180 lines, already written and unit-tested) onto the
opt tip. The open risk is not the patch but what comes after it: Blackhole ETH dispatch has never
run upstream (see Risks), so a third blocker or slower ETH dispatch could still eat the gain.
Only the device A/B can settle that.

## (a) Blocker 1: 14 listed ETH dispatch cores, 12 live

- `tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml:64,75` (2xharvested, 1 and 2 CQs)
  list `[0,0]..[0,13]`.
- `tt_metal/llrt/core_descriptor.cpp:244-258` only skips ETH cores with an **active link**
  (`logical_active_eth_cores`), never harvested ones, so logical (0,12) reaches the allocator and throws.

Smallest patch: the #387 overlay yaml cut to `[0,0]..[0,11]`. Already proven on bh-30: it gets past
`L1BankingAllocator::generate_config` (#387 attempt 2). It is safe: UMD numbers logical ETH cores
contiguously over the live channels, so on any p150 with at most 2 ETH channels harvested,
logical 0..11 all exist. Active-link cores are still skipped by the same loop. One device with
2 CQs uses 4 of them (prefetch and dispatch per CQ; dispatch_s is off for 2-CQ ETH,
`dispatch_query_manager.cpp:97-100`).

The proper fix, a harvest-aware filter in `core_descriptor.cpp:249-253` (also skip
`logical_coord.y >= number of logical ETH cores`), needs a libtt_metal rebuild. That is an
upstream note, not something we need.

## (b) Blocker 2: where 0x2ab0 comes from, and why it is not a real limit

The 0x2ab0 B figure is a **link-time bound left over from a non-XIP layout**:

- `tt_metal/hw/toolchain/main.ld:271-279` writes the per-region size limit the loader checks:
  `TEXT_SIZE - (__fw_export_text_end - TEXT_START)` for kernels. For idle ERISC,
  `TEXT_SIZE = MEM_IERISC_KERNEL_SIZE` (`main.ld:52`) = `MEM_ERISC_KERNEL_SIZE` = 24 KB
  (`dev_mem_map.h:50,309`). `TEXT_START` = `MEM_IERISC_FIRMWARE_BASE` = 0x3320.
- The BH idle-ERISC firmware text ends at 0x6870 (the address in #387's error), so it uses
  0x3550 = 13,648 B, and 0x6000 - 0x3550 = **0x2ab0**. The bound says: firmware + kernel text
  must fit in 24 KB.
- The check is `tt_metal/llrt/tt_elffile.cpp:390-399` (`TrimSegments`). Nothing else uses
  `MEM_IERISC_KERNEL_SIZE`: `grep` finds it only in `dev_mem_map.h` and `main.ld`. The host
  library does not compile it in.
- At run time the kernel text is **not** placed after the firmware. Blackhole idle-ETH kernels are
  execute-in-place from the kernel-config ring: `bh_hal_idle_eth.cpp:98` (`CONTIGUOUS_XIP`),
  `bh_hal.cpp:405-406` (`DISPATCH_IDLE_ETH_KERNEL_CONFIG_BUFFER` = true),
  `impl/program/dispatch.cpp:388-391` (text offset inside the config buffer), and the firmware
  jumps to `kernel_config_base + kernel_text_offset` (`hw/firmware/src/tt-1xx/idle_erisc.cc:175-184`).
  That ring is `MEM_ERISC_KERNEL_CONFIG_SIZE` = 25 KB at `MEM_IERISC_MAP_END` = 0xFD30
  (`dev_mem_map.h:232`, `bh_hal_idle_eth.cpp:56,78`).
- The real capacity is still checked at run time: `impl/program/program.cpp:2280-2287` `TT_FATAL`s
  if a program (RTAs + text) exceeds the config buffer. So relaxing the link bound cannot cause a
  silent overflow.

Fix: the JIT links kernels with the **pre-generated** linker scripts under
`runtime/hw/toolchain/blackhole/` read from the runtime root (`bh_hal.cpp:224,248`,
`jit_build/build.cpp:383`). In `kernel_ierisc.ld:88` and `kernel_subordinate_ierisc.ld:88` the
bound is the literal `LONG((24 * 1024)`. The overlay replaces those two files with copies where it
is `LONG((32 * 1024)`, which raises the link limit to 0x4ab0 (19,120 B). That covers the 0x3124 B
dispatch kernel with room to spare and stays below the 23,056 B safe part of the ring (next point).
No tt-metal rebuild. The firmware is unaffected (it links with `firmware_ierisc.ld`).

Caveat at 437bc366: the idle-ETH `UNRESERVED` base is computed from the *active*-ETH map
(`bh_hal_idle_eth.cpp:57-58`: align(0xF330 + 25 KB) = 0x15740), while the idle-ETH config ring
runs 0xFD30..0x16130. The last 2,544 B overlap. Upstream fixed this in 264c8fb9 (#54963). It is
harmless here as long as the dispatch core's config use stays under 0x15740 - 0xFD30 = 23,056 B
(we need ~12.6 KB text plus a few hundred bytes of args). The 32 KB link limit keeps any kernel
under 19.1 KB, so that holds.

Other options, not needed:
- Feature trim or opt level: the kernel is already `-Os`. It would need a 13% (1,648 B) cut, and
  1 CQ makes it larger (0x319c, #387). No compile-time switch does that.
- Splitting prefetch/dispatch: they already run on separate cores. Each is 0x3120/0x3124 B alone.
- Relayout of the idle-ERISC memory map (moving `MEM_IERISC_*` bases): needs a libtt_metal
  rebuild and host/device consistency, so it is riskier and unnecessary.
- Active ERISC: `core_descriptor.cpp:251` explicitly excludes active-link cores from dispatch. A
  standalone p150 has no active links anyway. Not an option, and not needed.

## (c) The p100a measurement box

The p100a has no usable Ethernet cores (#383 `render/host/dispatch_select.h:51-55`; #387 found
the card file reads `p100a`, and `auto` resolves to worker). ETH dispatch cannot run there, with
or without the patch. **The gain is p150-only** (also p300, not in our pool). Every A/B must run on
bh-30 under the 2026-10-07 viewer exception. The p100a stays on worker. Its numbers do not change.

## Risks

1. **Untested upstream (main risk).** No upstream test runs ETH dispatch on Blackhole: the
   fixtures only pick ETH on multi-chip Wormhole (`tests/tt_metal/tt_metal/dispatch/multi_command_queue_fixture.hpp:63-71`),
   and with both blockers no p150 could have opened in this mode. A third problem (a hang in
   the prefetcher/dispatcher on BH ERISC, NoC or multicast issues from ETH, wrong go-signal
   routing) is possible. A dispatch hang on bh-30 also takes the viewer down until the device is
   reset.
2. **ETH dispatch may be slower** than Tensix-worker dispatch (ERISC fetch/issue rate). Our
   pipeline launches many programs per view. If the per-launch cost rises, it eats into the
   modelled 0.50 ms (#382: 656.5 core-ms / 120 vs / 110).
3. The idle-ETH UNRESERVED/config overlap above (bounded, see (b)).
4. JIT cache: the overlay must use its own `TT_METAL_CACHE`, never the viewer's
   (`/localdev/smarton/viewer/tt-metal-cache`), so a relinked ELF never mixes with the viewer's build.

## Follow-up specs

### F1 (code, no device): overlay v2 + t383 on the opt tip
Branch `ttp/t391-eth-dispatch-overlay` from `origin/smarton/tt-project-opt`, in its own worktree of
~/dev/gstt2 (never edit ~/dev/gstt2). (1) Merge `ttp/t383-ethernet-dispatch-120-core-grid-on-the-p`
(f70118da) with a merge commit, no rebase. The default stays `worker`. (2) Copy
`docs/eth-dispatch-t387/make_overlay.sh` to `opt/eth/make_overlay.sh` and extend it: besides the yaml
cut (N = live ETH count, default 12), descend into `runtime/hw/toolchain/blackhole` the same way and
replace `kernel_ierisc.ld` and `kernel_subordinate_ierisc.ld` with copies where
`sed 's/LONG((24 \* 1024)/LONG((32 * 1024)/'` changed exactly one line each. The script must fail if
the pattern count is not exactly 1 per file, and print both diffs. (3) Add a run wrapper
`opt/eth/env.sh` that exports `TT_METAL_RUNTIME_ROOT=<overlay>` and `TT_METAL_CACHE=<own dir>`
and refuses to run if either equals the viewer's paths. (4) Unit tests stay green
(`tests/unit/test_dispatch_select.cpp`). Do not change tt-metal anywhere, do not push to tt-metal,
no PRs, no force push. Hand off with branch and head.

### F2 (device A/B on bh-30, after F1)
Run under the 2026-10-07 viewer exception. Hold resource `viewer` exclusive. No ird
reserve/extend/release. Use bh-30 only because no other free p150 exists. Before any sync, run
`ssh-preflight` (never disable host-key checking, and override the user's `StrictHostKeyChecking no`).
Build in its own tree (`p150bench/tree391`) with `nice -n 19 ionice -c3`, `-j` at half the cores,
while the viewer keeps running. Build the overlay from the viewer's tt-metal (read-only) into its own
dir, with its own `TT_METAL_CACHE`. Never touch the viewer's tt-metal, venv or cache.
Stop the viewer only for the bench window, restart it right away, check localhost:8091 = 200, and
tell the user the stop/start times.
Steps: (0) overlay sanity: open the device with `GSPLAT_TT_DISPATCH=eth` and check that the log
says `compute grid 12x10` and that dispatch kernels load (no `overflows region` and no
`too large for kernel config buffer`). Use a 120 s timeout. If it hangs, follow the tt-device
recovery ladder, restart the viewer, record the hang and stop (F2 = failed, with the log).
(1) Run 3 alternating rounds W/E/W/E/W/E of the bicycle 30-view bench, untraced,
`GSPLAT_PER_VIEW_STAGES=1`, both arms with the same tree and overlay root (worker through the overlay
too). Need md5 906e0435 30/30 in every round. (2) Take one eth-arm Tracy run with the per-core busy
over 120 cores (`docs/eth-dispatch-t387/ana210.py`). (3) If the eth arm is faster by at least 1%
on the mean of rounds: save the hero.png from the eth arm on device, a diff image and PSNR against
`benchmarks/reference_v2/hero.png`, and look at the image for tile artifacts (all 120 cores,
especially the new 12th column). Then add an iters.jsonl row + REPORT.html and a best-iter tag. Only
after that may `GSPLAT_TT_DISPATCH=auto` become the default, and only for p150/p300 cards (the p100a
stays worker). If the eth arm is not faster, record why from the Tracy data (dispatch latency
vs compute) and keep `worker`.

## Upstream note (not acted on)
tt-metal: (1) `core_descriptor.cpp` should drop harvested ETH cores from `dispatch_cores` (or the BH
eth-dispatch yaml should list only ≤12). (2) For XIP/config-buffer idle-ERISC kernels on Blackhole,
the `main.ld` text bound should be the kernel-config size, not `24 KB - firmware text`. As written,
the cq_dispatch kernel (12.3 KB) cannot load on BH idle ERISC even though the 25 KB ring has room.
