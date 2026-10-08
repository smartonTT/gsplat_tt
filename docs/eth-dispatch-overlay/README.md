# ETH dispatch overlay v2 (task #392, 2026-10-07)

Run-time-only route to Blackhole Ethernet dispatch on the p150: 12x10 = 120 compute cores instead
of 11x10 = 110, modelled at about -0.50 ms/view (#382). No tt-metal source edit or rebuild.
Background and line references: `docs/eth-dispatch-patch/README.md` on
`ttp/t390-eth-dispatch-patch-feasibility` (#390), first attempt `docs/eth-dispatch-t387/` on
`ttp/t387-ethernet-dispatch-120-core-grid-bh-30-p1` (#387).

The gain is p150-only. The p100a has no usable Ethernet cores and stays on worker dispatch.

## What is on this branch

- `GSPLAT_TT_DISPATCH=worker|eth|auto` from #383 (f70118da, merged with a merge commit).
  Default `worker`; `eth` opts in; `auto` picks eth only on p150/p300 cards. At device open the
  log prints `[DEV] dispatch <worker|eth> (...), command queues N, compute grid XxY`.
- `opt/eth/make_overlay.sh <tt-metal dir> <overlay dir> [N=12]` builds a `TT_METAL_RUNTIME_ROOT`
  overlay of a read-only tt-metal. Everything is a symlink into the tt-metal dir except three real
  files:
  - `tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml`: every 14-core dispatch list
    `[0,0]..[0,13]` cut to `[0,0]..[0,N-1]` (#387 blocker 1: stock lists name harvested ETH
    cores and throw at open).
  - `runtime/hw/toolchain/blackhole/kernel_ierisc.ld` and `kernel_subordinate_ierisc.ld`:
    `sed 's/LONG((24 \* 1024)/LONG((32 * 1024)/'`, exactly one line each (#387 blocker 2: the
    idle-ERISC kernel text link bound of 0x2ab0 B; cq_dispatch needs 0x3124 B. The kernel runs in
    place from the 25 KB config ring and `program.cpp` still fails loudly on a real overflow).

  It fails (exit 1) unless each `.ld` changes exactly one line and every 14-core yaml list is cut,
  and prints all diffs. It builds into a temp dir and swaps it in only on success, so a failed run
  leaves the old overlay alone; reruns give the same tree and output. It refuses (exit 2) an overlay
  under `/localdev/smarton/viewer` (the live viewer), inside the tt-metal dir, or over a non-empty
  dir it did not make (marker `.gsplat-eth-overlay`). It never writes into the tt-metal dir: the
  real files are made only after their parent dirs are real dirs in the overlay, and it checks that.
- `opt/eth/env.sh <overlay> [<cache>]`, sourced or as `bash opt/eth/env.sh <overlay> [<cache>] -- cmd`:
  exports `TT_METAL_RUNTIME_ROOT=<overlay>`, `TT_METAL_CACHE=<cache>/prod` and
  `TT_METAL_CACHE_RENDER=<cache>/render` (run.py moves the latter into `TT_METAL_CACHE`). Default
  cache `<overlay>-cache`. Refuses an overlay or cache under `/localdev/smarton/viewer`, the shared
  default caches (`/localdev/smarton/.cache/tt-metal-cache{,-render,-viewer}`), a cache inside the
  overlay, and a dir that is not an overlay. It does not set `GSPLAT_TT_DISPATCH`: both A/B arms
  run through the same overlay and cache.
- `opt/eth/test_eth_overlay.sh`: local test on a fake, read-only tt-metal tree (no device).

## Tests (local, no device)

    bash opt/eth/test_eth_overlay.sh          # 67 checks, "ALL PASSED"
    CXXFLAGS=-Irender/host bash tests/unit/run_cpp.sh tests/unit/test_dispatch_select.cpp

Also run once, read-only, against the vendored tt-metal on the Mac
(`backends/tt/tt-metal`, same `437bc366` files as the boxes): 6 yaml lists cut to 12 (lines
16, 27, 40, 51, 64, 75), line 88 changed in both `.ld` files, source files unchanged.

## Device A/B (follow-up task, bh-30 under the 2026-10-07 viewer exception)

Rules restated: hold resource `viewer` exclusive; no `ird reserve/extend/release`; run
`"$TTP_PROJECT/harness/bin/ssh-preflight" bh-30` first; every ssh uses
`-o BatchMode=yes -o StrictHostKeyChecking=yes` (the user's ssh config says `no` for bh-*), never
disable host-key checking. Build next to the running viewer with `nice -n 19 ionice -c3` and `-j`
at half the cores. Never touch `/localdev/smarton/viewer` (its tt-metal is only read through the
overlay's symlinks), `/localdev/smarton/gstt2` or `/home`. Stop the viewer only for steps 0-2 and
restart it right away; check localhost:8091 answers 200 and tell the user the stop/start times.

Sync and build a tree of this branch's head into `/localdev/smarton/p150bench/tree392` the way
`docs/eth-dispatch-t387/sync_bh30.sh` does (copy of an earlier p150bench tree, then the git diff,
then `cmake --build render/build-tt`), under `ttp lock p100 --`.

**Step 0 (overlay sanity, 120 s timeout), on bh-30 with the viewer stopped:**

    P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out392
    cd $T && source .venv/bin/activate && mkdir -p $O
    bash opt/eth/make_overlay.sh $V/tt-metal $P/ttm-eth12v2 12 2>&1 | tee $O/overlay.log
    export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
    source opt/eth/env.sh $P/ttm-eth12v2 $P/cache392
    GSPLAT_TT_DISPATCH=eth timeout 120 python3 render/run.py --no-ref --view-range 0:1 --iter-dir t392-s0 > $O/s0.log 2>&1; echo "s0 rc=$?"
    grep -E '^\[DEV\] dispatch|overflows region|too large for kernel config buffer|No core coordinate|TT_THROW|TT_FATAL|Traceback' $O/s0.log | cut -c1-300

Pass: rc 0 and `[DEV] dispatch eth (eth (forced)), command queues 1|2, compute grid 12x10`
(2 CQs with the default `GSPLAT_TT_MAT_CQ1`), with none of
`overflows region`, `too large for kernel config buffer`, `No core coordinate`. A timeout (rc 124)
is a dispatch hang: follow the tt-device recovery ladder (`tt-smi -r`), restart the viewer, record
the log and stop (A/B = failed). Any `TT_FATAL` stops the A/B the same way, with the log.

**Steps 1-3** as in #390 F2: 3 alternating rounds W/E/W/E/W/E of the untraced 30-view bench
(`GSPLAT_PER_VIEW_STAGES=1`, `--no-ref --dump-views`, md5 906e0435 30/30 every round), both arms
with the same tree, overlay and cache (`GSPLAT_TT_DISPATCH=worker` vs `eth`, env as in step 0);
then one eth-arm Tracy run with per-core busy over 120 cores (`docs/eth-dispatch-t387/ana210.py`).
`docs/eth-dispatch-t387/bench387.sh` is the template: replace its `TT_METAL_RUNTIME_ROOT` and
`TT_METAL_CACHE_RENDER` lines with `source opt/eth/env.sh $P/ttm-eth12v2 $P/cache392`. If eth wins
by at least 1% on the mean of rounds: device hero.png from the eth arm, diff image and PSNR
against `benchmarks/reference_v2/hero.png`, a visual check for tile artifacts (all 120 cores,
especially the 12th column), an iters.jsonl row + REPORT.html and a best-iter tag. Only then may
`auto` become the default, for p150/p300 cards only. Otherwise record why from the Tracy data and
keep `worker`.

## Incident during this task (fixed)

The first draft's directory walk returned early for `tt_metal/core_descriptors` because it was
reachable as a directory through the overlay's `tt_metal` symlink, and then wrote the yaml through
that symlink: it truncated the vendored copy on the Mac
(`backends/tt/tt-metal/tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml`, untracked).
It was restored byte-identical from `build_Release/libexec/tt-metalium/` (md5 98a13f24, all
sibling files identical between the two trees) with its original mode and mtime. The script now
realizes every parent first, checks the target is a symlink inside a real overlay dir before
replacing it, and the test runs on a read-only source tree and compares it before and after, so
this would fail the test. On bh-30 the same bug would have hit the viewer's tt-metal: run only
this version there.

## Upstream note (not acted on)

tt-metal: (1) `core_descriptor.cpp` should drop harvested ETH cores from `dispatch_cores`, or the
BH eth-dispatch yaml should list at most 12. (2) For idle-ERISC kernels that execute in place from
the config buffer on Blackhole, the `main.ld` text bound should be the kernel-config size, not
`24 KB - firmware text`.
