# AFR on 2 chips: host path (task #581)

2026-10-12 PDT. Lever L2 of docs/structural-plan-2/PLAN.md (bf864320, ttp/t578-structural-plan-2).
CPU only: compiled and unit-tested, **no device run yet**. Every number below is an estimate
from existing single-chip measurements, not a measurement of AFR.

## What it does

`GSPLAT_TT_AFR=1` (default off) with `render/run.py --back-to-back`: chip A renders the views at
even sweep positions (0, 2, ..., 28), chip B the odd ones (1, 3, ..., 29). Each chip has its own
DeviceState, programs, JIT-cached kernels and B4a fold chain. A chip's next view is two sweep
positions on, so the fold carries N -> N+2 on the same chip, and `GSPLAT_TT_XVIEW_OVERLAP`
hints the chip's own next view. Nothing crosses between the chips. Off (the default), run.py
and the C++ code take exactly the old path: the new hooks are `None` and `GSPLAT_TT_DEVICE_ID`
defaults to 0.

## Why one process per chip, not one thread per chip in one process

The plan asked for "two DeviceStates, one host thread per device". In this code base every
stage keeps process-wide state: `device_state.cpp` holds one MeshDevice, its buffer registry,
the K2 count rows and the sort->blend scalars, and the stage files hold about 250 statics or
anonymous-namespace objects (sort 82, tile_assign 48, gather 35, pfwc 29, blend 27, ...).
Making them per-device is the 4-6 day refactor the plan priced. One OS process per chip gives
each chip its own copy of all of them, and one host thread per device, without touching the
stages. The cost is a small coordinator protocol (below). AFR is a throughput mode, so nothing
needs both chips' state in one address space.

If MB1 (below) shows that two processes contend for something a single process could share,
the in-process version stays possible later. It would make `device_state::State` and the stage
statics per device id.

## Pieces

- `render/afr.py`: the view split (`split`, `merge`), the sweep md5 (`raw_md5`, the same
  formula as run.py), worker argv and env, the worker side (`WorkerLink`) and the coordinator
  (`coordinate`, `summary_line`).
- `render/run.py`:
  - `_afr_coordinate` runs when `GSPLAT_TT_AFR=1` and `--afr-worker` is absent. It opens no
    device. It refuses latency mode (exit 2): AFR latency is the single-chip latency.
  - `--afr-worker K/N` (hidden) makes a worker. It renders `order[K::N]`, and `_back_to_back`
    gets the `afr` link and the sweep positions `idx` for the dump file names.
- `render/host/device_state.cpp`: `GSPLAT_TT_DEVICE_ID` picks the device (default 0). The
  auto-dispatch card-type probe reads the sysfs entry of the first `TT_VISIBLE_DEVICES` id when
  that is set (`dispatch::card_index`). The `[DEV]` line now names the device and the card.

## Protocol and timing

1. The coordinator starts worker 0 and waits for its `ready`. A worker sends `ready` after its
   warm-up hero render and the BRISC stack check, before pass 0. Only then does the
   coordinator start worker 1, so the two never JIT-compile into the shared `TT_METAL_CACHE`
   at the same time. Worker 1 finds every kernel cached.
2. For each pass p (1 check pass + `--b2b-warmup` + `--b2b-passes`), the coordinator writes
   `GO p` to every worker's stdin. Each worker renders its 15 views and sends
   `pass {p, t0, t1, digests on p = 0}`. t0 and t1 come from `CLOCK_MONOTONIC`, which every
   process shares. Python's `time.monotonic` is per process on macOS, and the CPU test caught
   that.
3. Pass period = (max t1 - min t0) / 30, the union of the two chips' windows. The headline
   `ms_frame_median` is the median over the measured passes, as on one chip.
4. Workers send `done {rc, hero_md5}`. The coordinator prints one `B2B ...` line in run.py's
   format, plus these fields:
   - `afr_chips`, `afr_devices`
   - `afr_chip_ms_own_view`: each chip's ms per view it rendered
   - `afr_hero_equal`: both chips' hero frames match pixel for pixel. Chip A's hero is from its
     check pass; chip B's is its warm-up render.

   It then prints the `TTW_TIMING b2b_ms_frame(_median)` lines.

   Worker output is forwarded with an `[afrK] ` prefix, so parsers see one `B2B ` line.

Exit codes:

- a worker's own non-zero code: 5 for frames that differ across passes, 9 for the BRISC stack
  check.
- 8: a worker ended or broke the protocol early. The other worker is terminated by its own
  Popen handle.
- 5: the chips' hero frames differ.
- 2: a bad configuration.

Events are taken in order per worker. One chip's `done` can arrive before the other chip's
last `pass`, and the CPU test caught that race.

## Expected numbers (estimates, labelled; to be measured by the device task)

Base: best-iter-226, 8.615 ms/view b2b on one chip (bh-33, p100a). Per-view device work does
not change, so each chip's period per own view stays about 8.6 ms.

| term | estimate ms/view (over all 30 views) | source |
|---|---|---|
| two chips in parallel | 8.615 / 2 = **4.31** | per-chip work unchanged |
| even/odd imbalance (pass barrier waits for the slower chip) | +0.03 | docs/zerocopy-out/out/run-r3-zc.log: even views 170.4 ms vs odd 168.2 ms of 338.6 (+0.65 % for the slower half) |
| pass edge (each chip starts 2x as many passes per view; first view has no previous view to hide pfwc behind) | +0 to +0.06 | worst case 1.7 ms of pfwc per pass edge (docs/b2b-gap) / 30; the XVIEW hint across passes already hides most of it |
| coordinator (2 pipe writes + 2 JSON reads per pass, ~0.1-0.3 ms per pass) | +0.01 | one host round trip per 15 views per chip |
| **host serial work per view** (anything one chip's host work waits on the other's) | **~0.01-0.04** | separate processes: no GIL, no shared host thread. Only the per-pass barrier and coordinator are serial. The plan's 0.1-0.3 assumed one process. |
| **period, total** | **~4.35-4.45** (plan bound 4.4-4.6); gate <= 5.0 | |

Host CPU: each worker keeps its own host work per own view, about 1.3 ms of rtargs, bin
layout and emit, publish and pybind (B2B_STAGES in docs/vis-fast-t494/out/run-r4-off.log),
overlapped with its device. That is about 15 % of one core per chip, plus tt-metal's dispatch
threads. A 2-board host needs about 2 free cores per chip, which any IRD host has.

**PCIe D2H.** The output is 1024x1024 RGB, about 3.1 MB per view. The device writes it straight
into pinned host memory (zero-copy ring, out_ring.h). The host-side `d2h` stage is 0.009 ms
per view.

- At a 4.4 ms period: 3.1 MB / 4.4 ms = **0.71 GB/s in total**, two images (one per chip)
  every 8.8 ms, about 0.36 GB/s on each card's own PCIe link.
- A p150 link is Gen5 x16, and tt-metal's practical D2H is several GB/s per card. That puts
  each link at about 5 % utilisation, and the shared root complex and host DRAM at about
  0.7 GB/s.
- Contention is not expected to show. MB1 checks it: if `afr_chip_ms_own_view` climbs above
  the single-chip 8.6 ms, the extra is shared-host cost (PCIe, DRAM, CPU).

Memory: keep mode holds 15 + 2 output slots of 3.1 MB per worker
(`GSPLAT_TT_OUT_ZEROCOPY_SLOTS` = own views + 2), about 53 MB per chip.

**Viewer.** Not changed. Its page FPS is bound by the 12 MB/s link (plan L2 risk), and its
latency per frame stays 8.6 ms with one more frame in flight.

## Labeling

A 2-chip number is reported as "2 chips (AFR)" against the 1-GPU published reference
(INRIA 3DGS, RTX A6000, 10.75 ms/view, **published, not measured**). Whether such a number
can be the headline is the user's call (plan L2).

## How the device task runs it

- Same build as usual. The C++ change is in `device_state.cpp` and `dispatch_select.h`, so the
  render module must be rebuilt.
- Env as docs/land-t574/remote_ab.sh. `TTW_DEVRUN=1` is inherited by the workers.
- **Single chip** (baseline on the same host, chip 0):

  ```
  TT_METAL_CACHE_RENDER=$cache python3 render/run.py --no-ref --back-to-back --iter-dir afr-1chip
  ```

  For the other chip: `GSPLAT_TT_DEVICE_ID=1`, or `TT_VISIBLE_DEVICES=1`.
- **2 chips:**

  ```
  GSPLAT_TT_AFR=1 GSPLAT_TT_AFR_DEVICES=0,1 TT_METAL_CACHE_RENDER=$cache \
      python3 render/run.py --no-ref --back-to-back --iter-dir afr-2chip
  ```

  Allow a timeout of about 2x the single-chip one: two device opens and two warm-ups, run one
  after the other.
- **Per-view md5:** add `--dump-views <name>`. In AFR mode the check pass of each worker writes
  its frames to `tmp/<name>/view{i:02d}_{view}.png`, with i the sweep position, so the folder
  holds the same 30 file names as a single-chip run. `md5sum *` + `opt/md5_golden.py` work
  unchanged. `raw_md5` in the B2B line is the same formula as on one chip, so it must equal
  the single-chip b2b `raw_md5`.
- **Hero:**
  - chip A: `tmp/<iter-dir>/hero_clean.png`
  - chip B: `tmp/<iter-dir>/afr1/hero_clean.png`, its warm-up render of the same view
  - `afr_hero_equal=yes` means they match pixel for pixel. Take PSNR and the diff against
    `benchmarks/reference_v2/hero.png` from chip A's file.
- **Device selection:** by default worker K runs with `TT_VISIBLE_DEVICES=<K-th id of
  GSPLAT_TT_AFR_DEVICES>` and `GSPLAT_TT_DEVICE_ID=0`, so neither process opens the other's
  chip. If this tt-metal ignores `TT_VISIBLE_DEVICES`, the `[afrK] [DEV] device ... (card ...)`
  lines show it. Then set `GSPLAT_TT_AFR_VISIBLE=0`, and the worker uses
  `GSPLAT_TT_DEVICE_ID=<id>` with all chips visible.
- **Dispatch:** on a p150, `auto` picks ETH dispatch only when `TT_METAL_RUNTIME_ROOT` is an
  opt/eth overlay (else worker; see dispatch_select.h). Under ETH, the B4a fold defaults off
  (t562 hang). For the fold-on numbers comparable with best-iter-226, set
  `GSPLAT_TT_DISPATCH=worker`. Report ETH separately once the fold fix lands. ETH cores with
  an active link between the two boards are not available for dispatch, and the `[DEV]` line
  shows the grid.

## Microbenchmark MB1 (for the device task)

1. Run single-chip b2b on chip 0 and on chip 1.
2. Run AFR b2b.
3. Compare `afr_chip_ms_own_view` with each single-chip `ms_frame_median`. Any difference is
   cost from the two chips sharing the host (PCIe, DRAM, CPU).
4. Compare the AFR `ms_frame_median` with half the single-chip value. Any difference is the
   barrier, imbalance and coordinator cost.

## Tests (CPU)

`tests/unit/test_afr.py` (11 tests):

- the split and merge keep the sweep md5
- the period formula
- device, flag and worker parsing
- worker env and argv
- the coordinator against two fake worker processes:
  - the stagger
  - the merged md5 and the prefixed output
  - a worker that crashes before ready stops the other
  - differing hero frames give rc 5
- run.py `_afr_coordinate` end to end with an injected Popen
- `_back_to_back`'s worker hooks: GO per pass, the pass events, dump names by sweep position

`tests/unit/test_dispatch_select.cpp` covers `device_id_from_env` and `card_index`. The
existing `test_run_back_to_back.py`, `test_run_xview.py` and `test_run_error_exit.py` still
pass (AFR off path).
