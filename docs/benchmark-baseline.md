# Bicycle reference benchmark — reserve, run, baseline

Canonical, repeatable path for the 30-view bicycle benchmark used as the
optimization reference. Every performance claim in the status HTML must come
from this path.

## 1. Reserve a box

All clusters are reached through the `yyz-ird` management host.

```bash
ssh yyz-ird "ird list"                    # active reservations
ssh yyz-ird "ird list-machines"           # model + free-board count per machine
```

`ird list-machines` board column reads `model(free|total)`; the trailing column
lists the free chip ids. `p150(0|4)` means **zero** free p150 boards.

Reserve (flags before the `blackhole` positional, `--machine`/`--model` after it;
`--no-shell` is required or the command blocks on an interactive docker shell):

```bash
ssh yyz-ird "ird reserve --cluster tt_yyz --ssh-port 46483 --timeout 14:00:00 \
               --no-shell blackhole --machine yyzo-bh-07"
cd ~/dev && ./tt-workflows/devsync        # writes ~/.devsync/ssh-hosts, extends, syncs
ssh yyzo-bh-07 hostname                   # smoke test
```

Release with `ird release <SELECTION ID>` (the id from `ird list`, **not** the job id).

### Hardware reality check (2026-09-30)

- `yyzo-bh-07` — the box all recent optimization iterations (ttw-060…141) ran on —
  is a **p100a**, not a p150 (`tt-smi -s` → `board_type: p100a`).
- Only four p150 machines exist across all clusters: `bh-30` (tt_aus),
  `bh-48`/`bh-49`/`bh-50` (tt_bgd). At the time of writing none were obtainable:
  bh-30 had 0 of 4 boards free, bh-48 drained, bh-49 0 of 1 free, and every
  `ird reserve --cluster tt_bgd …` crashes inside ird
  (`utils.py:244 get_nodes_with_gres → ValueError: not enough values to unpack`).
- A queued p150 request sits at `Reason=Priority` with `TRES=…,mem=515000M`
  (ird asks for the node's whole memory), so it only starts when a p150 node is
  fully free.

**Therefore the baseline below is a p100a number.** It is the same box and the
same environment every prior iteration was measured on, so it is comparable with
the existing ledger — but it is not a p150 number. Re-measure on bh-30 before
publishing any p150 claim.

## 2. Build state on the box

`/localdev` is local xfs and survives reservation churn, so `yyzo-bh-07` still
carries the full prior tree:

```
/localdev/smarton/gstt2 -> /localdev/smarton/gstt2-clone   (branch smarton/stage2-hostfree-l1)
/localdev/smarton/tt-metal                                  (TT_METAL_HOME)
/localdev/smarton/gstt2/.venv                               (python 3.10)
/localdev/smarton/gstt2/render/build-tt                     (CMAKE_BUILD_TYPE=Release)
```

Rebuild only when sources changed:

```bash
cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release
cmake --build render/build-tt -j 16                 # -> render/render_clean.cpython-310-*.so
cmake --build build-avx2-tt --target _gsplat_cpu -j 16   # CPU reference (needed for PSNR only)
```

## 3. Run the benchmark (one command)

Device jobs go through `devrun.sh`, which takes the per-host device lock and
refuses timeouts above the 600 s reservation ceiling.

```bash
cd ~/dev/gstt2
~/dev/tt-workflows/scripts/devrun.sh --no-verify --timeout 580 --tag bench -- \
  "export TT_METAL_HOME=/localdev/smarton/tt-metal \
          TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
          TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100; \
   cd /localdev/smarton/gstt2; source .venv/bin/activate; \
   python3 render/run.py --iter-dir bench"
```

Add `--no-ref` to `run.py` to skip the CPU reference render (timing only; the
PSNR column then reports `nan` for `hero_vs_cpu`). Four `--no-ref` repeats fit
comfortably inside one 580 s reservation.

Output line:

```
SUMMARY scene=bicycle hero='hero' hero_vs_ref=100.00dB(8bit-vs-golden) \
        hero_vs_cpu=63.95dB(float-vs-cpu) avg_frame_ms=173.0 p50_ms=174.9 \
        min_ms=142.0 max_ms=210.7 n_views=30 warmup_s=2.2
```

`avg_frame_ms` over all 30 `cameras_v2.json` views (warmup excluded) is the
headline metric. Never quote the hero frame time or the amortized `ms/view`.

## 4. Recorded baseline (2026-09-30)

| field | value |
|---|---|
| host | `yyzo-bh-07`, board `p100a`, board_id `000004323191b005` |
| firmware | fw_bundle `19.12.0.0`, cm_fw `0.34.0.0`, dm_app_fw `0.28.0.0`, gddr_fw `2.16` |
| driver / tools | TT-KMD 2.9.0, tt-smi 5.2.0, tt_umd 0.9.5, pyluwen 0.8.5 |
| OS | Ubuntu 22.04.5, kernel 5.15.0-136 |
| repo | `smarton/stage2-hostfree-l1` @ `e28b5f93d19999a141b6873458f1dbfe95ac8178` |
| artifact | `render/render_clean.cpython-310-x86_64-linux-gnu.so` md5 `bcc3f2480f25875f8b7fbd2c7df0f81f` |
| tt-metal | `/localdev/smarton/tt-metal` @ `e77780fe4f6c86bd7de9e71cc499923da46f15f3` (describe `0.1.0`) |
| build flags | `CMAKE_BUILD_TYPE=Release`, empty `CMAKE_CXX_FLAGS` |

Five consecutive 30-view runs:

| run | avg_frame_ms | p50 | min | max |
|---|---|---|---|---|
| 1 (with CPU ref) | 173.0 | 174.9 | 142.0 | 210.7 |
| 2 | 173.5 | 175.3 | 142.3 | 211.3 |
| 3 | 173.3 | 175.1 | 142.3 | 211.0 |
| 4 | 173.1 | 175.0 | 141.9 | 210.8 |
| 5 | 173.6 | 175.2 | 141.7 | 212.5 |

- **mean 173.30 ms/frame, stdev 0.26 ms (0.15 %), range 173.0–173.6**
- **5.77 FPS** at the bench resolution
- per-view spread inside a run is much larger than run-to-run spread
  (142–212 ms, i.e. ±20 % view-to-view) — view content, not measurement noise
- quality: `hero_vs_ref = 100.00 dB` (8-bit output bit-identical to the golden
  image), `hero_vs_cpu = 63.95 dB` — matches the 63.85 dB anchor
- reproduces the recorded iter-140/141 plateau (173.1 / 173.3 ms) exactly

Anything below ~0.5 % is inside the noise band; a change must move
`avg_frame_ms` by more than that to be called a win.

## 5. What the current profiling tooling cannot see

Measured gaps, from the run-1 log (`~/dev/gstt2/.ttw/logs/t3-run1-*.log`):

1. **No per-stage split in the default run.** `context.md` promises
   `TTW_TIMING proj/ta/sort/cull/blend`; the run actually emits only
   `TTW_TIMING ms_view=` and `TTW_TIMING blend=`, both set to the same
   `avg_frame_ms`. So `blend` is not a blend measurement — it is the whole
   frame. The only genuine host-side stage timer is the `[SORT]` line
   (`bin` / `publish` / `total`, ~38 ms of the ~173 ms frame).
2. **No project / tile-assign / cull / blend host timers at all.** Roughly
   135 ms of every frame (78 %) is unattributed in the default log.
3. **No device-vs-host split.** Nothing in the default run distinguishes
   silicon time from dispatch, H2D/D2H and Python overhead. Only a Tracy
   capture gives device zones, and only through
   `python -m tracy --dump-device-data-mid-run` (`opt/profiler/capture_tracy.sh`)
   because gsplat never closes the device — a plain `capture-release` yields
   host/JIT-warmup zones only.
4. **No per-core / per-RISC utilization.** Nothing reports which of the ~110
   blend cores are idle, or the NCRISC-vs-TRISC-vs-BRISC balance, without a
   dedicated Tracy or DPRINT instrumentation pass.
5. **Tracy is not cheap.** The mandated capture is the full 30-view render
   (~1 M device-zone rows, ~16 MB csv) and it does not fit the 600 s devrun
   ceiling as a single job, so it has to be chunked.
6. **Structural counters exist but are not timers.** `[OVERFLOW-DIST]` and
   `[SUBCHUNK]` report pair counts, overflow tiles and payload pages per view —
   useful for reasoning about work volume, but they carry no time.

First profiling task for the optimization loop: add real per-stage host timers
(project / gather / tile-assign / sort / cull / blend) to `render/host/render.cpp`
and emit them as `TTW_TIMING <stage>=`, so the 135 ms of unattributed frame time
gets a name before any kernel is touched.
