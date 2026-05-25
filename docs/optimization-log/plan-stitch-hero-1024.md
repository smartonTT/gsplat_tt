# Plan — stitch hero @ 1024×1024 (Blackhole bh-30)

**Target:** end-to-end **5–10 ms/frame** (vs 981 ms Step-0 baseline, 1.75 ms realistic kernel peak).
**Device:** Blackhole P150 (`bh-30`, tt_aus).
**Benchmark:** `scripts/render_fixed.py stitch hero --resolution 1024x1024 --warmup 3 --frames 10`.

## IRD storage (read this first)

| Cluster | Host example | devsync target | Persistent? |
|---------|--------------|----------------|-------------|
| tt_yyz  | yyzc-wh-03   | `/proj_sw/user_dev/smarton` | weka — survives container restart |
| tt_aus  | bh-30        | `/localdev/smarton` | local NVMe — survives **container** restart, **not** reservation release |

**Never treat IRD-only paths as durable.** `/tmp`, ephemeral `/proj_sw/...` on tt_aus, and anything not under `~/dev` on the Mac can vanish when a container restarts or a reservation is released.

All iteration artifacts (PNGs, timing JSON, profiler summaries, diff images) **must land under `benchmarks/runs/` in the devsync-mirrored repo tree** — write them there on the IRD box (`/localdev/smarton/gsplat_tt/benchmarks/runs/...` on bh-30) or locally under `~/dev/gsplat_tt/benchmarks/runs/`. Mutagen syncs both directions automatically.

### Artifact layout (mandatory)

```
benchmarks/runs/
  step-000-baseline/
    stitch_hero.png
    stitch_hero.json          # render_fixed --json timings
    stitch_hero_diff.png      # if applicable
    profile_summary.json      # optional: parsed profiler CSV
  step-001-binary-ipc/
    ...
  iter-NNN-<name>/            # kernel-only iters @ 480×640 (archive)
    ...
```

**Rules for supervisor + workers:**

1. `--out`, `--amplified-diff`, `--json` → paths under `benchmarks/runs/<step-or-iter-id>/`, never `/tmp` on IRD.
2. Before reading artifacts on the Mac after a remote bench: `devsync is-finished bh-30` (waits for mutagen to finish).
3. Iter markdown in `docs/optimization-log/` references artifact paths relative to repo root.
4. Profiler CSV: copy a one-line summary JSON into the run dir; full CSV optional under same dir.

## Steps

| Step | Goal | Est. saving | Status |
|------|------|------------:|--------|
| **0** | 1024×1024 baseline + theoretical peak + viewer | — | **DONE** |
| **1** | Binary IPC + reused DRAM buffers | −100 ms total | **DONE** (−10.3%) |
| **2** | Static attrs in DRAM; gather by gid in reader | −200+ ms prep | in progress |
| **3** | Persistent kernel + mailbox dispatch | −25–80 ms daemon | pending |
| **4** | DST-resident R/G/B/T accumulators | −3–7 ms on-chip | pending |
| **5** | 16×16 tiles (conditional) | −3–10 ms | pending |
| **6** | Device-side `project_gaussians` (stretch) | −5–15 ms | pending |

## bh-30 workflow (tt_aus)

```bash
ssh yyz-ird
ird connect-to 2          # selection id from `ird list`
cd /localdev/smarton/gsplat_tt
source venv/bin/activate
export TT_METAL_HOME=$PWD/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$PWD/backends/tt/tt-metal
```

First seed on a fresh bh-30: from Mac, `devsync refresh bh-30` (after `devsync.conf` `[remote_root]` maps `bh-*` → `/localdev/smarton`).

## Supervisor loop

- Profile → pick idea → dispatch worker (Composer 2.5) → gate visuals → keep/reject → **save artifacts to `benchmarks/runs/`** → update `SUMMARY.md`.
- Check subagent status every ~5 min on long tasks.
- Keep web viewer running on bh-30; only brief restart when binary changes.
