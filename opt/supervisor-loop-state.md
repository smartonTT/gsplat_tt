# Supervisor loop state (device-resident → 1ms north star)

Updated: 2026-05-31 (bh-30 **RECOVERED** via `tt-smi -r`; loop running again)

## bh-30 recovery (RESOLVED)

The "wedge" was a hardware/PCIe wedge, not a code bug: warmup + profiler sync
succeed, then the first real compute dispatch hangs forever (host pins one core
~100% in a `Finish()` busy-wait, `futex_wait_queue`). Clean HEAD hung identically.
**Fix:** `tt-smi` 5.2.0 lives in the venv (not on PATH):
`ssh bh-30 'pkill -9 -f a003_verify; sleep 2; rm -f /dev/shm/tt_*; /localdev/smarton/gstt2/.venv/bin/tt-smi -r'`
(~60s; confirm `tt-smi -s` DDR_STATUS 0x5555). Procedure also in `opt/sfpu-cull-next.md`.

Known cosmetic issue: every run ends with an **atexit segfault (exit 139)** inside
tt-metal `SharedMemoryStatsProvider::update_from_allocator` during program dtor —
it fires AFTER a valid `SUMMARY` prints. Treat the printed SUMMARY as truth; fix
(decouple script pass/fail from exit code + try TT_METAL env to disable the stats
provider) bundled into next device session.

## Baseline (committed HEAD `0f4e72e`)

- Full resident chain: project→gather→ta→sort→blend in DRAM
- 1-view hero post-reset: **hero_vs_ref 63.64 dB** (0.21 dB under historical 63.85 —
  user confirmed irrelevant), frame **proj=60.6 ta=119.6 sort=57.6 blend=422.8 ms**
- **blend=422.8ms = 64% of frame** — scalar soft-float cull on mover (dominant)

## Active tracks (parallel)

Device is a SINGLE SERIAL resource (one job at a time; parallel verifies wedge
via CHIP_IN_USE) and a SINGLE working tree. So: ONE device-owning worker on the
top lever at a time; supervisor does device-free, non-file-conflicting prep.

| # | Track | Owner | Status |
|---|--------|-------|--------|
| 1 | **SFPU cull** (#1 lever, blend 422→~110ms, ~310ms/view) | **Opus subagent** (owns bh-30) | **IN PROGRESS** — verifying per-tile `tile_regs_acquire` fix + working +32 race follow-ups (`opt/sfpu-cull-next.md`). Gate: deterministic ≥63.6 dB ×2 + blend ~110ms. Edits blend/cull files only. |
| 2 | **tile_assign 119ms (#2 lever)** | supervisor (device-free prep) | Identified host bridges: D2H tpg + host prefix-sum + K3 per-gaussian `log` precompute + H2D. Added `GSPLAT_TT_TA_TIMING=1` resident-mode sub-stage dump in `tile_assign_device.cpp` to get k1/prefix/k2/cull split BEFORE committing to the on-device scan port (§8 item 2). **Do not build the scan until timing says it's the bottleneck.** |
| 3 | **Sort device publish** | `sort_publish.cpp` + `GSPLAT_TT_SORT_DEVICE_PUBLISH=1` | Wired; gate when device frees |
| 4 | **Stage C2 payload S2a** | `payload_pack.cpp` + `GSPLAT_TT_BLEND_PAYLOAD=1` | Built on bh-30; S2b reader TBD; CONFLICTS with track-1 blend files — sequence after SFPU lands |

## Uncommitted local diff

- `microblock_cull_compute.cpp`: one `tile_regs_acquire`/tile, batch pack with commit/wait
- `blend_device.cpp`: SFPU diag env (`DUMPALL`, `ALLONE`)
- `reader_alpha_blend_mb_devcull.cpp`: diag hooks

## bh-30 ops

- Kill: `ps aux | grep '[a]003_verify' | awk '{print $2}' | xargs kill -9` — **never** `pkill -f a003_verify` (kills the ssh command line).
- Verify: `scripts/a003_verify.py` with resident exports (see `scripts/bh30_recover_and_verify.sh`).
- **Never** run multiple verify in parallel.
- **2026-05-30 night:** TT device **WEDGED** — clean HEAD hangs at Metal init / profiler sync (~65 log lines, 10min timeout). CPU backend OK. **Do not run bh-30** until chip reset / idle cooldown. Verify loops **stopped** (local + remote). Next doc: `opt/sfpu-cull-next.md`.

## Next commits (when gated)

1. SFPU cull fix → default still OFF until 30-view ×2 @ 63.85 dB
2. Sort on-device publish (small kernel)
3. C2 payload writer + gated reader

## North star order (plan §8)

1. SFPU cull (D) → 2. on-device scan (B/C) → 3. C2 payload → 4. fuse C2+D+E → 5. persistent kernels / drop Finish() → 6. LPT → 7. FPU project → 8. pair-count reduction for sub-10ms
