# Post-L1 profile (task #297, section M of docs/next-levers-after-205.md)

Base: #293 head 51ac7cf9 (iter 206, fused mat+blend on by default). Branch
`ttp/t297-profile-postl1`, measured at 9a143d4. The only kernel change is a set of profiler-only cull
counters in `render/kernels/compute/mat_cull_compute.cpp`, compiled only under `PROFILE_KERNEL`.
Untraced builds are unchanged: md5 906e0435 on every run and in the screenshot (golden match,
max 0 LSB). Device: yyzo-bh-07 (p100a). Everything ran inside a single `ttp lock p100` after
ssh-preflight (`docs/profile-postl1-t297/drive.sh`). bicycle, 30 views, 1024x1024.

## 1. Untraced per-view latency (full defaults)

3 rounds × 2 arms (def / hp), alternating order, 30/30 views md5 906e0435 on every run:

| round | def ms/view | hp ms/view |
|---|---:|---:|
| r1 | 11.096 | 11.081 |
| r2 | 11.130 | 11.102 |
| r3 | 11.077 | 11.069 |

Mean def 11.10 and hp 11.08. The screenshot run gave
avg_frame 11.18.

## 2. Host bridge (untraced, defaults; #155 rule)

| stage | ms/view | notes |
|---|---:|---|
| project | 3.02-3.04 | gather_wait 2.94-2.96 is a device wait (pfwc + K2). pfwc rtargs 0.04, enqueue 0.03 |
| sort | 0.84-0.90 | bin_emit 0.55-0.57 (device), publish_host 0.16-0.18, bin_layout 0.05, mat 0.04 |
| blend | 6.90-6.95 | device wait (fused mat+blend) |
| d2h | 0.21-0.29 | |
| head / tail / tile_assign | ≤0.005 each | |
| resid_vs_frame | +0.03 | |

Host work on the serial path is about 0.4-0.5 ms/view: d2h ~0.25, pfwc dispatch 0.07, and part
of publish_host 0.17. Each piece is below the gate on its own (#171 shelved d2h overlap). The view
is device-bound.

## 3. Tracy: per program and RISC

Capture `t297-ns`. It needs `GSPLAT_TT_PFWC_WRITER_SPLIT=0`, because at defaults the Tracy kcfg
overflows the fused CBs. **Every pfwc number in this section is WRITER_SPLIT=0, not the
default.** 29 views (warmup dropped), 110 cores. Kernel span in ms, mean over cores / max over cores:

| program | BRISC | NCRISC | TRISC0 | TRISC1 | TRISC2 | window | tail (max−mean end) |
|---|---|---|---|---|---|---:|---:|
| pfwc (WS=0) | 2.343/2.626 | 2.209/2.473 | 2.288/2.588 | 2.296/2.595 | 2.284/2.584 | 2.595 | **0.300** |
| K2 | 0.872/0.956 | 0.905/0.980 | – | – | – | 0.965 | 0.065 |
| sort_ol | 1.290/1.423 | 1.341/1.432 | 1.233/1.356 | 1.234/1.355 | 1.233/1.354 | 1.438 | emit **0.084**, town 0.124 |
| mat+blend | 6.244/6.324 | 6.100/6.168 | 6.167/6.247 | 6.168/6.247 | 6.168/6.247 | 6.342 | blend 0.079 |

The traced device span is 11.42 ms/view (WS=0, profiler on). All-core idle between programs is
0.024 ms/view.

## 4. Fused mat+blend, per core

Times are from the first mat start on any core. Per-core values are the mean over views, shown as
min / p25 / median / p75 / max over the 110 cores:

| metric | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| mat end (= blend start) | 2.391 | 2.445 | 2.488 | 2.551 | 2.814 |
| gap mat end → blend start | 0 | 0 | 0 | 0 | 0 |
| blend end | 6.151 | 6.163 | 6.169 | 6.173 | 6.184 |
| blend TRISC1 span | 3.353 | 3.616 | 3.682 | 3.725 | 3.781 |
| mat mover busy BRISC (sort_subchunk_mat) | 2.278 | 2.369 | 2.377 | 2.388 | 2.450 |
| mat mover busy NCRISC | 2.227 | 2.438 | 2.481 | 2.537 | 2.766 |
| **TRISC idle in mat (mc_uw)** | **2.253** | **2.334** | **2.388** | **2.452** | **2.750** |
| TRISC cull busy in mat (unpack side) | 0.039 | 0.089 | 0.103 | 0.110 | 0.136 |
| cull batches | 139 | 192 | 193 | 196 | 209 |

Means over views: mat end 2.513 (max 2.989), gap 0.000, blend end 6.168 (max 6.247), blend tail
0.079. **TRISC idle inside the mat phase: mean 2.415, min 2.190, max 2.915 ms per core.**

L1 removed the barrier completely: blend starts the moment a core's mat ends. Blend ends within
0.08 ms on every core, because the dynamic claim balances it.

### How the idle is measured

`mc_uw` is the UNPACK thread's spin in `pick_stream` while it waits for a mover's coefficient
tile. `mc_utot` is the whole cull loop. `mc_uw + busy = 2.513`, which matches mat end exactly.
The counters also did not move mat end: 2.513 here and 2.513 in the uninstrumented #293 capture.

On the MATH thread, `mc_act` (copy + SFPU issue + commit) is 2.48 and `mc_mw` (mailbox wait) is
0.001. MATH sits in `copy_tile` waiting on the unpacker, so its issue time cannot separate SFPU
work from waiting.

What is proven: the cull TRISCs are **input-starved for 96% of the mat phase**. The mat phase is
bound by the movers: NCRISC busy 2.48 and BRISC 2.38 against a 2.49 phase. It is not bound by the
SFPU. The SFPU's own work in the mat phase is bounded above by the t147 estimate (~1.0 ms) and in
practice is far smaller. At ~193 batches per core over 2.49 ms, a batch arrives every ~12.9 µs,
and the unpacker spends ~0.5 µs on it.

## 5. Re-ranking A and C

**A. Blend lane during mat: not killed. Room is 2.4 ms per core against the 0.6 kill gate.** It
moves to the next model step as planned, with one new constraint the plan did not have: the
movers, not TRISC1, set the mat phase. NCRISC is busy 2.48 of 2.49 ms. The blend lane's loads
(`reader_alpha_blend_mb_devcull`, NCRISC, DRAM reads of tile ids, ranges and subchunk metadata)
would land on that same saturated NCRISC and stretch the mat phase.

The model must therefore charge the lane tiles' NCRISC load time to the mat phase. Gain per lane
tile ≈ blend TRISC time − NCRISC load time, with the TRISC side from the t147 fit
(87 + 0.027·rec + 0.197·live µs). An alternative is to give the lane's loads to BRISC, which has
~0.1 ms of slack in mat.

Upper bound unchanged at ~1.4 per core. Estimate 0.3-0.8 holds only if lane loads stay cheap
relative to blend TRISC time. Rank: still 2, after B.

**New observation (cheap lever to check).** Because mat is mover-bound with idle TRISCs, any cut
of mover work in `sort_subchunk_mat` shortens the mat phase 1:1. One option is to offload part of
the sort/scatter to TRISC0/TRISC2 (L1-local work), as L2 did for K2. L2 measured only −0.245, and
the emit TRISC-cursor idea was shelved (#215), so this needs a no-device op-count first. It is a
follow-up, not a rank change.

**C. Dynamic chunk claim: unchanged, bundle with B.** Tails measured:
- pfwc 0.300 (WRITER_SPLIT=0; the default split tail is not measured here and is likely smaller);
- emit 0.084;
- town 0.124;
- K2 0.065.

The recoverable sum is at most ~0.38 (pfwc + emit), and realistically 0.2-0.3 after claim cost.
That is below the 0.3 gate alone, so C stays rank 3, bundled with B.

## 6. Screenshot

`opt/metal-screenshots/t297-postl1/hero.png` and `hero_diff10.png`, device render at 9a143d4,
defaults. PSNR **42.51 dB vs benchmarks/reference_v2/hero.png**. The md5 badge is separate: sweep
md5 906e0435, golden_match=true, max 0 LSB vs golden. I looked at both images. The hero is clean,
with no missing, doubled or blocky 32-px tiles. The 10x diff shows only fine-edge residue
(spokes, foliage, bench slats) and no tile-aligned pattern. This is bit-identical to iter 205/206.

## Files

- `opt/profiler/postl1_cores.py <dev30.csv> [--percore out.csv]`: per-program/RISC spans,
  tails, per-core mat/blend timeline and mc_* counters.
- `render/kernels/compute/mat_cull_compute.cpp`: `MC_PROF` counters (profiler builds only).
- `docs/profile-postl1-t297/{drive.sh,remote_time.sh,remote_tracy.sh}`: the locked driver.
- `docs/profile-postl1-t297/out/`: untraced run logs and md5s, Tracy summaries
  (`tracy-t297-ns-{postl1,gaps,cores,zones}.txt`, percore.csv), and the analysis of #293's
  uninstrumented capture (`t293-ns-postl1.txt`).
- The raw dev30.csv (65 MB) is kept locally under tmp/t297 and is not committed.
