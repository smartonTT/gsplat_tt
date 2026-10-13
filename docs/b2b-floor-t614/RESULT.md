# t614: what sets the ~1.45 ms pfwc floor in the real b2b config (B4a fold on), bh-33

2026-10-13, bh-33 (p100 measurement reservation, 110 cores), branch ttp/t614-b2b-floor-profile.
Base best-iter-227 code; all probes default off. Headline metric: `render/run.py --back-to-back`,
30 bicycle views, 20 passes, median ms/view.

## Tooling (fits the FOLD=1 build; no device profiler, no Tracy)

- `GSPLAT_TT_HTL=1`: host timeline. A steady_clock stamp at every stage-timer span and host
  zone (render/host/htl.h), written once per pass. `analyze.py` (per-zone means) and
  `phases.py` (per-view phases A-D, medians) read it.
- Slack probes (the b2b change per inserted µs says whether that point is on the critical path):
  - `GSPLAT_TT_PFWC_SPIN_US` (dsN): the pfwc BRISC writer busy-waits N µs of wall clock
    after its last write, so the pfwc program ends N µs later.
  - `GSPLAT_TT_HTL_FOLD_SPIN_US` (hsN): host spin after the pfwc(N+1) enqueue, before the
    mat+blend(N) enqueue.
  - `GSPLAT_TT_HTL_PRE_SPIN_US` (psN): host spin just before the pfwc(N+1) enqueue.
  - `--view-gap-ms` (gN, N/10 ms): host spin at the view boundary, outside the render window.
- Drivers: `drive.sh` (one `ttp lock p100` per run; STEPS override), `remote.sh` (arms).
  Outputs: out-a (run 1730/1744), out-b (gap arms), out-c (pre-spin arms).

The HTL build is within noise of the uninstrumented one: base 8.452 / 8.447 / 8.470 / 8.463,
htl 8.464 / 8.463 / 8.447 / 8.496 (r0-r3, ms/view).

## Results, out-a (rounds r0-r3, rotating arm order)

| arm | r1 | r2 | median of base in round | delta |
|---|---|---|---|---|
| base | 8.447 | 8.470 | — | — |
| ds150 (pfwc ends +0.150 ms) | 8.579 | 8.582 | | **+0.132 / +0.112** |
| ds400 (pfwc ends +0.400 ms) | 8.827 | 8.841 | | **+0.380 / +0.371** |
| hs300 (host +0.3 ms after pfwc enqueue) | 8.484 | 8.467 | | +0.037 / -0.003 |
| hs700 (host +0.7 ms after pfwc enqueue) | 8.474 | 8.455 | | +0.027 / -0.015 |
| f0 (FOLD=0, r3) | 8.844 | | 8.463 | +0.381 |

- **pfwc's program end is on the b2b critical path 1:1, less a constant ~0.025 ms.** ds150 adds
  0.122 (mean), ds400 adds 0.376: delta = spin - 0.026 ± 0.01 in both. The 0.026 ms is the
  overlap of the mat+blend dispatch with pfwc's tail. No contention term appears: with one
  in-order CQ, the folded K2(N+1) runs inside mat+blend(N), after pfwc(N+1) ends, so
  **pfwc never shares DRAM/NoC with the folded K2.** That cause is ruled out by construction.
- **The host is not on the path after the pfwc enqueue.** A 0.7 ms host spin there costs
  ≤0.03 ms: pfwc keeps the device busy for >0.85 ms after its enqueue.
- So the "~1.452 ms floor" from #611's refit is not a hidden stall. It is pfwc's own program
  length, set by the movers (#608), and it counts 1:1 in b2b. TRISC cuts did not move b2b
  because they do not move the program end; any cut that moves the program end will.

## Per-view host phases (phases.py, median ms, FOLD=1 htl arms)

| phase | what the host does | r0 | r1 | r2 | r3 |
|---|---|---|---|---|---|
| A | view start -> gather(N+1) result (waits on device gather + D2H) | 0.270 | 0.263 | 0.242 | 0.271 |
| B | TA, sort(N+1) (onelaunch ~0.47, layout ~0.07, pub_host ~0.31), log/mat spans ~0.11, pfwc(N+2) rtargs+enqueue | 1.190 | 1.152 | 1.135 | 1.198 |
| C | rest of fold hook, fold rtargs, mat+blend(N+1) enqueue | 0.145 | 0.144 | 0.144 | 0.145 |
| D | host blocked until mat+blend(N+1) output is back | 6.899 | 6.933 | 6.949 | 6.890 |
| G | python between views | 0.056 | 0.056 | 0.055 | 0.055 |

FOLD=0 (r3-f0): A = 2.22 (the host waits for pfwc+gather), C = -0.11, D = 5.72.

**Open point the out-a data cannot settle:** in B the device has no queued work while the host
runs layout + pub_host + log/mat setup + pfwc rtargs (~0.5-0.6 ms/view) unless the sort or a
materialize program is still running. If the device idles there, that time is on the
critical path and pfwc(N+2) could run under it. The gN and psN arms (out-b, out-c) size it.

<!-- RESULTS-B-C -->

## Correctness and screenshot

Uninstrumented build (60475709, defaults, FOLD=1): device hero + 30-view sweep via
opt/ttw/screenshot.sh: sweep md5 **077caaba**, 30/30 views match golden (VIEWS_DIFFER 0 of 30),
max LSB vs golden 0. **PSNR 42.85 dB** vs benchmarks/reference_v2/hero.png. Images:
opt/metal-screenshots/ttw-t614/hero.png and hero_diff10.png (device render, not the CPU
reference). Checked by eye: no tile artifacts; the diff is edge-only. No dashboard iteration
(no speedup kept).
