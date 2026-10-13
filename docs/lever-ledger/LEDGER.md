# Lever ledger and stop review (task #617, 2026-10-13)

This is an independent review of every lever tried so far. It was done without a device. The sources are
`opt/ttw/iters.jsonl` (223 rows), `docs/profile-t227/PROFILE.md` (#595),
`docs/structural-plan-3/PLAN.md` (#592), `docs/structural-plan-4/PLAN.md` (#603),
`docs/pfwc-compact-resize-t611/RESULT.md`, `docs/b2b-floor-t614/RESULT.md`,
`docs/b2b-triage-t468/README.md`, `docs/hw-ceilings.md`, `docs/l2-2chip/RESULT.md`,
`docs/split-frame-2chip/README.md`, and the project's decision memory.

**Headline metric:** back-to-back (b2b) ms/view from `render/run.py --back-to-back`, using the 30 bicycle
views at 1024x1024. The current best is **best-iter-227 at 8.454 ms/view (118.3 FPS) on bh-33**, a p100
with 11x10 = 110 cores. The md5 is 077caaba (30/30) and the hero PSNR is 42.85 dB. For comparison, the
published GPU figure is **10.75 ms/view** (INRIA 3DGS, RTX A6000). That number is published, not
measured here.

Abbreviations:
- M: measured b2b, i.e. a median of alternating A/B rounds on one board.
- L: latency mode. Before iteration 211 (xview), latency was about equal to b2b (#275). After 211,
  latency with `--dump-views` hides about 1.7 ms of pfwc (#464), so it is not valid for keep/shelve decisions.
- Mod: a model, a CPU simulation, or a bound derived from a Tracy profile. No device A/B was run.
- F0: a bound derived from a FOLD=0 traced profile. The shipped b2b runs with FOLD=1.
- OB: an old baseline. The judgement predates the fold (226), the prune (225) or EXP1 (227), so the
  baseline frame was 9.1 to 18 ms instead of 8.45 ms.

## 1. Lever table

### 1a. Landed in the b2b era (all M, keep gate ≥0.1 ms/view, md5 30/30)

| Iter | Task | Lever | b2b on | b2b off | Δ ms/view |
|---|---|---|---|---|---|
| 221 | REC32 | 32 B mat records | 9.229 | 9.434 | -0.205 |
| 222 | #274 re-check | K2 on idle TRISCs (re-measured after #468 triage) | 9.409 | 9.707 | -0.298 |
| 223 | | pfwc ACQ fuse | 9.235 | 9.406 | -0.171 |
| 224 | | mat early cull | 9.131 | 9.367 | -0.236 |
| 225 | | opacity prune (5.56M of 6.13M kept) | 8.998 | 9.131 | -0.133 |
| 226 | | B4a fold: K2(N+1) inside mat+blend(N) | 8.615 | 8.990 | -0.375 |
| 227 | #584 | degree-1 minimax exp (BLEND_EXP1) | 8.454 | 8.596 | -0.142 |

Iterations 217 to 220 re-baselined best-iter-216 at 9.574 (bh-30) and 9.821 (yyzo-bh-04). The b2b era has
cut 1.12 ms/view so far, a 12% cut.

### 1b. Landed before b2b was the headline (iterations ≤216; judged in L before xview, which matched b2b)

These are part of the 227 baseline, so they need no re-check:
- 196 blend schedule
- 198 decode-ahead
- 199 pfwc NoC1 balance
- 205 reciprocal via Newton
- 206 mat+blend fusion
- 207 mat-cull TRISC fill
- 209 K2 rows on CQ1
- 210 DRAM bank-coprime stride
- 211 zero-copy output
- 212 xvpin
- 213 zerocopy+xv
- 215 packed radix
- 216 NoC loopback permute
- #155 SORT_OL_EARLY
- lever C (#140/#142), the opacity-aware tile rectangle, which is in `project_pfwc_compute.cpp`

### 1c. Shelved or killed

The Flags column marks shelvings that should not be treated as settled.

| Lever | Task / iter | Result | Basis | Flags | Reason given |
|---|---|---|---|---|---|
| 16 B record | #23 / it150 | +2.37 ms, -9.3 dB | M (L, pre-xview) | OB | Slower and loses precision. Settled. |
| contrib_floor 1/255 to higher | it152, it200-204, #260 | floor 2/255 gives 37.7 dB | M + PSNR | | Breaks the 42.4 dB gate. Settled. |
| Inter-program idle cut | it158, #514 | 0.013 | M | | Too small. Settled. |
| pfwc cov2d on SFPU | it197 / t228 | rejected | L pre-xview | OB | pfwc mover-bound. TRISC cuts do not move pfwc's end (t614). Settled. |
| K2 into sort_ol | #298 | +0.19 slower | L pre-xview (=b2b) | OB | Walk lands on critical movers. Settled. |
| p150 emit mover table | #358 | +0.69 slower | L pre-xview | OB | Settled. |
| Emit pack on idle TRISCs | #165 | +0.056 slower | L pre-xview | OB | Settled. |
| Shared big-tile sort | #168 | -0.038 | L pre-xview | OB | Superseded by #425. |
| U2 blend coefficients | #252 | +3.63 slower | L pre-xview | OB | Settled. |
| v2 mover table + t164 fold | #187 | -0.196 then; ~0.06 scaled | L pre-xview | OB, Mod scaling | Emit rewritten (#418/#425). Branch conflicts. |
| Tail-chained mask walk | #191 | -0.163 then; ~0.11 scaled | L pre-xview | **OB, Mod scaling** | TRISC1 rebuilt since. 0.11 is a scaled guess near the gate. |
| Mat mid-tile split | #176 | 0.26-0.30 cap on old mat | Mod + L | OB, Mod | #418/#425 cut big-tile sort since. |
| Blend per-tile fixed cost | #172 | ~0.1 cuttable | Mod + Tracy | **OB, Mod** | See §2: about 27% of mat+blend is fixed today (1.42 ms intercept). The old 0.1 bound is stale. |
| Mask-0 record drop | #185 | -0.13 to +0.16 | Mod | Mod | Not md5-exact. |
| Emit TRISC cursors | #215 | 0.10-0.20 upper bound | Mod | Mod | The similar #165 lost. |
| Blend-waste work cut | #148 | 0.04-0.08 | counters + Mod | Mod | |
| Host residue / D2H overlap | #171 | — | host profile pre-xview | OB | D2H is gone. Host slack was re-measured by t614 (see L-A). |
| K2 pairs from pfwc writers | #384 | net ~-0.05 | Mod (Tracy) | Mod | Writer growth lands on pfwc's mover end, which is 1:1 critical (t614). Stays shelved. |
| FPU quadratic blend | #511 / t111 / t148 | +0.45 worse | M | | Settled. |
| B2 ellipse cull | #538 | 9.168 vs 9.284 (-0.116) | M | **OB** (pre-226/227) | Shelved as small and costly. The gain was over the 0.1 bar at the time; it was not re-measured after the fold. |
| Chunk cull (ply order); Morton reorder + chunk cull | #486, #502; #169/#433 (GSPLAT_TT_CHUNK_CULL), #470, #479, #493 | ply order ~0; Morton +0.18..+0.41 ms/view (loss) | M / Mod | | PLY order has no spatial locality. The Morton reorder + chunk cull was already built and device-measured in b2b: net +0.18..+0.41 ms/view (loss) (#470/#479/#493: 0.309, 0.413, 0.215, 0.177 ms), caused by sort emit bursts on Morton gids. See U3 (sized by #631). |
| Speed-weighted pfwc deal | #517 | noise | M | | Settled. |
| pfwc N+1 under blend tail | #512 | ≤0.070 fused | Mod | Mod | Fold (226) took most of this. |
| Mat-start TRISC idle fill | #509 | ~0 | Mod | Mod | T1 idle-fill bound 0.298 (PROFILE t227). Only pays if mat is critical, which it is not. |
| Metal trace replay | #526 | ≤0.083 | Mod (code study) | **Mod** | Bound rests on a host-cost model. t614 later measured host-before-enqueue fully hidden, so the bound looks right. |
| Coalesced emit (B1) | #534 | 0-0.15 | CPU Mod | **Mod** | Never on device. Sort is 1.44 ms on scalar movers. |
| Blend lane packing | #538 / t538 | ≤0.09 | Mod | Mod | |
| Cross-view sort reuse | #529 | 0 on bench; viewer 0.2-0.4 | CPU Mod | Mod | The bench views are not temporally coherent. Real gain only in the viewer. |
| B4' SIB | #573 | 11.505 vs 8.670 | M | | Settled. |
| L4 persistent kernel | #579 | code 2.37x the kcfg limit | Code size | **not a perf result** | Never timed. Blocked by the kernel-config size limit, not by speed. |
| Precision cuts (alpha floor, T_EPS, coarser T) | #580 | 37.7 dB; coarser T slower | M + PSNR | | Settled. 0.45 dB margin. |
| B4 cross-frame | #591 | +0.073 | Mod | **Mod, no device run** | |
| Spec A | #596 | killed | Mod | **Mod, F0** | |
| Specs B / C | #595 | 0.02-0.18 / 0.037 | Mod from FOLD=0 trace | **Mod, F0** | Spec B's top end (0.18) is above the bar. |
| RECFAST + VIS_FAST | #606 / t611 | 0.071 / 0.022 / bundle 0.070 | M | md5 changed | RECFAST alone is 0.071, below the 0.1 gate. Its slowest mover was 1.30 vs a 1.45 window. |
| A' compaction (pfwc) | #611 | killed at step 0 | **Mod (M3 fit on 2-4 points), F0 windows** | **Mod, F0** | "Floor F = 1.452". That floor comes from FOLD=0 windows. b2b gains did not follow those windows. |
| Early pfwc enqueue | t614 | killed | M (spin probes) | | Host before enqueue is hidden by 0.13-0.23 ms of queued sort. Settled. |
| Host work in b2b loop | plan 3 drop list | "0.04 idle, nothing to win" | F0 trace (t224) | **F0, contradicted** | t614 measured up to 0.30 ms/view of device idle at the view boundary in b2b. That idle is lever L-A (#628). |
| Scene pre-filter (never-visible 60% of N) | plan 3 drop list | dropped as "gaming the bench" | judgement | **no measurement** | A spatial or LOD cull computed from the camera is legitimate (U3). Only a filter fitted to the 30 bench views is gaming. |
| NoC/DRAM in blend and sort | plan 3 drop list | 4% / ≤50% | Mod | Mod | |
| cov_cam on FPU | plan 3 drop list | ≤0.26 TRISC ms | Mod | Mod | TRISC time in pfwc is off the critical path (t614). Settled. |

### 1d. What the flags say

- **Seven shelvings rest only on a model:** B4 #591, A #596, A' #611, B1 #534, B/C #595, metal trace #526,
  and mat-start #509. Four of them (A, A', B, C) take their bounds from FOLD=0 profiles. t611 showed
  that b2b gains do not track FOLD=0 windows, so those bounds are unreliable in both directions.
  The A' kill is the weakest of these, and it sits on the one stage (pfwc) whose end is 1:1 critical.
- **One drop was contradicted by later measurement:** plan 3's "host work: nothing to win" (F0, t224)
  is contradicted by t614 (≤0.30 ms/view of boundary idle). It is being measured as L-A (#628).
- **No live shelving rests on latency with `--dump-views`.** #468 already re-checked the pre-xview list,
  and #433/#439 were re-checked elsewhere.
- **OB shelvings that were near the bar** and could change on today's 8.45 ms frame: B2 ellipse
  (#538, -0.116 measured), #191 (~0.11 scaled) and #172 (fixed-cost bound, stale). Of these, only B2 has
  a measured gain over the bar.
- **L4 was never timed.** Its shelving is about code size, not speed.

## 2. Time budget at best-iter-227 vs hardware floor

All figures are ms/view. The b2b total is 8.454. The split comes from PROFILE t227 (a FOLD=0 trace,
mapped onto the shipped FOLD=1 frame by subtracting the folded K2 window and adding its net cost). Scene
sizes:
- N = 5.56M Gaussians read
- M = 1.675M visible
- P = 2.614M tile pairs

Fits across views (PROFILE t227):
- pfwc = 1.063 + 0.274·M
- sort = 0.329 + 0.424·P
- mat+blend = 1.421 + 1.443·P
- K2 = 0.250·P

Hardware (bh-33, a p100): 110 cores at 1350 MHz and 7 DRAM channels. The design figure for the DRAM
floor is 420 GB/s. Measured scattered 64 B reads cap at about 35 GB/s (`hw-ceilings.md`). Data movers
use soft-float at 84-90 cycles per op. The FPU is about 0% busy.

| Stage | b2b share | What bounds it now | Floor | Floor basis | Gap |
|---|---|---|---|---|---|
| pfwc (fused projection + cull) | 1.52 | Mover program end, which is 1:1 critical (t614). The slowest mover after RECFAST is 1.30. 70% of the stage is fixed in M because it scales with N. | ~0.80 | 334 MB per view / 420 GB/s streaming (plan 3) | ~0.7 |
| sort (one launch) | 1.44 | Scalar movers: about 164 core-cycles per pair (1.44 ms × 110 × 1.35 GHz / 2.614M). Fixed part 0.33. | ~0.2-0.4 | Assumes ~30-60 mover cycles per pair for a 2-pass radix on 8 B keys. A model, not measured. | ~1.0 |
| K2 (folded) | ~0.27 net | Runs inside mat+blend(N) | 0 if fully hidden | The fold already hides about 0.37 | ~0.27 |
| mat (inside mat+blend) | not on the critical path (mean 1.42, max 2.04, on TRISC) | Overlaps blend | 0 | | 0 |
| blend (SFPU) | ~5.2 of the 5.46 window (mean 3.51, max 3.78; reader 3.51) | SFPU: about 214 core-cycles per pair (1.443·P). Fixed intercept 1.42 ms. | not derived | Needs an SFPU instruction count per pair from the compiled kernel (U9) | Unknown. The imbalance tail alone is 0.27 (max - mean). The window is 1.4 above the slowest blend core. |
| view boundary (gather, host) | ≤0.30 | Host gather and enqueue (t614: G 0.055 + A 0.246) | ~0.03 | Program gaps are 0.03 | ≤0.27 |

**Fixed costs dominate the remaining headroom.** The fitted intercepts add up to about 2.8 ms per view:
- pfwc 1.06
- sort 0.33
- mat+blend 1.42

That is about 33% of the frame, and none of it has been taken apart. The mat+blend window is 5.46 ms
against a slowest blend core of 3.78 ms and a mean of 3.51 ms. So about 1.4 ms of that window is not
the busiest SFPU. It is the folded K2 (about 0.27), plus program start and end, the output write, and
time the reader or mat keep a core waiting. PROFILE t227 does not attribute that time.

## 3. Ideas not yet sized

Bounds are rough and come from the §2 budget. None of them is measured.

| # | Stage | Idea | Rough bound (b2b ms/view) | Notes |
|---|---|---|---|---|
| U1 | pfwc | Mover transfer pattern: both NoCs, bursts ≥256 B, fewer barriers, more outstanding reads | 0.2-0.5 | The mover end is 1.30-1.45 and the DRAM floor is 0.80. This is the only stage with a known critical path at 1:1. Being sized in #629. |
| U2 | pfwc | Smaller per-Gaussian record (fp16/bf16 SoA for scale, rotation and colour; 60 B down to ~32 B) | ≤0.3 | Plan 3 listed it as "≤0.3 once DRAM-bound" but never built it. It pays only after U1 makes pfwc DRAM-bound. Needs a precision check (0.45 dB margin). |
| U3 | pfwc | Morton reorder + chunk frustum cull: **not unsized, sized by #631** (docs/morton-cull-t631/README.md on ttp/t631-morton-cull-size) | gross 0.44 bound; net expected <=0.15 unless sort emit de-burst is solved | Already built (#169/#433, GSPLAT_TT_CHUNK_CULL) and device-measured in b2b (#470, #479, #493: 0.309, 0.413, 0.215, 0.177 ms): net +0.18..+0.41 ms/view (loss), cause sort emit bursts on Morton gids. Re-open only with an emit de-burst fix. |
| U4 | view boundary | L-A: enqueue gather(N+2) right after mat+blend(N+1) | ≤0.30 | #628 is measuring this now. |
| U5 | view boundary | Fold gather into pfwc (one program less per view) | 0.05-0.25 | Overlaps U4. Size it after #628. |
| U6 | mat+blend | Break down the 1.42 ms fixed intercept: per-tile T init, the output write (1024² RGB), CB setup, and the window minus the slowest core (~1.4 ms) | 0.2-0.8 | The largest piece of unattributed time. Needs a FOLD=1 trace with per-phase zones (profiling, not a lever). |
| U7 | blend | Dynamic tile stealing across cores (a NoC semaphore counter instead of static LPT) | ≤0.27 | Equals max - mean of blend SFPU time. Mat imbalance (2.04 vs 1.42) is hidden. |
| U8 | blend / sort / K2 | Fewer pairs: 16x16 screen tiles, or a tighter ellipse-to-tile test (B2 re-measured at 227) | B2 0.1-0.15 (it measured -0.116 at 9.2 ms); 16x16 tiles unknown, ± | Every pair costs about 1.44 + 0.42 + 0.25 = 2.1 ms per million pairs, so a 5% cut in pairs saves about 0.27. |
| U9 | blend | SFPU instruction-count floor per pair (disassemble the TRISC1 loop) | profiling | Without this, the blend gap cannot be bounded. Cheap to do and needs no device. |
| U10 | sort | Shorter keys: 16-bit depth plus a tile index packed into 32 bits, so fewer radix passes; or a fixed-cost cut (0.33 intercept) | 0.2-0.4 | Packed radix (215) landed, but the pass count and key width were not revisited after the prune. A sort floor of about 0.3 against 1.44 measured is the largest relative gap. |
| U11 | sort | Sort tail (last core) and host prep overlapped with blend (t614 unsized ideas) | 0.05-0.2 | Being sized in #629. |
| U12 | dispatch | p150 120 cores with ETH dispatch: re-measure 227 on a p150 (t577 measured 8.119 with an older build) | headline only, ~0.3-0.4 vs p100 | Not a code lever. The p100 is the gate box. Report it as a second headline when a p150 is free. |
| U13 | multi-chip | AFR on 2 chips | measured 4.226 (1.99x, pre-227 build) | Viewer AFR with page FPS is in #620. Re-measure on best-iter-227. |
| U14 | multi-chip | Split frame across 2 chips (latency) | model 6.2 ms latency; b2b 5.6-6.7 (worse than AFR) | Build done (#623/#624). Device measurement #627 is queued. |
| U15 | precision | fp16 instead of bf16 for the alpha-blend intermediates in blend CBs | unknown, ± | It would only free precision margin to spend on U8 or contrib_floor. #580 found no direct gain. Low priority. |
| U16 | algorithmic | Level of detail: merge distant small Gaussians at load into a coarser set chosen per view | 0.5-1.5, if PSNR holds | Large but changes the image. Needs a precision study first. The 42.4 dB gate decides. |
| U17 | pipeline | Core-partition pipelining (sub-devices): pfwc+sort for N+1 on k cores while blend N runs on 110-k cores | ≤0.4 | It only gains the tails and fixed costs that do not scale with core count. Related shelvings: L4 (#579, code size) and #512. |
| U18 | viewer | Cross-view sort reuse in the live viewer | 0.2-0.4 realistic | Not counted for the bench (0 on 30 views). It helps page FPS only. |

## 4. Verdict

**A stop is not justified now.** The reasons:
1. About 2.8 ms per view of fitted fixed cost (§2) and the ~1.4 ms mat+blend window overhang have
   never been broken down. They are bigger than every lever landed in the b2b era combined.
2. The last pfwc kill (A', #611) and four other kills (A, B, C, B4) rest on models built from FOLD=0
   windows. t611 itself showed that b2b does not track those windows, while t614 showed pfwc's
   mover end is 1:1 critical.
3. The plan 3 drop of host work was contradicted by measurement (t614), and #628 is now sizing it.
4. Sort runs at about 5x its rough floor and has no lever that has been sized since the prune.
5. Three items in the user's stop checklist (msg #332) are still open: AFR in the viewer (#620),
   split-frame measurement (#627), and this review's ranked list.

**Ranked levers to size next (each ≥0.1 ms/view b2b):**

1. **U6 + U9: profile the unexplained mat+blend overhang and the fixed intercepts in FOLD=1**, with
   per-phase zones and an SFPU instruction count per pair. This is profiling, not a lever, but it decides
   whether 0.2-0.8 ms is reachable. Do it first, because levers 3, 5 and 6 depend on it.
2. **U1 pfwc mover transfer pattern** (0.2-0.5). The critical path is proven 1:1 and the DRAM floor
   is 0.80. #629 is sizing it. Stack RECFAST (0.071 measured) on top of any landing, since together
   they can clear the bar.
3. **U4 L-A view-boundary enqueue** (≤0.30). #628 is running it. Follow with U5 (gather into pfwc) only
   if L-A leaves idle.
4. **U3 Morton reorder + chunk frustum cull**: sized by #631 (docs/morton-cull-t631/README.md on
   ttp/t631-morton-cull-size): gross bound 0.44, expected net <=0.15. Already device-measured as a net
   loss of +0.18..+0.41 ms/view (#470/#479/#493, sort emit bursts on Morton gids); worth a retry only if emit de-burst is solved.
5. **U10 sort key width / pass count and fixed cost** (0.2-0.4). The stage has the largest gap to its floor.
6. **U8 pair reduction**: re-measure B2 ellipse at 227 (0.1-0.15, measured once at -0.116) and size
   16x16 tiles on a CPU pair count.
7. **U7 dynamic blend tile stealing** (≤0.27).
8. Multi-chip and headline items run in parallel and do not compete for this list: U13 (AFR viewer, #620),
   U14 (split frame, #627), and U12 (p150 120-core re-measure at 227).

**When to stop:** stop only if 1 to 7 come back below 0.1 each, or are measured and do not clear the
gate, and the U6 breakdown leaves no fixed component above 0.1.

Shelvings that should not be re-queued without new evidence: everything in §1c with no flag, plus
#384, #298, #358, #165, #252, #511, #573, #580 and the contrib_floor family.
