# Next levers after iter 205 and the fused mat+blend (task #295, no device)

Plan only. No device run. All numbers come from docs already on origin; each one names its
source. The GPU numbers are published, not measured.

## Where we stand

| item | ms/view | source |
|---|---:|---|
| iter 205 (default chain, untraced) | 11.657 | best-iter-205, t290 |
| fused mat+blend (L1) A/B, untraced | 11.13 vs 11.75 base (-0.62) | t289, docs/matblend-ready-t273/README.md (ttp/t293 branch) |
| expected default after #293 | ~11.1 | iter 205 minus the t289 gain |
| GPU G1: INRIA 3DGS, RTX A6000, bicycle 1080p, 93 FPS (published, not measured) | 10.75 | REPORT.html |
| GPU G2: G1 pixel-normalized to 1024² (published, not measured) | 5.44 | REPORT.html |

The last full traced breakdown is from iter 199 (t267, docs/profile-iter199-t267.md, yyzo-bh-07
p100a, 110 cores). It does not include L1:

| program | traced ms | inside |
|---|---:|---|
| pfwc | 2.008 | BRISC 1.795 / 2.006 mean / max, tail 0.21 |
| K2 (pairs) | 0.981 | BRISC/NCRISC only, TRISCs idle |
| sort_ol | 1.432 | emit movers ~1.2 / 1.3, town TRISC 1.085 / 1.23, tail 0.09-0.15 |
| mat + blend | 6.895 | mat_cull_mask 2.512 / 3.001, blend TRISC1 3.732, barrier wait 0.489 (L1 removes it) |
| host serial | ~0.39 | d2h 0.234, pfwc dispatch 0.08, rtargs ~0.08 |

No post-L1 traced profile exists yet. #293's Tracy capture of the fused build failed: its CBs were
5888 B over L1 under the profiler. drive2 retries with GSPLAT_TT_KCFG_EXTRA_KB=26.

## Plain verdict

Big levers are running out. After L1 we are at about 11.1 ms, roughly level with G1 (10.75).
Three levers are left that clear the 0.3 ms gate on paper, and only one of them (B) has a solid
model. If all three land at their realistic values, the total is about 0.9-1.6 ms, giving
~9.5-10.2 ms/view. That beats G1 by 5-12%. Nothing on this list gets near G2 (5.44 ms). That
would need about 2x on blend itself, and every blend lever we know of is shelved or measured
below the gate (see "Rejected" below). After these three, the project should stop or reframe the
goal. Do not keep chasing sub-0.3 ms items.

## Ranked list

| rank | lever | estimate, ms/view untraced | basis | risk | cost |
|---:|---|---:|---|---|---|
| 0 | M: post-L1 profile (prerequisite, not a lever) | - | - | low | 1 device run |
| 1 | B: fold K2 into sort_ol (no pair pages; difference-array counts) | 0.4-0.7 | t267 K2 0.981 traced; t274 91.9 cyc/pair x 2.64 M pairs; t170 count cost 0.16 | medium (md5 order, L1 in sort_ol) | medium: 1 kernel merge plus a host model |
| 2 | A: per-core mat/blend overlap inside the fused program | 0.3-0.8 (gated on M) | t267: per core mat 2.5 then blend 3.73 in series; TRISC floor cull ~1.0 + blend 3.73; t147 slab model | high (L1 is full; #147 shelved the full version) | high |
| 3 | C: dynamic chunk claim for pfwc and the sort_ol emit (L4) | 0.2-0.35 | t267 pfwc tail 0.21, emit tail 0.09-0.15 | medium (output order) | medium; bundle with B |

By value per cost: B, then C bundled into B's sort_ol rewrite, then A. A has the largest upside,
but it only goes ahead if M shows the room.

---

### M. Post-L1 profile (prerequisite for A and C, also checks B)

Why first: every estimate above uses iter-199 data. L1 moved the mat to blend boundary, and A
depends on what is left there. The #155 rule applies: before targeting any Tracy gap, check the
untraced host-bridge numbers. Tracy adds 0.24 ms (t267: untraced minus traced span).

Spec (device). If #293's drive2 already produced a post-L1 Tracy capture, reuse it and skip the
device run.
1. No-device prep. Branch `ttp/<task>-profile-postl1` off smarton/tt-project-opt at the #293
   head (fused default on). Reuse the t267 Tracy recipe (docs/profile-iter199-t267.md). If the
   profiler build overflows L1, use GSPLAT_TT_KCFG_EXTRA_KB=26 as #293 drive2 does. Do not
   shrink the production CBs.
2. Device, all inside one `ttp lock p100 -- ...`. Run `tt-project/harness/bin/ssh-preflight
   <host>` first. Steps:
   - untraced 30-view bicycle sweep: ms/view and md5 (expected 906e0435);
   - one Tracy capture;
   - untraced host-bridge timings (the GSPLAT_TT host timers t267 used).
3. Report, per program:
   - mean and max busy for each RISC;
   - for the fused mat+blend: per core, the time mat ends and the time blend starts, the gap,
     blend TRISC1 busy, and mat mover busy;
   - the TRISC1 idle time inside the mat phase on each core (this is A's room);
   - the pfwc and emit tails.
4. Screenshot: `opt/ttw/screenshot.sh <iter> <rev>`. It produces the device hero.png, the 10x
   diff against benchmarks/reference_v2/hero.png, and PSNR against that same reference. Look at
   the hero and the diff for tile seams, blocky tiles and empty tiles. The md5/golden match is
   a separate badge, never shown as the PSNR.
5. Write docs/profile-postl1-<task>.md. In the hand-off, re-rank A and C using the measured
   numbers.

Restrictions to restate in the spec:
- own worktree, never edit ~/dev/gstt2;
- `ttp lock p100` around every sync+build+run;
- no ird reserve/release; never disable host-key checking; no scanning;
- push only the new ttp/* branch, no force, no PRs.

---

### B. Fold K2 into sort_ol (rank 1)

What it is. Today K2 (tile_assign_scatter_seg.cpp → pfwc_fuse::emit_pairs_diet) walks each
visible gaussian's tile rectangle and writes one (gid, tid) pair per tile into DRAM pair pages,
2,640,222 pairs/view (t274). It also counts `cnt[tid]++` for every pair (the fold count rows).
sort_ol then:
- takes the prefix over those count rows;
- re-reads every pair page in the emit, together with the blend record, and scatters a 32 B
  record into each tile bucket.

The pair set is a pure AABB rectangle walk; there is no per-tile ellipse test (pfwc_fuse.h,
`tid = (miny + dy) * tiles_x + minx + dx`). So both halves can be done without materializing
pairs:
- **Count:** each mover covers a pair range [pg0, pg0 + npg) of its segment. For the whole
  rectangles in that range, use a 2D difference array: 4 updates per gaussian, then one 2D
  prefix over the 1024 screen tiles. Partial rectangles at the two range ends are counted per
  pair, as now. The count rows come out identical to the fold's, so the prefix and the bucket
  positions do not change.
- **Emit:** the emit movers walk the same rectangles in the same pair order (the emit_pairs_diet
  row walk, with one divide per gaussian) instead of reading pair pages. Each pair gets the
  same bucket slot as today, so the output stays md5-identical.
- K2 disappears as a separate program and launch.

Estimate (traced, then untraced). The saving is:
- K2 0.981 (t267);
- the emit's pair-page reads. These are ~21 MB of 8 B pair data read back through the sort's
  1024-page window. t170 measured that window fill at ~26 MB of 64 B reads, and it explained
  most of sort_ol_prefix (0.037 → 0.373 ms).

The new costs are:
- the difference-array count, ~0.05 (4 updates per visible gaussian plus a 1024-tile prefix per
  mover row);
- the rectangle walk moved into emit: t170 put the walk without the count at roughly
  0.76 - 0.16 of K2's mover time, about 0.6 ms of mover time spread over the existing movers.
  The walk is integer increments, and it replaces a page read per 16 pairs, so the net emit
  change is about +0.1 to +0.3.

Net: about 0.5-0.8 traced, or **0.4-0.7 untraced**.

If #291 keeps K2_TRISC on by default, B's baseline becomes K2 ~0.74, and B's saving shrinks by
about 0.25, to 0.2-0.45. B then only clears the gate together with C.

Risks:
1. **md5 order.** The bucket order must stay canonical: core, then BRISC pages, then NCRISC
   pages, gaussian-major. The emit walk must use the exact pair ranges K2 used.
2. **L1 in sort_ol.** The difference array per mover row is 1024 x 4 B = 4 KB. The lofs/box
   read ring (2 x 8 x 64 B x 2) replaces the pair window, so L1 likely goes down.
3. **Emit mover time rises.** If the walk costs more than the page reads it replaces, emit
   becomes the bound. Step 1 measures this before any device time is spent.
4. **Overlap with #291.** #291 rebalances K2's TRISC job split, and B makes it obsolete. Wait
   for #291's verdict first.

Spec.
1. No-device prep. Branch `ttp/<task>-k2-into-sortol` off smarton/tt-project-opt at the #293
   head, after #291 has a verdict.
   - **Host model.** Extend the host model of emit_pairs_diet (tests/unit/test_pfwc_fuse.cpp) with a
     difference-array count. Prove that its per-(core, mover) count rows equal the fold's
     rows, using the hero fixtures and 3 bench views.
   - **Emit walk model.** Model the emit walk and prove that every pair maps to the same
     (tile, slot) as today.
   - **Kernel change.** Change sort_bin_onelaunch.cpp: it reads lofs/box pages and runs count,
     then prefix, then emit with the walk. Keep K2 behind a kill switch,
     `GSPLAT_TT_K2_FOLDED=0` restoring the current path. The host stops launching K2 when the
     switch is on.
   - **Counter.** Add a cycle counter for the emit walk under the existing MB_STATS-style
     opt-in.
   - **Gate.** Build it host-side.
2. Device A/B. Use the same build, alternating, with ≥3 runs per arm of the 30-view bicycle
   sweep, all inside `ttp lock p100 -- ...`. Run `tt-project/harness/bin/ssh-preflight <host>`
   first. Pass conditions:
   - untraced ms/view at least 0.3 below the switch-off arm;
   - md5 906e0435, or the current golden on the base head;
   - 30/30 views.
3. Screenshot of the switch-on build: `opt/ttw/screenshot.sh <iter> <rev>`. It produces the
   device hero.png, the 10x diff and PSNR against benchmarks/reference_v2/hero.png. Look at the
   hero and the diff for tile artifacts. Keep the golden md5 badge separate from the PSNR.
4. Keep:
   - default the switch on;
   - add the iteration to opt/ttw/iters.jsonl with the screenshot, then rebuild REPORT.html;
   - tag best-iter-<N> and never move it;
   - merge forward into main (no force).
5. If the gain is < 0.3: keep the switch default off, document the measured split (count, walk
   and emit times), and shelve.

Restrictions to restate in the spec:
- own worktree, never edit ~/dev/gstt2;
- `ttp lock p100` for device work;
- no ird reserve/release;
- no force push, rebase, amend or tag moves;
- no PRs;
- never disable host-key checking.

---

### A. Per-core mat/blend overlap inside the fused program (rank 2, gated on M)

What it is. L1 lets each core start blending as soon as tiles are ready, which removed the
global barrier (0.489 mean wait at iter 199). Each core still runs its own mat phase and then
its own blend phase in series.

Inside the mat phase:
- the movers (BRISC/NCRISC) are busy 2.375 / 2.491;
- TRISC only runs the cull, ~1.0 ms of SFPU (t147: ~0.04 µs per record);
- so TRISC1 idles for ~1.5 ms per core.

Inside the blend phase, the movers mostly wait. The per-core lower bound is the TRISC work,
cull ~1.0 + blend 3.73 ≈ 4.8 ms, against about 6.2 in series. That is 1.4 ms of room per core
in the best case.

Why it was shelved before. #147 fused mat into blend and found that the sort, mat and blend
slabs do not fit in L1 together. #293 makes L1 fit only by aliasing the blend CBs onto the mat
CBs (CB4 512 KB bucket, CB2/5/6/20/21). mat needs ~1.37 MB and blend ~0.56 MB.

The new angle is a **small** blend lane during the mat phase, not full co-residency. Reserve a
small blend ring (64-128 KB, carved from CB4 by capping the mat bucket at the size of the
largest tiles that actually occur after the big-tile split). TRISC1 then blends ready **small**
tiles between culls. The large tiles still blend after mat. The descending-count claim already
puts the big tiles first, so the small ones arrive late, which suits a lane that only takes
small tiles.

Estimate: **0.3-0.8 untraced**, model only. Upper bound ~1.4 per core. It is limited by:
- how many small tiles are ready during a core's mat phase;
- the cull and blend contention for TRISC1;
- the ring size.

M gives the per-core TRISC1 idle time inside the mat phase. If that idle time is under 0.6 ms
mean, kill A.

Risks:
1. **L1 headroom.** A smaller mat bucket may force more big-tile splits, which costs mat time.
2. **TRISC interleaving.** Cull and blend are different SFPU programs, so switching between
   them costs reconfiguration time (unpacker/SFPU config). That cost is unmeasured.
3. **Code complexity** in the already-complex fused kernel.
4. **L1 under the profiler.** It already overflows by 5.9 KB; Tracy captures need
   KCFG_EXTRA_KB.

Spec.
1. No-device model, after M. Inputs:
   - per-core timelines from M's capture;
   - per-tile record and live counts (the t147 fit: 87 + 0.027·rec + 0.197·live µs per tile);
   - the tile size histogram (bench views).

   Simulate a per-core blend lane of ring size R ∈ {64, 96, 128} KB. It takes ready tiles of at
   most R/32 B records during the mat phase, with a per-switch cost of {2, 5, 10} µs. Output the
   per-view makespan versus M's measured makespan.

   Gate: a modeled gain of ≥0.4 ms at the 5 µs switch cost. Otherwise shelve A with the model.
2. Code prep. Branch `ttp/<task>-matblend-lane` off the B head (or off #293 if B was shelved).
   - Cap CB4, add the blend lane CB, and claim small tiles from the existing ready-flag buffer.
   - Kill switch `GSPLAT_TT_MATBLEND_LANE=0`.
   - Gate: a host build and an L1 fit check for the production config and for the profiler
     config with KCFG_EXTRA_KB.
3. Device A/B, then screenshot and keep/shelve rules exactly as in B. That means:
   - wrap everything in `ttp lock p100 -- ...` and run ssh-preflight first;
   - ≥3 alternating runs per arm;
   - md5 must equal the base head's golden;
   - screenshot via opt/ttw/screenshot.sh, with the diff and PSNR against
     benchmarks/reference_v2/hero.png;
   - visual tile-artifact check (this lever reorders tile work, so check closely for missing
     or doubled tiles);
   - gate 0.3.

Restrictions to restate in the spec: the same as in B.

---

### C. Dynamic chunk claim for pfwc and the sort_ol emit (rank 3; bundle with B)

What it is. pfwc and the emit split their work into static ranges. t267 measured:
- pfwc tail (window minus mean) 0.21;
- emit tail 0.09-0.15.

Replace the static ranges with a chunk queue claimed by atomic increment (a NoC semaphore on
one core). Determinism stays: each chunk writes to its own fixed output position. pfwc
segments are already per chunk. In the emit, a chunk's bucket slots come from the prefix over
per-chunk count rows instead of per-(core, mover) rows. That makes C a natural part of B: B
already rewrites the count rows.

Estimate: **0.2-0.35 untraced**. Tail recovery is at most the measured 0.21 plus ~0.12. The
claim costs ~1-2 µs of NoC round trip per chunk at 476 cycles for an exposed remote access
(hw-ceilings.md).

It is below the gate alone. It clears the gate only bundled with B, or if M shows the tails
grew after L1.

Risks:
1. **md5.** The record order inside a bucket must not depend on which core claims a chunk.
   Per-chunk count rows fix this, but they multiply the count rows (more chunks than cores),
   which makes the prefix bigger. Keep chunks at ~4 per core.
2. **The pfwc writer** is already the bound (t267 BRISC 1.795 / 2.006). Dynamic claim only
   trims the tail and does not lower the mean.

Spec.
1. No-device prep, after B lands or as B's second commit.
   - Host model: replay the claim with M's per-core busy times and the t197 per-gaussian cost.
     Print the modeled tail at 2, 4 and 8 chunks per core.
   - Then implement it behind `GSPLAT_TT_DYN_CLAIM=0`. Prove count-row and bucket identity on
     the host model.
2. Device A/B, then screenshot and keep/shelve rules exactly as in B. That means:
   - `ttp lock p100` and ssh-preflight;
   - ≥3 alternating runs per arm;
   - md5 must equal the base head's golden;
   - screenshot via opt/ttw/screenshot.sh, with the diff and PSNR against
     benchmarks/reference_v2/hero.png;
   - visual check;
   - gate 0.3, counting B+C against B-only if bundled.

Restrictions to restate in the spec: the same as in B.

---

## Rejected or marginal (with numbers; do not open tasks)

| idea | numbers | why not |
|---|---|---|
| FPU matmul for the blend quadratic form (q) | It would save ~14 SFPU ops x 2.81 microblocks per live record ≈ 39 cycles, about 0.62 ms if free (t205/t230 op counts; 1 cycle per live record = 0.0158 ms). But a dense 32x32x32 HiFi tile per gaussian-tile costs 32-64 cycles. The dense form throws away the microblock sparsity (~14x more pair work, t111), and it needs DEST, which blend holds. | Net ~0 or negative. tf32 is not md5-identical. t111's ablation ceiling was 2.0 of 10.6 ms at the time, and S2 and other work have since taken most of that room. The ablation flag BLEND_FPU_QF_ABL hits an #error with BLEND_SCHED, so even re-measuring needs a build. |
| FPU for the pfwc transform / cov_cam | pfwc TRISC ~1.76 vs writer ~1.8 (t232) | The writer is the bound. Faster TRISC math saves nothing. |
| Blend TRISC1 per-record cost (~255 vs a 210 SFPU bound) | Headroom ≤ 0.71 ms if the bound were reached. Remaining staging is 31 pushes and 14 loads per live record, and the replay buffer is full (32 slots). | Unpacker staging (U2, #252) measured +3.63, and S2 (DECODE_AHEAD=2) already landed. A realistic further cut of ~10-15 cycles is 0.16-0.24, below the gate. |
| Host serial work (rtargs 0.08, dispatch 0.08) | 0.16 together (t267) | Below the gate. d2h overlap is shelved (#171), and back-to-back throughput equals latency (t275: 11.618 vs 11.639). |
| Whole-pipeline single program (pfwc → K2 → sort → mat+blend) | The launch gaps are 0.031 traced (t267). | Gaps are already near zero. The L1 cost of keeping every stage's CBs resident is the #147 problem again. B is the useful piece of this. |
| K2 on idle TRISCs (L2) | -0.245 measured (t274) | Owned by #291. Superseded if B lands. |

Not re-proposed (shelved by earlier tasks): mat mid-tile split, blend per-tile fixed cost, mask-0
record drop, v2 mover table, tail-chained blend mask walk, emit-loop TRISC cursors, U2 staging,
emit pack on idle TRISCs, shared big-tile sort, chunk frustum cull, host residue/d2h overlap,
blend-waste work-cut.

## Sources

- docs/profile-iter199-t267.md (ttp/t267-profile-iter199 4f54951)
- docs/matblend-ready-t273/README.md (ttp/t293 branch: t273 model, t289 A/B, #293 Tracy overflow)
- docs/k2-split-t274.md (ttp/t274 branch), docs/k2-diet-t170/README.md
- docs/rerank-17ms.md (row 8: K2 count difference array, bundle only), docs/fuse-matblend-t147/
- t205/t230/t231 blend op counts; t111 FPU ablation; t197/t232 pfwc; docs/hw-ceilings.md; t275 throughput
- render/kernels/dataflow/pfwc_fuse.h (emit_pairs_diet: a pure rectangle walk)
- REPORT.html GPU rows G1/G2 (published, not measured)
