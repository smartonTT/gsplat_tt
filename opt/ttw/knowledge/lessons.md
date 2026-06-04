# gsplat project lessons (DONT_DO list)

**This is the gsplat (`gsplat_tt` / gstt2) project lesson pack.** It is read by
the tt-workflows loop at loop start **in addition to** the framework-generic
`tt-workflows/knowledge/lessons.md` (wired via `ttw.toml [project]
knowledge_dir`). The framework file holds codebase-agnostic rules; THIS file
holds the gsplat/render-specific lessons, anecdotes, and concrete values that
back those generic rules. Newest on top; each entry keeps its date.

---

## 2026-06-03 unified subchunk plan — one step per iter; 3 tries then split

- **Ship plan steps A→B→C→D→E as separate kept iterations, not one mega-diff.** The
  in-flight iter-53 monolith (materialize + directory + unified reader, D still
  open) is hard to bisect; prefer splitting on failure. User: *"If it can't be
  fixed or near fixed in three tries, then split it."*
- **Three tries per step:** repro → fix → re-verify on device; if still broken
  after three attempts, split into a smaller step or revert — do not loop
  indefinitely on the same combined changeset. See
  `~/.cursor/plans/unified_l1_chunk_pipeline_9cc07f6a.plan.md`.

## 2026-06-02 timing metric (gsplat specifics)

- **The gsplat timing metric is the AVERAGE FRAME TIME over ALL 30 bench views
  (`cameras_v2.json` order), warmup excluded; HERO is screenshot+PSNR only.**
  Backs the generic "timing = representative average over the project's full
  bench set" rule. User: *"the whole reason I gave you a 30-view test is to
  measure average frame time. Start reporting average frame time from now on,
  always rendering all 30 test frames. Hero is for screenshot, not timing."* A
  single hero frame is one (possibly unrepresentative) view; the amortized `@8v`
  (8-view ms/view) hides per-frame host bubbles that overlap across views — a PR
  once claimed "~170 ms/view @8v" while the real single-frame latency was ~300
  ms. → Harness renders every one of the 30 views, excludes warmup, reports
  `avg_frame_ms` (+ p50/min/max); `render/run.py` updated to the 30-view average.

## 2026-06-02 git / report-asset policy (gstt2 specifics)

- **DON'T blanket-ignore `opt/` — the report's linked assets must be TRACKED so
  the report stays valid forever; only genuine junk is ignored and belongs in
  `tmp/`.** A blanket `opt/` line in `gstt2/.git/info/exclude` (added so
  `iterate.sh`'s `git add -A` wouldn't sweep the 2.7G `opt/`) plus
  `opt/*.png`/`opt/REPORT.html` `.gitignore` rules meant NONE of the report's
  assets were committed — shared report links pointed at local-only files and
  broke for anyone else (and would vanish on a clean checkout). User: *"change
  skills to not gitignore opt … make sure all linked assets end up in there, so
  they stay valid forever … the junk should be in tmp, gitignored, but report
  stuff in opt."* Fix: removed the blanket exclude; rewrote `gstt2/.gitignore` to
  ignore-all-of-`opt/`-then-re-include exactly the report deliverables
  (`REPORT.html`, plan `*.md`, `opt/ttw/iters.jsonl`,
  `opt/metal-screenshots/ttw-*/*.png`, `opt/profiler/ttw-*/render.tracy` ≈ 240M);
  added `gstt2/.gitattributes` git-LFS rules for `opt/**/*.png`, `opt/**/*.tracy`,
  `*.ply` (the repo already used LFS for `tests/fixtures/hero/*.npz`); stale
  pre-ttw experiment screenshot dirs (`metal-iter-*`, `amendment-*`, ~1.2G) and
  `*.npy`/logs stay ignored. The surgical re-include keeps `git add -A` from
  sweeping the 4519 loose junk files. → Backs the generic "track linked report
  assets; never blanket-ignore the report-output dir; binaries via LFS; junk to
  tmp/" rule. See `gstt2/.gitignore`, `gstt2/.gitattributes`.

- **DON'T track `.ttw/` in gstt2.** A loop's `git add -A` once committed 541
  per-project `.ttw` files (build ids, locks, device logs) into `gstt2`. `.ttw/`
  is machine-local + transient. → The 541 already-tracked files get `git rm
  --cached` once the live loop is stopped (to avoid racing the loop's
  per-iteration `git add -A`). Backs the generic "never track `.ttw/`" rule.

## 2026-06-02 build / device-build-root (gstt2 specifics)

- **DON'T `build.sh` without first pushing source to the device's BUILD root —
  gstt2 builds in `/localdev` but mutagen only live-mirrors edits to `/proj_sw`,
  so a bare `cmake --build` recompiles a STALE tree (`ninja: no work to do`), the
  binary hash does NOT move, and the iter is a no-delta fake.** A worker editing
  kernel `.cpp`s saw `build.sh` report success while `bin=` stayed identical — the
  device's `/localdev` work-tree never received the edits (mutagen syncs the
  Mac→`/proj_sw` path, not the harness build path). Recovery: run
  `gstt2/scripts/sync_to_bh30.sh` before `build.sh` each iter and verify the
  `bin=` fingerprint moved. The safety net held: `done.sh require_build_delta=1`
  REJECTS an identical-`bin=` keep. → Workers are told verbatim to
  `sync_to_bh30.sh` before `build.sh`. Backs the generic "build succeeded ≠ device
  built YOUR source when build root ≠ mirror root" rule. See
  `gstt2/scripts/sync_to_bh30.sh`.

- **DON'T accept a "milestone PASS" without checking the binary fingerprint
  actually CHANGED — a stale binary scoring the prior path's PSNR was logged as
  both M0 AND M1 of Stage 2, so the loop "advanced" two milestones while ZERO new
  code ran on-device.** The Stage-2 host-free rewrite logged iter 40 "M0
  32B-record = 63.85 dB keep" and iter 41 "M1 L1-radix = 63.85 dB keep", both
  bit-identical (`bin=d193d161`) to iter 39's pre-Stage-2 ideal path. The M0/M1
  SOURCE was genuinely committed (and `_gsplat_cpu.so` is the correct artifact —
  `gsplat_tt` static-links into it), but the device `.so` never rebuilt, so each
  "keep" ran iter 39's ideal-path binary mislabeled as the new milestone. devrun's
  build-ID check PASSED because it only asks "live == stamped?" (both old), not
  "did anything change vs the last iter?". Fixes: (1) the build-ID now also hashes
  `kernel_src_dir` (the on-device kernel `.cpp` tree), not just `bin_artifact`;
  (2) `done.sh require_build_delta=1` rejects a kept `cpp` iter whose `bin=` equals
  the prior iter. The false iter-41 was reverted; M0 itself is unverified and must
  be genuinely rebuilt+measured. Backs the generic build-delta integrity rule. See
  `opt/host-free-l1-render-plan.md`.

## 2026-06-02 profiling / Tracy (gsplat specifics)

- **The Tracy capture is ALWAYS the FULL 30-view render** (`opt/profiler/capture_tracy.sh
  <iter>`, `ttw.toml [profile]`, `python -m tracy --dump-device-data-mid-run` — the
  only path that streams DEVICE zones since gsplat never closes the device); the
  fast gate `verify_cmd` stays `--views 1`. Verify 30-view coverage via the dumped
  `profile_log_device.csv` row count (~30× the ~65.6k 1-view baseline; ttw-043 =
  1,016,801 rows / 16.3 MB / 1.58M zones). A 1-view or warmup-only `.tracy` is NOT
  a satisfied gate. Context: the profiler-instrumented build (`TT_METAL_DEVICE_PROFILER=1`)
  once failed to JIT-compile `gather_visible_scatter` and `sort_bin` so iters
  41/42/43 `.tracy` were partial JIT-warmup captures — root cause was a runtime
  ternary passed to `DeviceZoneScopedN` (`count_only ? "proj_count" : "proj_scatter"`
  / `mode==0 ? "sort_bin_hist" : "sort_bucket_emit"`); FIXED 2026-06-02 commit
  `b85fbf9` by opening a static-string-literal zone inside each branch (63.85 dB
  bit-identical). Backs the generic `DeviceZoneScopedN` compile-time-literal quirk
  in `quirks.md`.

- **DON'T trust the SUMMARY `sort=` field as the sort STAGE's cost — on the
  L1_RECORD / sort→blend-pipe path it LUMPS cull+blend into the sort host-timer
  (`blend=0.0`), so the true DRAM radix is only ~7.5 ms, not the ~232 ms the timer
  shows.** Stage-2 M2 was planned around "drop the DRAM radix ⇒ sort 232→0"; on
  measuring (device zones, pipe-OFF for a clean split) the 232 ms is a lumped pipe
  artifact and the real radix is ~7.5 ms + publish 1.3 ms — dropping it recovers
  only ~9 ms, AND it's structurally blocked (overflow tiles `max_tile_n=26496 >
  BUCKET_FIT=8192` take a gather fallback needing depth-order `sort_sorted_ids` +
  `cull_masks`). The REAL ms/view cost is **blend=179 ms (cull ~80 SFPU + blend
  ~95)**, so M2's lever is the **microblock-major blend + transmittance early-out**
  (blend is currently gaussian-major with NO early-out), not the sort. iter 43
  banked a real win: dropped the dead 64 B `tile_recs` dense scatter (~216 MB/frame)
  the L1_RECORD path never reads (bin 43→41 ms, 63.85 dB bit-identical). Backs the
  generic "when a stage timer looks huge, confirm what it actually drains before
  framing a milestone around it" rule. See `opt/host-free-l1-render-plan.md` §6/§12.

## 2026-06-01 corrections (gsplat render path)

- **DON'T profile or optimize FUSED_TILE/gather "production" when the user
  mandated the IDEAL path — TILE_BUCKET full-record scatter + L1-resident blend,
  host out of loop.** We kept shipping/gating the gather path, concluded "blend is
  SFPU-bound" from a flawed A/B, and opened `render-labeled.tracy` (FUSED_TILE)
  while the user asked for L1-resident. The screenshot DOES show device zone names
  (`proj`, `cull`, `blend`, `blend_rd`, `fused`) — the `???` rows are Tracy
  GPU-context thread names. The ~90ms gap between bursts is host `Finish`/setup
  between stages/views (needs HOST Tracy zones). L1 at 63.85 dB is valid
  (cull-fold config, not the old 42 dB bug); 430MB scatter is acceptable. Soft-float
  cull on BRISC (`MB_DEVCULL`) is NOT the ideal path — SFPU `cull_masks` only. →
  Loop verify_cmd + default = TILE_BUCKET, no FUSED_TILE, no MB_DEVCULL; ideal path
  only until host+locks gone; zone-sum evidence before any "bound" claim.

- **DON'T conclude a fused-kernel host-free refactor is "ms-neutral" until you've
  removed EVERY host barrier in the chain.** iter-36's on-device scan looked
  neutral only because it still kept the mid-chain M-read drain AND the cap-sized
  depth D2H; finishing the fusion (pre-size to the N ceiling `padded_n`, drop the
  mid M read, drop the unused depth D2H) flipped it to a real proj −4.1ms / −2.1%
  ms/view win. The compact visible count M is ALWAYS ≤ N, so pre-sizing outputs to
  the per-frame safe ceiling lets the scatter dispatch WITHOUT the host knowing M
  first; the host-free render consumes NONE of the compact attr VALUES so the
  post-chain readback can be a single 1-page `proj_M` read. A/B @8v same binary:
  OFF `proj=32.4 ms/view=173.2`, ON `proj=28.3 ms/view=169.6`, 63.85 dB
  bit-identical → DEFAULT-ON `GSPLAT_TT_PROJ_DEVICE_SCAN`. Backs the generic "when
  a host-free block measures neutral, check for a residual sizing/sync barrier
  still pinning the device idle; the safe-ceiling trick removes a need-the-count
  dependency without an extra D2H" rule. See `opt/plan-high-utilization-pipeline.md`
  §9.4.

- **DON'T default-on a host-free refactor that holds the gate but is ms-NEUTRAL —
  measure the bridge's real wall-time first.** STEP-3 on-device exclusive scan for
  the project gather compaction (`gather_scan_bases.cpp`, gated
  `GSPLAT_TT_PROJ_DEVICE_SCAN`, cpp#105): the count pass D2H's only 110 per-core
  counts (≈7 KB) and the host exclusive-scan is a 110-iter loop (≈µs); the one
  `Finish` it drops is below the 0.1 ms stage-timer resolution. A/B same binary:
  OFF `proj=32.4 ms/view=173.1`, ON `proj=32.4 ms/view=173.2`, 63.85 dB
  bit-identical — CORRECT but ms-neutral → kept gated OFF + banked as a host-free
  building block. Backs the generic "delete-a-host-bridge only pays when the bridge
  is on the measured critical path; quantify with a ledger first; ms-neutral
  host-free progress stays gated-off until surrounding barriers fall" rule. See
  `opt/plan-high-utilization-pipeline.md` §9 / §9.3.

- **DON'T re-upload a CONSTANT array to the device every frame — tile_assign's
  all-ones keep-mask was a 13.5 MB/frame H2D of pure 1s.** The `GSPLAT_TT_TA_TIMING`
  breakdown showed `k4 = 8–11 ms (1-view)` was NOT cull compute but the `ta_no_cull`
  default path re-uploading an all-ones keep mask (`cap_p_elems` u32 ≈ 13.5 MB of
  constant 1s) H2D + a `Finish()` EVERY frame. The mask is constant (cull off ⇒
  every AABB pair kept) → fill `buf_keep` with 1s ONCE per (re)allocation/grow and
  cache a `buf_keep_all_ones` flag; skip the per-frame H2D+Finish. Result (cpp#103):
  k4 11.3 ms alloc-frame → 0.0 ms later frames; ta 21.2→10.2 ms, ms/view
  183.4→173.5 (−5.4%) @8v, 63.85 dB bit-identical. Backs the generic "any per-frame
  H2D/D2H of unchanging data is dead host work — hoist to the alloc path; mine the
  per-substage breakdown for the real bottleneck when the assigned lever is a
  non-win" rule. See `opt/plan-high-utilization-pipeline.md` §8.9.

- **DON'T assume a load-rebalance that won on one stage transfers to a "same-shape"
  sibling — MEASURE the per-core work proxy first; the tile_assign K2 scatter is
  ALREADY balanced (write 1.0004×, set_g 1.037×), so the project striding win does
  NOT apply.** Added a gated host-only per-core work proxy (`GSPLAT_TT_TA_STATS`,
  cpp#100/101) instead of building the scheduler. K2 splits the already-compacted
  uniform pair domain EQUALLY across 110 cores (write max/mean = 1.0004); the only
  variable cost `set_g` is max/mean = 1.037. A strided split drops the gspan skew
  only 1.037→1.027 while ADDING ~239 binary-searches/core = a wash/loss. NO strided
  K2 shipped. The aggregate "1.54×" was cross-stage BRISC-busy noise. (Real lever
  found instead → the all-ones keep-mask H2D above.) Backs the generic "same
  contiguous-N-chunk shape does NOT imply same skew — measure the per-core proxy
  AND the strided-projection bound before writing a scheduler" rule. See
  `opt/plan-high-utilization-pipeline.md` §8.8.

- **DON'T attribute an aggregate-profiler "stage imbalance" to the whole stage —
  break it into sub-kernels; for `project` the 1.56/33 ms was really a 2.12× WRITE
  skew on ONE sub-kernel (gather scatter), fixed for free by striding.** A gated
  host-only per-core visible-count dump (`GSPLAT_TT_GATHER_STATS`, cpp#96) proved
  the dominant cost is `gather_visible_scatter` (~51.6 ms of 58.7 ms 1-view proj;
  means_cam/pfwc are uniform/cheap), and the scene is spatially clustered ⇒ min/core
  9,235, max/core 36,300, mean 17,126 = max/mean 2.12. FIX = STRIDED tile→core
  assignment (`gather_visible_device.cpp::split_strided` + `t_stride` arg): skew
  2.12→1.21×, gather 51.6→36.6 ms, proj 37.0→32.4 ms, ms/view 191.9→183.7 (−4.3%)
  @8v, 63.85 dB bit-identical → DEFAULT-ON `GSPLAT_TT_PROJ_BALANCE` (cpp#97). Backs
  the generic "a stage imbalance can hide in ONE sub-kernel — split + per-core dump
  to find which/what kind of skew; cheap STRIDING captures spatially-clustered skew"
  rule. See `opt/plan-high-utilization-pipeline.md` §8.7.

- **DON'T set out to "implement LPT load balancing" without first checking whether
  it's ALREADY SHIPPED — for gsplat cull+blend it is, and it's within 1.3% of
  optimal.** `sort_device.cpp::build_lpt` (mirrored in
  `blend_device.cpp::compute_lpt_assignment`) already does textbook LPT (sort
  non-empty tiles by descending candidate count, greedily assign to least-loaded
  core, published as `sort_lpt_tile_ids`/`sort_lpt_meta`). Proof (gated
  `GSPLAT_TT_LPT_STATS`, cpp#95): 1024 non-empty tiles / 110 cores, mean 39,151
  cand/core, max/core 39,664 ⇒ makespan/mean = 1.013; heaviest tile = 26,544 =
  0.678× the per-core mean. Realized SFPU imbalance ~1.067 (cull) / 1.067–1.072
  (blend). NO load-balance win to ship for cull/blend; the real lever is SHRINKING
  the SFPU candidate count. project (1.56) and tile_assign (1.54) show real
  imbalance but are N-chunk splits off the cull/blend critical path. Backs the
  generic "before building an optimization, grep for it — measure realized per-core
  spread AND the cost-proxy bound first" rule. See `opt/blend-data-movement-plan.md`
  §15 + `opt/plan-high-utilization-pipeline.md` §8.6.

- **DON'T assume the ~80 ms `CULL_SPLIT` SFPU pass is a host/dispatch bubble — it
  is irreducible DEVICE SFPU compute, and it shares SFPU cores with the blend, so
  the only thing overlappable is the 26 ms sort.** Measured three ways: (1)
  ROUTE-C sequential-record cull (no gather) was 74 ms vs 80 ms — only 6 ms is
  gather; (2) gating the standalone host `Finish` (`GSPLAT_TT_CULL_PIPELINE`)
  recovered just ~1 ms (300.0→298.8 ms/view); (3) profiler shows NCRISC ~93% /
  TRISC ~84% busy. cull-SFPU (80) and blend-SFPU (~93) run on the SAME SFPU cores
  ⇒ 173 ms serial SFPU even on separate command queues; fusing is blocked by the
  iter-26 DEST hazard. Banked gated OFF. Backs the generic "confirm a cost is
  latency/idle (overlappable) vs a saturated shared compute resource (not
  overlappable) before chasing an overlap win" rule. See
  `opt/blend-data-movement-plan.md` §14.

- **DON'T assume a gated "fast path" that is slower end-to-end is paying for the
  EXPENSIVE stage you think — check for a DUPLICATED pass first.** The L1-resident
  bucket blend was 372.9 ms/view vs production 303.9; the `BUCKET_MASK` (ROUTE C)
  config ran TWO full SFPU cull passes per frame — `[BUCKET_CULL]≈74 ms` (sort-stage
  RMW into record word 10) AND `[CULL_SPLIT]≈80 ms` (the production standalone cull
  filling `cull_masks`). The bucket tiles read `recp[10]` and ignore `cull_masks`,
  so the 74 ms is pure overhead. FIX = drop `BUCKET_MASK`: bucket reader reads the
  shared `cull_masks` the single `CULL_SPLIT` already fills. Result: sort
  99.4→26.2 ms, 298.8 ms/view at 63.85 dB — BEATS production 303.4. Corollary
  (Option-1 trap): re-folding the soft-float cull INLINE into the L1-resident blend
  reader BLEW blend to 416.9 ms — production's inline cull is free only because it
  hides in the random-gather latency shadow; with L1-resident records the cull
  becomes fully-exposed serial work, so a pre-computed SFPU `cull_masks` is
  ESSENTIAL once the gather shadow is gone. Backs the generic "when a fused/folded
  stage and a standalone stage both compute the same thing, delete the duplicate"
  rule. See `opt/blend-data-movement-plan.md` §13.

- **DON'T assume `cb_wait_front`/`cb_push_back` protects a compute kernel that
  reads the CB row in L1 DIRECTLY on the MATH thread** (gsplat bucket-blend
  Lb>64 42 dB race). `alpha_blend_compute_mb.cpp` reads each coeff row from L1 on
  MATH via `get_tile_address`, but `cb_wait_front`/`cb_pop_front` are UNPACK-only,
  so on the throttle-free bucket feed the producer overwrites the slot before MATH
  loads it ⇒ torn rows. FIX (gated `GSPLAT_TT_BUCKET_CB_FENCE=1`): (1)
  `invalidate_l1_cache()` on MATH before the row loads (Blackhole L1 is
  write-through); (2) a MATH→UNPACK back-pressure ack over the hardware mailbox so
  UNPACK can't free a slot before MATH read it. Result: bucket fast path 63.85 dB,
  blend 178.9 ms (faster than production's 192.9 ms blend), no EMIT_SPIN. The
  generic mechanism is now in `tt-workflows/knowledge/quirks.md` (direct-L1-read on
  MATH needs an explicit consumer→producer handshake). See
  `opt/blend-data-movement-plan.md` §11.

- **DON'T trust a clean "small-N passes / large-N fails" FIT split to mean the DATA
  ASSEMBLY is wrong — for the L1-bucket blend the Lb>64 42 dB is a fast-PRODUCER
  timing race, and the dense records + baked mask are bit-perfect.** Three on-device
  experiments refute the §10 "sort-stage record assembly" theory:
  `BUCKET_FORCE_INLINE` (ignore recp[10], recompute mask inline) → 63.85 dB;
  `BUCKET_MASK_DEBUG` (mask=recp[10] + inline compare) → 63.85 dB BSUM mism=0;
  `BUCKET_EMIT_SPIN` 200→37 dB, 2000→63.85 dB and deeper `CB_MB_COEFF` = WORSE ⇒ a
  producer-ahead race. Backs the generic "when more buffering makes it worse and a
  per-iteration delay makes it correct, it's a producer-ahead race, not data
  assembly" debug rule. See `opt/blend-data-movement-plan.md` §11.

- **DON'T blame a gated path's PSNR regression on the new code without bisecting
  the SET of inputs it touches — the L1-resident bucket-blend 42 dB is a
  PRE-EXISTING bug for tiles with Lb>64, not a ROUTE C defect.** Bisection (one var
  each, via devrun): `BUCKET_MASK=0` → 42 (not the mask); `NOSORT` → 16 (sort does
  real work); `BUCKET_FIT=16`→63.85 AND `FIT=64`→63.85 (bucket plumbing + insertion
  sort + LSD radix correct for Lb≤64); `FIT=8192`→42. Only tiles differing between
  passing FIT=64 and failing FIT=8192 are Lb∈(64,8192] ⇒ the dense / multi-binning
  regime. Backs the generic "when a gated path regresses, threshold-sweep to find
  WHICH inputs are wrong before touching the new code" rule. See
  `opt/blend-data-movement-plan.md` §10.

- **MB_CULL_SPIN is a reader READ-COMPLETION window, not a DRAM write-settle
  artifact** (gsplat blend masks). iter 15 (67c7e31) did the two-program L1 mask
  handoff: masks landed bit-identically in L1 (0 mismatches, 63.85 dB), yet
  dropping the per-candidate spin still gave 30 dB; spin sweep (0→30.1 dB … 512→
  63.85 dB) shows the gate is met only where blend==baseline, regardless of mask
  storage. The spin is the blend reader consuming a freshly `noc_async_read`'d page
  too soon after the barrier (a per-candidate read-completion stall on RANDOM
  single-page reads), NOT a DRAM-bank settle. The real bottleneck is the ~1.9 GB
  random attr gather; the win is the Stage C2 contiguous per-tile payload (a pure
  SEQUENTIAL stream). Scaffold gated `GSPLAT_TT_L1_MASKS=1`. See
  `opt/host-free-fusion.md`.

## 2026-05-31 corrections (gsplat render path)

- **DON'T treat "device kernels exist" as "0 host in the render loop."** With all
  `GSPLAT_TT_*=1` flags, `render_full_py` still runs host bridges: tile_assign
  prefix-sum + K3 `m2_thresh` + compaction (H1/H2), sort CPU fallback vectors,
  blend `build_tile_assignment` + per-tile count scan + image readback + `Finish()`
  between cull/blend. Spin tuning is NOT progress until the orchestration path is
  host-free. → First goal: `render_full_py` calls ONLY device entry points; no
  `gsplat_cpu::tile_assign` / `sort_and_bin` / host attr-id build; fuse per-tile
  work if it deletes stage-locks. Final image D2H once per view is OK; everything
  between project and blend must be device-resident. Backs the generic north-star
  "host-free hot path" rule.

- **DON'T over-complicate a first kernel bring-up — SIMPLIFY to a no-op, then build
  up.** Debugging the full SFPU cull (Mahalanobis math) at once burned hours. The
  correct first step is a pass-through no-op (every microblock keeps for every
  splat) to prove data flows end-to-end, THEN add math one term at a time,
  re-verifying each step. Backs the generic "simplify to a no-op then build up"
  bring-up rule.

- **Anti-thrash anecdote:** a Sonnet worker ran ~17 device probes in 10 min chasing
  a faster sort it couldn't land, reused `--iter 41` (clobbering the kept
  iteration's screenshot with a 4.35 dB garbage re-run), and churned ~65 min past a
  banked milestone before intervention. → workers switched Sonnet 4.6 → Opus 4.8
  (`ttw.toml [loop] worker_model`); Sonnet repeatedly mislabeled milestones and
  could not self-correct. Backs the generic anti-thrash watchdog + bounded-effort +
  Opus-default rules.

- **First milestone anecdote:** after iter 1 (SFPU cull spin fence, commit
  `7e5bced`) the supervisor idled — a kept iteration is one lap toward ~1 ms/frame,
  not mission complete. Backs the generic "a milestone/commit/subagent completion is
  not a stop" rule.

## gsplat env-flag glossary

- `GSPLAT_TT_DEVICE_*` / `GSPLAT_TT_RESIDENT_*` — device-resident project / gather /
  tile_assign / sort / blend stages (the host-free path).
- `GSPLAT_TT_SFPU_CULL=1` — SFPU microblock cull (fills `cull_masks`). The ideal
  path; soft-float `MB_DEVCULL` is NOT.
- `GSPLAT_TT_TILE_BUCKET=1` + `GSPLAT_TT_BUCKET_FIT=8192` — L1-resident bucket blend.
- `GSPLAT_TT_FUSED_TILE=0` — the FUSED_TILE "production" path is OFF; the loop
  optimizes the TILE_BUCKET / L1-resident IDEAL path.
- `GSPLAT_TT_L1_RECORD=1` — on-device record format.
- `GSPLAT_TT_PROJ_DEVICE_SCAN` / `GSPLAT_TT_TA_DEVICE_SCAN` / `GSPLAT_TT_PROJ_BALANCE`
  — banked host-free building blocks (see lessons above).
- gated diagnostics (default OFF, host-only, zero production effect):
  `GSPLAT_TT_GATHER_STATS`, `GSPLAT_TT_TA_STATS`, `GSPLAT_TT_LPT_STATS`,
  `GSPLAT_TT_TA_TIMING`, `GSPLAT_TT_MB_TIMING`.
