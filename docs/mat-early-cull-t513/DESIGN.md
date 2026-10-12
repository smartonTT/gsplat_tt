# t513: mat lever — early cull + select-ranks per subchunk (design, not built yet)

Base: best-iter-223 (5ca019ae) + t494/t493 env-gated default-off changes. Branch
ttp/t513-lever-stack-mat-fill-jobs-big-tile-split.

## Why not literal fill jobs
#506/#508 model (runs/1400/m, crit.py, 15 views): the critical mover is always NCRISC of
one core, makespan 1.97-2.26 ms; the 20th core ends ~1.66-1.88 ms, the other mover of the
critical core ends 0.93-1.42 ms. On the critical core the time is serial on one mover:
- big-item cores (10/15 views, e.g. 2-2): keys ~250-325 us, full radix sort 1116-1365,
  gather ~290-300, cull_wait 303 (TRISC cull starts only after gather), write 6.
- whole-tile cores (14-2, 1-3): read 45-156, sort 992-1085, permute ~255 (2 subchunks),
  cull_wait ~580-605 (2 jobs, posted after permute), write ~12.
Both waits are TRISC cull time that could overlap the sort; the big-item sort sorts the
whole tile to get one subchunk's ranks. So the lever attacks the critical mover's serial
chain first (judgment call vs. the spec's "fill jobs"): it removes the work rather than
moving it. Splitting/fill jobs (stage 4) only if the spread is still wide after this.

## Gate
`GSPLAT_TT_MAT_EARLY_CULL=1` -> define `MAT_EARLY_CULL=1`, only with MATCULL_TRISC_FILL and
SORT_ONELAUNCH (not OL_MAT_SELECT), set in render/host/sort_device.cpp
add_mat_cbs_and_defines before `mat_defines_m0 = mat_defines;` (line ~661) so both movers get it.

## Stage 1: whole-tile early cull (sort_subchunk_materialize.cpp ~635-690)
1. read_bucket as now.
2. Gather keys from buck word3 into kA (+kmin/kmax) BEFORE posting the cull (the cull
   patches the mask into word3).
3. E = post_job(buck, count) (cull is per-record, order-independent; <=2 jobs in flight
   counting pend).
4. Same kB placement; sorted = sort_ids_gathered(kA, count, kmin, kmax, ...).
5. flush_pending(); wait_job(E).
6. Per subchunk: permute_records(buck, slab, sorted + s*fit, Ls) -> emit_at directly (masks
   already in word3, travel with the record); no pend.
Expected: ~580 us of cull_wait off whole-tile critical cores.

## Stage 2+3: big item, select ranks of one subchunk (~694-830, N <= MAX_N path)
1. keys as now (k[] in bs, kmin/kmax).
2. s = select_bins(k, N, sc_off, sc_off + L_sub, hist).
3. If s.m <= 16384 (buck cap; 4*m u32 fits the slab):
   a. collect candidate indices cv in index order (in place in bs);
   b. flush_pending;
   c. gather candidates: read_bucket chunks of 8192 records into slab, monotone pointer over
      cv, loopback copy slab[cv[j]-r0] -> buck[j]; barrier;
   d. extract candidate keys from buck word3 into slab arrays k/v/k2/v2 (4m u32) BEFORE
      post_job;
   e. E = post_job(buck, m); res = sort_ids_gathered(...); ids[i-lo] = res[i-s.base];
   f. wait_job(E); permute_records(buck, slab, ids, L); emit.
4. Else the old path (k intact: select_bins only reads it).
Estimate: big critical core ~2250 -> ~1400-1600 us. Then update the host LPT cost for
onelaunch big items (cnt + l_sub) to match.
Note #121: OL_MAT_SELECT measured +0.15 ms because it disabled matblend fusion
(`!ol_mat_select()` at sort_device.cpp:877); this design keeps fusion (no OL_MAT_SELECT).

## Build check
Host unit test like tests/unit/test_sort_onelaunch_v2.cpp: candidate gather + candidate
sort ids == full stable sort ids for random/tied keys and all subchunk offsets.

## Device plan (bh-33, p100, IRD 138447; every step inside `ttp lock p100`)
Driver modeled on docs/land-t490/drive.sh + remote_ab.sh (ssh/scp shims forcing
StrictHostKeyChecking=yes, ssh-preflight, opt/sync_remote.sh bh-33 /localdev/smarton/gstt2-t513
<rev>, devrun.sh --host bh-33 --no-verify --timeout 560), MESH_DEVICE=P100,
TT_METAL_CACHE_RENDER per task. Modes:
- prof: Tracy b2b MATCULL capture, views 0:10 (`--back-to-back --b2b-passes 1 --b2b-warmup 0`,
  GSPLAT_TT_MATCULL_PROF=1, GSPLAT_TT_PROFILE_READ_EVERY=11, as docs/mat-job-zones-t413/
  remote_tracy.sh), once for A at the start and once for B; analyse with runs/1400/m
  extract.py/ana506 + crit.py for the per-core finish spread.
- ab: >=3 alternating b2b rounds A (unset) / B (GSPLAT_TT_MAT_EARLY_CULL=1).
- dump: one --dump-views pass per arm, md5 vs 906e0435 30/30 (opt/md5_golden.py).
- hero: opt/ttw/screenshot.sh with H=bh-33 T=/localdev/smarton/gstt2-t513 (it defaults to
  yyzo-bh-04), PSNR vs benchmarks/reference_v2/hero.png, look for seams.
bh-33 load ~15 from t510's CPU model (nice 19): alternate arms, note the noise.
