# t740: why the T2 FPU reduce could not be landed "in the trainer path" (2026-10-14 08:55 PDT)

No device run, no lock. Findings from the code at 4f1d4451 (t726 head) and origin/ttp/t728 @32d726ec.

1. There is no on-device trainer path. `git grep t2_bench|blend_bwd_bench` finds only the two
   standalone benches (render/train/{t2_bench,blend_bwd_bench}.cpp), their drive.sh scripts and
   the oracle helpers. Nothing chains forward -> loss -> T1 blend backward -> T2 param pass, and
   no code trains bicycle on device. So "real bicycle training ms/view" (spec step 3) and a 30k-
   iteration test-view PSNR/SSIM/LPIPS (step 4) cannot be measured yet; the only device numbers
   that exist are the T2 synthetic bench (#726: K=8 6.13 vs 13.85 ms) and the T0 dump oracle.
   T2 even reads host-prepared slot pages (gsplat/t2_dev.py write_inputs), not T1's output.

2. The T1 producer cannot emit the block-float pieces as specified. The shared grid is per T2
   slot page and stream column: E = exponent of the largest |x| over the page's 64 slots
   (t2_bench.cpp:103-133). T1 (bwd_writer.cpp) writes each slot as its own 64 B record to slot
   P[g] + j, from whichever core owns the pair's screen tile; the 64 slots of one T2 page come
   from many cores and times, so no T1 core knows the column max when it writes.
   Feasible places for the split:
   a) T2 on device: the reader pulls the fp32 slot page (4 KB instead of 3 x 2 KB bf16), compute
      takes the column max (stream c sits in tile columns c and c+16 of all 32 rows, so a column
      max-reduce), exponent via SFPU exexp, then three truncations to the 2^(E-7-8k) grid
      (mantissa mask) into CB_K. Cost is new SFPU work per page that #726's 6.13 ms did not pay
      (its split ran on the host), so the K=8 speedup must be re-measured with it.
   b) A fixed per-stream exponent known before T1 runs (e.g. a bound from the previous step):
      not exact when a column exceeds it, so it would need an overflow fallback; not recommended.

3. #728's latest RESULTS say the hero FAIL was fp32 cancellation and a T_min threshold tie, not
   a cross-chunk carry (commits 80f7c2e2, bd36d918); its fix is oracle-side tie handling.
   Merging it here is only needed once a T1 -> T2 chain exists.
