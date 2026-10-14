# t780 working notes (unpack-thread staging of the T1 pair scalars)

- PLAN t773 (6dd38e8c) does not cover this lever: it defers pair setup to #772, whose
  follow-up is this task.
- Dst is full (fp32, full sync, 8 tiles): 0 T, 1 BH, 2-4 g, 5 x ramp + consts, 6 y ramp, 7 ACC.
  SFPU vector i of a tile = Dst rows 2i, 2i+1; face f = vectors 8f..8f+7.
  Free whole faces: tile 6 faces 1 and 3 (y ramp reads only V & ~9), tile 5 face 3 (x ramp
  reads V & 9 in 0..15, consts at 16..23). No 12 contiguous free vectors without moving data.
- The planes and ramps already reach Dst by unpack-to-dest fp32 (copy_tile, UnpackToDestFp32).
- Design picked for step 2 (if the probe passes): single staging region filled by the UNPACK
  thread (2-face fp32 unpack-to-dest), MATH copies 12 vectors staging -> DR_S/P0/LX/LY with
  SFPLOAD/SFPSTORE (24 Tensix instr, no RISC-V L1 loads) and frees the staging with a semaphore;
  bodies unchanged. Estimated MATH setup ~172 -> ~50 cyc/pair if the copy issues at ~1 cyc each.
- Step 1 (this probe, docs/t780/probe.sh): reader writes the replicated words (1.5 KB/pair) and
  the MATH record wait (rec) and hero ms must not grow.
