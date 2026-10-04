# t166 gate (yyzo-bh-07 p100a, 3 interleaved rounds x 30 views, bicycle 1024x1024, rev 23be983)

Arms: tip = PRECULL=1, no split (current branch default); old = PRECULL=1 + split 400/440/490;
base = new default (PRECULL=2 + split 400/440/490). view_total ms/view:

| round | tip    | old    | base   | base-tip |
|-------|--------|--------|--------|----------|
| r1    | 18.570 | 18.906 | 18.293 | -0.277   |
| r2    | 18.639 | 18.737 | 18.352 | -0.287   |
| r3    | 18.515 | 18.635 | 18.372 | -0.143   |
| mean  | 18.575 | 18.759 | 18.339 | -0.236   |

Stages (mean): base sort 4.61 vs tip 4.38 (+0.23), blend 8.90 vs 9.29 (-0.39).
Split also costs PRECULL=1 +0.18 ms/view (old vs tip).

Correctness: tip and old arms ALL_VIEWS_IDENTICAL to md5-r82new.txt. Base deterministic
(md5 equal across r1-r3, 46a725ab...), every diff pixel is 1 LSB, PSNR 74.24-77.81 dB.

Verdict: gate (>= 0.3 ms/view vs tip) missed: -0.236. Base only ties PRECULL=2 with fast
emit off (#162: 18.30/18.38), so the split recovers the fast-emit regression but the fast
loop still adds nothing under PRECULL=2. Nothing pushed.
