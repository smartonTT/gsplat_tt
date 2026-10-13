# t637 working state (delete when README is final)
- Runs (detached, run 1773): t637model (views 6:30 all 9 opts -> out-C120-v{12-18,18-24,24-30}.json),
  then t637extra (today,u64_s8,u64g1_s8,u64g32_s8 --tag s8 -> out-C120-v{0-10,10-20,20-30}-s8.json).
  Logs/rc: tt-project/state/runs/1773/t637{model,extra}.{log,rc}.
- Next: python3 docs/emit-burst-model/analyze.py 4 > tables.txt; fill README placeholders.
- 12-view result (C120, L4, R4D8 emit Δ raw / x0.75 calibrated): mcull +0.63 (skip 0.41, gross 0.43);
  mcull_s8 +0.46; u64 +0.20 (skip 0.58, gross 0.62); u64g1 +0.18, bursts p99 2 / max 4 (fits ring 4/8).
  Residual +0.18 in u64g1 = TRISC owner imbalance (few tiles per 512-pair window); _s8 runs test that.
- Net u64g1 R4D8: gross 0.62 - emit 0.13..0.18 - host unit list ~0.19 (t493 chunk list) - pfwc reader
  gather (16x256 B reads per plane per tile; unmeasured, 0..0.25) - writer mask transpose (~0.01)
  = about 0.0..0.3, central ~0.15.
- g1 lane interleave is cheap: pfwc writer walks set mask bits per lane (writer_pfwc_split.cpp:387);
  g1 order = walk a 16x64 bit-transposed mask, no L1 data shuffle.
- t493 "no pfwc saving on p150" was latency --dump-views at aiclk 800 (dump hides pfwc, #464);
  t470 b2b p150 aiclk 1350 measured gross -0.525. Model vs t493 c48: predicts loss ~+0.33 vs +0.177 meas.
- Followup if verdict holds: device probe of pfwc unit-gather (u64 lists, b2b) to measure gross - gather
  before building the writer transpose; gate pfwc saving >= 0.35 ms.
