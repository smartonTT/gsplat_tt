id=95
sha=b744d2a
ts=2026-06-11T09:49:16-0700
desc=iter 128: sort_bucket_emit: hoist per-gaussian-invariant 32B record packing (cov/depth_key/UNORM16 op+color) out of the per-pair scatter loop. iter-128 ablation MEASURED pack_rec=27ms=71% of the 40.9ms scatter (the scattered-DRAM-write premise was REFUTED: 32B writes only 1.6ms); only the tile-local mean varies per pair, so the invariant words are computed once per gaussian (pack_invariants in the blendrec-read branch). Bit-identical hero md5 e3fefb11, 63.95dB. sort_bucket_emit 40.9->30.6, NCRISC-KERNEL 144.6->134.2, BRISC-FW ~179->169; ms_view 195.6->187.4 avg / 163.7->157.6 min.
bin=2737d8866f393d17
