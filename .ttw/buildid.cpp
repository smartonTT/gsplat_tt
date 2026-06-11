id=93
sha=de6d1db
ts=2026-06-11T08:19:44-0700
desc=iter 125: S5.4: parallelize on-device sort bin-layout Pass-2 base emit across 16 cores (new sort_bin_emit.cpp). Coordinator (sort_bin_layout.cpp) publishes per-worker running-prefix checkpoints (page_acc pages + rec_acc counts at each worker first source-core) to a small DRAM ckpt buffer; emit workers replay the exact Pass-2 accumulation over disjoint source-core ranges (contention-free) -> bit-identical bin2d/l1_rec_base bases. CQ-ordered after the coordinator (no semaphores; follows the proven program-ordering pattern). Recovers part of the iter-121 +13ms single-core layout regression: A/B (matched thermal) min_ms 213.0 -> 211.5 (-1.5), BRISC-FW 226.2 -> 224.7, NCRISC-KERNEL 191.4 -> 189.9, TRISC unchanged. Bit-exact (layout_verify=true: 33/33 IDENTICAL across 30 views) + hero 63.95dB. Launch overhead caps the 2-program win now; folds away under Metal Trace.
bin=c70695503e7573cf
