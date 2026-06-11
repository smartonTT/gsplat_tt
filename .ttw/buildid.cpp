id=92
sha=d754416
ts=2026-06-11T07:37:36-0700
desc=iter 124: S5.3 tile_assign P-read deletion — pair buffers + K2 scatter work-split sized to static pair_ceiling() (4,718,592); K2 reads clamped P from resident ta_pairs_P ctrl page (InterleavedAddrGen<true>); scan_bases clamps published P + publishes overflow/true-P; sort_device hard-fails on overflow. Behind host_free_mp_enabled (reversible). Frame-neutral trace-prerequisite (removes one of three mid-frame host blocking-read drains).
bin=ba1fecf45513c2a7
