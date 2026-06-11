id=89
sha=56d8f42
ts=2026-06-11T04:57:06-0700
desc=iter 121: S5.1: enable on-device sort bin-layout (sort_device_layout_enabled=true). Recovered the missing sort_bin_layout.cpp kernel (host referenced it but it was absent from render/kernels/ — the real reason the flag was off), rewrote it as an efficient single-core bulk-row pass, and extended it to populate l1_rec_base (the L1_RECORD scatter's per-(core,tile) bucket base) which the legacy kernel lacked. Removes host hist-read + host-LPT + 6 H2D re-uploads; host now reads only resident result buffers + a tiny ctrl page.
bin=241ea500aa9bdc9d
