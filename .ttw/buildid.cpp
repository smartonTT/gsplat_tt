id=94
sha=ecba17b
ts=2026-06-11T09:20:14-0700
desc=iter 127: recover frame to baseline: disable on-device sort bin-layout (sort_device_layout_enabled=false) AND the S5.3 host-free M/P over-provisioning (host_free_mp_enabled=false) now that iter-126 MEASURED the Metal Trace endgame as a no-go (replay removes <0.5ms; the ~92ms BRISC-FW is on-device firmware+NCRISC dataflow trace replays unchanged). Restores the host host_bin_layout_from_hist+build_lpt path and the real-M/P-read tile_assign (pre-iter-121 working path); on-device layout (S5.1-S5.5 sort_bin_layout/sort_bin_emit) + static-ceiling over-provision code KEPT gated-off for a future fusion lever. Bit-identical: hero md5 e3fefb116d860f99d92bba1ef51d820c, 63.95dB. ms_view 236.7(iter-125 ledger)/~205-211(steady) -> 195.6 avg, min 163.7; BRISC-FW 224.7->179.1, NCRISC-KERNEL 189.9->144.4 == iter-116 best-known baseline (full regression recovery).
bin=3a2c4a729a3b7f56
