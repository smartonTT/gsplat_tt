id=102
sha=faebfa0
ts=2026-06-11T11:18:04-0700
desc=iter 131: Stage-2b: pre-pack op/color UNORM16 into the 32B blend record at BIRTH (gather_visible_scatter) so sort_bin pack_invariants and the depth-sorted materialize overflow gather COPY the two packed words instead of re-deriving 4 fp32->UNORM16 conversions (per-gaussian / per-overflow-record). Phase-1 fresh 30-view ranking (ttw-131): BRISC-FW 165.86, NCRISC-KERNEL 131.42; top reducible NCRISC zones sort_bucket_emit 30.58, sort_subchunk_mat 19.62 (overflow re-read/re-pack), readers ~98% backpressure. Ablation MEASURED the gather re-pack at frame -7.1 (skip-pack) vs read -4.5 (skip-read). Bit-identical hero md5 e3fefb116d860f99d92bba1ef51d820c, 63.95dB. NCRISC-KERNEL 131.42->113.80 (-17.6), BRISC-FW 165.86->162.12 (-3.74); sort_bucket_emit 30.58->18.83, sort_subchunk_mat 19.62->13.74; tradeoff proj_scatter 20.83->35.40 (+14.6, pack relocated to BRISC producer). ms_view 181.6->178.9 avg / 147.0->143.8 min.
bin=de6db8435d93defa
