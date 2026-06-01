id=24
sha=67c7e31
ts=2026-06-01T08:17:00-0700
desc=iter 16: Stage C2 contiguous per-tile blend payload: on-device pack pass (payload_pack) writes 64B/candidate rows (attrs+conic-raw+microblock mask) into blend_payload indexed by global candidate idx; sequential streaming blend reader (reader_alpha_blend_mb_payload) replaces the ~1.9GB random gather. Blend exec drops 191->26ms with byte-correct data proven end-to-end (pack->DRAM->reader L1->CB->compute all verified identical to baseline via readback+DPRINT) but final image regresses to 11.23dB via an as-yet-unexplained path; gated OFF (GSPLAT_TT_BLEND_PAYLOAD default off) to hold the 63.85dB baseline.
bin=76c1343b5166443f
