id=59
sha=2ff081b
ts=2026-06-05T00:08:16-0700
desc=iter 103: M4a: strip dead code from cull/blend (no ABI change) — remove unused blend-reader helpers (read_soa_u32/gather/load_ids/load_mask/pack_blendrec), dead per-tile bucket_meta read in blend reader, dead CB consts (CB_MB_COEFF/CB_BUCKET/CB_BSORT/CB_BMASK in readers+compute), dead host cb_cfg (CB_MB_COEFF, blend+cull CB_BSORT), and debug scaffolding (MB_CULL_SPIN/MB_C1_PAYLOAD_DEBUG/MB_RD_ROW_SUPPRESS_DPRINT defines + dprint include)
bin=81e2917b5cb8f378
