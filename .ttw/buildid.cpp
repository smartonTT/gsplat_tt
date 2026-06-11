id=91
sha=f1a8eab
ts=2026-06-11T06:30:43-0700
desc=iter 123: S5.3 M-domain GROUNDWORK (gated prerequisite, behind host_free_mp_enabled): tile_assign now reads the visible-count M from the resident proj_M control page via a runtime InterleavedAddrGen<true> (bbox K1 + scan_reduce + scan_add) and over-provisions the M-domain offsets/scan buffers + K1 page-split to a static page-aligned ceiling n_ceil (= proj_m_px capacity) instead of the host-read M. Reversible (host_free_mp_enabled() default true). Frame-neutral at the steady-state floor (~204-205 ms/view; the higher verify avg is thermal from back-to-back runs). Bit-identical (hero md5 e3fefb11, 63.95 dB across 30 views). NOTE: the three mid-frame host blocking read-deletions (tile_assign P-read, sort P-read, gather M-read) are NOT removed yet — this lands only the resident-read + static-ceiling pattern they depend on.
bin=67660b4ac6a5febd
