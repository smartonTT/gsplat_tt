id=156
sha=f18c3c3
ts=2026-06-02T20:03:39-0700
desc=iter 47: MEASURE-FIRST blend breakdown (device per-tile-class zones rd_bk_emit/rd_overflow/cp_inb/cp_ovf + host TILE_HISTO). Hero: in-budget 914 tiles/2.01M cand (59.8%), overflow 110 tiles/1.36M cand (40.2%), max_tile_n=25699>FIT. Blend makespan ~93ms (reader~=compute co-equal) + separate cull ~80ms = lumped host blend 182.5. In-budget CO-LIMITED: reader-emit 824 ~= compute-SFPU 844 (load+sort only ~286). Busiest cores are OVERFLOW-gather-dominated (top core overflow 449 vs emit 181, 30v totals). Tailskip scaffold gated-off (deadlock-free 63.85 but blend 185.8>182.5, not a win). Measurement-only, PSNR 63.85 bit-identical.
bin=325ef2bf2c6a5942
