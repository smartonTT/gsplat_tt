id=62
sha=a0e5989
ts=2026-06-05T01:01:44-0700
desc=iter 106: M5: REVERT cull double-buffer prefetch (iters 104-105) to baseline. Tracy proof: tile_l1_cull_rd (NCRISC reader) 27.46ms == tile_mb_mask (TRISC SFPU) 27.47ms -> cull stage already ALU/SFPU-co-limited and fully overlapped; reader is busy-bound (issues every slab read + emits every coeff row + per-gaussian logf), not stall-bound, so double-buffer cannot lower it. Same for blend (reader 12.33 <= SFPU 13.08). Restores 98.4ms baseline.
bin=81e2917b5cb8f378
