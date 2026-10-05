# t187: fast emit cursors off the stack (cur_lm -> CB_CUR in L1)

#186 (docs/emit-prof-stack-t186) found kernel_main of `sort_bin_onelaunch.cpp` at ~0 B stack margin on
the fast emit path: frame ~4.9 KB + lambda + _start ~5072 B vs 5072/5088 B of room (BRISC/NCRISC),
4 KB of it the local-memory array `cur_lm[OL_RING_TILES]`.

Change (e8754ac): the fast loop counts in `curp` itself (CB 5, `BIN_ROW_BYTES` per mover, already
allocated by the host CB allocator, so it cannot overlap the sort scratch, the window, the rings or
the slabs; no new L1). A plain (non-volatile) pointer lets the compiler schedule the loads. The
startup copy loop is gone; the tail drain reads `curp` for both paths.

## Frame sizes (riscv-tt-elf-g++ -O3 -fstack-usage, fast path EMIT_PUBOC=1 OL_RING=2 OL_RING_TILES=1024)

Built on yyzo-bh-07 with #186's build.sh (`/localdev/smarton/gstt2-t187-scratch`), kernel from
f4d91df (old) and e8754ac (new):

| kernel | PROF=0 | PROF=1 |
|---|---|---|
| old (BRISC = NCRISC) | 4896 B | 4896 B |
| new (BRISC = NCRISC) | 784 B | 800 B |

-4112 B: the stack margin goes from ~0 to ~4.1 KB.

Fold kernel (t177 cd707df, issue_brec fold) without and with the fix (fba971e = cd707df + e8754ac):

| kernel | F=0 | F=1 (PROF 0 or 1) |
|---|---|---|
| cd707df | 4912 B | 4928 B |
| fba971e | 816 B | 848 B |

## Device (yyzo-bh-07 p100a, bicycle 30 views 1024x1024, untraced)

Driver `drive.sh` (detached, one `ttp lock p100` per step), remote scripts `remote_run.sh`,
`remote_tracy.sh`. Results: pending.
