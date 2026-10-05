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
`remote_tracy.sh`. Run 2026-10-04 (task run 507). a = tip f4d91df (stack cur_lm), b = 1f6274e (fix).

| round (order) | a view_total | b view_total | b - a | a sort | b sort |
|---|---|---|---|---|---|
| r1 (b, a) | 17.410 | 17.149 | -0.261 | 4.087 | 3.882 |
| r2 (a, b) | 17.304 | 17.002 | -0.302 | 4.091 | 3.800 |
| r3 (b, a) | 17.205 | 16.982 | -0.223 | 4.017 | 3.781 |
| mean | 17.306 | 17.044 | **-0.262** | 4.065 | 3.821 |

ms/view, 30 views. Project and blend unchanged (4.11 / 8.87-8.91); the gain is all in the sort
(bin_emit 3.43 vs 3.66 on a). Likely cause (not profiled): the fast loop no longer copies 1024 counts into the stack array at
start and back at the end. All 6 runs ALL_VIEWS_IDENTICAL and md5-identical to md5-r82new.txt
(46a725ab set).

### Fold hang (t164 issue_brec fold under the K2 count fold)

| tree | GSPLAT_TT_OL_EMIT_FOLD=1 untraced | EMIT_PROF=1 Tracy |
|---|---|---|
| c = cd707df (no fix) | **hangs** (rc=124 after `[SORT] ONELAUNCH k2_fold=1`, as in #177) | not run |
| f = fba971e (cd707df + fix) | runs, 16.915 ms/view, md5 46a725ab | runs (rc=0), emit zone 2.706 ms/view |

Verdict: the fold hang was the stack overrun; with cur_lm off the stack the fold runs traced and
untraced. Single unpaired f run vs b mean: -0.13 ms/view (v2 table + fold over the fix); the
paired fold + v2 gate is in GATE.md.

### Gate: t177 v2 mover table + t164 fold vs the new tip (shelved)

Driver `gate.sh` (task run 521). t = tip dfc7548 (fix landed), c = a8f7d19 (tip + t177 v2 table +
t164 fold, `GSPLAT_TT_OL_EMIT_FOLD=1`). Untraced, 30 views, avg_frame_ms.

| round (order) | t | c | c - t | t bin_emit | c bin_emit |
|---|---|---|---|---|---|
| g1 (c, t) | 16.591 | 16.524 | -0.067 | 3.386 | 3.233 |
| g2 (t, c) | 16.704 | 16.422 | -0.282 | 3.370 | 3.210 |
| g3 (c, t) | 16.688 | 16.449 | -0.239 | 3.439 | 3.212 |
| mean | 16.661 | 16.465 | **-0.196** | 3.398 | 3.218 |

All 6 runs ALL_VIEWS_IDENTICAL and md5-identical to md5-r82new.txt (46a725ab). The fold no longer
hangs. Mean -0.196 ms/view misses the -0.3 gate (about the same as #177's v2 table alone, -0.200),
so the fold adds nothing measurable once the fix is in. **Shelved, not landed.** Branch
ttp/t187-fold-gate keeps the code; the fold stays opt-in.
