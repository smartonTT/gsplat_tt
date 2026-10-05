# t160: sort_ol_emit pack-loop instruction diet

Fast path in `render/kernels/dataflow/sort_bin_onelaunch.cpp` for the default config
(ring + PUBOC + power-of-two tiles_x). `GSPLAT_TT_OL_EMIT_FAST=0` is the kill switch.

Changes against the old loop (see docs/emit-zones-t154/README.md for the baseline split):
- register locals in a hand-inlined loop instead of by-reference lambda captures;
- per-tile cursors in a 4 KB RISC local-memory array instead of L1 `curp` read-after-write;
- `sub_int32` always-inline 32-bit path, `sub_int_cold` fallback (exhaustive fp32 test: 0 mismatches
  over 4.45e9 fast-path inputs);
- plain keep/gid/tid plane loads after a compiler barrier, P bound hoisted per batch.

## Measured (yyzo-bh-07, Blackhole p100a, not a p150; bicycle 30 views 1024x1024, untraced)

Code: e74a2dc (on b54c54e, PRECULL=1 default). Two swapped rounds:

| round | fast (new default) | FAST=0 (old loop) | delta |
|---|---|---|---|
| 1 (fast first) | 18.563 | 19.287 | -0.724 |
| 2 (off first)  | 18.621 | 19.110 | -0.489 |
| mean           | 18.59  | 19.20  | -0.61 ms (-3.2%) |

Stages (means): sort 5.02 -> 4.40, of which bin_emit 4.61 -> 3.99; project and blend unchanged.
All 4 runs md5-identical to md5-r82new.txt (PRECULL=1 golden) over 30 views.

## Emit sub-zones with the fast loop (Tracy, GSPLAT_TT_OL_EMIT_PROF=1, ms/view, mean mover)

| part | ms/view |
|---|---|
| sort_ol_emit zone | 2.61 (busiest 2.76) |
| issue_brec (keep/gid scan + brec read issue) | 0.51 |
| process_batch | 1.97 |
| - pack/key build | 1.82 |
| - run write issue | 0.13 |
| tail drain | 0.09 |

263 cycles per record for the whole emit (was 347 in t154, pre-PRECULL). The issue_brec scan
(0.51 ms) was not folded into the previous batch; it is the next lever here.

Files: `out/emit_parts.txt`, `out/zones.txt`, `out/gaps.txt`, `out/run-r*.log`, `out/md5-r*.txt`;
drivers `drive.sh`, `remote_time.sh`, `remote_tracy.sh`.
