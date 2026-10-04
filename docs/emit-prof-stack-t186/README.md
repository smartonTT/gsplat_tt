# t186: is the EMIT_PROF=1 + emit fold hang a stack overrun?

Static check only, no device runs. Fold kernel from branch ttp/t177-re-derive-emit-mover-speed-table-from-t1,
built on yyzo-bh-07 with `build.sh` (sfpi g++ -O3 -fstack-usage, fast emit path: EMIT_PUBOC=1 OL_RING=2 OL_RING_TILES=1024).

kernel_main frame (BRISC = NCRISC):

| PROF | FOLD | frame |
|---|---|---|
| 1 | 1 | 4928 B |
| 1 | 0 | 4912 B |
| 0 | 1 | 4928 B |
| 0 | 0 | 4912 B |

Plus lambda (line 596) 112 B and _start 32 B: ~5072 B. Stack room: 0xffb02000 - __stack_base 0xffb00c30 = 5072 B (BRISC), ~5088 B (NCRISC).
`cur_lm[1024]` (4 KB) is most of the frame.

Verdict: PROF does not change the frame, so the hang is not a PROF-specific static overrun.
But every fast-path build sits at ~0 B margin, so any callee (noc helpers, Tracy markers) can overrun.
Fix not committed (spec: only on a confirmed PROF overrun). Recommended follow-up: move cur_lm to L1
for all fast-path builds, then re-test PROF+FOLD on device.
