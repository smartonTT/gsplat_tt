# t215: faster TRISC emit loop after t202 (model, no device)

Question: after t202 (656a1fa, 14.235 ms/view) can (A) per-tile cursors in TRISC local memory
or (B) the two movers as 4th/5th tile owners cut the emit pack makespan by >= 0.3 ms/view?

**Verdict: SHELVE.** A saves ~0.10 ms/view, B ~0.14, both together ~0.20 (best case 0.23).
None reaches 0.3. Model: `model.py` -> `model.txt`.

## Data

`docs/emit-trisc-own-t200/out/tracy-town-*.txt` (Tracy views 0:10, `GSPLAT_TT_OL_EMIT_PROF=1`,
yyzo-bh-07 p100a). Mean core, ms/view:

- emit zone 1.228 (busiest mover 1.328, which sets the sort_ol end).
- TRISC: 7925 records each, 178.8 net cycles/record = 1.050 ms net pack; process() 1.113 with
  fl waits. Zone minus TRISC process = 0.115 (prologue, drain 0.097, tail service).
- Movers: busy 0.881 (list build + read issue 0.468, queue service 0.291, drain 0.097,
  prologue 0.023), idle 0.347. Their `ep_wfl` 0.55 is the slot wait loop and contains most of
  the 0.29 queue service, so the real mover slack is ~0.35, not 0.55 + 0.29.
- The TRISCs are the critical path: pack makespan ~= 0.115 + TRISC process.

## A: cursors in TRISC local memory

Each TRISC keeps its ~342 tiles x 2 streams of cursors locally (2.7 KB as 32-bit, 1.4 KB as
16-bit), loads them at GO and writes them back for the mover's drain (~1 us each way). This
removes one L1 load and one L1 store per record. On the movers the same move was worth 17
cycles/record (t200 README, #160 vs #187). The other ~160 cycles stay: list load, 8 blendrec
loads on 92% of records (ng 7284 / rec 7927), sub_int32, 8 ring stores, run words.

| cycles saved / record | 12 | 17 | 25 | 35 |
|---|---|---|---|---|
| saving ms/view, mean (busiest) | 0.07 (0.08) | 0.10 (0.11) | 0.15 (0.16) | 0.21 (0.22) |

It would need >= 52 cycles/record (29% of the loop) for 0.3 ms. That is 3x the measured cursor
cost. Not verified: the TRISC local memory size on Blackhole (the tt-metal headers were not
reachable from the paths I may search). 1.4-2.7 KB plus stack may not fit.

## B: movers as 4th/5th owners

Mover m owns a share of its own stream's tiles (cursors are per stream, so there is no sharing).
It packs those records inside its list build and writes their runs directly. Extra mover cost per
owned record: 130-200 cycles (the base mover loop was 216 cycles/record, minus the ~45 the list
build already pays). Each record taken off a TRISC saves 178.8/3 cycles of TRISC makespan.
Balance point: mover lane 0.881 + x * cm equals TRISC lane 0.115 + 0.063 + (23775 - 2x)/3 * 178.8.

| TRISC cyc/rec | mover cm | records/mover | zone | saving mean (busiest) |
|---|---|---|---|---|
| 178.8 | 130 | 1880 | 1.062 | 0.17 (0.18) |
| 178.8 | 170 | 1620 | 1.085 | 0.14 (0.16) |
| 178.8 | 200 | 1468 | 1.098 | 0.13 (0.14) |
| 161.8 (A+B) | 170 | 1201 | 1.032 | 0.20 (0.21) |

These are upper bounds. The model leaves out the batch coupling: a mover that packs is later
to publish READY, and the slot ring is only 4 deep. It also needs a per-stream owner map and a
second ring-reuse path (writes_flushed on the mover) next to the fl handshake.

## Why not more

The mover slack is only ~0.35 ms per mover, and moving one record costs about as many mover
cycles as it saves on a TRISC. So 5 lanes cannot do much better than 0.881 + share. The TRISC
loop is mostly L1 traffic that A does not touch. Other emit items seen in the same data, also
under the gate: the serial drain (0.097 ms; could overlap if a TRISC queued a tile's partial
run when it writes the tile's last record, untested) and core imbalance (TRISC busy max core
1.278 vs mean 1.129 ms). Untraced savings tracked traced ones in t202 (traced sort_ol -1.02,
untraced -0.96), so the traced numbers above are a fair estimate.

Better targets remain the larger programs: blend 5.57, mat 3.02, pfwc 2.95 ms/view.
