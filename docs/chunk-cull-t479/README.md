# t479: chunk cull with Morton gids — b2b A/B and diagnosis

Board: p100a (yyzo-bh-04, measurement reservation), sha bda3949b, `run.py --back-to-back`,
30 bicycle views, 3 passes per arm, 3 rotated rounds. Logs in `out/b2b/`.

| arm | env | r1 | r2 | r3 | median ms/frame | list md5 | hero md5 |
|---|---|---|---|---|---|---|---|
| off | (none) | 9.826 | 9.840 | 9.844 | **9.840** | 906e0435 (golden) | 86524912 |
| off48 | OL_RING=4 OL_RING_DEPTH=8 | 9.769 | 9.767 | 9.781 | **9.769** | 906e0435 (golden) | 86524912 |
| cull | CHUNK_CULL=1 | 10.252 | 10.254 | 10.253 | **10.253** | 5f542ec0 | aa232dbd |
| cull48 | CHUNK_CULL=1 OL_RING=4 OL_RING_DEPTH=8 | 9.988 | 9.984 | 9.983 | **9.984** | 5f542ec0 | aa232dbd |

Gate (cull beats off by >= 0.2 ms): **fails**. cull is +0.413 ms over off; cull48 is +0.215 ms
over off48 and +0.144 over off. Decision: the chunk cull stays off; nothing landed.

## Profile findings (Tracy, views 0:10, max-core ms/view, off / ro / ro48)
- Blend kernels do not slow down: tile_blend_sfpu 4.29 / 4.29 / 4.28. The host "blend" stage
  growth seen in t470 is the sort tail that runs inside it.
- The cost is the onelaunch sort: sort_ol_town 1.32 / 2.01 / 1.83, sort_ol_emit 1.42 / 2.07 / 1.88.
  Emit counters: per-tile run wait (wfl) 0.06 / 0.46 / 0.25 ms per TRISC launch. Cause: Morton
  gids send bursts of records to the same screen tile, so the mover stalls on the tile's single
  run buffer. `GSPLAT_TT_OL_RING_DEPTH` (ring of D >= R records per tile) halves the wait, but
  about 0.3-0.5 ms of sort makespan stays above off.
- pfwc per-core balance is not worse with Morton gids (CPU model `imbalance.py`).

## Tie-break (spec part 1)
Not implemented. Equal fp32 depth keys of distinct gaussians change order (254 px on hero); restoring
today's order needs a per-record original rank through pfwc writer, emit and materialize. Deferred
because the cull loses the gate even without it.

## Side result
off48 (ring R=4 D=8, cull off) is bit-identical (906e0435) and 0.071 ms faster than off: under the
0.1 ms keep gate alone; a candidate to bundle with another sort-side lever.
