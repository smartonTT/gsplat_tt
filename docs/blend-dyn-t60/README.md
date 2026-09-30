# Task #60 — blend tail: dynamic tile claiming (lever L3)

**Board: yyzo-bh-07 (Blackhole p100a), not a p150.** Bicycle, 30 views, 1024x1024.

## Change
The blend program (reader NCRISC / SFPU compute / writer BRISC) used the sort's
static LPT split, cost = padded pair count. Real blend cost also depends on the
kept microblocks and the transmittance early-out, so the blend was tail-bound
(iter-162: tile_blend_sfpu mean 18.09, max 21.48 ms, balance 0.84).

Now every core claims tiles from one shared counter (semaphore on the first
core, NoC atomic increment-and-get, return value into a reader L1 slot). Claim
order = the sort LPT lists interleaved by rank (approx. descending cost). The
reader queues each claimed tile id for the writer (CB 13) and ends both streams
with a DONE flag / 0xFFFFFFFF. Per-tile output does not depend on the core, so
the image is byte-identical by construction.

Why not re-weight the static LPT by pairs x kept microblocks: the mask is only
known after the cull, and it still misses the T-saturation early-out. Dynamic
claiming uses the real durations. Splitting the blend reader over BRISC+NCRISC
was not done: the reader zone (`tile_blend_load`, `rd_l1_bulk`) includes its
wait on the 2-slot bulk CB, so it tracks the SFPU time; the blend is SFPU-bound.

## Scripts
`run_all.sh <base> <cand>` (detached): verify (30-view md5) -> 3 interleaved A/B
rounds -> 10-view Tracy for both trees. Remote trees /localdev/smarton/gstt2-t60{base,}.
