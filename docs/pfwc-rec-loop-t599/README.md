# t599: pfwc writer record loop, cycles per survivor (Step 1: microbench)

Measured 2026-10-13 08:25 PDT, bh-33 (p100a, IRD 138447), 1 core, tick 1349.7 MHz, branch
ttp/t599-pfwc-record-loop-microbench-cut-survivor.

Probe `render/bench/kernels/p11_pfwc_rec.cpp`, suite `MB_SUITE=pfwcrec` (MB_NB = banks):
64 chunks of 1024 gaussians, ~28 % survivors (LCG mask), the REC32 record loop with real NoC
page / record flushes. cycles/survivor = (loop ticks - flush ticks) / survivors, the t484
"rec - fl" figure (production capture: 108.5 B / 112.4 N).

| variant | nb | BRISC | NCRISC | checksum |
|---|---|---|---|---|
| v0 production loop (ChunkPages::put + Rec32Stage::next) | 8 | 117.16 | 117.74 | 891cbeab |
| v1 `pfwc_wsplit::rec32_chunk` (register cursors) | 8 | **66.95** | **67.79** | 891cbeab |
| v0 | 7 | 117.38 | 117.90 | c60057fd |
| v1 | 7 | **66.84** | **67.70** | c60057fd |

v1 cuts ~50 cycles/survivor (-43 %), under the 90-cycle kill line; same flushed bytes
(checksum) and host equivalence (`tests/unit/test_pfwc_rec_fast.cpp`, 1.93 M survivors, flush
order / staging / cursor state identical). Why: v0 keeps the ChunkPages / Rec32Stage state in
stack slots (about 10 stack loads per survivor), uses each load at once (load-use stalls) and
branches every other record on `half`. v1 keeps a page pointer, a record pointer with an
alternating step and two countdowns in registers, issues the ten input loads before the eleven
stores, and leaves the struct code only on a page end (every 16) or bank-row end (every 2 nb).

The flush ticks in this probe (~193/survivor) are inflated by per-page writes and the checksum
reads; they are excluded and not a production figure.

t598 model (docs/pfwc-mover-t598): rec at 70 cyc/survivor -> +0.198 ms/view (fitted lead),
less with a smaller lead. Step 2 (wire into writer_pfwc_split.cpp behind a flag, b2b A/B
against best-iter-227) is the next task.

Reproduce: `MB_NB=8 MB_SUITE=pfwcrec TT_METAL_CACHE=<dir off /home> render/bench/build/risc_microbench`
(bh-33's /home is full, so the default cache fails).

## Step 2: device A/B (bh-33 p100, 2026-10-13 01:32-01:44 PDT)

`writer_pfwc_split.cpp` uses `rec32_chunk` when built with `GSPLAT_TT_PFWC_RECFAST=1`
(default 0). Same commit 0d4dff8f both arms, separate kernel caches; driver `drive.sh`,
outputs in `out/` (run logs gzipped). `render/run.py --back-to-back`, 30 bicycle views,
20 timed passes after 1 warm-up, ms/view:

| round | order | RECFAST=0 | RECFAST=1 | gain |
|---|---|---|---|---|
| r0 (warm-up round) | A B | 8.442 | 8.391 | 0.051 |
| r1 | A B | 8.446 | 8.392 | 0.054 |
| r2 | B A | 8.459 | 8.387 | 0.072 |
| r3 | A B | 8.447 | 8.373 | 0.074 |
| r4 | B A | 8.468 | 8.384 | 0.084 |
| **mean r1-r4** | | **8.455** | **8.384** | **0.071 (0.84 %)** |

The new arm wins all 5 rounds; passes within a run spread about +-0.01 ms. md5 both arms
077caaba = golden, 30/30 (11x10 grid); b2b raw_md5 1d372b94 identical in both arms.
RECFAST=1 device hero (`opt/metal-screenshots/ttw-t599-recfast/`): 42.85 dB vs
benchmarks/reference_v2/hero.png (same as best-iter-227), golden match, max 0 LSB; hero and
diff checked by eye: no tile seams or block artifacts, diff only on spoke/leaf edges as before.

Verdict: **shelved** by the spec gate (keep needs >= 0.1 ms/view b2b). The gain is real and
bit-exact but 0.07 ms/view, below the t598 model's 0.20 lead estimate: the mover is not the
whole pfwc critical path. The flag stays default 0; flipping it is a one-line, zero-risk
change if a later pfwc lever is stacked with it.
