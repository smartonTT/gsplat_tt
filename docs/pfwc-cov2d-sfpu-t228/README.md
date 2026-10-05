# t228: P2, cov2d a/b/c + conic + radii as two SFPU sections

Build of "P2" from [docs/pfwc-trisc-model-t226](../pfwc-trisc-model-t226/README.md)
(modeled -0.48 ms/view, range 0.39-0.50). Knob `GSPLAT_TT_PFWC_COV2D_SFPU` (default 0
while building, `=1` on).

## What changed

Steps 7-11 of `render/kernels/compute/project_pfwc_compute.cpp` (cov2d a, b, c, the
conic fold and the two radii) ran as six DEST acquires: 38 `copy_tile` and 87 SFPU tile
ops per chunk plus the conic pass, because j00, j02, j11, j12 were rebuilt in five of
them and a and c twice. With the knob on they run as two acquires:

- **S_AC**: copy cc00, cc02, cc22, cc11, cc12, inv_tz, tx, ty into slots 0-7; one looped
  SFPU pass (`pfwc_cov2d::run_ac`, `render/kernels/compute/pfwc_cov2d_sfpu.h`) computes
  j00, j02, a, then j11, j12, c and stores a to slots 0 and 6 and c to slots 3 and 7;
  then the same `relu_tile`, `sqrt_tile`, `mul_unary_tile(k)`, `ceil_tile` as before on
  slots 6 and 7. Packs: 0 -> CB_TMP_A, 3 -> CB_TMP_C, 6 -> CB_RX (+ CB_TMP_RX),
  7 -> CB_RY (+ CB_TMP_RY).
- **S_BC**: copy cc02, cc01, cc12, cc22, inv_tz, tx, ty into slots 0-6; `run_b` computes
  b into slots 1 and 7; copy CB_TMP_A -> 0 and CB_TMP_C -> 2; the unchanged conic fold
  (`pfwc_conic_unroll<0>`); packs 0/1/2 -> CB_A/B/C and 7 -> CB_TMP_B.

The passes are raw `TTI_SFPLOAD/SFPMUL/SFPADD/SFPSTORE` with fixed tile addresses and
`sfpi::dst_reg++` per vector, in a `#pragma GCC unroll 0` loop. Every product
(SFPMUL, + 0.0) and sum (SFPADD, * 1.0) is rounded on its own in the default path's
order, so the result is bit-identical (no SFPMAD fusion, as in t206's
`pfwc_covcam_sfpu.h`). Two loop-invariant constants per pass live in LREG0 / LREG7
(2.0 and 0.3 in run_ac, fx and fy in run_b); the rest come per vector from `SFPLOADI`
pairs. Per chunk: copies 38 -> 17, SFPU tile ops 87 -> 8 (the radii's relu / sqrt / k /
ceil) plus the two passes, packs unchanged (10).

## Host check

`tests/unit/run_cpp.sh tests/unit/test_cov2d_sfpu.cpp`: the header's real instruction
stream on a model of DEST, LREGs and the DEST counter, LREGs starting as garbage, against
steps 7-9 op by op. 3000 chunks (scene-like, any finite bits, specials):
18,432,000 lane checks, 0 mismatches, inputs the passes must keep are kept, the DEST
counter ends at 64 rows. A planted wrong load is caught. Per chunk: run_ac 1316 TTI +
256 runtime SFPLOADI + 32 INCRWC, run_b 896 + 132 + 32 (~2.7k SFPU instructions in all).
`tests/syntax_stub/check.sh` compiles the kernel for all three TRISCs with the knob, with
and without VIS / PRECULL / WSPLIT / STEPCYC.

## Device run

`dev/drive.sh <rev>` (yyzo-bh-07 p100a, tree /localdev/smarton/gstt2-t228, every step under
`ttp lock p100`): sync, 30-view smoke with the knob on (md5 gate; a "too large" program
retries at `GSPLAT_TT_KCFG_EXTRA_KB=32`), Tracy STEPCYC=1 split for knob 1 and knob 0
(views 0-4, +32 KB), then 4 swapped untraced 30-view rounds base / cov2d. Logs in
`dev/out/`, summary in `dev/out/summary.txt` (`dev/summ.py`).

Gates (task spec): md5 46a725ab on every arm; TRISC_1 steps 6-11 (S_AC + S_BC) <= 16k
cycles/chunk; paired untraced drop >= 0.3 ms/view.

## Results (2026-10-05, 543869b, yyzo-bh-07 p100a, bicycle 30 views 1024x1024)

**Decision: SHELVED, default stays 0.** The compute cut is real, but pfwc is then bound by
the NCRISC writer on the right-hand grid columns, and the frame gets slower.

| gate | result |
|---|---|
| md5 46a725ab, every arm | **pass**: smoke + 4 rounds x 2 arms, 30/30 views identical |
| program size | fits the default auto +24 KB kernel-config open (Tracy at +32) |
| TRISC_1 steps 6-11 <= 16k cycles/chunk | **pass**: 24.7k -> 11.5k |
| paired untraced drop >= 0.3 ms/view | **fail**: +0.256 ms/view (slower) |

Paired untraced rounds, base = today's default (knob 0), cov2d = knob 1
(`dev/out/summary.txt`):

| round | base view_total | cov2d view_total | delta | base project | cov2d project | delta |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 13.662 | 13.883 | +0.221 | 3.384 | 3.640 | +0.256 |
| 2 | 13.574 | 13.883 | +0.309 | 3.350 | 3.596 | +0.246 |
| 3 | 13.578 | 13.824 | +0.246 | 3.350 | 3.595 | +0.245 |
| 4 | 13.597 | 13.844 | +0.247 | 3.338 | 3.588 | +0.250 |
| mean | 13.603 | 13.858 | **+0.256** | 3.356 | 3.605 | **+0.249** |

73.5 FPS -> 72.2 FPS.

TRISC steps 6-11 (Tracy STEPCYC=1, views 0-4, mean over 110 cores, cycles/chunk at 54.4
chunks per core per view):

| | T0 unpack | T1 math | T2 pack | TRISC wall (mean) |
|---|---:|---:|---:|---:|
| knob 0 | 27.6k | 24.7k | 19.5k | 2.20 ms |
| knob 1 | 15.5k | 11.5k | 11.3k | 1.86 ms |

T0 is now the slowest of the three in these steps; T1's S_AC is 0.290 ms/view and S_BC
0.171.

### Where the time went: NCRISC record writes on the right columns

Per grid column (NoC0 x), core wall = slowest RISC of the core, mean over the 5 traced
launches (`dev/percol.py`, `dev/out/percol.txt`):

| x | knob 0 wall | knob 1 wall | knob 1 max | BRISC rec k0 / k1 | NCRISC rec k0 / k1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.234 | 1.751 | 1.818 | 0.711 / 0.746 | 0.756 / 0.832 |
| 3 | 2.241 | 1.750 | 1.837 | 0.704 / 0.736 | 0.716 / 0.812 |
| 5 | 2.248 | 1.773 | 1.813 | 0.704 / 0.737 | 0.729 / 0.866 |
| 6 | 2.246 | 1.826 | 1.892 | 0.675 / 0.709 | 0.768 / 0.955 |
| 11 | 2.242 | 1.805 | 1.899 | 0.676 / 0.710 | 0.763 / 0.955 |
| 12 | 2.250 | 1.901 | 2.031 | 0.709 / 0.742 | 0.769 / 1.070 |
| 13 | 2.260 | 2.114 | 2.296 | 0.739 / 0.768 | 0.836 / 1.235 |
| 14 | 2.262 | 2.332 | 2.552 | 0.728 / 0.750 | 0.882 / 1.400 |
| 15 | 2.261 | 2.247 | 2.489 | 0.721 / 0.747 | 0.851 / 1.344 |

- With knob 0 every core is TRISC-bound and the walls are flat (2.23-2.26 ms, max 2.316).
  With knob 1 the median core drops 0.435 ms (2.247 -> 1.812) and columns 1-5 drop
  ~0.48 ms, but the program ends with the slowest core: 2.316 -> 2.552 ms.
- On the slow cores the writers are the bound. NCRISC (odd chunks, NoC1, also the input
  reader) spends 1.4-1.6 ms in its record loop against BRISC's ~0.75 on the same core.
  BRISC then waits for NCRISC's OPEN hand-off (`opn` 0.6-0.95 ms) and the TRISCs wait for
  output CB space (T2 `depth` / T0, T1 `means` ~0.8 ms on the slowest core, 0.05-0.13
  with knob 0).
- NCRISC `rec` grows with x (0.83 at x=1 -> 1.40 at x=14) while BRISC `rec` stays flat
  (0.71-0.77), and `rd` (the reader's own polling) is flat at ~0.3. So the extra time is in
  NoC1 record and page writes from the right-hand columns. The same gradient exists with
  knob 0 (0.71 -> 0.88). It was hidden while compute was the bound; with compute 0.34 ms
  faster, the busier NoC makes it worse.

What would unlock P2 (not measured): if columns 12-15 ran like columns 1-5 (~1.75 mean,
1.85 max), the knob-1 pfwc wall would be ~1.85-1.9 ms against today's 2.32. The
untraced project stage tracked the Tracy wall here (+0.25 untraced vs +0.24 traced), so
that would be about 0.4 ms/view. Candidate fixes for a follow-up, in the order to try:
move the 10-tile input reader from NCRISC (NoC1) to BRISC (NoC0) on the right-hand
columns; deal 2:1 BRISC:NCRISC chunks on those columns (t226's shelved option; it needs
compute's output-set choice to follow the deal); NoC0 for NCRISC's record writes (dynamic
NoC mode). Before building any of these, check the NoC1 hypothesis with
`dev/percol.py` on one Tracy run.

The P2 code stays in the tree behind `GSPLAT_TT_PFWC_COV2D_SFPU=1` (md5-safe, fits the
default open) for that follow-up to re-gate.

Artifacts: `dev/out/` (chain log `chain-run1.log`, round logs and md5 lists, Tracy CSVs
`prof-k{0,1}-dev.csv.gz`, `prof-k{0,1}-pc_split.txt`, `percol.txt`, `summary.txt`).
