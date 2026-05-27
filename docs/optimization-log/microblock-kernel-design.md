# Microblock Alpha-Blend Kernels — Design Spec

Status: design (no code yet)
Target: device_kernel time on Blackhole (P300, yyzo-bh-14), 1080p stitch.
Author trail: derived from a verified read of `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h`, `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`, the existing `programming_examples/gaussian_splatting/{alpha_blend.cpp, alpha_blend_host.h, kernels/{compute,dataflow}/*.cpp}`, and `gsplat/rasterization.py::_assign_gaussians_to_tiles`.

---

## 0. TL;DR

The current kernel does full 32×32 tile compute for every Gaussian-tile pair, even when a Gaussian only meaningfully touches a few microblocks of the tile. Real (not-AABB) Gaussian footprints are highly skewed: a 3σ ellipse with σ ≈ 1–4 px touches 10–40% of pixels in the tiles it overlaps. We are paying ~3–10× the SFPU work we need on the inner Mahalanobis/exp/alpha/blend chain.

This spec defines a hybrid pipeline:

1. **Host bins to microblocks** (4-row × 8-col, 32 lanes each — 32 microblocks per tile), not just to tiles. Each per-Gaussian payload now carries a 32-bit microblock-active mask.
2. **Compute kernel keeps 32×32 tile dispatch** (matches existing CB / pack / writer / DST budget) and uses **per-microblock SFPU compute** through `TT_SFPLOAD/TT_SFPMAD/TT_SFPSTORE` against DST addresses, skipping zero-mask microblocks entirely.
3. **Tile-level FPU work that is invariant under the microblock mask** (the 6 basis tiles `x², xy, y², x, y, 1` derived from `px/py`, plus state-CB I/O) stays on the FPU as full-tile `mul_tiles_bcast_*` ops. Tile-level work is paid once per tile, not per Gaussian.

Result: per-Gaussian cost scales with `popcount(mask) / 32` × the inner microblock cost, plus a small constant. Expected end state: device_kernel proportional to true Gaussian footprint, not AABB footprint.

---

## 1. Hardware ground truth (Blackhole)

These facts are **quoted verbatim** from `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h` lines 22–113. They are the authoritative source for DST layout on Blackhole — we will not invent any other addressing.

### 1.1 DST tile layout in addr units

```
A 32x32 tile is 4 faces of 16x16 values, arranged:
    Face 0 (tile rows  0-15 | tile cols  0-15)
    Face 1 (tile rows  0-15 | tile cols 16-31)
    Face 2 (tile rows 16-31 | tile cols  0-15)
    Face 3 (tile rows 16-31 | tile cols 16-31)

One SFPLOAD/SFPSTORE moves 4 dest rows x 8 cols (32 lanes). Dest addresses
are in units of "4-row x 8-col" slots:
    addr +0  -> rows 0-3,  cols  0-7  of the face
    addr +2  -> rows 0-3,  cols  8-15
    addr +4  -> rows 4-7,  cols  0-7
    ...
    addr +14 -> rows 12-15, cols 8-15
    addr +16 -> start of next face
```

Layout constants (same file, lines 95–112):

```cpp
constexpr uint32_t DEST_TILE_SIZE_RAW = 64;        // 4 faces * 16 addr/face
constexpr uint32_t FACE0_BASE = 0;                 // upper-left
constexpr uint32_t FACE1_BASE = 16;                // upper-right  (cols 16-31)
constexpr uint32_t FACE2_BASE = 32;                // lower-left   (rows 16-31)
constexpr uint32_t FACE3_BASE = 48;                // lower-right
constexpr uint32_t ROW_BAND_STRIDE = 4;            // +4 = next 4-row band
constexpr uint32_t NUM_ROW_BANDS_PER_FACE_HALF = 4;// 16 rows / 4-row-band
constexpr uint32_t ODD_COLS_OFFSET = 2;            // +2 = cols 8..15 of a face
```

### 1.2 Microblock = 4r × 8c = 32 SFPU lanes = 1 SFPLOAD/SFPSTORE

```
SFPU lane numbering in a register (32 lanes = 4 sub-rows x 8 columns):
    lane[i * 8 + c]  i in {0..3}, c in {0..7}
After SFPLOAD of rows r..r+3 cols c0..c0+7:
    LReg[i * 8 + c] = data[r + i][c0 + c]
```

So the entire 32×32 tile is **exactly 32 microblocks** (4 faces × 4 row-bands × 2 col-bands). One microblock fits in one LREG, fully. This is the unit we'll skip on / compute over.

### 1.3 Microblock → DST addr formula

Given microblock index `m ∈ [0, 32)` decomposed as
- `face_row ∈ {0,1}` (which 16-row tile half: upper / lower)
- `face_col ∈ {0,1}` (which 16-col tile half: left / right)
- `band    ∈ {0,1,2,3}` (which 4-row band inside the face)
- `colhi   ∈ {0,1}` (cols 0..7 vs 8..15 inside the face)

then the DST addr for that microblock in tile slot `dst_tile_idx` is:

```cpp
uint32_t face = (face_row << 1) | face_col;       // 0=UL, 1=UR, 2=LL, 3=LR
uint32_t face_base = face * 16;                   // 0, 16, 32, 48
uint32_t band_off  = band * 4;                    // 0, 4, 8, 12
uint32_t col_off   = colhi * 2;                   // 0 or 2
uint32_t mb_addr   = dst_tile_idx * 64 + face_base + band_off + col_off;
```

We will use a flat enumeration `m = 0..31` chosen so that the natural raster
order (rows 0,1,2,...,31) corresponds to a known mapping `m -> mb_addr`. We
materialize this mapping as a precomputed `constexpr` table so the inner
loop never does the decode arithmetic.

### 1.4 SFPU instruction primitives we need

From the same file (lines 296–349) and `relu`, `silu`, `recip` kernels, the
real, in-tree primitives are:

```cpp
// Load 32 lanes (= one microblock) from DST addr -> LREG
TT_SFPLOAD (LREG_DST, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_addr);

// Store 32 lanes from LREG -> DST addr
TT_SFPSTORE(LREG_SRC, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_addr);

// 32-lane fused multiply-add: dst = a * b + c   (all LREGs)
TTI_SFPMAD (LREG_A, LREG_B, LREG_C, LREG_DST, 0);

// Multiply: dst = a * b
TTI_SFPMUL (LREG_A, LREG_B, p_sfpu::LCONST_0, LREG_DST, 0);

// Add: dst = a + b   (encoded as a*1 + b)
TTI_SFPADD (LREG_A, p_sfpu::LCONST_1, LREG_B, LREG_DST, 0);

// Constants we can reference as operands without loads:
//   p_sfpu::LCONST_0     = 0.0
//   p_sfpu::LCONST_1     = 1.0
//   p_sfpu::LCONST_neg1  = -1.0

// 2-cycle latency on every SFPMUL/SFPADD/SFPMAD output. Issuing a dependent
// op next cycle requires SFPNOP. Independent ops (distinct dest LREGs) issue
// back-to-back. See lines 240-292 for the canonical pipelining pattern:
// LREG_BCAST + LREG_DATA0..3 are pairwise independent, so 4 SFPMADs run
// back-to-back with no NOPs between.
```

`InstrModLoadStore::DEFAULT` and `ADDR_MOD_7` are taken straight from the
bcast kernel — they are the "load/store as-is into FP32 dest" mode used
when the kernel runs with `fp32_dest_acc_en = true`, which our existing
host program already configures (`alpha_blend.cpp` constructs CBs with
`tt::DataFormat::Float32` for fp32 paths).

### 1.5 sfpi-style equivalent (simpler, same hardware)

For sections of the kernel where we don't need fine-grained pipelining we
use the higher-level `sfpi` API, which wraps the same instructions but
hides the LREG allocation and NOPs:

```cpp
#include "sfpi.h"
using namespace sfpi;

// Same dst_reg cursor as relu/silu use (see ckernel_sfpu_relu.h):
//   dst_reg[0]  -> 32 lanes at the current cursor (one microblock)
//   dst_reg++   -> advance cursor by one microblock (8 cols within face)
//
// IMPORTANT: dst_reg is SCOPED to the currently selected face inside an
// _llk_math_eltwise_unary_sfpu_*_  wrapper. To address microblocks across
// faces we use the lower-level TT_SFPLOAD/TT_SFPSTORE form with explicit
// dst_addr (section 1.4), not dst_reg++.
```

We will use the explicit-addr form everywhere we need cross-face addressing
(which is everywhere in this design).

### 1.6 Replay buffer (`lltt::record/replay`)

Defined in `runtime/sfpi/include/lltt.h` (verified path; the header lives
in the sfpi include set, NOT in `tt_metal/hw/inc/`). Used in production
by `ckernel_sfpu_binary_bcast.h` (line ~250) and `ckernel_sfpu_recip.h`
(line ~236). It records a fixed-length sequence of SFPU instructions
into a hardware replay slot once, then re-issues them with a single
`lltt::replay(slot, len)` call. We will use this for the per-microblock
inner sequence so the per-Gaussian inner loop body shrinks to a few
instructions.

The actual signatures (verbatim from `lltt.h`):

```cpp
namespace lltt {

enum ExecBool : bool { NoExec, Exec };

// Records LENGTH instructions starting at slot START. By default (NoExec)
// the recorded instructions are NOT executed during recording; pass
// lltt::Exec to also run them now (handy when the recording itself does
// useful work the first time, e.g. lane-mask setup).
template<ExecBool E = NoExec>
[[gnu::always_inline]] inline void record(unsigned start, unsigned length);

// Re-issues the LENGTH instructions previously recorded at slot START.
[[gnu::always_inline]] inline void replay(unsigned start, unsigned length);

}  // namespace lltt
```

Usage pattern (matches the bcast kernel exactly):

```cpp
#include "lltt.h"

// Record once, at kernel init. Slots are global to the SFPU; pick a slot
// number that doesn't collide with the LLK kernels you also call (the
// stock SFPU exp/recip implementations occupy slots 0-3).
constexpr uint32_t REPLAY_SLOT_MB_INNER = 4;
constexpr uint32_t REPLAY_LEN_MB_INNER  = 16;  // count actual TTI_* ops

lltt::record(REPLAY_SLOT_MB_INNER, REPLAY_LEN_MB_INNER);
TTI_SFPMUL(...);   // 1
TTI_SFPMAD(...);   // 2
// ... 14 more ops; total must equal REPLAY_LEN_MB_INNER ...

// Issue per microblock:
lltt::replay(REPLAY_SLOT_MB_INNER, REPLAY_LEN_MB_INNER);
```

Caveat: replay does not parameterize the SFPLOAD/SFPSTORE addresses. If
the recorded sequence contains `TT_SFPLOAD(..., dst_addr)` then `dst_addr`
is baked at record time. Per-microblock varying `dst_addr` either uses
the inline (non-replayed) form (`TT_SFPLOAD` taken from `ckernel_sfpu_binary_bcast.h`
line 311) OR splits the recorded slot into the parts that are
addr-independent (the multiply/add chain) and the addr-dependent
SFPLOADs/SFPSTOREs which stay inline.

### 1.7 What we will NOT use (ruled out by hardware)

- **`mul_tiles` / `mul_tiles_bcast_*` per microblock** — these are full-tile (1024-lane) FPU ops, indivisible. They cannot skip microblocks. We use them only at the tile-level (basis prep) and full-tile inner paths.
- **Per-microblock subtile dispatch from the host** — TT-Metal does not expose subtile-granularity tile dispatch. CBs operate in tile units. Subtile work is internal to the compute kernel.
- **`tile_regs_acquire` per microblock** — `tile_regs_acquire/commit/release` is the SYNC primitive between unpack/math/pack threads; one acquire spans the full inner section of work that ends in a `pack_tile`. We acquire once per inner segment, do all microblock work, then commit/pack.

---

## 2. Pipeline overview

```
HOST (gsplat/rasterization.py + backends/tt/backend.py)
  1. project_gaussians                 (existing)
  2. AABB tile assignment              (existing, _assign_gaussians_to_tiles)
  3. PER-PAIR mahalanobis cull         (existing, contrib_floor)
  4. NEW: per-Gaussian microblock mask (32-bit popcounted active mask
                                        for each surviving (g, tile) pair)
  5. depth sort + per-tile bin         (existing, sort_and_bin)
  6. pack scalars                      (existing, +1 uint32 per Gaussian
                                        for the microblock mask -> payload
                                        grows from 64B to 64B; we pack the
                                        mask into the existing 64B page)
  7. pack px/py tiles                  (existing)
  8. write tile_offsets, tile_ids,     (existing; LPT load balancing
     packs to DRAM                     unchanged)

DEVICE
  READER (NCRISC, NoC1, kernels/dataflow/reader_alpha_blend.cpp)
    Per assigned tile:
      - g_count = tile_offsets[tile+1] - tile_offsets[tile]
      - push g_count to CB_TILE_META
      - push px tile (1) to CB_PX
      - push py tile (1) to CB_PY
      - for g in [g_start, g_end):
          push 1 64B page to CB_SCALARS (now contains mb_mask)

  COMPUTE (kernels/compute/alpha_blend_compute.cpp)
    Once per program launch:
      - prefill CB_CONST_ZERO, CB_CONST_099                         (existing)
      - record replay slot for the per-microblock SFPU inner sequence  (NEW)
      - prefill CB_BASIS_X, CB_BASIS_Y, CB_BASIS_X2, CB_BASIS_XY,
                CB_BASIS_Y2, CB_BASIS_ONE  (per-tile basis tiles)       (NEW)

    Per tile:
      A. Init R/G/B/T/sat state CBs                                 (existing)
      B. Fast-skip: if tile_g_count == 0, pack zeros and continue   (existing+)
      C. Build per-tile basis tiles from CB_PX, CB_PY               (NEW)
         - x_local = px - tile_origin_x  (FPU, mul_tiles_bcast or sub)
         - y_local = py - tile_origin_y
         - x², xy, y²                                               (FPU)
         - These are PER-TILE constants; we compute once and hold in
           L1 across all g_count Gaussians of this tile. Saves
           3 SFPU mul_unary_tile + 1 mul_tiles_bcast per Gaussian.
      D. Per-Gaussian inner loop. The 32-bit mb_mask drives compute:
         - if popcount(mb_mask) == 32:  full-tile FPU/SFPU path     (existing)
         - if popcount(mb_mask) < THRESH: per-microblock path       (NEW)
            - basis-form Q evaluation per active microblock only
            - exp, alpha, contrib, R/G/B accum, T update
              all per-microblock against THE SAME state CB (we
              load/modify/store only the microblocks we touch)
      E. Per-tile finalize: pack R/G/B state to CB_COLOR_OUT        (existing)

  WRITER (BRISC, NoC0, kernels/dataflow/writer_alpha_blend.cpp)     (existing)
```

### 2.1 Why hybrid full-tile + per-microblock

- **Full-tile path is faster per byte of work** when the mask covers the
  whole tile, because `mul_tiles_bcast_*` is FPU (matrix engine) and beats
  any SFPU loop on dense work. Real workloads have a heavy tail of
  Gaussians that *do* cover the full tile (the central tile of a big
  splat). We must keep this path.
- **Per-microblock path** only competes when the mask is sparse — at that
  point the FPU full-tile cost becomes a constant ceiling we want to
  break through.

The crossover threshold (`THRESH`) is empirical; design this as a runtime
or compile-time knob. First implementation: `THRESH = 24` (i.e., switch to
the microblock path only when 24+ of 32 microblocks are *inactive*; that's
≥75% sparsity, a regime where SFPU cost ≈ 25% of full-tile cost).

---

## 3. Host binning to microblocks

### 3.1 Where the change lands

`gsplat/rasterization.py::_assign_gaussians_to_tiles` already produces
`(gaussian_ids, tile_ids, tiles_per_gaussian)`. We extend it to emit one
additional output:

```python
mb_masks: torch.Tensor   # uint32, shape (P,) where P = sum(tiles_per_gaussian)
                         # bit `m` (m in 0..31) is 1 iff microblock `m` of
                         # tile `tile_ids[i]` has at least one pixel that
                         # passes the per-microblock contribution test.
```

### 3.2 Per-microblock contribution test

Reuse the existing closest-pixel-in-bbox Mahalanobis trick from the
per-pair `contrib_cull` (rasterization.py lines 268–290), but evaluate it
**32 times per surviving (g, tile) pair**, once per microblock.

Each microblock is a 4×8 axis-aligned rectangle inside the tile. For
microblock `m`, the closest point to the Gaussian mean is

```python
# Per-tile pixel origin of the tile, broadcast over P
tx_tile = (tile_ids % tiles_x).float() * tile_size            # (P,)
ty_tile = (tile_ids // tiles_x).float() * tile_size           # (P,)

# Per-microblock origin within the tile, table of 32 entries.
# Constructed from the (face_row, face_col, band, colhi) decode in §1.3.
mb_origin_x = ...  # shape (32,) values in {0, 8, 16, 24}
mb_origin_y = ...  # shape (32,) values in {0, 4, 8, 12, 16, 20, 24, 28}

# Closest point inside the m-th microblock of the i-th pair to mean[i]:
#   cx[i, m] = clamp(mean_x[i], tx_tile[i] + mb_origin_x[m],
#                                tx_tile[i] + mb_origin_x[m] + 8)
# (likewise cy; 4 instead of 8 for y)
# Then evaluate
#   m2[i, m] = (c*dx² - 2b*dx*dy + a*dy²) / det
#   keep_mb[i, m] = opacity[g] * exp(-0.5 * m2) >= contrib_floor
# and
#   mb_masks[i] = sum_m (keep_mb[i, m] << m)
```

This is a single `(P, 32)` vectorized `torch` block, no Python loops. Cost
on CPU is `O(P)` with constant 32 — at 1080p (~2k tiles, ~50k pairs after
existing culls) this is microseconds, dwarfed by everything else in
prep.

### 3.3 Mask aliasing rule

We do not drop pairs where `mb_masks[i] == 0` (they should be 0 already
because `contrib_cull` runs first; if any survive, it means the AABB-level
test passed but every microblock failed — drop them here too).

Conservative invariant: `popcount(mb_masks[i]) >= 1` for every pair we
keep. Sanity-check this on the host, fail loudly if it fires.

### 3.4 Microblock enumeration

We commit to a single canonical mapping `m ↔ (face, band, colhi)` that is
shared between host and device. Define it once in
`alpha_blend_host.h` so neither side guesses. Recommended raster order:

```cpp
// In alpha_blend_host.h
//
// Microblock index m enumerates the tile in raster order: top-to-bottom
// over 4-row bands, left-to-right within a row.
//
//   m = (tile_row_band) * 4 + (tile_col_group)
//   where tile_row_band in [0, 8),  tile_col_group in [0, 4)
//   tile_row_band -> face_row * 4 + band
//   tile_col_group -> face_col * 2 + colhi
//
// Microblock 0 = rows 0..3,  cols 0..7   (face 0, band 0, colhi 0)
// Microblock 1 = rows 0..3,  cols 8..15  (face 0, band 0, colhi 1)
// Microblock 2 = rows 0..3,  cols 16..23 (face 1, band 0, colhi 0)
// Microblock 3 = rows 0..3,  cols 24..31 (face 1, band 0, colhi 1)
// Microblock 4 = rows 4..7,  cols 0..7   (face 0, band 1, colhi 0)
// ...
// Microblock 31 = rows 28..31, cols 24..31 (face 3, band 3, colhi 1)
constexpr uint32_t mb_to_dst_addr(uint32_t m, uint32_t dst_tile_idx) {
    uint32_t row_band   = m >> 2;          // 0..7
    uint32_t col_group  = m & 0x3;         // 0..3
    uint32_t face_row   = row_band >> 2;   // 0 or 1 (upper / lower half)
    uint32_t band       = row_band & 0x3;  // 0..3
    uint32_t face_col   = col_group >> 1;  // 0 or 1 (left / right half)
    uint32_t colhi      = col_group & 0x1; // 0 or 1
    uint32_t face       = (face_row << 1) | face_col;     // 0..3
    return dst_tile_idx * 64 + face * 16 + band * 4 + colhi * 2;
}
```

This function is `constexpr` so the compiler folds it for any compile-time
`m`. The per-microblock loop body uses it as a precomputed table of 32 addr
values for `dst_tile_idx = 0` (the inner-loop slot we always pack from).

---

## 4. Payload + reader changes

### 4.1 Existing 64B scalar pack

`alpha_blend_host.h::SCALAR_PACK_BYTES = 9 * 4 = 36 bytes`, padded to
`SCALAR_PACK_PAGE_BYTES = 64`. The 9 fp32 scalars are
`(mean_x, mean_y, a, b, c, opacity, color_r, color_g, color_b)`.

### 4.2 New 64B scalar pack

The 64-byte page has 28 bytes of slack. We pack the new mb_mask into one
of those uint32 slots, raising the layout to **10 lanes** with one of them
being a uint32 mask:

```cpp
// alpha_blend_host.h additions:
constexpr uint32_t SCALAR_PACK_LANES = 10;
constexpr uint32_t SCALAR_PACK_BYTES = SCALAR_PACK_LANES * 4;  // 40 bytes
constexpr uint32_t SCALAR_PACK_PAGE_BYTES = 64;                // unchanged

// Lane assignment (host-side packs; device-side reads identical layout):
//   0: mean_x   (fp32)
//   1: mean_y   (fp32)
//   2: a        (fp32, cov_inv[0,0])
//   3: b        (fp32, cov_inv[0,1])
//   4: c        (fp32, cov_inv[1,1])
//   5: opacity  (fp32)
//   6: color_r  (fp32)
//   7: color_g  (fp32)
//   8: color_b  (fp32)
//   9: mb_mask  (uint32, bit m = microblock m active)  *** NEW ***
```

Page size is unchanged at 64 bytes — no NoC-bandwidth change, no CB
reconfig. Reader and host pack changes are both trivial.

### 4.3 Reader kernel diff

`kernels/dataflow/reader_alpha_blend.cpp` does not need to look at the
mask at all — it just streams 64B pages. **Zero changes** to the reader.

The compute kernel reads the mask via `read_tile_value(CB_SCALARS, 0, MB_MASK_LANE)`,
exactly the same way it reads the existing fp32 lanes. See `read_tile_value`
calls in the existing compute kernel for `mean_x`, `opacity`, etc.

`MB_MASK_LANE` depends on which payload layout is in effect:
- 9 fp32 + mb_mask uint32 = 10 lanes total (mask at lane 9). Used during
  Stage 2 of the validation plan (§9), before the basis-form refactor.
- 6 fp32 (A..F) + 4 fp32 (opacity, color_r, color_g, color_b) + mb_mask
  uint32 = 11 lanes total (mask at lane 10). Used after the §5.1
  basis-form refactor and from §6 onward.

Define `MB_MASK_LANE` as a single host-shared constant in `alpha_blend_host.h`
so neither side guesses.

---

## 5. Tile-level basis prep (FPU, once per tile)

We compute six per-tile basis tiles and hold them in CBs across the entire
per-Gaussian inner loop. They are **invariant under the microblock mask**
and they encode all of the per-pixel x/y dependence of the inner Q form.

Per-tile basis tiles (6 tiles at depth 1, allocated as new CBs):

```
CB_BASIS_X    : (px - tile_origin_x)         (fp32, 32x32 tile, "x_local")
CB_BASIS_Y    : (py - tile_origin_y)         (fp32)
CB_BASIS_X2   : x_local * x_local            (fp32)
CB_BASIS_XY   : x_local * y_local            (fp32)
CB_BASIS_Y2   : y_local * y_local            (fp32)
CB_BASIS_ONE  : constant 1.0                 (fp32, prefilled once at init)
```

Build sequence at the top of the per-tile loop, **before** the per-Gaussian
inner loop starts:

```cpp
// Construct CB_BASIS_X = px - tile_origin_x (per-tile constant).
// tile_origin_x is a tile-id-derived runtime arg, broadcast as a scalar
// via sub_unary_tile (SFPU unary, accepts a uint32 fp32-bits immediate).
// 4 SFPU passes, ~tens of cycles -- amortized across all g_count Gaussians.
tile_regs_acquire();
copy_tile_to_dst_init_short(CB_PX);
copy_tile(CB_PX, 0, 0);
sub_unary_tile(0, tile_origin_x_bits);
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(CB_BASIS_X, 1);
pack_tile(0, CB_BASIS_X);
cb_push_back(CB_BASIS_X, 1);
tile_regs_release();

// Same for CB_BASIS_Y from CB_PY.

// Then x², xy, y² with FPU mul_tiles (3 full-tile FPU ops, ~3 * 32 = 96
// matrix-cycles total -- one-time cost per tile).
tile_regs_acquire();
mul_tiles_init(CB_BASIS_X, CB_BASIS_X);
mul_tiles(CB_BASIS_X, CB_BASIS_X, 0, 0, 0);   // dst[0] = x²
mul_tiles_init(CB_BASIS_X, CB_BASIS_Y);
mul_tiles(CB_BASIS_X, CB_BASIS_Y, 0, 0, 1);   // dst[1] = xy
mul_tiles_init(CB_BASIS_Y, CB_BASIS_Y);
mul_tiles(CB_BASIS_Y, CB_BASIS_Y, 0, 0, 2);   // dst[2] = y²
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(CB_BASIS_X2, 1);
cb_reserve_back(CB_BASIS_XY, 1);
cb_reserve_back(CB_BASIS_Y2, 1);
pack_tile(0, CB_BASIS_X2);
pack_tile(1, CB_BASIS_XY);
pack_tile(2, CB_BASIS_Y2);
cb_push_back(CB_BASIS_X2, 1);
cb_push_back(CB_BASIS_XY, 1);
cb_push_back(CB_BASIS_Y2, 1);
tile_regs_release();
```

Cost: ~5 FPU/SFPU full-tile ops × 32 microblocks × 1 cycle/microblock + acquire/pack overhead = a few hundred cycles per tile, paid once. **Saves 4 SFPU mul_unary_tile + 1 sub_unary chain per Gaussian.** Break-even at ≥ 2 Gaussians per tile (typical: 100s).

### 5.1 Per-Gaussian Q in basis form

Once the basis tiles exist, every Gaussian-inner Q computation reduces to

```
Q   = a*x² + 2b*xy + c*y² - 2*(a*mx + b*my)*x - 2*(b*mx + c*my)*y
      + (a*mx² + 2b*mx*my + c*my²)
    = A*x² + B*xy + C*y² + D*x + E*y + F      (basis-form expansion)
```

where `A..F` are 6 fp32 coefficients computed per Gaussian on the host
from `(mean, cov_inv)`. We extend the scalar pack to carry these 6
coefficients **in place of** the 5 raw `(mean_x, mean_y, a, b, c)`
scalars (still 6 fp32 lanes; no payload growth). New layout:

```
  0: A          (= a)
  1: B          (= 2b)
  2: C          (= c)
  3: D          (= -2*(a*mx + b*my))
  4: E          (= -2*(b*mx + c*my))
  5: F          (= a*mx² + 2b*mx*my + c*my²)
  6: opacity
  7: color_r
  8: color_g
  9: color_b
 10: mb_mask    (uint32; we have to extend the pack by 4 bytes here --
                 still 11 lanes = 44 bytes, well inside the 64B page)
```

This is the 057b basis-form refactor that the project previously tried
with bf16 basis tiles and which failed because of `mul_bcast_rows_init_short`
re-init overhead. Here we use **fp32 basis tiles + fp32 dest acc** (the
existing config), so there is no row-bcast re-init churn — the 6 ops are
straight `mul_tiles` against tile-resident operands.

### 5.2 Full-tile inner Q (mask = 0xFFFFFFFF case)

```cpp
// dst[0] = A*x² + B*xy + C*y² + D*x + E*y + F  (basis-form Q)
// All FPU mul_tiles + acc_to_dest accumulation.
tile_regs_acquire();
mul_tiles_init(CB_BASIS_X2, CB_BASIS_ONE);
mul_tiles(CB_BASIS_X2, CB_BASIS_ONE, 0, 0, 0);   // dst[0] = x² * 1
mul_unary_tile(0, A_bits);                        // dst[0] *= A   (SFPU 4 pass)

// We then want dst[0] += B * xy + C * y² + D * x + E * y + F.
// Each += can be done with an FPU mul + binary_dest_reuse_tiles<ADD>:
//   tmp_slot = mul_tiles(BASIS_XY, BASIS_ONE);
//   mul_unary_tile(tmp_slot, B_bits);
//   binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCB>(tmp_slot, 0);
// 5 such += chains -- this is the 057b code path. fp32 tile multiplies
// against an (almost) ones tile is a single FPU pass; no SFPU bcast init
// overhead because the dest format is fp32 throughout.

// ... 5 MAC chains ...
// followed by power = -0.5 * Q ; weight = exp(power) ; alpha = ... ;
// (existing C/D2/E pipeline, unchanged)

tile_regs_commit();
tile_regs_wait();
// pack alpha, contrib, R/G/B, T -- existing pattern.
tile_regs_release();
```

This is only 1.5x faster than today's `dx²/dxdy/dy²` recompute path
(the savings come from not recomputing `dx, dy, dx*dy` per Gaussian,
~6 SFPU ops/Gaussian → 0). It is necessary to make the per-microblock
path cheap (we'll reuse the basis tiles there).

---

## 6. Per-microblock inner kernel (the SFPU sparse path)

This is the heart of the design. Given a Gaussian with `mb_mask`, we
update the per-tile state (R/G/B/T) **only on the active microblocks**,
loading basis values from the basis tiles and per-Gaussian coefficients
from the SFPU constant registers we set up once per Gaussian.

### 6.1 DST register slots used

We allocate the per-Gaussian inner work to 5 DST tile slots:

```
slot 0:  Q (Mahalanobis quadratic) being built up across the 5 += stages
slot 1:  alpha (after exp + cap)
slot 2:  contrib = alpha * T * sat_mask
slot 3:  Q scratch / temporary
slot 4:  T_new scratch
```

DST budget on Blackhole with `fp32_dest_acc_en = true` is 8 tiles. We use
5 → safe.

### 6.2 Setup phase per Gaussian

Once per Gaussian, before iterating microblocks, we:

1. Read the 11-lane scalar pack into 11 LREG-resident fp32 constants
   using `read_tile_value(CB_SCALARS, 0, lane)` → bit-reinterpret to
   fp32.
2. Convert each fp32 scalar coefficient to its uint32 bit pattern for use
   as an immediate to `_unary` SFPU ops (the existing kernel already does
   this — see `NEG_HALF_BITS` etc.).
3. Lift the 6 basis coefficients `A..F` and `opacity`, `color_r/g/b` into
   per-Gaussian uint32 immediates.
4. Read `mb_mask`. If `popcount(mb_mask) >= 32 - SPARSE_SKIP_THRESH`, take
   the full-tile path (§5.2) and skip the per-microblock loop. Else go to
   §6.3.

### 6.3 Per-microblock loop body

The inner loop iterates `m = 0..31`, skipping inactive microblocks:

```cpp
constexpr uint32_t MB_TO_DST_ADDR[32] = {
    /* m =  0 */ 0,  /* m =  1 */ 2,  /* m =  2 */ 16, /* m =  3 */ 18,
    /* m =  4 */ 4,  /* m =  5 */ 6,  /* m =  6 */ 20, /* m =  7 */ 22,
    /* m =  8 */ 8,  /* m =  9 */ 10, /* m = 10 */ 24, /* m = 11 */ 26,
    /* m = 12 */ 12, /* m = 13 */ 14, /* m = 14 */ 28, /* m = 15 */ 30,
    /* m = 16 */ 32, /* m = 17 */ 34, /* m = 18 */ 48, /* m = 19 */ 50,
    /* m = 20 */ 36, /* m = 21 */ 38, /* m = 22 */ 52, /* m = 23 */ 54,
    /* m = 24 */ 40, /* m = 25 */ 42, /* m = 26 */ 56, /* m = 27 */ 58,
    /* m = 28 */ 44, /* m = 29 */ 46, /* m = 30 */ 60, /* m = 31 */ 62,
};
// Verified by hand against §3.4 enumeration; addresses are within
// dst_tile_idx = 0 (we acquire one DST tile slot for the inner loop).

uint32_t mb_mask = ckernel::read_tile_value(CB_SCALARS, 0, 10);

tile_regs_acquire();

// First we need the basis tiles loaded into a known set of DST slots. The
// FPU does this with copy_tile -- one full-tile copy per basis tile.
copy_tile_to_dst_init_short(CB_BASIS_X);
copy_tile(CB_BASIS_X,  0, 0);   // dst tile 0 = x_local
copy_tile_to_dst_init_short(CB_BASIS_Y);
copy_tile(CB_BASIS_Y,  0, 1);   // dst tile 1 = y_local
copy_tile_to_dst_init_short(CB_BASIS_X2);
copy_tile(CB_BASIS_X2, 0, 2);   // dst tile 2 = x²
copy_tile_to_dst_init_short(CB_BASIS_XY);
copy_tile(CB_BASIS_XY, 0, 3);   // dst tile 3 = xy
copy_tile_to_dst_init_short(CB_BASIS_Y2);
copy_tile(CB_BASIS_Y2, 0, 4);   // dst tile 4 = y²
// State CBs already loaded into dst tiles 5..7 in the OUTER acquire/commit
// flow we describe in §6.4.

// We will write the per-microblock result to dst tile slot SLOT_OUT.
constexpr uint32_t SLOT_OUT = 5;          // R_state in-place update target
constexpr uint32_t TILE_BASE_X  = 0 * 64;
constexpr uint32_t TILE_BASE_Y  = 1 * 64;
constexpr uint32_t TILE_BASE_X2 = 2 * 64;
constexpr uint32_t TILE_BASE_XY = 3 * 64;
constexpr uint32_t TILE_BASE_Y2 = 4 * 64;
constexpr uint32_t TILE_BASE_OUT = SLOT_OUT * 64;
// Note: TILE_BASE_* + MB_TO_DST_ADDR[m] = absolute SFPLOAD/SFPSTORE addr
// for microblock m of that DST tile slot.

// Pre-load the 11 per-Gaussian fp32 coefficients into LREGs that survive
// across all 32 microblock iterations (LREG10..15 are not touched by the
// per-microblock body, see §6.5):
//   LREG10 = A
//   LREG11 = B   (overrides default; see §6.5 caveat about LREG_11 = -1.0)
//   LREG12 = C
//   LREG13 = D
//   LREG14 = E
//   LREG15 = F
// Done via 6 SFPLOADI sequences (each = 2 instructions: UPPER + LOWER).
// (Helper: see _build_lane_mask_col0_ in ckernel_sfpu_binary_bcast.h
// lines 192-200 for the exact SFPLOADI pattern.)

// Iterate 32 microblocks; skip those with mb_mask bit clear.
for (int m = 0; m < 32; m++) {
    if ((mb_mask & (1u << m)) == 0) continue;

    uint32_t a_x  = TILE_BASE_X  + MB_TO_DST_ADDR[m];
    uint32_t a_y  = TILE_BASE_Y  + MB_TO_DST_ADDR[m];
    uint32_t a_x2 = TILE_BASE_X2 + MB_TO_DST_ADDR[m];
    uint32_t a_xy = TILE_BASE_XY + MB_TO_DST_ADDR[m];
    uint32_t a_y2 = TILE_BASE_Y2 + MB_TO_DST_ADDR[m];
    uint32_t a_out = TILE_BASE_OUT + MB_TO_DST_ADDR[m];

    // === Q = A*x² + B*xy + C*y² + D*x + E*y + F  (microblock-local) ===
    // 5 FMAs + 1 base load. All independent across distinct LREG dests.
    // LREG6 = scratch accumulator. LREG7 = temporary.
    TT_SFPLOAD(p_sfpu::LREG6, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_x2);
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG10 /*A*/, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);

    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_xy);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG11 /*B*/, p_sfpu::LREG6, p_sfpu::LREG6, 0);

    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_y2);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG12 /*C*/, p_sfpu::LREG6, p_sfpu::LREG6, 0);

    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_x);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG13 /*D*/, p_sfpu::LREG6, p_sfpu::LREG6, 0);

    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_y);
    TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG14 /*E*/, p_sfpu::LREG6, p_sfpu::LREG6, 0);

    // dst[6] += F  (1.0 * F + LREG6)
    TTI_SFPADD(p_sfpu::LREG15 /*F*/, p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG6, 0);

    // === alpha = clamp(opacity * exp(-0.5 * Q), 0, 0.99) ===
    // Per-microblock exp via the inline SFPU exp polynomial. We do NOT
    // call exp_tile here -- exp_tile is a full-tile op. Instead we use
    // the lower-level _calculate_exponential_body_<APPROXIMATION_MODE>
    // entry point that operates on dst_reg[0] (= 32 lanes = 1 microblock).
    //
    // Plan A (simpler): SFPU exp polynomial inlined here. Implementation
    //   mirrors the body of _calculate_exp_inline_<true> in
    //   ckernel_sfpu_exp.h with dst_reg pointing at our LREG_TMP. About
    //   ~10 SFPU ops -- the same the full-tile exp pays per microblock,
    //   so no microblock skip benefit lost.
    //
    // Plan B (faster): use exp_tile<true>() in the FULL-TILE inner-Q
    //   path only and exit the microblock path BEFORE exp -- i.e. we
    //   only do per-microblock work for the Q evaluation, then promote
    //   the result back to a full DST tile (the inactive microblocks
    //   remain whatever the previous Gaussian left, but they will be
    //   overwritten safely because contrib will multiply them by an
    //   in-active-mask predicate before adding to state).
    //
    // First impl: Plan A. See §6.6 for the inline exp expansion.

    // === inline exp on LREG6 (= power) ===
    // IMPORTANT: there is no built-in LCONST_neg1_half. Only LCONST_0,
    // LCONST_1, LCONST_neg1 are guaranteed (verified via grep of
    // tt_llk_blackhole/common/inc/ckernel_instr_params.h). Two options:
    //
    //   (a) Pre-multiply A..F by -0.5 on the host so that the basis-form
    //       Q evaluation directly produces -0.5*Q. Then "power" is the
    //       output of the §6.3 Q chain with no additional multiply.
    //       This is the recommended path -- host coefficients become:
    //         A' = -0.5*A, B' = -0.5*B, C' = -0.5*C,
    //         D' = -0.5*D, E' = -0.5*E, F' = -0.5*F.
    //       Saves 1 SFPMUL per microblock per Gaussian.
    //
    //   (b) Load -0.5 into a reserved LREG once per Gaussian via
    //         TTI_SFPLOADI(LREG_NEG_HALF, SFPLOADI_MOD0_UPPER, 0xBF00);
    //         TTI_SFPLOADI(LREG_NEG_HALF, SFPLOADI_MOD0_LOWER, 0x0000);
    //       and multiply per-microblock. 2 cycles per Gaussian setup +
    //       1 SFPMUL per microblock.

    /* with option (a): no scalar multiply here -- LREG6 is already -0.5*Q */
    /* ... ~8 SFPU ops for exp polynomial, see ckernel_sfpu_exp.h ... */
    /* result in LREG6 = weight */

    // alpha = min(opacity * weight, 0.99)
    TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG_OPACITY, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    /* TTI_SFPMIN against 0.99 (see ckernel_sfpu_binary_max_min.h for the
       exact instruction sequence) -- result in LREG6 = alpha */

    // === contrib = alpha * T_state[m] * sat_mask[m] ===
    // T_state and sat_mask are state CBs -- we keep them in DST tiles 6 and 7.
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, /* T_state's MB addr */ 6 * 64 + MB_TO_DST_ADDR[m]);
    TTI_SFPMUL(p_sfpu::LREG7, p_sfpu::LREG6 /*alpha*/, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
    /* ... * sat_mask similarly ... result LREG6 = contrib */

    // === R/G/B accum (3 FMAs) and T_new = T - contrib (1 SUB) ===
    // Fused like iter-045 D2+E mega-fuse, but per microblock.
    // R_state[m] += contrib * color_r
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, /* R_state's MB addr */ 5 * 64 + MB_TO_DST_ADDR[m]);
    TTI_SFPMAD(p_sfpu::LREG6 /*contrib*/, p_sfpu::LREG_COLOR_R, p_sfpu::LREG7, p_sfpu::LREG7, 0);
    TT_SFPSTORE(p_sfpu::LREG7, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 5 * 64 + MB_TO_DST_ADDR[m]);

    /* ... G_state, B_state, T_state likewise ... */
}

tile_regs_commit();
// note: the state CBs are updated IN-PLACE in DST. We pack them BACK to
// the same state CB slots after the WHOLE inner Gaussian loop ends -- not
// per Gaussian. See §6.4.
```

### 6.4 State CB I/O strategy (key design choice)

The current kernel pops + reserves + pushes the state CBs **inside the
per-Gaussian loop** (one full-tile pack/unpack pair per Gaussian). For
the microblock path this would be catastrophic: per-microblock SFPU
work is fast, but pack/unpack is full-tile.

The fix: **lift state I/O out of the per-Gaussian loop**.

```
Per-tile flow:
  Acquire DST.
    Load R/G/B/T/sat state CBs into DST tile slots 5/6/7/... once.
    For each Gaussian g in this tile:
      Per-Gaussian setup: load scalar pack, prep LREG10..15.
      If full-tile path: existing dx²/dxdy/dy² flow, write to slots 5..7
                         in-place via dest_reuse_tiles binary ops.
      If microblock path: per-microblock SFPU work, in-place on slots 5..7.
    Store R/G/B/T/sat state from DST tile slots back to state CBs.
  Commit / pack / release.
```

This means **one DST acquire per tile** (not per Gaussian). It is the
single largest reduction in overhead this design enables.

⚠ Constraint: DST acquire holds the math-pack synchronization point. With
8 fp32 dest tile slots (Blackhole `fp32_dest_acc_en`), we have 5 basis
tiles + 5 state tiles = 10 — too many. Resolution: load basis tiles **per
Gaussian** (5 copy_tile per Gaussian, 4 of which are recycled, total
amortized ~few SFPU ops/Gaussian) and keep state tiles resident across
the whole tile loop. Slot map:

```
DST slots:
  0: scratch / Q accumulator      (per-Gaussian, transient)
  1: alpha                         (per-Gaussian, transient)
  2: contrib                       (per-Gaussian, transient)
  3: x_local | x²                  (per-Gaussian basis, reloaded with copy_tile)
  4: xy | y² | y_local             (per-Gaussian basis, reloaded)
  5: R_state                       (PERSISTENT across Gaussians)
  6: G_state                       (PERSISTENT)
  7: B_state                       (PERSISTENT)
```

T_state and sat_mask cannot stay in DST — they don't fit. They live in
their CBs and are pulled into a transient DST slot when needed (slot 0 or
2). For `T_new = T*sat - contrib` we mux through slot 0 with one
copy_tile from CB_T_STATE per Gaussian.

This is a very real DST budget trade-off and IS the rate-limiting
constraint on this design. We may need to:
  - Drop sat_mask (revert iter-013 effort: just eat the no-op alpha
    contributions on saturated pixels). Cost: a few extra SFPU ops per
    saturated pixel per Gaussian; but most pixels are non-saturated so
    the average is ~0%.
  - Or split R+G+B into a single 3-channel "color_state" tile by
    storing them as 3 contiguous 32×32 tiles in adjacent DST slots and
    sharing the `contrib` LREG across all 3 microblock SFMADs.

First implementation: **drop sat_mask**, keep R/G/B/T as 4 PERSISTENT
DST tile slots (slots 4..7), use slots 0..3 as transient.

### 6.5 LREG allocation contract

Within the per-microblock loop body we use:

```
LREG0..LREG5    : per-microblock transient scratch (Q accumulator, exp
                  polynomial intermediates, contrib, etc). Free at start
                  of each iteration; need not be preserved across iterations.

LREG6           : per-microblock load/Q/contrib working scratch.
LREG7           : per-microblock multiplicand load (basis values, state
                  values).

LREG8, LREG9    : reserved scratch for inline exp polynomial (matches
                  the pattern in ckernel_sfpu_exp.h _calculate_exp_body_).

LREG10..LREG15  : PER-GAUSSIAN persistent fp32 immediates:
                  10 = A
                  11 = B            (overrides hardware default of -1.0;
                                     restore at end of Gaussian if any
                                     downstream code expects LREG11 = -1.0)
                  12 = C
                  13 = D
                  14 = E
                  15 = F

  ALSO USED AS PERSISTENT:
  LREG_OPACITY, LREG_COLOR_{R,G,B}: aliased onto LREG10..15 by sharing
  slots after the Q evaluation (which only needs A..F) -- once we've
  computed Q for this microblock and moved on to the alpha/contrib chain
  we can clobber LREG10..14 with opacity/color_r/g/b and pull them back
  in for the NEXT microblock from the constant scalar pack we keep
  re-reading via SFPLOADI.

  This is a tight register budget; if we run short, fall back to
  reloading 1-2 fp32 immediates per microblock via SFPLOADI (4 cycles
  per immediate, ~ amortized ~negligible against the 5 SFMADs).
```

Restoring LREG11 to -1.0 at end-of-Gaussian: `TTI_SFPLOADI(LREG11, SFPLOADI_MOD0_UPPER, 0xBF80); TTI_SFPLOADI(LREG11, SFPLOADI_MOD0_LOWER, 0x0000);` — same pattern as `_build_lane_mask_col0_` lines 213–215 of the bcast kernel.

### 6.6 Inline exp polynomial

For the per-microblock path we cannot call `exp_tile<true>(0)` (full-tile
op). We need a per-microblock SFPU exp. Two options:

**Option A: inline the body of `_calculate_exponential_<APPROXIMATION_MODE=true>`** — see `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`. The approximate-mode polynomial is ~10 SFPU ops on `dst_reg[0]`. We adapt it to operate directly on `LREG6` (input/output) using the explicit-LREG `TTI_SFPMAD/TTI_SFPMUL/TTI_SFPADD` form. No `dst_reg++` cursor advance because we're processing exactly 1 microblock per call. This is the path of least resistance.

**Option B: lltt::replay** — record the exp polynomial body to a replay slot once at kernel init (`lltt::record(REPLAY_SLOT_EXP, REPLAY_LEN_EXP)`) and issue `lltt::replay(REPLAY_SLOT_EXP, REPLAY_LEN_EXP)` per microblock. Saves on dispatch cost in the inner loop.

Recommendation: ship Option A first; A→B if profiling shows replay overhead winnable.

The `_init_exponential_<APPROXIMATION_MODE>()` call must still be issued **once per kernel init** so the SFPU approximation constants are loaded.

### 6.7 Sparse loop structure & branch overhead

A naive `for (m = 0; m < 32; m++) if (mb_mask & (1<<m)) { ... }` adds a
branch + bit-test per microblock — 64 cycles/Gaussian wasted on inactive
microblocks. Alternative: use **bit-manipulation iteration** (`__builtin_ctz`):

```cpp
uint32_t remaining = mb_mask;
while (remaining) {
    uint32_t m = __builtin_ctz(remaining);
    remaining &= remaining - 1;
    /* ... per-microblock body using MB_TO_DST_ADDR[m] ... */
}
```

Loop body executes exactly `popcount(mb_mask)` times. Loop overhead
~3 cycles/iteration. For typical 25% sparsity (8/32 microblocks active)
this is ~24 cycles/Gaussian of loop overhead — < 5% of inner work.

---

## 7. Quality / correctness invariants

These MUST hold for any candidate change:

1. **Mask completeness**: every pixel where the Gaussian's actual
   contribution exceeds `contrib_floor` (1/255 = 3.92e-3) must be inside
   at least one microblock with `mb_mask[m] == 1`. Use the existing
   closest-point-in-microblock Mahalanobis test (§3.2) — it is
   conservative (overestimates contribution because closest-point
   minimizes Mahalanobis distance) so this holds by construction.

2. **State coherence**: a pixel that is NOT in any active microblock for
   Gaussian `g` must have its state (R/G/B/T) preserved exactly across
   that Gaussian. The per-microblock SFPU path satisfies this because
   we only `SFPSTORE` into microblocks that are masked active.

3. **PSNR ≥ 35.0 dB** on the validation suite (clean-keep gate from
   the SUPERVISOR-LOOP). Microblock culling tightens the per-pair
   contribution test; PSNR should *improve* slightly relative to the
   current AABB+per-pair pipeline.

4. **No tile-boundary artifacts**: pixels on a microblock boundary are
   computed identically whether their microblock is active or not (the
   compute is the same per-pixel formula). The only difference is
   whether their result gets written. The closest-point test is
   monotonic in distance to mean → no chequerboard tearing across
   microblock seams.

5. **DST budget**: 8 fp32 dest tiles total. We use 8 (4 transient + 4
   persistent state). Adding a 9th will silently corrupt — verify with
   `tile_regs_acquire`/`commit` always paired and never spanning a
   reconfig.

---

## 8. Performance model

### 8.1 Cycles per microblock (estimated, BH SFPU @ 1 GHz)

Per active microblock (Q + exp + alpha + contrib + R/G/B/T):

```
Q evaluation         : 5 SFPMAD + 5 SFPLOAD + 1 SFPLOAD + 1 SFPADD  = ~14 cycles
exp polynomial (apx) : ~10 SFPU ops                                  = ~10 cycles
opacity * weight     : 1 SFPMUL                                      =  ~1 cycles
min(., 0.99)         : 2 SFPU ops (SFPMIN)                           =  ~2 cycles
contrib chain        : 2 SFPMUL + 1 SFPLOAD                          =  ~3 cycles
R/G/B accum (3 ch)   : 3 SFPMAD + 3 SFPLOAD + 3 SFPSTORE             =  ~9 cycles
T update             : 1 SFPMUL + 1 SFPMAD + 1 SFPLOAD + 1 SFPSTORE  =  ~4 cycles
                                                              total  = ~43 cycles
```

### 8.2 Cycles per Gaussian (full-tile path = today)

```
dx, dy, dx², dy², dx*dy   : ~5 SFPU full-tile ops × 4 microblocks/face
                            × 4 faces = ~80 cycles SFPU work + FPU mul_tiles
Q via FPU mul_tiles_bcast  : ~3 FPU passes × 32 cycles                    = ~100
exp_tile<true>             : ~ 100 cycles (per existing kernel docs)
alpha cap                  : ~30 cycles
contrib + R/G/B + T (D2/E) : ~3 FPU passes × 32 cycles                   = ~100

Per Gaussian ≈ 400-500 SFPU+FPU cycles.
```

Currently observed: ~25.4 ms / 1080p frame / ~50k Gaussian-tile pairs ≈
500 ns/pair @ 1 GHz = 500 cycles/pair. Matches.

### 8.3 Per-microblock total

```
For mask sparsity factor k = popcount(mb_mask) / 32:
  cycles_per_g ≈  k * 32 * 43   +   constant (basis copy + scalar load + branch)
              =  k * 1376       +   ~100 cycles
              ≈  100 + 1376*k   cycles

Crossover with 500 cycles full-tile:  k ≈ 0.29.

So:
  k = 0.10 (3 active microblocks): ~240 cycles/g  (2.1x speedup)
  k = 0.25 (8 active microblocks): ~440 cycles/g  (1.1x speedup)
  k = 0.50 (16 active microblocks): ~790 cycles/g (0.6x speedup -- USE FULL TILE)
  k = 1.00 (full tile):            ~1500 cycles/g (TAKE FULL TILE PATH)
```

Conclusion: the per-microblock path **only wins for k < 0.3** (at most
9 active microblocks). The threshold check in §6.2 step 4 must
guarantee we go full-tile above that.

### 8.4 Expected end-to-end speedup

The histogram of `popcount(mb_mask) / 32` across pairs is the input we
need from a profiling pass. Hypothesis (to validate first iteration):
~half of pairs are k > 0.5 (full-tile path, no change in cost),
~half are k < 0.3 (microblock path, 2-3× speedup). Net ~1.5× kernel
speedup → 25.4ms → ~17ms.

This is a conservative model; we can do better with §10 follow-ups.

---

## 9. Validation plan

Three stages, each gated:

### Stage 1: host binning correctness, no kernel changes

1. Implement §3 in `gsplat/rasterization.py`. Emit `mb_masks` alongside
   the existing outputs.
2. Property test: for every `(g, tile)` pair in the existing pipeline,
   assert `popcount(mb_mask[i]) >= 1` and `popcount(mb_mask[i]) <= 32`.
3. Visual test: rasterize on a CPU reference (use the existing CPU
   backend) using `mb_masks` as a "render only these microblocks" mask
   for each Gaussian (any pixel outside an active microblock keeps the
   pre-Gaussian state). PSNR vs no-mask reference must be ≥ 100 dB
   (rounding-only deltas).
4. Gate: PSNR ≥ 100 dB AND `popcount` distribution matches expectations
   (eg. heavy tail at 32 for big splats, mode at 8-16 for typical).

### Stage 2: full-tile path with basis tiles (no microblock skip)

1. Implement §5 (basis tile prep).
2. Implement §5.2 full-tile inner Q via basis form.
3. Run the existing iter validation: `device_kernel` ms, PSNR.
4. Gate: PSNR ≥ 35.0 dB ON ALL VALIDATION VIEWS, kernel ms ≤ current
   median (no regression).

### Stage 3: per-microblock SFPU path

1. Implement §6 with a runtime knob `GSPLAT_TT_USE_MICROBLOCK_PATH=0/1`.
2. With knob = 0: must reproduce Stage 2 results bit-for-bit.
3. With knob = 1: PSNR drift ≤ 0.5 dB vs Stage 2; kernel ms reduction.
4. Gate: PSNR ≥ 35.0 dB AND device_kernel ms reduction.

### Bisect on failure

If PSNR drops below the gate, the bisect order is:
  a. Per-microblock state I/O coherence bug (most likely; probe with
     a small targeted test that runs ONE Gaussian against ONE tile
     with mb_mask = 0x0000_0001 and checks every pixel).
  b. Inline exp polynomial difference vs `exp_tile<true>` (diff
     2-3 ULP per microblock should be safe; if it isn't, switch to
     full-tile exp_tile and only microblock-skip the Q evaluation).
  c. LREG11 not restored to -1.0 across Gaussians (run with restore).

---

## 10. NOT YET DESIGNED (future iterations)

These are explicitly out-of-scope for the first microblock implementation
but are obvious follow-ups once §1-9 ship:

- **Multi-microblock packing of dependent Gaussians**: if two adjacent
  Gaussians both have mask `m`, their per-microblock work could share
  the basis SFPLOAD. Saves ~5 SFPLOADs per microblock per Gaussian.
- **lltt::replay-based inner kernel** (§6.6 Option B). Estimated 10-20%
  speedup on inner-loop dispatch overhead.
- **Tile-level FP16 basis** (revisit iter-057b): once microblocks are
  in place, the basis tiles' precision tradeoff is bounded by a single
  microblock's worth of pixels (32 lanes), making bf16 basis tile error
  much smaller in absolute terms than the previous global-coords
  attempt. Can save a 2x in basis tile L1 footprint.
- **Microblock dispatch from host**: instead of one CB push per tile,
  push one CB entry per **active microblock-Gaussian** pair (host
  expands the bin to per-microblock granularity). Loses the current
  per-tile state coherence model — would need a redesign of the inner
  loop to be (microblock, gaussian) instead of (gaussian, microblock).
  Likely net loss until L1 bandwidth becomes the bottleneck.
- **Persistent kernel + mailbox dispatch**: orthogonal to microblocks
  but composes; queued for after §1-9.
- **8x8 microblocks** (vs 4x8): would require a SFPLOAD that reads 64
  lanes; the SFPU is 32-lane wide so this is hardware-impossible.
  We are at the natural minimum granularity already.

---

## 11. Concrete file diffs (preview, not yet applied)

```
gsplat/rasterization.py
  + _assign_gaussians_to_tiles_microblock_masks()  (§3)
  + return mb_masks alongside (gaussian_ids, tile_ids, tiles_per_gaussian)

gsplat/backend.py + backends/tt/backend.py
  + plumb mb_masks into prepare_kernel_inputs
  + pack mb_mask into the 11th uint32 lane of the 64B scalar pack

backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/
  alpha_blend_host.h
    + SCALAR_PACK_LANES bumped to 11
    + mb_to_dst_addr() constexpr (§3.4)
    + MB_TO_DST_ADDR[32] static table
    + CB_BASIS_X / Y / X2 / XY / Y2 / ONE indices

  alpha_blend.cpp
    + 6 new CB allocations (basis tiles), all depth=1, fp32
    + (no kernel-args change; existing dispatch path reuses)

  kernels/dataflow/reader_alpha_blend.cpp
    (no changes; pack is still 64B/Gaussian)

  kernels/compute/alpha_blend_compute.cpp
    + at init: build basis tiles ONCE per program (§5)... wait, basis
      tiles are PER-TILE, not per-program. Build in the per-tile loop
      head (§5).
    + per-tile: compute basis tiles, hold across inner Gaussian loop
    + per-Gaussian: dispatch microblock vs full-tile path on
      popcount(mb_mask)
    + per-microblock body using TT_SFPLOAD/TTI_SFPMAD/TT_SFPSTORE
      against MB_TO_DST_ADDR
    + state CBs: lift acquire/commit out of the per-Gaussian loop
      (acquire ONCE at start of tile, commit ONCE at end)
```

---

## 12. References (verbatim source files used)

These were read to ground every snippet in this document:

- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h`
  (DST layout, SFPU instruction patterns, replay buffer usage,
  pipelining rules).
- `tt_metal/hw/inc/api/compute/sfpu_binary_bcast.h`
  (high-level binop bcast API, `sfpu_*_bcast_col_init` / `_bcast_col`).
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`
  `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h`
  (sfpi-style `dst_reg[0] / dst_reg++` pattern, ITERATIONS = 8 per face).
- `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h`
  (`calculate_exponential` template signature, init signature).
- `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_recip.h`
  (lltt::record/replay usage in production; `lltt::replay(slot, len)` form).
- `runtime/sfpi/include/lltt.h`
  (canonical `lltt::record<ExecBool>(start, length)` /
  `lltt::replay(start, length)` signatures; included via `#include "lltt.h"`).
- `tt_metal/programming_examples/gaussian_splatting/alpha_blend_host.h`
  (existing CB layout, existing 9-fp32 scalar pack).
- `tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp`
  (existing 5-stage F/A/B/C/D/E pipeline; iter-045 D2+E mega-fuse pattern).
- `tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp`
  (existing reader pipeline; LPT tile-id slice, scratch L1 layout).
- `gsplat/rasterization.py::_assign_gaussians_to_tiles`
  (existing tile binning; per-pair Mahalanobis cull pattern).

When implementing this design, **re-open the corresponding source file
each time you reach for an API**. Do not reproduce these snippets from
memory; the SFPU instruction encodings have arch-specific edge cases
that the production headers handle and that this design intentionally
delegates rather than re-derives.
