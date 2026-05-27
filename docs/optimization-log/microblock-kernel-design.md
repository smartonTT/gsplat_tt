# Microblock Alpha-Blend Kernels — Design Spec

Status: design (no code yet)
Target: device_kernel time on Blackhole (P300, yyzo-bh-14), 1080p stitch.
Author trail: derived from a verified read of `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_binary_bcast.h`, `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_silu.h`, the existing `programming_examples/gaussian_splatting/{alpha_blend.cpp, alpha_blend_host.h, kernels/{compute,dataflow}/*.cpp}`, and `gsplat/rasterization.py::_assign_gaussians_to_tiles`.

---

## 0. TL;DR

The current kernel does full 32×32 tile compute for every (Gaussian, tile) pair, even when a Gaussian only meaningfully touches a few of the tile's 32 microblocks (4-row × 8-col SFPU slots). Real Gaussian footprints are highly skewed: a 3σ ellipse with σ ≈ 1–4 px touches 10–40% of the pixels in the tiles it overlaps. We are paying ~3–10× the inner SFPU work we need.

This spec defines a **microblock-major** pipeline:

1. **Host bins to microblocks**, not just to tiles. Per tile it emits (a) one coefficient table for that tile's gaussians and (b) 32 ordered, depth-sorted gaussian-index lists, one per microblock.
2. **Compute kernel keeps 32×32 tile dispatch** (matches existing CB / pack / writer model) but its outer loop is `for microblock m in tile`. Inner loop is `for gaussian g in microblock_list[m]`.
3. **Inner body is per-microblock SFPU math** via `TT_SFPLOAD/TTI_SFPMAD/TT_SFPSTORE` against the microblock's DST address (one 4-row × 8-col, 32-lane slot).
4. **State (R/G/B/T for this microblock) lives in 4 LREGs across the microblock's gaussian inner loop.** DST is touched only at microblock boundaries (32× per tile, not G× per tile).
5. **Basis values for this microblock (x², xy, y², x, y) live in 5 LREGs across the same inner loop.** Loaded once per microblock from per-tile basis tiles in DST; computed from x, y via 3 SFPMUL.
6. **Per-Gaussian coefficients (A..F + opacity + color) come from a per-tile L1 table.** Indexed by the microblock's gaussian-list entry; loaded fresh per (microblock, gaussian) iteration.
7. **Inner SFPU sequence is recorded into the replay buffer once** and reissued per (microblock, gaussian) via `lltt::replay`.

Per-microblock SFPLOADs/SFPSTOREs are amortized over the microblock's gaussian list; per-gaussian coefficient loads are amortized over the 11 SFPLOADIs constant cost. Memory ops drop ~10× relative to the g-major + mask-skip alternative we considered first. Expected end state: device_kernel time proportional to active (microblock, gaussian) pairs.

This design is informed by two prior proposals (see §13) — one had the right outer loop and the wrong inner math (full-tile compute then microblock-extract), the other had the right inner math (real `TT_SFPLOAD`/`TTI_SFPMAD`/`TT_SFPSTORE`) and the wrong dispatch granularity (per-face). Neither fit DST budget. This spec is the verified synthesis.

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
  1. project_gaussians                  (existing)
  2. AABB tile assignment               (existing, _assign_gaussians_to_tiles)
  3. PER-PAIR mahalanobis cull          (existing, contrib_floor)
  4. NEW: per-microblock cull            for each surviving (g, tile) pair,
                                        compute the 32-microblock activity
                                        mask (closest-point Mahalanobis per
                                        microblock).
  5. depth sort + per-tile bin          (existing, sort_and_bin)
  6. NEW: expand to per-microblock      from each (g, tile) pair's mask,
     gaussian-index lists                emit one (microblock_idx,
                                        local_gaussian_idx) pair per active
                                        microblock. Group these per
                                        microblock per tile -> 32 lists/tile.
  7. NEW: pack per-tile coefficient     for each tile, write the contiguous
     table to DRAM                       table of {A..F, opacity, color_rgb}
                                        for the L = g_count gaussians of
                                        this tile.
  8. NEW: pack per-tile microblock      for each tile, write 32 (offset,
     headers + per-microblock streams    count) entries followed by the
                                        flat concatenated stream of L'
                                        local-gaussian-indices (sum of
                                        per-microblock counts; L' = sum of
                                        popcount(mask) across the tile's
                                        gaussians).
  9. pack px/py tiles, tile_ids         (existing)

DEVICE
  READER (NCRISC, NoC1, kernels/dataflow/reader_alpha_blend.cpp)
    Per assigned tile:
      - push px, py tiles to CB_PX, CB_PY                          (existing)
      - DMA per-tile coefficient table to CB_COEFF_TABLE           (NEW)
      - DMA per-tile microblock-header to CB_MB_HEADER             (NEW)
      - DMA per-tile microblock-stream to CB_MB_STREAM             (NEW)
        (one shot per tile -- no per-gaussian DMA churn)

  COMPUTE (kernels/compute/alpha_blend_compute.cpp)
    Once per program launch:
      - prefill CB_CONST_ZERO, CB_CONST_099                         (existing)
      - record replay slot for the per-microblock SFPU inner body   (NEW)

    Per tile:
      A. Acquire DST.
      B. Build per-tile basis tiles X, Y in DST slots 4, 5:        (NEW)
         - X = px - tile_origin_x         (SFPU sub_unary_tile)
         - Y = py - tile_origin_y
         - These are PER-TILE constants; held in DST across the
           whole 32-microblock outer loop.
      C. Init state DST slots 0,1,2,3:
         - R = G = B = 0, T = 1                                    (fill_tile)
      D. Outer microblock loop:  for m in [0, 32):
           1. count = mb_header[m].count
              if count == 0: continue
           2. Read 5 basis values for microblock m into LREGs:
              X^2_lreg, XY_lreg, Y^2_lreg via SFPMULs of
              X_lreg, Y_lreg loaded from DST slots 4, 5 at m's
              addr.
           3. Load microblock m's R, G, B, T state from DST slots
              0..3 (at m's addr) into 4 LREGs.
           4. Inner gaussian loop:  for i in [0, count):
                gauss_idx = mb_stream[mb_header[m].offset + i]
                load (A..F, opacity, color_rgb) for gauss_idx from
                  CB_COEFF_TABLE into LREGs
                lltt::replay(REPLAY_SLOT_MB_BODY) -- runs Q, exp,
                  alpha, contrib, R/G/B FMA, T update entirely in
                  LREGs (no DST touch).
                periodic saturation check: if T-all-lanes saturate,
                  break.
           5. SFPSTORE the 4 state LREGs back to DST slots 0..3 at
              m's addr.
      E. Commit DST. Pack DST slots 0..2 to CB_COLOR_OUT (R/G/B).
      F. Release DST.

  WRITER (BRISC, NoC0, kernels/dataflow/writer_alpha_blend.cpp)     (existing)
```

### 2.1 No "hybrid full-tile path" anymore

The earlier draft kept a full-tile FPU fallback for dense `mb_mask`. With
mb-major dispatch the dense case is exactly: a microblock whose count
equals the total tile gaussian count L (every gaussian touches it).
The per-microblock SFPU path already handles this — its cost in that case
is `L × ~30 cycles` per microblock, which is competitive with the FPU
full-tile path at any realistic L. We delete the hybrid threshold; there
is one code path. (We revisit this in §8 if the performance model says
otherwise.)

---

## 3. Host binning to microblocks

### 3.1 What the host emits

For each tile, the host produces three blobs in DRAM:

```python
# Per-tile coefficient table: contiguous fp32 rows, one per local-gaussian.
# Indexed by `local_gaussian_idx in [0, L)` where L = number of gaussians
# of this tile.
coeff_table[tile]: np.ndarray         # shape (L, 11), dtype=np.uint32
                                       # 10 fp32 lanes + 1 unused pad
                                       # lane layout per row:
                                       #   0..5: A, B, C, D, E, F (basis-form coeffs)
                                       #   6:    opacity
                                       #   7..9: color_r, color_g, color_b
                                       #   10:   pad (round to 44B -> 48B page)

# Per-tile microblock header: 32 (offset, count) entries.
mb_header[tile]: np.ndarray            # shape (32, 2), dtype=np.uint32
                                       # mb_header[m][0] = offset into mb_stream
                                       # mb_header[m][1] = count

# Per-tile microblock stream: flat list of local_gaussian_idx values, mb-major.
mb_stream[tile]: np.ndarray            # shape (L',), dtype=np.uint16 or uint32
                                       # L' = sum_m mb_header[m][1] (active pairs)
                                       # mb_stream[mb_header[m][0] : mb_header[m][0] + mb_header[m][1]]
                                       #   = local_gaussian_idx values in DEPTH ORDER
                                       #     for microblock m
```

Total host-side memory cost per tile (typical L=200, K_avg=8):
- coeff_table: 200 × 48 B = 9.6 KB
- mb_header: 32 × 8 B = 256 B
- mb_stream: 1600 × 4 B = 6.4 KB
- Sum: ~16 KB / tile. At 1080p × 2040 tiles: ~33 MB total. Fits DRAM with
  room to spare; comparable to today's per-gaussian packs.

### 3.2 Per-microblock cull (the binning math)

Same closest-pixel-in-rectangle Mahalanobis trick as the existing
per-pair `contrib_cull` (`rasterization.py` lines 268–290), but evaluated
once per (g, tile, microblock) triple. Vectorized over all 32
microblocks at once via a `(P, 32)` torch op:

```python
# Per-microblock origin within a tile, table of 32 entries. Order matches
# §3.4 enumeration.
mb_origin_x = torch.tensor([...32 entries: 0, 8, 16, 24, 0, 8, ...])  # (32,)
mb_origin_y = torch.tensor([...32 entries: 0, 0,  0,  0, 4, 4, ...])  # (32,)

# Per-tile pixel origin, broadcast across the P (g, tile) pairs:
tx_tile = (tile_ids % tiles_x).float() * tile_size                    # (P,)
ty_tile = (tile_ids // tiles_x).float() * tile_size                   # (P,)

# Microblock origin in image coords for each (pair, microblock):
mb_ox = tx_tile[:, None] + mb_origin_x[None, :]                       # (P, 32)
mb_oy = ty_tile[:, None] + mb_origin_y[None, :]                       # (P, 32)

# Closest point inside the m-th microblock to gaussian g's mean:
cx = torch.clamp(means_2d[gaussian_ids, 0:1], min=mb_ox, max=mb_ox + 8)
cy = torch.clamp(means_2d[gaussian_ids, 1:2], min=mb_oy, max=mb_oy + 4)
dx = cx - means_2d[gaussian_ids, 0:1]
dy = cy - means_2d[gaussian_ids, 1:2]

# Closest-point Mahalanobis distance for each (pair, microblock):
m2 = (c[:, None] * dx*dx - 2.0*b[:, None] * dx*dy + a[:, None] * dy*dy) / det[:, None]
keep_mb = opacities[gaussian_ids, None] * torch.exp(-0.5 * m2) >= contrib_floor   # (P, 32)
```

Then build the per-microblock streams. The trick: `keep_mb[i, m] == True`
means pair `i` (a (g, tile) pair) contributes to microblock `m` of its
tile. For each tile we want, per-microblock, the depth-sorted list of
`local_gaussian_idx` (i.e., the position of this gaussian in the tile's
gaussian list).

```python
# After sort_and_bin, gaussian_ids and tile_ids are sorted so that all
# pairs of one tile are contiguous and depth-sorted within. Per tile:
for t in range(num_tiles):
    s, e = tile_offsets[t], tile_offsets[t + 1]
    pairs_in_tile = slice(s, e)
    L = e - s
    keep_tile = keep_mb[pairs_in_tile]               # (L, 32) bool
    mb_stream_t = []
    mb_header_t = np.zeros((32, 2), dtype=np.uint32)
    for m in range(32):
        local_idx = np.where(keep_tile[:, m])[0]     # depth-sorted by construction
        mb_header_t[m, 0] = len(mb_stream_t)
        mb_header_t[m, 1] = len(local_idx)
        mb_stream_t.extend(local_idx.tolist())
    # write coeff_table_t, mb_header_t, mb_stream_t to DRAM-bound buffers
```

This Python loop over `num_tiles` is a real cost (~2 ms at 1080p). Easy
torch-vectorize: `keep_mb` is `(P, 32)`; flatten to `(P*32, 2)` with a
column for `m`; sort by `(tile_id_per_pair, m, original_pair_position)`;
emit. We'll port to vectorized form once Stage 1 host correctness lands.

### 3.3 Drop-pair rule

A pair with `keep_mb[i, :].sum() == 0` (passed AABB+per-pair cull but
fails every microblock) is dropped from the bin entirely. Sanity-check
on the host: this should be rare (<1% of pairs); fail loudly if >5%
because it means our per-pair `contrib_cull` is wrong.

### 3.4 Microblock enumeration (host ↔ device agreement)

We commit to a single canonical mapping `m ↔ (face, band, colhi)` shared
between host and device. Define it in `alpha_blend_host.h` so neither
side guesses. Recommended raster order (rows 0..3 first, left-to-right;
then rows 4..7; etc):

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
// Microblock 0  = rows  0..3,  cols  0..7   (face 0, band 0, colhi 0)
// Microblock 1  = rows  0..3,  cols  8..15  (face 0, band 0, colhi 1)
// Microblock 2  = rows  0..3,  cols 16..23  (face 1, band 0, colhi 0)
// Microblock 3  = rows  0..3,  cols 24..31  (face 1, band 0, colhi 1)
// Microblock 4  = rows  4..7,  cols  0..7   (face 0, band 1, colhi 0)
// ...
// Microblock 31 = rows 28..31, cols 24..31  (face 3, band 3, colhi 1)
constexpr uint32_t mb_to_dst_addr(uint32_t m, uint32_t dst_tile_idx) {
    uint32_t row_band   = m >> 2;          // 0..7
    uint32_t col_group  = m & 0x3;         // 0..3
    uint32_t face_row   = row_band >> 2;   // 0 or 1
    uint32_t band       = row_band & 0x3;  // 0..3
    uint32_t face_col   = col_group >> 1;  // 0 or 1
    uint32_t colhi      = col_group & 0x1; // 0 or 1
    uint32_t face       = (face_row << 1) | face_col;
    return dst_tile_idx * 64 + face * 16 + band * 4 + colhi * 2;
}
```

The kernel uses a precomputed 32-entry table indexed by `m` directly to
avoid the decode in the inner loop (see §6.3).

---

## 4. Payload + reader changes

The existing per-gaussian 64B scalar pack and `CB_SCALARS` stream go
away. Reads are reorganized around per-tile blobs instead of per-gaussian
pages.

### 4.1 New CB layout

```cpp
// alpha_blend_host.h additions:

// COEFFICIENT TABLE: per-tile, fp32. One row per gaussian of this tile.
// Row size 48 B (= 10 fp32 lanes + 2 fp32 pad to NoC-align), L rows.
constexpr uint32_t COEFF_LANES_PER_GAUSSIAN = 10;  // A..F, opacity, color_rgb
constexpr uint32_t COEFF_ROW_BYTES          = 48;  // NoC-aligned
constexpr uint32_t COEFF_LANE_A             = 0;
constexpr uint32_t COEFF_LANE_B             = 1;
constexpr uint32_t COEFF_LANE_C             = 2;
constexpr uint32_t COEFF_LANE_D             = 3;
constexpr uint32_t COEFF_LANE_E             = 4;
constexpr uint32_t COEFF_LANE_F             = 5;
constexpr uint32_t COEFF_LANE_OPACITY       = 6;
constexpr uint32_t COEFF_LANE_COLOR_R       = 7;
constexpr uint32_t COEFF_LANE_COLOR_G       = 8;
constexpr uint32_t COEFF_LANE_COLOR_B       = 9;

// MICROBLOCK HEADER: per-tile, uint32. Always exactly 256 B (32 entries x 8 B).
struct MicroblockHeader {
    uint32_t offset;   // into mb_stream
    uint32_t count;    // gaussians in this microblock
};
static_assert(sizeof(MicroblockHeader) == 8, "8B per entry assumed");
constexpr uint32_t MB_HEADER_BYTES = 32 * 8;       // 256 B

// MICROBLOCK STREAM: per-tile, uint32. L' = sum(mb_header[m].count) entries,
// each a local-gaussian-index into the per-tile coeff_table.
//
// Sized to a worst-case L' bound per tile (e.g. L_max * 32 if every
// gaussian touches every microblock); host writes only the prefix that's
// actually used.

// New CB indices:
constexpr uint32_t CB_PX            = 0;   // unchanged
constexpr uint32_t CB_PY            = 1;   // unchanged
constexpr uint32_t CB_COEFF_TABLE   = 2;   // *replaces* CB_SCALARS
constexpr uint32_t CB_MB_HEADER     = 3;   // *replaces* CB_TILE_META
constexpr uint32_t CB_MB_STREAM     = 4;   // NEW

constexpr uint32_t CB_COLOR_OUT     = 16;  // unchanged
// State and constant CBs (CB_CONST_ZERO, CB_CONST_099) keep their indices
// from the existing host header; the per-Gaussian intermediate CBs
// (CB_DX/DY/DX2/DY2/DXDY/Q/POWER/ALPHA/CONTRIB/T_TMP/...) are DELETED
// because the inner loop no longer round-trips through them.
```

### 4.2 CB sizing

| CB | Page size | Pages (depth) | Total L1 |
|---|---|---|---|
| CB_PX | 2048 B (bf16 32×32) | 2 (double-buffered) | 4 KB |
| CB_PY | 2048 B | 2 | 4 KB |
| CB_COEFF_TABLE | 48 B (per gaussian row) | L_max (~512) | 24 KB |
| CB_MB_HEADER | 256 B (fixed) | 2 | 512 B |
| CB_MB_STREAM | 4 B per index | L_max·K_max (~16384) | 64 KB |
| CB_COLOR_OUT | 2048 B | 3 (R/G/B push) | 6 KB |
| State CBs (R/G/B/T persistent intra-tile) | — none, in DST | — | — |

Total per-core L1 cost: ~100 KB, well within the ~1.5 MB L1 budget.

### 4.3 Reader kernel

`kernels/dataflow/reader_alpha_blend.cpp` is rewritten:

```cpp
// Pseudocode for the per-tile work; same outer loop over tile_ids[]
// as the existing reader.
for tile_id in this core's tile slice:
    // 1. Streams as today.
    noc_async_read_tile(tile_id, px_acc, get_write_ptr(CB_PX));
    noc_async_read_tile(tile_id, py_acc, get_write_ptr(CB_PY));

    // 2. Per-tile blob fetches. Each tile's coefficient table and
    //    microblock metadata live at known DRAM offsets indexed by
    //    tile_id (TensorAccessor with row-stride = COEFF_ROW_BYTES * L_max
    //    for coeff_table; fixed 256 B for mb_header; row-stride =
    //    MB_STREAM_MAX_BYTES for mb_stream).
    noc_async_read(coeff_table_dram_for(tile_id),
                   get_write_ptr(CB_COEFF_TABLE), L * 48);
    noc_async_read(mb_header_dram_for(tile_id),
                   get_write_ptr(CB_MB_HEADER), 256);
    noc_async_read(mb_stream_dram_for(tile_id),
                   get_write_ptr(CB_MB_STREAM), L_prime * 4);
    noc_async_read_barrier();

    cb_push_back(CB_PX, 1);
    cb_push_back(CB_PY, 1);
    cb_push_back(CB_COEFF_TABLE, L);       // depth-multi push
    cb_push_back(CB_MB_HEADER, 1);
    cb_push_back(CB_MB_STREAM, L_prime);   // depth-multi push
```

Reader is dramatically simpler than today — one DMA per blob, no
per-gaussian `for g in g_count { noc_async_read_tile(packs_acc, ...) }`
loop. Total NoC transactions per tile drops from `~L + 4` to `5`.

DRAM bandwidth is the same: `L × 48 B` (coeff_table) vs today's
`L × 64 B` (packs); plus the small mb_header (256 B) and mb_stream
(~L × K × 4 B = roughly `2 × L × K` B). At L=200, K=8 this is
`9.6 KB + 256 B + 6.4 KB = 16.3 KB/tile` vs today's `12.8 KB/tile`.
~25% more DRAM bandwidth in exchange for the compute reduction we model
in §8. Worth it.

---

## 5. Tile-level basis prep (DST-resident X, Y; per-microblock X²/XY/Y²)

The previous draft maintained 6 basis tiles (X, Y, X², XY, Y², ONE) in
CBs. With mb-major we hold only **2 basis tiles in DST** for the whole
tile (X and Y), and compute the 3 product values (X², XY, Y²) **per
microblock into LREGs** (3 SFPMUL each, reused across the microblock's
gaussian inner loop).

Why:
- Frees 3 DST tile slots (we recover budget to fit 4 state + 2 basis + 2
  transient = 8 tiles total — see §6.5).
- ONE basis tile is unnecessary: the `F` term in `Q = A*x² + B*xy + C*y²
  + D*x + E*y + F` is added as an SFPADD with the LREG-resident `F`
  coefficient, no tile multiply needed.
- Per-microblock product compute is 3 SFPMULs amortized over the
  microblock's gaussian list (G_mb gaussians × 5 product accesses
  saved). Net win for G_mb ≥ 1.

### 5.1 Build sequence

At the top of the per-tile compute loop, **before** the 32-microblock
outer loop:

```cpp
tile_regs_acquire();

// Slot 4 = X basis tile = (px - tile_origin_x)
copy_tile_to_dst_init_short(CB_PX);
copy_tile(CB_PX, 0, 4);
sub_unary_tile(4, tile_origin_x_bits);

// Slot 5 = Y basis tile = (py - tile_origin_y)
copy_tile_to_dst_init_short(CB_PY);
copy_tile(CB_PY, 0, 5);
sub_unary_tile(5, tile_origin_y_bits);

// Slots 0..3 = state tiles, initialized R = G = B = 0, T = 1.
fill_tile_init();
fill_tile(0, 0.0f);    // R
fill_tile(1, 0.0f);    // G
fill_tile(2, 0.0f);    // B
fill_tile(3, 1.0f);    // T

// DST is now:
//   0..3 : R, G, B, T  (persistent across the 32-microblock outer loop)
//   4..5 : X, Y        (persistent)
//   6..7 : transient scratch for per-microblock work

// We do NOT release DST here. The outer microblock loop runs inside this
// same acquire so state and basis remain DST-resident.
```

### 5.2 Per-microblock basis-product LREG init (inside the outer loop)

```cpp
// For each microblock m at the top of its inner gaussian loop:
constexpr uint32_t addr_X  = 4 * 64 + MB_TO_DST_ADDR[m];   // X basis MB addr
constexpr uint32_t addr_Y  = 5 * 64 + MB_TO_DST_ADDR[m];   // Y basis MB addr

// Load X_mb, Y_mb into persistent LREGs (kept across this MB's gaussian loop).
TT_SFPLOAD(LREG_X,  InstrModLoadStore::DEFAULT, ADDR_MOD_7, addr_X);
TT_SFPLOAD(LREG_Y,  InstrModLoadStore::DEFAULT, ADDR_MOD_7, addr_Y);

// Compute X², XY, Y² products into 3 more persistent LREGs.
TTI_SFPMUL(LREG_X, LREG_X, p_sfpu::LCONST_0, LREG_X2, 0);
TTI_SFPMUL(LREG_X, LREG_Y, p_sfpu::LCONST_0, LREG_XY, 0);
TTI_SFPMUL(LREG_Y, LREG_Y, p_sfpu::LCONST_0, LREG_Y2, 0);

// Now LREG_X, LREG_Y, LREG_X2, LREG_XY, LREG_Y2 are all live for the
// inner gaussian loop. Cost = 5 SFPU ops per microblock, ~5 cycles.
```

### 5.3 Basis-form Q

The basis form is unchanged from the previous draft:

```
Q = A·x² + B·xy + C·y² + D·x + E·y + F
power = Q       (host already absorbs the -0.5 factor into A..F)
```

Per gaussian per microblock, Q evaluation is 5 FMAs against the 5 basis
LREGs plus one SFPADD with `F`:

```cpp
// Q = A*x² + B*xy + C*y² + D*x + E*y + F
TTI_SFPMUL(LREG_X2, LREG_A, p_sfpu::LCONST_0, LREG_Q, 0);  // Q  = A*x²
TTI_SFPMAD(LREG_XY, LREG_B, LREG_Q,           LREG_Q, 0);  // Q += B*xy
TTI_SFPMAD(LREG_Y2, LREG_C, LREG_Q,           LREG_Q, 0);  // Q += C*y²
TTI_SFPMAD(LREG_X,  LREG_D, LREG_Q,           LREG_Q, 0);  // Q += D*x
TTI_SFPMAD(LREG_Y,  LREG_E, LREG_Q,           LREG_Q, 0);  // Q += E*y
TTI_SFPADD(LREG_F,  p_sfpu::LCONST_1, LREG_Q, LREG_Q, 0);  // Q += F
```

6 SFPU ops, 32 lanes each. With `fp32_dest_acc_en = true` and a
host-precomputed A..F, this evaluates the entire Q form for this
microblock's 32 pixels in ~12 cycles (6 ops × 2-cycle latency, fully
pipelined back-to-back because each writes/reads LREG_Q which we
serialize on; the dependency chain limits parallelism here but we can
hide the latency under the next gaussian's coefficient SFPLOADIs).

---

## 6. Per-microblock compute kernel (mb-major)

This is the heart of the design. One DST acquire per tile spans the
whole 32-microblock outer loop. Within each microblock the inner gaussian
loop is pure LREG math — state and basis live in LREGs across the inner
loop, DST is touched only at microblock boundaries.

### 6.1 DST slot map (8 total on BH `fp32_dest_acc_en = true`)

```
slot 0:  R_state    (persistent across all 32 microblocks of this tile)
slot 1:  G_state    (persistent)
slot 2:  B_state    (persistent)
slot 3:  T_state    (persistent; init 1.0)
slot 4:  X basis    (persistent; px - tile_origin_x)
slot 5:  Y basis    (persistent; py - tile_origin_y)
slot 6:  transient (free for pack_tile staging at end of tile)
slot 7:  transient
```

All 8 slots used; 6 persistent, 2 transient. No room left — adding any
new persistent tile requires evicting one of the above. Sat_mask is
dropped (handled per-microblock by the saturation early-out in §6.7).

### 6.2 LREG allocation contract (16 total on BH SFPU)

```
PERSISTENT (one microblock's lifetime, ie. across that microblock's gaussian loop):
  LREG0  = R           per-microblock R-channel state
  LREG1  = G           per-microblock G-channel state
  LREG2  = B           per-microblock B-channel state
  LREG3  = T           per-microblock transmittance state
  LREG4  = X           per-microblock x basis value (x_local for this MB)
  LREG5  = Y           per-microblock y basis value
  LREG6  = X²          (= X*X, computed once per microblock)
  LREG7  = XY          (= X*Y)
  LREG8  = Y²          (= Y*Y)

TRANSIENT (overwritten per gaussian inner iteration):
  LREG9  = Q           Mahalanobis quadratic for this (g, m) pair
  LREG10 = coeff/tmp1  bf16-immediate destination + scratch
  LREG11 = coeff/tmp2  (overrides hw default -1.0; restored at tile end)
  LREG12 = exp_scratch inline exp polynomial scratch (1 of 2)
  LREG13 = exp_scratch                              (2 of 2)
  LREG14 = alpha       per-(g, m) alpha after exp + cap
  LREG15 = contrib     per-(g, m) contrib = alpha * T

Total: 9 persistent + 7 transient = 16, fits exactly. No overflow margin —
if any extension needs an additional persistent LREG, evict X²/XY/Y² and
recompute per gaussian inside Q evaluation (cost +3 SFPMUL per gaussian).
```

The trick: A, B, C, D, E, F, opacity, color_r/g/b (10 per-gaussian
coefficients) are loaded into the transient LREGs **per gaussian**, used
immediately, and overwritten. We never need more than ~3 coefficient
LREGs live at a time because the Q evaluation chain reads one coefficient
per FMA and we can interleave loads with FMAs.

### 6.3 Microblock → DST addr table

```cpp
// Verified by hand against §3.4 enumeration: every entry checked.
constexpr uint32_t MB_TO_DST_ADDR[32] = {
    /* m =  0 */  0, /* m =  1 */  2, /* m =  2 */ 16, /* m =  3 */ 18,
    /* m =  4 */  4, /* m =  5 */  6, /* m =  6 */ 20, /* m =  7 */ 22,
    /* m =  8 */  8, /* m =  9 */ 10, /* m = 10 */ 24, /* m = 11 */ 26,
    /* m = 12 */ 12, /* m = 13 */ 14, /* m = 14 */ 28, /* m = 15 */ 30,
    /* m = 16 */ 32, /* m = 17 */ 34, /* m = 18 */ 48, /* m = 19 */ 50,
    /* m = 20 */ 36, /* m = 21 */ 38, /* m = 22 */ 52, /* m = 23 */ 54,
    /* m = 24 */ 40, /* m = 25 */ 42, /* m = 26 */ 56, /* m = 27 */ 58,
    /* m = 28 */ 44, /* m = 29 */ 46, /* m = 30 */ 60, /* m = 31 */ 62,
};

constexpr uint32_t TILE_BASE(uint32_t slot) { return slot * 64; }
```

### 6.4 The outer microblock loop (one acquire spans the whole tile)

```cpp
// Once per program: record the inner replay slot. See §6.6.
ckernel::sfpu::init_inline_exp_approx();   // wraps _init_exponential_<true>
lltt::record(REPLAY_SLOT_INNER, REPLAY_LEN_INNER);   /* body in §6.6 */

// Per tile:
tile_regs_acquire();

// §5.1 basis/state init: DST slots 0..5 populated.
build_basis_and_state_in_dst();

// Read per-tile microblock header (32 entries) into local L1 stack.
cb_wait_front(CB_MB_HEADER, 1);
volatile MicroblockHeader* mb_hdr = reinterpret_cast<volatile MicroblockHeader*>(
    get_read_ptr(CB_MB_HEADER));
MicroblockHeader hdr[32];
for (int i = 0; i < 32; ++i) { hdr[i].offset = mb_hdr[i].offset; hdr[i].count = mb_hdr[i].count; }
cb_pop_front(CB_MB_HEADER, 1);

// Per-tile L1 pointers for stream + coeff table.
cb_wait_front(CB_MB_STREAM, L_prime);          // host knows L_prime; passed as arg
cb_wait_front(CB_COEFF_TABLE, L);              // L gaussians of this tile
auto mb_stream    = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_MB_STREAM));
auto coeff_table  = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_COEFF_TABLE));
constexpr uint32_t ROW_U32 = COEFF_ROW_BYTES / 4;   // 12 (= 10 lanes + 2 pad)

// Outer microblock loop.
for (uint32_t m = 0; m < 32; ++m) {
    const uint32_t count = hdr[m].count;
    if (count == 0) continue;

    const uint32_t mb_addr  = MB_TO_DST_ADDR[m];
    const uint32_t a_X      = TILE_BASE(4) + mb_addr;   // X basis MB addr
    const uint32_t a_Y      = TILE_BASE(5) + mb_addr;   // Y basis MB addr
    const uint32_t a_R      = TILE_BASE(0) + mb_addr;
    const uint32_t a_G      = TILE_BASE(1) + mb_addr;
    const uint32_t a_B      = TILE_BASE(2) + mb_addr;
    const uint32_t a_T      = TILE_BASE(3) + mb_addr;

    // Load X, Y for this microblock into persistent LREGs.
    TT_SFPLOAD(p_sfpu::LREG4, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_X);
    TT_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_Y);

    // Compute X², XY, Y² products into persistent LREGs.
    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);  // X²
    TTI_SFPMUL(p_sfpu::LREG4, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);  // XY
    TTI_SFPMUL(p_sfpu::LREG5, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG8, 0);  // Y²

    // Load this microblock's R, G, B, T from DST -> persistent LREGs.
    TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_R);
    TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_G);
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_B);
    TT_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_T);

    // Inner gaussian loop for this microblock.
    const uint32_t base = hdr[m].offset;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t gidx = mb_stream[base + i];
        const volatile uint32_t* row = coeff_table + gidx * ROW_U32;

        // Load A..F into transient LREGs via bf16-immediate loads:
        //   TTI_SFPLOADI(L, FP16_B_MOD0, fp32_bits >> 16);
        // Saves vs SFPLOADI UPPER+LOWER pair (1 instr vs 2 per coeff).
        // See _build_lane_mask_col0_ for the SFPLOADI pattern and
        // ckernel_sfpu_binary_bcast.h lines 192-200 for the encoding.
        //
        // The Q chain reads each coeff exactly once into LREG10/11 (alternating)
        // and immediately consumes it; coefficient LREGs are recycled
        // across the 6 ops of Q evaluation.
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_A]);
        TTI_SFPMUL(p_sfpu::LREG6, p_sfpu::LREG10, p_sfpu::LCONST_0, p_sfpu::LREG9, 0);  // Q  = A*X²
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_B]);
        TTI_SFPMAD(p_sfpu::LREG7, p_sfpu::LREG10, p_sfpu::LREG9, p_sfpu::LREG9, 0);     // Q += B*XY
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_C]);
        TTI_SFPMAD(p_sfpu::LREG8, p_sfpu::LREG10, p_sfpu::LREG9, p_sfpu::LREG9, 0);     // Q += C*Y²
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_D]);
        TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG10, p_sfpu::LREG9, p_sfpu::LREG9, 0);     // Q += D*X
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_E]);
        TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG10, p_sfpu::LREG9, p_sfpu::LREG9, 0);     // Q += E*Y
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_F]);
        TTI_SFPADD(p_sfpu::LREG10, p_sfpu::LCONST_1, p_sfpu::LREG9, p_sfpu::LREG9, 0);  // Q += F

        // Q in LREG9 is already -0.5 * (true Q) because host folded -0.5
        // into A..F. So "power" = LREG9; no extra SFPMUL needed.

        // INLINE EXP POLYNOMIAL (~10 ops; LREG12/13 are exp scratch).
        // Adapted from ckernel_sfpu_exp.h _calculate_exponential_<true> body,
        // operating on explicit LREG9 instead of dst_reg[0].
        inline_exp_approx(p_sfpu::LREG9, p_sfpu::LREG12, p_sfpu::LREG13);
        // LREG9 now holds exp(power) = weight.

        // alpha = min(opacity * weight, 0.99). Cap via SFPMIN
        // (TTI_SFPSWAP_MOD1_VEC_MIN_MAX form, see ckernel_sfpu_recip.h
        // line ~199 for the encoding).
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_OPACITY]);
        TTI_SFPMUL(p_sfpu::LREG9, p_sfpu::LREG10, p_sfpu::LCONST_0, p_sfpu::LREG14, 0);   // = opacity*weight
        load_bf16_imm(p_sfpu::LREG10, 0x3F7AE148u);                                       // 0.99f as bf16-upper
        sfpu_min_inplace(p_sfpu::LREG14, p_sfpu::LREG10);   // LREG14 = alpha = min(.,0.99)

        // contrib = alpha * T   (LREG3 = T persistent)
        TTI_SFPMUL(p_sfpu::LREG14, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG15, 0);

        // R += contrib * color_r
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_COLOR_R]);
        TTI_SFPMAD(p_sfpu::LREG15, p_sfpu::LREG10, p_sfpu::LREG0, p_sfpu::LREG0, 0);
        // G += contrib * color_g
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_COLOR_G]);
        TTI_SFPMAD(p_sfpu::LREG15, p_sfpu::LREG10, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        // B += contrib * color_b
        load_bf16_imm(p_sfpu::LREG10, row[COEFF_LANE_COLOR_B]);
        TTI_SFPMAD(p_sfpu::LREG15, p_sfpu::LREG10, p_sfpu::LREG2, p_sfpu::LREG2, 0);

        // T = T - contrib  (= T * (1 - alpha) because contrib = T*alpha)
        TTI_SFPMAD(p_sfpu::LREG15, p_sfpu::LCONST_neg1, p_sfpu::LREG3, p_sfpu::LREG3, 0);

        // Periodic T-saturation check (every K_SAT gaussians, eg. 16):
        // if all 32 lanes of T < 1e-4, break out of this microblock's loop.
        // See §6.7 for the reduction pattern.
    }

    // SFPSTORE state LREGs back to DST persistent slots for this MB.
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_R);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_G);
    TT_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_B);
    TT_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::DEFAULT, ADDR_MOD_7, a_T);
}

// All 32 microblocks done. Persistent DST slots 0..2 hold final R/G/B.
tile_regs_commit();
tile_regs_wait();

cb_reserve_back(CB_COLOR_OUT, 3);
pack_tile(0, CB_COLOR_OUT);   // R
pack_tile(1, CB_COLOR_OUT);   // G
pack_tile(2, CB_COLOR_OUT);   // B
cb_push_back(CB_COLOR_OUT, 3);

tile_regs_release();
cb_pop_front(CB_COEFF_TABLE, L);
cb_pop_front(CB_MB_STREAM,   L_prime);
cb_pop_front(CB_PX, 1);
cb_pop_front(CB_PY, 1);
```

### 6.5 The replay-buffer optimization

The inner per-gaussian body (everything between "Load A..F" and "T = T - contrib")
is ~30 SFPU ops, identical across (m, g) pairs except for the
coefficient values loaded via `load_bf16_imm`. Because `load_bf16_imm`
emits `TTI_SFPLOADI` with a per-call immediate, the **immediate is baked
at record time** — replay cannot parameterize it.

Two approaches:

- **(a) Skip replay** for the coefficient load + multiply pair (keep these
  inline), record only the cross-coefficient chains that are address-
  and coefficient-independent. Net win: ~10 of 30 ops compressed.
  Modest improvement; simple to implement.

- **(b) Use `lltt::record<lltt::Exec>` once at the top of the FIRST
  gaussian's processing in a fresh tile**, capturing the coefficient
  loads with the *first gaussian's values baked in*. Replay then replays
  the first gaussian's coefficients onto subsequent gaussians — which is
  **wrong**. (b) doesn't work; replay isn't a function-call abstraction.

Recommendation: option (a). Implementation deferred to Stage 3 of §9
(measure the dispatch overhead first; if it's <5% of inner cost, replay
buys nothing and we don't bother).

### 6.6 Inline exp polynomial helper

We need `exp(x)` on a single 32-lane LREG. The full-tile `exp_tile<true>`
in `api/compute/eltwise_unary/exp.h` calls down to
`_calculate_exponential_<APPROXIMATION_MODE=true, SCALE_EN=false,
ITERATIONS=8>`, which iterates the polynomial body 8 times for a
full tile and advances `dst_reg` between iterations.

We extract the per-iteration body and bind it to an explicit LREG. The
body (per
`tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h`)
is the standard `2^x` polynomial: clamp the input, split into integer
and fractional parts, polynomial-approximate `2^frac`, scale by `2^int`
via an exponent-field add. The approximate path is ~10 SFPU ops on a
32-lane vector.

```cpp
// Skeleton — exact instruction sequence transliterated from
// _calculate_exponential_<APPROXIMATION_MODE=true>'s inner body in
// ckernel_sfpu_exp.h. Inputs: x in LREG_X; scratch in LREG_S0, LREG_S1.
// Output: x replaced with exp(x).
inline void inline_exp_approx(uint32_t LREG_X, uint32_t LREG_S0, uint32_t LREG_S1) {
    // 1. Clamp x to >= -88 (avoid underflow + match LLK ClampToNegative semantics).
    //    TTI_SFPSWAP(0, LCONST_clamp, LREG_X, MOD1_VEC_MAX) — see recip kernel L201.
    // 2. x = x * log2(e)        (TTI_SFPMUL against a precomputed bf16-imm in LREG_S0)
    // 3. n = floor(x); f = x - n
    //    SFPIADD / SFPSHFT2 sequence per the LLK reference.
    // 4. poly(f) = 1 + f*(c1 + f*(c2 + f*c3))    (3 SFPMADs, coeffs as bf16 imms)
    // 5. result = poly(f) * 2^n  via SFPSHFT2 on the exponent field.
    // ~10 ops total.
}
```

The exact op sequence is left as a transliteration exercise during
implementation — read `ckernel_sfpu_exp.h` and convert each `dst_reg[0]`
reference to the named LREG. This is mechanical, but we should NOT make
up the polynomial coefficients here; pull them from the LLK source so
they match `exp_tile<true>` bit-for-bit (which preserves the PSNR delta
profile we already validated in iter-040).

### 6.7 Saturation early-out

Per-microblock T eventually drops below 1e-4 for foreground microblocks
covered by many opaque gaussians. Once all 32 lanes of LREG3 (T) are
below that threshold, remaining gaussians in this microblock's list
contribute <1/255 to any pixel — we can break.

```cpp
// Every K_SAT gaussians (eg. 16), check all-lanes saturation.
if ((i & (K_SAT - 1)) == (K_SAT - 1)) {
    // Build "any lane > 1e-4" via SFPCMP and SFPCCC (set-condition,
    // copy-cc-out). See ckernel_sfpu_relu.h for v_if pattern.
    bool any_active = sfpu_any_gt(p_sfpu::LREG3, T_THRESH_BITS);
    if (!any_active) break;
}
```

`sfpu_any_gt` is the reduction primitive — there isn't a direct one-op
"any lane true" in the SFPU. The cheapest approach is a 5-level
sub-vector OR reduction (4 ROR1 stages + 1 SFPSETCC, see
`_record_broadcast_replay_` in `ckernel_sfpu_binary_bcast.h` for the
shuffle-ROR pattern adapted from broadcast to reduce). ~10 ops, paid
every K_SAT=16 gaussians. Negligible.

Per-microblock saturated-skip is the per-tile early-out generalized.

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

### 8.1 Per (microblock, gaussian) inner cost (BH SFPU @ ~1 GHz)

Body inside the `for i in count` loop, per (m, g) iteration:

```
6 bf16 coeff loads (LREG10 reused)   : 6 SFPLOADI                = ~6 cycles
Q chain                              : 1 SFPMUL + 4 SFPMAD + 1 SFPADD = ~6 cycles
exp polynomial (approx)              : ~10 SFPU ops              = ~10 cycles
opacity load + mul                   : 1 SFPLOADI + 1 SFPMUL     = ~2 cycles
alpha clamp (min vs 0.99)            : 1 SFPLOADI + 2 SFPU ops   = ~3 cycles
contrib = alpha * T                  : 1 SFPMUL                  = ~1 cycle
R/G/B FMAs (3 colors, 3 loads)       : 3 SFPLOADI + 3 SFPMAD     = ~6 cycles
T -= contrib                          : 1 SFPMAD                  = ~1 cycle
                                                            total = ~35 cycles per (m, g) pair
```

Pipeline note: SFPU ops have 2-cycle latency on dependent reads. The Q
chain is a strict serial dependency through LREG9 — the 5 SFPMADs each
read the previous one's output, so they cost ~10 cycles in pipeline
terms, not 5. The exp polynomial and the R/G/B/T tail are similarly
serialized on their respective accumulators. Realistic estimate: **~50
cycles per (m, g) pair** after dependency stalls.

### 8.2 Per-microblock overhead (paid once per non-empty microblock)

```
basis SFPLOAD (X, Y)                 : 2 cycles
basis-product SFPMUL (X², XY, Y²)    : 3 cycles
state load (R, G, B, T)              : 4 cycles
state store (R, G, B, T)             : 4 cycles
loop overhead                        : ~5 cycles
                                  total = ~18 cycles
```

### 8.3 Per-tile overhead (paid once per tile)

```
basis tile build (§5.1)              : ~50 cycles  (sub_unary_tile × 2)
state tile init                      : ~40 cycles  (fill_tile × 4)
mb_header copy to L1 stack           : ~30 cycles
final pack (R/G/B → CB_COLOR_OUT)    : ~50 cycles
acquire/commit/release               : ~50 cycles
                                  total = ~220 cycles
```

### 8.4 Total per tile

```
cycles_per_tile ≈ 220                            (per-tile)
                + Σ_m≠empty  18                  (per non-empty MB)
                + Σ_(m,g)    50                  (per (m, g) pair)

≈ 220 + N_mb_active * 18 + N_pairs * 50

where:
  N_mb_active = number of non-empty microblocks (≤ 32)
  N_pairs     = Σ_m count[m] = total (mb, g) pairs in this tile
              = L * K_avg   (L gaussians, K_avg average MBs each touches)
```

For typical (L = 200, K_avg = 8, all 32 microblocks non-empty):
```
≈ 220 + 32 * 18 + 1600 * 50
= 220 + 576 + 80000
≈ 80800 cycles/tile
```

### 8.5 Baseline comparison

Existing kernel (measured): ~25.4 ms / 1080p frame / 2040 tiles
≈ 12.5 µs/tile = ~12500 cycles/tile @ 1 GHz.

Hmm — that's lower than our model predicts (80800). Two reconciling
notes:

1. The existing kernel does NOT do per-gaussian work for all (g, tile)
   pairs equally — many pairs hit the iter-013 "T saturated"
   early-skip, the iter-050 power-clamp elision, etc. The effective
   pair count per tile is well below 200 × 1.0 = 200.
2. Our model assumes worst-case sequential pipeline stalls; in reality
   instruction-level parallelism between coefficient loads and FMAs
   recovers a factor of 1.5-2×.

We expect the mb-major design to **measure faster than today** in the
sparse regime (K_avg ≤ 10), but the absolute speedup factor depends on
the workload's K distribution, which we will profile in Stage 1 (§9).

### 8.6 Conservative end-to-end target

Plausible range based on (1) host pair count after per-microblock cull
(should drop ~30% — only pairs whose Mahalanobis still passes the
tighter per-microblock test survive) and (2) per-pair cost reduction
from SFPU LREG residency (~2× vs today's CB round-trips):

```
new_kernel_ms ≈ today * (0.7 pair_cull) * (0.5 per_pair_speedup)
              ≈ 25.4 * 0.35
              ≈ 9 ms / frame
```

Stretch target if §10's follow-ups land: ~5 ms / frame.

---

## 9. Validation plan

Four stages, each gated. Strictly sequential — no stage starts until the
previous gate has passed on stitch + at least 2 other scenes.

### Stage 1: host binning correctness, no kernel changes

1. Implement §3 in `gsplat/rasterization.py`. Emit `(coeff_table[tile],
   mb_header[tile], mb_stream[tile])` for every tile.
2. Property tests, run on every frame of the validation set:
   - For each tile: `sum(mb_header[m].count for m in 0..31) == L'`
   - For each (m, gidx) appearing in mb_stream: that gidx, plus opacity
     & cov for the corresponding gaussian, gives `peak_in_mb ≥
     contrib_floor` at the closest pixel.
   - Per-microblock streams are depth-sorted (no inversions vs the
     tile's global depth-sorted gaussian order).
   - Drop rate (pairs that pass per-pair AABB cull but fail every
     microblock) < 5%.
3. Visual test: rasterize on the CPU reference using the per-microblock
   lists as a "render only these microblocks for this gaussian" mask.
   PSNR vs the unmasked CPU reference ≥ 100 dB (rounding-only deltas).
4. Gate: all property tests pass + PSNR ≥ 100 dB + `K` distribution
   reasonable (median ≈ 4-12, p99 ≤ 32).

### Stage 2: per-tile coefficient table + reader plumbing, no compute change

1. Implement §4 (new CBs, new reader). Leave compute kernel unchanged —
   it still consumes the OLD per-gaussian scalar packs from a parallel
   shadow CB to avoid behavioral change.
2. Validate: reader correctly DMAs coeff_table / mb_header / mb_stream
   into their CBs. End-to-end PSNR is unchanged (compute hasn't seen
   the new data yet).
3. Gate: PSNR bit-identical to current; reader cycles per tile within
   2× of today (we expect SIGNIFICANTLY faster — single-shot DMAs vs
   per-gaussian loop — but a 2× regression caps the worst case).

### Stage 3: §5 basis-form + §6 mb-major compute kernel

1. Implement §5.1 basis-build, §6 outer microblock loop, §6.4 inner body
   (without inline exp — use `exp_tile<true>` on a full tile and extract
   the microblock; intentionally suboptimal, just for correctness).
2. Run validation: PSNR per view, kernel ms.
3. Gate: PSNR ≥ 35.0 dB on every validation view; kernel ms ≤ current
   median × 1.2 (a 20% regression is tolerable because we haven't
   landed inline exp yet).

### Stage 4: §6.6 inline exp + §6.7 saturation early-out

1. Replace full-tile exp_tile with the inline polynomial.
2. Add the per-microblock T saturation check.
3. Run validation.
4. Gate: PSNR ≥ 35.0 dB on every view; kernel ms ≤ 50% of current
   median (the 2× speedup target).

### Bisect on failure

If PSNR drops below the gate, the bisect order is:
  a. **mb_stream ordering** (most likely): one (m, g) pair processed
     out-of-depth-order. Probe: capture mb_stream for one frame, verify
     monotonic depth indices per microblock.
  b. **DST/LREG state coherence** between microblocks: did SFPSTORE
     write to the wrong addr? Add a per-microblock checksum tile that
     captures the pre-store LREG values + the post-store DST values
     after every 100 microblocks. Compare.
  c. **Inline exp polynomial drift** vs `exp_tile<true>`: should be
     ≤2 ULP per call; if more, revert to full-tile exp for this stage
     and recheck.
  d. **bf16-immediate truncation** of A..F coefficients: if Q drifts,
     the coeffs lost too much precision via fp32→bf16 imm. Switch the
     6 coeffs to full fp32 SFPLOADI (UPPER+LOWER) at +1 cycle/coeff/
     gaussian cost.
  e. **LREG11 = -1.0 hardware default**: ckernel SFPU code assumes
     LREG11 holds -1.0 after restore; ensure we restore at tile end
     (`SFPLOADI 0xBF80; SFPLOADI 0x0000`).

---

## 10. NOT YET DESIGNED (future iterations)

Explicitly out-of-scope for the first mb-major implementation; follow-ups
once §1-9 ship:

- **lltt::replay for the address-independent slice of the inner body**
  (§6.5 option (a)). Estimated 5-15% inner-loop dispatch saving.
- **Tile-level bf16 basis** (revisit iter-057b): the bf16 precision
  trade-off is bounded by a single microblock's 32 lanes (much better
  numerics than the failed global-coords attempt). Cuts L1 basis
  footprint 2×.
- **Coefficient compression**: A..F + opacity + color in 11 fp32 = 44 B
  per gaussian per tile. Many of these have low dynamic range across
  gaussians of one tile (color especially). Quantize to int8 + per-tile
  scale → 11 B per gaussian; saves 75% DRAM bandwidth on the coeff
  table. Worth doing if Stage 2 measures coeff DMA as a bottleneck.
- **Multi-gaussian coefficient pipelining**: prefetch next gaussian's
  coefficients into shadow LREGs (LREG11) while the current gaussian's
  Q chain runs. Hides 6 SFPLOADI latency per gaussian. ~10% speedup.
- **Persistent kernel + mailbox dispatch**: orthogonal to microblocks;
  composes cleanly.
- **8×8 microblocks**: would need a 64-lane SFPLOAD; SFPU is 32-lane,
  so this is hardware-impossible. We are at the natural minimum
  granularity already.
- **Per-microblock saturation MASK propagation**: once a microblock
  saturates, mark it in a tile-local bitmask. Cheap, but unclear
  benefit beyond §6.7's per-microblock break — the break already
  exits that microblock's loop; the only further saving is skipping
  the start-of-microblock SFPLOAD basis/state setup, which we already
  guard on `count > 0`. Track if profiling shows residual overhead.

---

## 11. Concrete file diffs (preview, not yet applied)

```
gsplat/rasterization.py
  + per-microblock cull (§3.2): vectorized (P, 32) Mahalanobis test
  + per-tile output assembly: coeff_table[tile], mb_header[tile],
    mb_stream[tile] (§3.1)
  + return these alongside (gaussian_ids, tile_ids, tiles_per_gaussian)

gsplat/backend.py + backends/tt/backend.py
  + replace prepare_kernel_inputs's old packs/offsets/px/py with
    {coeff_tables, mb_headers, mb_streams, px, py, tile_ids}
  + host packs A..F coefficients (with -0.5 already folded in) into the
    coeff_table rows; opacity + color_rgb in lanes 6..9

backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/
  alpha_blend_host.h
    + MicroblockHeader struct (§4.1)
    + COEFF_* lane constants
    + CB_COEFF_TABLE, CB_MB_HEADER, CB_MB_STREAM indices
    + mb_to_dst_addr() constexpr (§3.4)
    + MB_TO_DST_ADDR[32] static table (§6.3)
    - removed: SCALAR_PACK_*, CB_SCALARS, CB_TILE_META, the per-gaussian
      intermediate CBs (CB_DX/DY/.../CB_T_TMP, etc.)

  alpha_blend.cpp
    + allocate CB_COEFF_TABLE / CB_MB_HEADER / CB_MB_STREAM with sizes
      per §4.2
    - removed: per-Gaussian intermediate CB allocations
    + reader/writer args: drop packs_addr/offsets_addr; add
      coeff_table_addr/mb_header_addr/mb_stream_addr per tile (or pass
      a single TensorAccessor that strides by tile_id)

  kernels/dataflow/reader_alpha_blend.cpp
    + rewrite: per-tile single-shot DMA for coeff_table, mb_header,
      mb_stream (§4.3). No per-gaussian inner loop.

  kernels/compute/alpha_blend_compute.cpp
    + REWRITE around the mb-major outer/inner loop (§6.4)
    + per-tile: build X, Y basis tiles (§5.1); init R/G/B/T state tiles
    + outer microblock loop: load X/Y, compute X²/XY/Y² into LREGs;
      load R/G/B/T into LREGs; iterate inner gaussian loop
    + inner body: bf16-imm coeffs, basis-form Q, inline exp, alpha,
      contrib, R/G/B FMA, T update -- all in LREGs (§6.4)
    + per-microblock store R/G/B/T LREGs back to DST
    + per-tile pack R/G/B to CB_COLOR_OUT

  kernels/dataflow/writer_alpha_blend.cpp
    (unchanged; still consumes CB_COLOR_OUT 3-tile pushes)
```

### 11.1 Estimated review-sized PRs

This is a large change. Suggested PR sequence (each is independently
reviewable and shippable as a no-functional-change refactor, except the
last):

1. **PR1**: §3 host binning — add mb_masks / coeff_table / mb_header /
   mb_stream emission alongside the existing per-gaussian outputs. No
   reader/compute changes; existing pipeline unchanged. Property tests
   only. ~200 LOC.
2. **PR2**: §4 reader rewrite using new CBs in PARALLEL with existing
   CBs. Compute still consumes old scalars. PSNR unchanged. ~300 LOC.
3. **PR3**: §5 + §6 compute rewrite. Uses the new CBs from PR2; deletes
   the old scalar consumption path. PSNR validation. ~600 LOC.
4. **PR4**: §6.6 inline exp + §6.7 saturation early-out. Performance
   measurement. ~200 LOC.

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
