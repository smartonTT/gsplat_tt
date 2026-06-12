# Iter 71 (ttw-071) — revert iter70 + blend LRA trim

**Status:** **keep** (quality gate); fuse still **blocked** for active path.

## Hypothesis

1. Revert iter-70 mask kernel changes (return to iter-68/69 baseline perf ledger).
2. Trim non-essential SFPU-adjacent work in `alpha_blend_compute_mb.cpp` for TRISC0 LRA headroom.
3. If headroom: minimal fuse compile hook (`MB_FUSE_TILE_L1_CULL`).

## Revert iter 70

**Yes.** `microblock_cull_compute.cpp` + `writer_tile_l1_mask.cpp` restored to `53ac306` (iter-68 anchor).

## Trim applied (iter 71)

| Change | Rationale |
|--------|-----------|
| Drop `DeviceZoneScopedN("cp_inb")` / `("cp_l1_blend")` inner zones | Profiler scaffolding in per-tile hot loop |
| Drop `mb_cb_consume_fence()` in `process_tile_l1_blend` | L1 bulk has no CB slot recycle race (unlike in-budget coeff stream) |
| `noinline` DEVCONIC helper | **Reverted** — SFPI gcc ICE on `blend_conic_from_cov` (attempt 1) |

## LRA (TRISC0 seg[1] memsz from JIT `trisc0.elf`, yyzo-bh-07)

| Kernel hash | Config | seg[1] memsz | vs 0x720 limit |
|-------------|--------|--------------|----------------|
| `2850661745767925136` | iter 69 active fuse | **0x768** | **overflow** (iter 69 blocked) |
| `13186597283002392372` | iter 71 trim, fuse OFF | **0x4** | OK — gate verify |
| `7254253703035741498` | iter 71 trim + `MB_FUSE_TILE_L1_CULL=1`, dead `fuse_l1_cull_compile_hook` | **0x4** | OK — hook not linked (dead-stripped) |

**Bytes freed vs iter-69 fuse image:** not directly comparable (0x768 includes active fuse stack/batch). Trim alone does **not** enable active `cull_dispatch` in the blend hot path — iter-69 class overflow expected if hook is wired.

## Device (yyzo-bh-07)

| Attempt | bin | Result |
|---------|-----|--------|
| 1 | (noinline conic) | JIT **fail** — SFPI gcc ICE in `blend_conic_from_cov` |
| 2 | `1c9522cbcc848062` | **PASS** — `hero_vs_ref=63.63`, `ms_view=296.868` (30v) |
| 3 | same + `MB_FUSE_TILE_L1_CULL=1` | Compile+run OK; fuse hook dead-stripped — not active fuse |

## Gate vs iter 68

| Metric | iter 68 | iter 71 | Δ |
|--------|--------:|--------:|---|
| `hero_vs_ref` | 63.63 | 63.63 | 0 |
| `ms_view` | 296.58 | 296.87 | +0.29 ms (+0.1%) |

## Decision

**keep** — gate holds; iter-70 mask opts reverted; trim is measurement-only / fuse-prep (profiler zones + l1_bulk fence). **reject** active fuse retry until a compile-safe SFPU shrink (not noinline conic) frees ≥0x48 LRA for wired `tile_l1_cull_sfpu`.

## Next

- Reader-side mask skip / overlap `tile_l1_cull_rd` (iter 61 hang risk) OR host preconic emit to shrink blend DEVCONIC without noinline helper.
- Do not re-attempt iter-70 mask-path micro-opts without ≥2% Tracy bar.
