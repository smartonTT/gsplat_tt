# Iter 85 — C1 materialize PACK2 overlap fix (yyzo-bh-07)

**Branch:** `smarton/stage2-hostfree-l1` · **Build:** `cpp#21` `bin=497be364272dfde7`

## Root cause

Iter 76 batched overflow materialize packed each blendrec into the **upper 32 B of the same 64 B staging slot** (`slot + PACK_OFF`). In `proj_m_blendrec`, `aos[8]` (cb) sits at byte offset 32 and `aos[9]` (depth_key) at offset 36 — the same region as PACK2 `splat[0]` / `splat[1]`. In-place stores clobbered those fields before `splat[3]=aos[9]` and `splat[7]` (cg|cb unorm) were read.

Symptoms matched iter 84 DPRINT: `pay0==gat0` (cov_a), `mism=4` at words **3** (depth) and **7** (colors), `pay7` often `0xFFFF…` garbage.

The working blend gather path uses a **separate** `pack_blendrec_to_l1` destination (`volatile uint32_t splat[8]` or `l1_splat_words`), not in-place on the blendrec page.

## Fix

`sort_subchunk_materialize.cpp` overflow gather: pack into **CB_PACK** (`pack_l1`, 32 B) and `noc_async_write` from there — same as pre–iter-76. Removed unused `PACK_OFF` in-slot pack.

`reader_alpha_blend_mb_devcull.cpp`: `use_payload = (num_subchunks > 1u)` for overflow subchunks.

## Gate

| Config | `hero_vs_ref` | `ms_view` | Result |
|--------|---------------|-----------|--------|
| `use_payload=true` (overflow) | **63.63 dB** | **295.5** | **PASS** ≥63.6 |
| `use_payload=false` (gather fallback) | 63.63 dB (iter 83/84 baseline) | ~295–297 | unchanged |

## Build

`iterate.sh --iter 85 --build cpp` → `cpp#21 bb7a8c4` `bin=497be364272dfde7` (kernel+reader; includes iter 84 DPRINT hooks when `MB_C1_PAYLOAD_DEBUG` defined in blend build).
