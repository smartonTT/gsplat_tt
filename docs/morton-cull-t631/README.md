# t631: ledger U3 — Morton scene reorder + per-chunk frustum cull for pfwc (CPU sizing, no device)

**Answer.** On today's scene (bicycle after the #536 opacity prune, N = 5,559,909), a load-time
Morton reorder with 1024-gaussian chunks lets a conservative per-view test skip
**38-46% of N** (bounding sphere: min 0.377 / median 0.419 / mean 0.418 / max 0.460; the
`chunk_cull.h` box: 0.336 / 0.388 / 0.382 / 0.419). No visible gaussian is lost on any of the 30 views.
That clears the spec's 25% bar, so a design and bound follow, **but the lever is not new**:
the ledger's premise ("chunk cull failed on PLY order") is wrong. The Morton reorder plus chunk cull
was already built (#169, ported in #433, `GSPLAT_TT_CHUNK_CULL=1`, still in HEAD) and measured on
device three times in b2b. Each time it was a net loss (table below), because Morton order makes
the sort's emit stall, not because the cull skips too little.

## Skip fractions (`size631.py`, output `size631-out.txt`)

`python3 docs/morton-cull-t631/size631.py scenes/bicycle.ply` (repo root, md5 3745b7a6, 30 views of
`benchmarks/cameras_v2.json`, 1024x1024). Order and bounds depend only on the scene (built once,
nothing fitted to the bench views). Visibility = the #140/#169 per-gaussian model (t466 `cullshare.py`).

| test (Morton, chunk 1024, pruned N) | skip min | median | mean | max | visible lost |
|---|---|---|---|---|---|
| box, `render/host/chunk_cull.h` (K 1.5, 5 px pad) | 0.336 | 0.388 | 0.382 | 0.419 | 0 |
| bounding sphere (R = max \|m - c\| + 1.5 rho, same s_K guard) | 0.377 | 0.419 | 0.418 | 0.460 | 0 |
| empty-chunk bound (no visible gaussian; any per-chunk test) | 0.573 | 0.633 | 0.641 | 0.716 | - |
| PLY order, any test (t466) | 0 | 0 | 0 | 0 | - |

As a share of the unpruned 6.13M: box 0.346, sphere 0.379 (mean). The sphere is tighter than the box
here because the box pads each mean by a K·rho cube, the sphere by a K·rho ball; it also costs one
dot product per side plane per chunk (5430 spheres per view) instead of 8 corner transforms.

## Bound (b2b ms/view)

pfwc = 1.063 + 0.274·M (PROFILE t227 fit); the 1.063 part scales with N read, and pfwc's mover end is
1:1 critical in b2b (t614). The cull removes only invisible gaussians, so M, pairs, sort and blend
work stay the same.

- **Gross pfwc saving: 0.418 × 1.063 ≈ 0.44 ms/view** (sphere; 0.40-0.49 over the views; box 0.41).
  Measured gross in b2b on an older stack: 0.525-0.573 (#470, p150, full N).
- **Known costs, all measured:**

| task | board, metric | net cull vs off | where it goes |
|---|---|---|---|
| #433 | p100a, latency | +0.69..+0.74 | blend stage (sort tail), +0.1 host list |
| #470 | p150 bh-30, b2b | +0.309 (pfwc -0.525, Morton order +0.834) | sort emit |
| #479 | p100a, b2b | +0.413; with emit ring 4/8: +0.215 vs off48 | sort_ol_emit 1.42 → 1.88-2.07 ms: Morton gids burst into one screen tile's run buffer |
| #493 | p150 bh-30, b2b | +0.177 (ring 4/8); ring 16 does not fit L1 | emit burst + host list +0.19 |

  Also: the opacity prune (#536, -0.133) is switched off under `GSPLAT_TT_CHUNK_CULL`
  (`render/opacity_prune.py prune_blocker`); a new path must prune first, then Morton-order (as modelled).
- **Net bound: ≤ 0.44 ms/view only if the Morton-order emit cost goes to zero; with today's best
  measured emit fix (ring 4/8) the order cost is still ~0.3-0.5 ms of sort makespan, so the expected
  net is about -0.05 to +0.15.** The ledger's 0.3-0.6 is not reachable without a new emit design.

## Design, if a device follow-up is wanted

1. Load (per scene): prune opacity < 1/255, Morton-order (render/run.py `morton_order`), chunk 1024,
   store per chunk (centre xyz, R, rho_max): 5430 × 5 floats.
2. Per view (host, before the pfwc enqueue): near test + 4 padded side planes per sphere, survivor
   list dealt strided over the pfwc cores (`chunk_cull::deal`, existing). C++ cost ~tens of µs.
3. **Precondition: de-burst the sort emit for a spatially ordered record stream.** Today the writer
   compacts visible records in processing order, which becomes the gid order sort and blend use;
   Morton chunks send ~300 consecutive records into one or two screen tiles. Options: a deeper
   per-tile run ring that fits L1 (16 does not), an emit that reads each core's segment at a stride
   (interleaving distant chunks), or cull units of 64 gaussians (256 B bursts) gathered from distant
   survivors into each pfwc tile. All change the depth-tie order (md5 changes; PSNR must hold).

**Recommendation:** do not re-queue a device A/B of the existing `GSPLAT_TT_CHUNK_CULL` path (measured
net-negative three times; standing rule msgs #299/#300). Queue device work only after a CPU model of the
emit stream (records per screen tile per burst, under each option in step 3) shows the burst stays
within a ring depth that fits L1. Without that, U3's realistic value is ≤0.15 ms/view.
