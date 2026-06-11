# Render alternate (removed) code paths

Single-live-path policy: each branch keeps ONE live implementation (no `#ifdef`
alternates). When an alternate is replaced, its prior form is recorded here with
the git ref it was live at, so it can be recovered if a regression is found.

## sort_subchunk_materialize — in-budget per-record DRAM scatter emit

- Replaced by: iter 113 (sort Stage 1) — L1->L1 permute into CB_SLAB scratch +
  coalesced `SLAB_PAGE_BYTES` page writes.
- Live at git ref: `d33d3da` (`render/kernels/dataflow/sort_subchunk_materialize.cpp`).
- What it did: for in-budget tiles (`count <= BUCKET_FIT`) it bulk-read the tile
  bucket into L1, radix-sorted the indices in L1, then emitted the depth-sorted
  PACK2 slab with **one `noc_async_write` per record** (up to `BUCKET_FIT`=8192
  separate 32B NoC writes from the permuted L1 source, single barrier at end):

```cpp
for (uint32_t k = 0; k < L; ++k) {
    const uint32_t idx = sorted[k];
    const uint32_t out_page = sc_page + (k / SLAB_RECS_PER_PAGE);
    const uint32_t out_off = (k % SLAB_RECS_PER_PAGE) * L1_SPLAT_BYTES;
    const uint32_t src_page = (idx >> 1);
    const uint32_t src_half = (idx & 1u) * L1_SPLAT_BYTES;
    noc_async_write(
        buck + src_page * L1_PACK_PAGE_BYTES + src_half,
        get_noc_addr(out_page, payload_acc) + out_off,
        L1_SPLAT_BYTES);
}
noc_async_write_barrier();
```

- Why removed: the per-record scatter posts up to 8192 tiny NoC write
  descriptors per in-budget tile; building the slab in an L1 scratch and writing
  it in ≤128 page-sized transfers keeps the record format/slab layout
  bit-identical while collapsing the descriptor count. Overflow tiles
  (`> BUCKET_FIT`) keep the existing `sort_sorted_ids` blendrec gather fallback.
