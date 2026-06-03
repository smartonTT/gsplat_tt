// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SFPU microblock-cull WRITER (GSPLAT_TT_SFPU_CULL).
//
// Consumes the keep-tiles microblock_cull_compute.cpp packs (one per 32-gaussian
// batch) and emits the 32-bit per-gaussian microblock masks into the resident
// cull_masks buffer (indexed identically to sort_sorted_ids: global candidate
// index == id_start + batch*32 + gaussian).
//
// LANE -> MASK (the fallback primitive). The compute kernel writes, for the
// g-th gaussian of a batch, the per-microblock keep flag (1.0/0.0) into SFPU
// vector g. pack_tile untilizes DEST -> CB at CB-linear positions, the exact
// inverse of the copy_tile the compute used to load the box-origin ramps. So a
// keep flag for (vector g, microblock m) lands at the SAME CB-linear position
// the host wrote microblock m's box origin to. That position is
//   PERM(g,m) = m<16 ? (2*(g>>1))*32   + (g&1) + 2*m
//                    : (2*(g>>1)+1)*32 + (g&1) + 2*(m-16)
// (derived from the VECMAP geometry vector(r,c)=2*(r/2)+(c&1) used by
// mb_perm_img_of_dev). We read keep[PERM(g,m)] (fp32; pure integer != 0 test,
// no float on the mover) and set mask bit m. The intra-vector lane order never
// has to be known: it cancels in the copy/pack round-trip.
//
// RUNTIME ARGS
//   0: cull_masks base (uint32, 64B / 16-elem pages)
//   1: sort_tile_ranges base   2: tile_ids base
//   3: sort_lpt_meta base   4: core_index
//   5: cull_mask_base base (uint32, 64B SoA: per-tile PAGE-ALIGNED mask offset)
// COMPILE-TIME: 5 DRAM-interleaved TensorAccessorArgs: cull_masks, ranges,
// tile_ids, sort_lpt_meta, cull_mask_base.
//
// cull_masks is laid out per-tile PAGE-ALIGNED (cull_mask_base[tile] is a
// multiple of 16): the masks for tile t occupy [base, base+L). This guarantees
// every DRAM write below starts on a 16-element (64B) boundary and spans whole
// pages -- NoC->DRAM writes that are not 16-element aligned land shifted (the
// hardware rounds the unaligned tail to the next 16B boundary), which silently
// corrupted the dense (sort-range-indexed) layout.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t CB_MASK_SCR = 6;    // writer-private scratch
constexpr uint32_t CB_KEEP     = 16;   // compute -> writer keep tiles

constexpr uint32_t SOA_PAGE_BYTES = 64;
constexpr uint32_t IDS_PAGE_BYTES = 64;
constexpr uint32_t BATCH = 32;
constexpr uint32_t NUM_MB = 32;
constexpr uint32_t MASKS_PER_PAGE = 16;  // cull_masks: 16 u32 / 64B page

inline uint32_t perm(uint32_t g, uint32_t m) {
    const uint32_t cp = g & 1u;
    if (m < 16u) {
        return (2u * (g >> 1)) * 32u + cp + 2u * m;
    }
    return (2u * (g >> 1) + 1u) * 32u + cp + 2u * (m - 16u);
}

template <typename Acc>
inline uint32_t read_soa_u32(const Acc& acc, uint32_t elem, uint32_t scratch_addr) {
    noc_async_read_tile(elem >> 4, acc, scratch_addr);
    noc_async_read_barrier();
    return reinterpret_cast<volatile uint32_t*>(scratch_addr)[elem & 0xF];
}

}  // namespace

void kernel_main() {
    const uint32_t masks_addr     = get_arg_val<uint32_t>(0);
    const uint32_t ranges_addr    = get_arg_val<uint32_t>(1);
    const uint32_t tile_ids_addr  = get_arg_val<uint32_t>(2);
    const uint32_t lpt_meta_addr  = get_arg_val<uint32_t>(3);
    const uint32_t core_index     = get_arg_val<uint32_t>(4);
    const uint32_t base_addr      = get_arg_val<uint32_t>(5);

    constexpr auto masks_args  = TensorAccessorArgs<0>();
    constexpr auto ranges_args = TensorAccessorArgs<masks_args.next_compile_time_args_offset()>();
    constexpr auto tids_args   = TensorAccessorArgs<ranges_args.next_compile_time_args_offset()>();
    constexpr auto lpt_meta_args = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto base_args   = TensorAccessorArgs<lpt_meta_args.next_compile_time_args_offset()>();

    const auto masks_acc  = TensorAccessor(masks_args,  masks_addr,    SOA_PAGE_BYTES);
    const auto ranges_acc = TensorAccessor(ranges_args, ranges_addr,   SOA_PAGE_BYTES);
    const auto tids_acc   = TensorAccessor(tids_args,   tile_ids_addr, IDS_PAGE_BYTES);
    const auto lpt_meta_acc = TensorAccessor(lpt_meta_args, lpt_meta_addr, SOA_PAGE_BYTES);
    const auto base_acc   = TensorAccessor(base_args,   base_addr,     SOA_PAGE_BYTES);

    constexpr uint32_t META_ELEMS_PER_PAGE = 16u;
    const uint32_t meta_elem0 = core_index * 2u;
    const uint32_t meta_page0 = meta_elem0 / META_ELEMS_PER_PAGE;
    const uint32_t meta_ip0   = meta_elem0 % META_ELEMS_PER_PAGE;

    const uint32_t scratch_addr = get_write_ptr(CB_MASK_SCR);
    auto scratch_ptr_meta = reinterpret_cast<volatile uint32_t*>(scratch_addr);
    noc_async_read(get_noc_addr(meta_page0, lpt_meta_acc), scratch_addr, 64);
    noc_async_read_barrier();
    uint32_t tile_ids_start = scratch_ptr_meta[meta_ip0];
    uint32_t tile_ids_count = 0;
    if (meta_ip0 + 1u < META_ELEMS_PER_PAGE) {
        tile_ids_count = scratch_ptr_meta[meta_ip0 + 1u];
    } else {
        noc_async_read(get_noc_addr(meta_page0 + 1u, lpt_meta_acc), scratch_addr, 64);
        noc_async_read_barrier();
        tile_ids_count = scratch_ptr_meta[0];
    }

    if (tile_ids_count == 0) {
        return;
    }
    auto scratch_ptr = reinterpret_cast<volatile uint32_t*>(scratch_addr);

    constexpr uint32_t MAX_TILE_IDS_PER_CORE = 256;
    uint32_t tile_ids[MAX_TILE_IDS_PER_CORE];
    {
        const uint32_t ids_per_page = IDS_PAGE_BYTES / 4;  // 16
        uint32_t page_idx = tile_ids_start / ids_per_page;
        uint32_t in_page  = tile_ids_start % ids_per_page;
        uint32_t remaining = tile_ids_count;
        uint32_t out_idx = 0;
        while (remaining > 0) {
            noc_async_read_tile(page_idx, tids_acc, scratch_addr);
            noc_async_read_barrier();
            uint32_t take = ids_per_page - in_page;
            if (take > remaining) take = remaining;
            for (uint32_t i = 0; i < take; i++) tile_ids[out_idx + i] = scratch_ptr[in_page + i];
            out_idx   += take;
            remaining -= take;
            page_idx  += 1;
            in_page    = 0;
        }
    }

    for (uint32_t ti = 0; ti < tile_ids_count; ti++) {
        const uint32_t tile_id = tile_ids[ti];
        uint32_t id_start = read_soa_u32(ranges_acc, tile_id * 2u + 0u, scratch_addr);
        uint32_t id_end   = read_soa_u32(ranges_acc, tile_id * 2u + 1u, scratch_addr);
        const uint32_t L = id_end - id_start;
        // Per-tile PAGE-ALIGNED base in cull_masks (multiple of 16).
        const uint32_t base = read_soa_u32(base_acc, tile_id, scratch_addr);

        uint32_t processed = 0;
        while (processed < L) {
            uint32_t nb = L - processed;
            if (nb > BATCH) nb = BATCH;

            cb_wait_front(CB_KEEP, 1);
            auto keep = reinterpret_cast<volatile uint32_t*>(get_read_ptr(CB_KEEP));

            // Pack masks for this batch into the scratch, then write them out.
            for (uint32_t g = 0; g < nb; g++) {
                uint32_t mask = 0u;
                for (uint32_t m = 0; m < NUM_MB; m++) {
                    if (keep[perm(g, m)] != 0u) {
                        mask |= (1u << m);
                    }
                }
                scratch_ptr[g] = mask;
            }
            // Write the batch as WHOLE PAGES from the tile's page-aligned base.
            // base is a multiple of 16 and `processed` is a multiple of the batch
            // size (32) for every batch except possibly the last, so base_k is
            // always 16-element (64B) aligned. Round the dword count up to a full
            // page (zero-padding the tail of the final partial batch) so the write
            // size is a multiple of 64B and never starts/ends mid-page.
            const uint32_t base_k = base + processed;
            uint32_t nb_pad = (nb + (MASKS_PER_PAGE - 1u)) & ~(MASKS_PER_PAGE - 1u);
            for (uint32_t g = nb; g < nb_pad; g++) scratch_ptr[g] = 0u;
            const uint32_t npg = nb_pad / MASKS_PER_PAGE;
            for (uint32_t pp = 0; pp < npg; pp++) {
                noc_async_write(scratch_addr + pp * MASKS_PER_PAGE * 4u,
                                get_noc_addr(base_k / MASKS_PER_PAGE + pp, masks_acc),
                                MASKS_PER_PAGE * 4u);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_KEEP, 1);

            processed += nb;
        }
    }
}
