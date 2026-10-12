// SPDX-License-Identifier: Apache-2.0
// Task #544 (B4' step 1): in the fused mat+blend program the blend CBs CB_MB_COUNTS,
// CB_OUT and CB_IMG_U8 share a mat CB's storage through one multi-index config
// (sort_device.cpp alias_cb), and such a config gives every index the config's total
// size as its fifo size (circular_buffer_init.h; #289). So these blend rings wrap
// through the whole mat region (CB_OUT: 129 pages instead of 6). Every RISC that
// touches one of them calls blend_fifo_shrink at the start of its blend body, before
// the first use, so producer and consumer wrap at the same place and the rest of the
// mat region is unused during blend (docs/b4-sort-l1/RESULT.md section 1).
// Host side: GSPLAT_TT_BLEND_ALIAS_CUT (default on, =0 off) defines BLEND_ALIAS_CUT and
// the BLEND_*_BYTES sizes (blend_device.cpp).
// The interface is a template so tests/unit/test_blend_alias_fifo.cpp can check it on a
// stand-in struct.
#pragma once

#include <cstdint>

// c: the RISC's own LocalCBInterface for the blend id, not yet used by this RISC
// (fifo_rd_ptr / fifo_wr_ptr still at the base). bytes: the ring's real size, a
// multiple of page_bytes. shift: cb_addr_shift (fifo fields are in 16 B words on the
// TRISCs, bytes on the movers). Keeps the base, so the read and write pointers stay valid.
template <typename Iface>
inline void blend_fifo_shrink_iface(Iface& c, uint32_t bytes, uint32_t page_bytes, uint32_t shift) {
    const uint32_t size = bytes >> shift;
    const uint32_t base = c.fifo_limit - c.fifo_size;
    if (size == 0u || size >= c.fifo_size) return;  // already that small (not fused)
    c.fifo_size = size;
    c.fifo_limit = base + size;
    c.fifo_num_pages = bytes / page_bytes;
}

#if defined(BLEND_ALIAS_CUT) && !defined(BLEND_ALIAS_FIFO_HOST_TEST)
#ifdef TRISC_MATH
// The math TRISC has no cb_interface (its link fails on any reference) and never
// moves a fifo pointer, so there is nothing to shrink.
FORCE_INLINE void blend_fifo_shrink(uint32_t, uint32_t, uint32_t) {}
#else
FORCE_INLINE void blend_fifo_shrink(uint32_t cb, uint32_t bytes, uint32_t page_bytes) {
    blend_fifo_shrink_iface(get_local_cb_interface(cb), bytes, page_bytes, cb_addr_shift);
}
#endif
#endif
