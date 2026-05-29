// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign K2 — pair-centric scatter (Phase 3).
//
// Single data-movement kernel (scalar C++). Splits the P (gaussian, tile)
// pairs OVER CORES in contiguous, 16-pair-page-aligned ranges. For each pair
// index p in its range a core finds the owning Gaussian g (offs[g] <= p <
// offs[g+1]) and emits:
//     gid = g
//     tid = (min_y[g] + dy) * tiles_x + (min_x[g] + dx)
//   where local = p - offs[g], dy = local / w, dx = local % w.
//
// This reproduces the CPU Phase-3 scatter's gaussian-major order EXACTLY while
// letting every core write whole, page-aligned 64B output pages (no per-core
// boundary sharing, no atomics). The owning Gaussian is found once per range
// via binary search on offs[], then advanced linearly. The AABB (min_x/min_y/
// width) is recomputed from px/py/rx/ry with the identical formula K1 uses, so
// the (dx,dy) iteration matches the tiles_per_gaussian K1 fed the prefix-sum.
//
// RUNTIME ARGS
//   0: offs_addr     int32 SoA exclusive prefix-sum, M+1 entries
//   1: px_addr   2: py_addr   3: rx_addr   4: ry_addr     (fp32 SoA inputs)
//   5: gids_addr     int32 SoA output (P, padded to 16)
//   6: tids_addr     int32 SoA output
//   7: page_start    first 16-pair page this core handles
//   8: page_count    number of pages this core handles
//   9: P             real pair count (entries >= P are padding -> 0)
//  10: M
//  11: tiles_x
//  12: tiles_y
//  13: tile_size
//
// COMPILE-TIME ARGS: 7 TensorAccessorArgs (offs, px, py, rx, ry, gids, tids).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}

inline int clampi(int v, int lo, int hi) {
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

}  // namespace

void kernel_main() {
    const uint32_t offs_addr  = get_arg_val<uint32_t>(0);
    const uint32_t px_addr    = get_arg_val<uint32_t>(1);
    const uint32_t py_addr    = get_arg_val<uint32_t>(2);
    const uint32_t rx_addr    = get_arg_val<uint32_t>(3);
    const uint32_t ry_addr    = get_arg_val<uint32_t>(4);
    const uint32_t gids_addr  = get_arg_val<uint32_t>(5);
    const uint32_t tids_addr  = get_arg_val<uint32_t>(6);
    const uint32_t page_start = get_arg_val<uint32_t>(7);
    const uint32_t page_count = get_arg_val<uint32_t>(8);
    const int P               = static_cast<int>(get_arg_val<uint32_t>(9));
    const int M               = static_cast<int>(get_arg_val<uint32_t>(10));
    const int tiles_x         = static_cast<int>(get_arg_val<uint32_t>(11));
    const int tiles_y         = static_cast<int>(get_arg_val<uint32_t>(12));
    const float tsf           = static_cast<float>(get_arg_val<uint32_t>(13));

    constexpr auto offs_args = TensorAccessorArgs<0>();
    constexpr auto px_args   = TensorAccessorArgs<offs_args.next_compile_time_args_offset()>();
    constexpr auto py_args   = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto rx_args   = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();
    constexpr auto ry_args   = TensorAccessorArgs<rx_args.next_compile_time_args_offset()>();
    constexpr auto gids_args = TensorAccessorArgs<ry_args.next_compile_time_args_offset()>();
    constexpr auto tids_args = TensorAccessorArgs<gids_args.next_compile_time_args_offset()>();

    const auto offs_acc = TensorAccessor(offs_args, offs_addr, PAGE_BYTES);
    const auto px_acc   = TensorAccessor(px_args,   px_addr,   PAGE_BYTES);
    const auto py_acc   = TensorAccessor(py_args,   py_addr,   PAGE_BYTES);
    const auto rx_acc   = TensorAccessor(rx_args,   rx_addr,   PAGE_BYTES);
    const auto ry_acc   = TensorAccessor(ry_args,   ry_addr,   PAGE_BYTES);
    const auto gids_acc = TensorAccessor(gids_args, gids_addr, PAGE_BYTES);
    const auto tids_acc = TensorAccessor(tids_args, tids_addr, PAGE_BYTES);

    if (page_count == 0) {
        return;
    }

    // Scratch CBs (declared in tile_assign_device.cpp scatter program).
    constexpr uint32_t CB_OFFS = 0;
    constexpr uint32_t CB_PX   = 1;
    constexpr uint32_t CB_PY   = 2;
    constexpr uint32_t CB_RX   = 3;
    constexpr uint32_t CB_RY   = 4;
    constexpr uint32_t CB_GID  = 5;
    constexpr uint32_t CB_TID  = 6;

    const uint32_t offs_l1 = get_write_ptr(CB_OFFS);
    const uint32_t px_l1   = get_write_ptr(CB_PX);
    const uint32_t py_l1   = get_write_ptr(CB_PY);
    const uint32_t rx_l1   = get_write_ptr(CB_RX);
    const uint32_t ry_l1   = get_write_ptr(CB_RY);
    const uint32_t gid_l1  = get_write_ptr(CB_GID);
    const uint32_t tid_l1  = get_write_ptr(CB_TID);

    auto offs_ptr = reinterpret_cast<volatile int32_t*>(offs_l1);
    auto pxp = reinterpret_cast<volatile uint32_t*>(px_l1);
    auto pyp = reinterpret_cast<volatile uint32_t*>(py_l1);
    auto rxp = reinterpret_cast<volatile uint32_t*>(rx_l1);
    auto ryp = reinterpret_cast<volatile uint32_t*>(ry_l1);
    auto gidp = reinterpret_cast<volatile int32_t*>(gid_l1);
    auto tidp = reinterpret_cast<volatile int32_t*>(tid_l1);

    // 1-page caches keyed by page index.
    int32_t offs_cached_page = -1;
    int32_t attr_cached_page = -1;

    auto read_offs = [&](int idx) -> int {
        const int pg = idx / static_cast<int>(ELEMS_PER_PAGE);
        const int ip = idx - pg * static_cast<int>(ELEMS_PER_PAGE);
        if (pg != offs_cached_page) {
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), offs_acc),
                           offs_l1, PAGE_BYTES);
            noc_async_read_barrier();
            offs_cached_page = pg;
        }
        return offs_ptr[ip];
    };

    // Per-Gaussian state recomputed on advance.
    int cur_g = -1;
    int cur_offs_g = 0;
    int cur_offs_g1 = 0;
    int cur_minx = 0;
    int cur_miny = 0;
    int cur_w = 1;

    auto load_attrs = [&](int g) {
        const int pg = g / static_cast<int>(ELEMS_PER_PAGE);
        const int ip = g - pg * static_cast<int>(ELEMS_PER_PAGE);
        if (pg != attr_cached_page) {
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), px_acc), px_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), py_acc), py_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), rx_acc), rx_l1, PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), ry_acc), ry_l1, PAGE_BYTES);
            noc_async_read_barrier();
            attr_cached_page = pg;
        }
        const float px = bits_to_f(pxp[ip]);
        const float py = bits_to_f(pyp[ip]);
        const float rx = bits_to_f(rxp[ip]);
        const float ry = bits_to_f(ryp[ip]);
        const int min_x = clampi(static_cast<int>((px - rx) / tsf), 0, tiles_x - 1);
        const int min_y = clampi(static_cast<int>((py - ry) / tsf), 0, tiles_y - 1);
        const int max_x = clampi(static_cast<int>((px + rx) / tsf), 0, tiles_x - 1);
        cur_minx = min_x;
        cur_miny = min_y;
        cur_w = max_x - min_x + 1;
    };

    auto set_g = [&](int g) {
        cur_g = g;
        cur_offs_g = read_offs(g);
        cur_offs_g1 = read_offs(g + 1);
        load_attrs(g);
    };

    const int p_start = static_cast<int>(page_start) * static_cast<int>(ELEMS_PER_PAGE);
    const int p_end   = p_start + static_cast<int>(page_count) * static_cast<int>(ELEMS_PER_PAGE);

    // Binary search: largest g in [0, M-1] with offs[g] <= p_start.
    {
        int lo = 0;
        int hi = M - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (read_offs(mid) <= p_start) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        set_g(lo);
    }

    uint32_t out_idx = 0;
    uint32_t out_page = page_start;
    for (int p = p_start; p < p_end; p++) {
        int gid = 0;
        int tid = 0;
        if (p < P) {
            while (p >= cur_offs_g1) {
                set_g(cur_g + 1);
            }
            const int local = p - cur_offs_g;
            const int dy = local / cur_w;
            const int dx = local - dy * cur_w;
            gid = cur_g;
            tid = (cur_miny + dy) * tiles_x + (cur_minx + dx);
        }
        gidp[out_idx] = gid;
        tidp[out_idx] = tid;
        out_idx++;
        if (out_idx == ELEMS_PER_PAGE) {
            noc_async_write(gid_l1, get_noc_addr(out_page, gids_acc), PAGE_BYTES);
            noc_async_write(tid_l1, get_noc_addr(out_page, tids_acc), PAGE_BYTES);
            noc_async_write_barrier();
            out_page++;
            out_idx = 0;
        }
    }
}
