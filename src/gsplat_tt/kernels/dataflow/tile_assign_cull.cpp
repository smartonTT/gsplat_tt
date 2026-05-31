// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// tile_assign K4 — per-pair constrained-min Mahalanobis cull (Phase 4).
//
// Single data-movement kernel (scalar C++). Splits the P pairs OVER CORES in
// 16-pair-page-aligned ranges. For each pair (g, tile_id) it reproduces
// gsplat_cpu::tile_assign Phase 4 BIT-EXACT (covs_2d direct quadratic form,
// constrained-min over the tile rectangle in Gaussian-centered coords) and
// writes keep_mask[p] in {0,1}. The host computes the per-Gaussian
// m2_thresh / opacity-floor (K3) and compacts the kept pairs (H2).
//
//   det = max(a*c - b*b, 1e-6)
//   tx_tile = (tile_id % tiles_x) * tile_size ; ty_tile = (tile_id/tiles_x)*ts
//   u_lo = tx_tile - px ; u_hi = u_lo + ts ; v_lo = ty_tile - py ; v_hi = v_lo+ts
//   inside -> keep ; else constrained-min num vs det * m2_thresh[g]
//
// The per-Gaussian opacity floor is folded into m2thr (host K3): when
// op <= contrib_floor the host writes the SENTINEL m2thr = -1.0f. Legitimate
// m2thr = -2*log(contrib_floor/op) for op > contrib_floor is always >= 0
// (ratio < 1 -> log <= 0), so `!(m2thr >= 0)` exactly reproduces the old
// `opacok == 0` drop test (and also rejects any NaN) WITHOUT a separate
// opacok buffer — halving the K3 H2D. The check runs BEFORE the inside-test,
// matching the old opacok short-circuit, so a low-opacity pair that happens
// to be inside its tile is still dropped.
//
// RUNTIME ARGS
//   0: gids_addr  1: tids_addr                (int32 SoA pair arrays)
//   2: a_addr  3: b_addr  4: c_addr           (fp32 SoA cov a,b,c)
//   5: px_addr  6: py_addr                    (fp32 SoA means)
//   7: m2thr_addr (fp32; -1.0f sentinel == opacity-floor drop)
//   8: keep_addr (int32 output)
//   9: page_start  10: page_count  11: P
//  12: tiles_x  13: tile_size
//
// COMPILE-TIME ARGS: 9 TensorAccessorArgs in the above order.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// The JIT compiles dataflow kernels with -ffast-math, which fuses a*b+c into
// FMA and uses reciprocal approximations — that diverges from the host cull
// (gsplat_cpu is built with -ffp-contract=off), flipping keep decisions at the
// threshold boundary and perturbing P'. Disable fast-math + FMA contraction
// for this translation unit so the per-pair comparison is bit-faithful.
#pragma GCC optimize("no-fast-math", "fp-contract=off")

namespace {

constexpr uint32_t PAGE_BYTES = 64;
constexpr uint32_t ELEMS_PER_PAGE = 16;
constexpr float kInf = 1e30f;

inline float bits_to_f(uint32_t b) {
    float f;
    __builtin_memcpy(&f, &b, 4);
    return f;
}
inline float fmaxf2(float a, float b) { return a > b ? a : b; }
inline float clampf(float v, float lo, float hi) {
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    return v;
}

}  // namespace

void kernel_main() {
    const uint32_t gids_addr   = get_arg_val<uint32_t>(0);
    const uint32_t tids_addr   = get_arg_val<uint32_t>(1);
    const uint32_t a_addr      = get_arg_val<uint32_t>(2);
    const uint32_t b_addr      = get_arg_val<uint32_t>(3);
    const uint32_t c_addr      = get_arg_val<uint32_t>(4);
    const uint32_t px_addr     = get_arg_val<uint32_t>(5);
    const uint32_t py_addr     = get_arg_val<uint32_t>(6);
    const uint32_t m2thr_addr  = get_arg_val<uint32_t>(7);
    const uint32_t keep_addr   = get_arg_val<uint32_t>(8);
    const uint32_t page_start  = get_arg_val<uint32_t>(9);
    const uint32_t page_count  = get_arg_val<uint32_t>(10);
    const int P                = static_cast<int>(get_arg_val<uint32_t>(11));
    const int tiles_x          = static_cast<int>(get_arg_val<uint32_t>(12));
    const float tsf            = static_cast<float>(get_arg_val<uint32_t>(13));

    constexpr auto gids_args   = TensorAccessorArgs<0>();
    constexpr auto tids_args   = TensorAccessorArgs<gids_args.next_compile_time_args_offset()>();
    constexpr auto a_args      = TensorAccessorArgs<tids_args.next_compile_time_args_offset()>();
    constexpr auto b_args      = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto c_args      = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto px_args     = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto py_args     = TensorAccessorArgs<px_args.next_compile_time_args_offset()>();
    constexpr auto m2thr_args  = TensorAccessorArgs<py_args.next_compile_time_args_offset()>();
    constexpr auto keep_args   = TensorAccessorArgs<m2thr_args.next_compile_time_args_offset()>();

    const auto gids_acc   = TensorAccessor(gids_args,   gids_addr,   PAGE_BYTES);
    const auto tids_acc   = TensorAccessor(tids_args,   tids_addr,   PAGE_BYTES);
    const auto a_acc      = TensorAccessor(a_args,      a_addr,      PAGE_BYTES);
    const auto b_acc      = TensorAccessor(b_args,      b_addr,      PAGE_BYTES);
    const auto c_acc      = TensorAccessor(c_args,      c_addr,      PAGE_BYTES);
    const auto px_acc     = TensorAccessor(px_args,     px_addr,     PAGE_BYTES);
    const auto py_acc     = TensorAccessor(py_args,     py_addr,     PAGE_BYTES);
    const auto m2thr_acc  = TensorAccessor(m2thr_args,  m2thr_addr,  PAGE_BYTES);
    const auto keep_acc   = TensorAccessor(keep_args,   keep_addr,   PAGE_BYTES);

    if (page_count == 0) {
        return;
    }

    constexpr uint32_t CB_GID  = 0;
    constexpr uint32_t CB_TID  = 1;
    constexpr uint32_t CB_A    = 2;
    constexpr uint32_t CB_B    = 3;
    constexpr uint32_t CB_C    = 4;
    constexpr uint32_t CB_PX   = 5;
    constexpr uint32_t CB_PY   = 6;
    constexpr uint32_t CB_M2T  = 7;
    constexpr uint32_t CB_KEEP = 8;

    const uint32_t gid_l1  = get_write_ptr(CB_GID);
    const uint32_t tid_l1  = get_write_ptr(CB_TID);
    const uint32_t a_l1    = get_write_ptr(CB_A);
    const uint32_t b_l1    = get_write_ptr(CB_B);
    const uint32_t c_l1    = get_write_ptr(CB_C);
    const uint32_t px_l1   = get_write_ptr(CB_PX);
    const uint32_t py_l1   = get_write_ptr(CB_PY);
    const uint32_t m2t_l1  = get_write_ptr(CB_M2T);
    const uint32_t keep_l1 = get_write_ptr(CB_KEEP);

    auto gidp = reinterpret_cast<volatile int32_t*>(gid_l1);
    auto tidp = reinterpret_cast<volatile int32_t*>(tid_l1);
    auto ap   = reinterpret_cast<volatile uint32_t*>(a_l1);
    auto bp   = reinterpret_cast<volatile uint32_t*>(b_l1);
    auto cp   = reinterpret_cast<volatile uint32_t*>(c_l1);
    auto pxp  = reinterpret_cast<volatile uint32_t*>(px_l1);
    auto pyp  = reinterpret_cast<volatile uint32_t*>(py_l1);
    auto m2tp = reinterpret_cast<volatile uint32_t*>(m2t_l1);
    auto keepp = reinterpret_cast<volatile int32_t*>(keep_l1);

    int32_t attr_cached_page = -1;

    auto load_attrs = [&](int g) {
        const int pg = g / static_cast<int>(ELEMS_PER_PAGE);
        if (pg != attr_cached_page) {
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), a_acc),      a_l1,   PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), b_acc),      b_l1,   PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), c_acc),      c_l1,   PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), px_acc),     px_l1,  PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), py_acc),     py_l1,  PAGE_BYTES);
            noc_async_read(get_noc_addr(static_cast<uint32_t>(pg), m2thr_acc),  m2t_l1, PAGE_BYTES);
            noc_async_read_barrier();
            attr_cached_page = pg;
        }
    };

    const uint32_t p_start_page = page_start;
    const uint32_t p_end_page = page_start + page_count;

    for (uint32_t pg = p_start_page; pg < p_end_page; pg++) {
        // Load this page of 16 pairs.
        noc_async_read(get_noc_addr(pg, gids_acc), gid_l1, PAGE_BYTES);
        noc_async_read(get_noc_addr(pg, tids_acc), tid_l1, PAGE_BYTES);
        noc_async_read_barrier();

        for (uint32_t j = 0; j < ELEMS_PER_PAGE; j++) {
            const int p = static_cast<int>(pg) * static_cast<int>(ELEMS_PER_PAGE) +
                          static_cast<int>(j);
            if (p >= P) {
                keepp[j] = 0;
                continue;
            }
            const int g = gidp[j];
            const int tid = tidp[j];
            load_attrs(g);
            const int ip = g - (g / static_cast<int>(ELEMS_PER_PAGE)) *
                                   static_cast<int>(ELEMS_PER_PAGE);

            // Opacity-floor drop folded into m2thr: the host writes the -1.0f
            // sentinel when op <= contrib_floor. Legitimate m2thr is always
            // >= 0, so !(m2thr >= 0) reproduces the old `opacok == 0` test
            // (and rejects NaN) — checked BEFORE the inside-test exactly like
            // the old short-circuit.
            const float m2thr = bits_to_f(m2tp[ip]);
            if (!(m2thr >= 0.0f)) {
                keepp[j] = 0;
                continue;
            }

            const float a = bits_to_f(ap[ip]);
            const float b = bits_to_f(bp[ip]);
            const float c = bits_to_f(cp[ip]);
            const float px = bits_to_f(pxp[ip]);
            const float py = bits_to_f(pyp[ip]);

            const float det = fmaxf2(a * c - b * b, 1e-6f);

            const float tx_tile = static_cast<float>(tid % tiles_x) * tsf;
            const float ty_tile = static_cast<float>(tid / tiles_x) * tsf;

            const float u_lo = tx_tile - px;
            const float u_hi = u_lo + tsf;
            const float v_lo = ty_tile - py;
            const float v_hi = v_lo + tsf;

            const bool x_inside = (u_lo <= 0.0f) && (0.0f <= u_hi);
            const bool y_inside = (v_lo <= 0.0f) && (0.0f <= v_hi);

            const float scaled_thresh = det * m2thr;

            if (x_inside && y_inside) {
                keepp[j] = 1;
                continue;
            }

            float m2_v = kInf;
            if (!x_inside) {
                const float u_fix = (u_lo > 0.0f) ? u_lo : u_hi;
                const float a_safe = fmaxf2(a, 1e-12f);
                float v_star = (b * u_fix) / a_safe;
                v_star = clampf(v_star, v_lo, v_hi);
                m2_v = c * u_fix * u_fix - 2.0f * b * u_fix * v_star +
                       a * v_star * v_star;
            }
            float m2_h = kInf;
            if (!y_inside) {
                const float v_fix = (v_lo > 0.0f) ? v_lo : v_hi;
                const float c_safe = fmaxf2(c, 1e-12f);
                float u_star = (b * v_fix) / c_safe;
                u_star = clampf(u_star, u_lo, u_hi);
                m2_h = c * u_star * u_star - 2.0f * b * u_star * v_fix +
                       a * v_fix * v_fix;
            }
            const float num = (m2_v < m2_h) ? m2_v : m2_h;
            keepp[j] = (num <= scaled_thresh) ? 1 : 0;
        }

        noc_async_write(keep_l1, get_noc_addr(pg, keep_acc), PAGE_BYTES);
        noc_async_write_barrier();
    }
}
