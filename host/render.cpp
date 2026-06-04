// SPDX-License-Identifier: Apache-2.0
//
// render.cpp — the clean, linear orchestrator for the production Tenstorrent
// gsplat render pipeline, plus a small pybind module (`render_clean`).
//
// This is the host-free production path that the original render_full_py drives
// under the verify_cmd flag set (GSPLAT_TT_BLEND_MODE=2 + the full resident /
// device stack). Every stage runs on the Blackhole device and keeps its outputs
// device-resident; the host only hands pointers between stages. See config.h for
// the baked-in production configuration and README.md for the stage overview.
//
// Pipeline (execution order):
//   1. project    transform_means_cam -> pfwc -> gather_visible  (resident SoA)
//   2. tile_assign                                               (resident pairs)
//   3. sort       per-tile depth sort                            (resident order)
//   4. cull (SFPU)  ─┐ fused into the sort -> blend continuation
//   5. blend        ─┘ writes the final RGB image
//
// Stages 4+5 run inside the sort driver's continuation (sort_and_bin_tt's
// SortBlendContinuation), exactly as production does, so frame setup overlaps
// the device window on a single command queue.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

// Device-profiler dump hook. Pulls in the public
// ReadMeshDeviceProfilerResults(MeshDevice&, ...) entry point. The dump is a
// strict no-op unless TT_METAL_DEVICE_PROFILER=1 enabled the profiler at device
// init, so these includes are safe in normal (non-profiling) builds/runs.
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include "blend.h"
#include "config.h"
#include "device_state.h"
#include "gather_visible.h"
#include "jit_warmup.h"
#include "pfwc.h"
#include "project.h"
#include "sort.h"
#include "tile_assign.h"

#include "gsplat_cpu/project.h"
#include "gsplat_cpu/sort.h"
#include "gsplat_cpu/thread_pool.h"
#include "gsplat_cpu/tile_assign.h"

namespace py = pybind11;

namespace {

// Single shared worker pool for the host-side bridges (SoA pack, etc.). The
// production renderer caps at 48 workers on the 96-thread bh hosts (measured
// sweet spot); honour GSPLAT_TT_NUM_THREADS if set, else cap at 48.
gsplat_cpu::ThreadPool& worker_pool() {
    static gsplat_cpu::ThreadPool pool([] {
        unsigned hw = std::max(2u, std::thread::hardware_concurrency());
        return static_cast<std::size_t>(std::min(hw, 48u));
    }());
    return pool;
}

// Repack the scene cov3d (N*9 row-major 3x3) into the unique upper-triangular
// (N*6) layout [c00 c01 c02 c11 c12 c22] the device pfwc kernel consumes. cov3d
// is view-invariant across the bench, so cache on (pointer, N) — one repack per
// scene. Bit-identical to the production project_via_device cache.
const std::vector<float>& cov3d_unique(const float* cov3d, std::size_t N) {
    static std::vector<float> cache;
    static const float* cached_ptr = nullptr;
    static std::size_t cached_N = 0;
    if (cached_ptr == cov3d && cached_N == N && cache.size() == N * 6) {
        return cache;
    }
    cache.resize(N * 6);
    float* __restrict cu = cache.data();
    for (std::size_t i = 0; i < N; ++i) {
        cu[i * 6 + 0] = cov3d[i * 9 + 0];
        cu[i * 6 + 1] = cov3d[i * 9 + 1];
        cu[i * 6 + 2] = cov3d[i * 9 + 2];
        cu[i * 6 + 3] = cov3d[i * 9 + 4];
        cu[i * 6 + 4] = cov3d[i * 9 + 5];
        cu[i * 6 + 5] = cov3d[i * 9 + 8];
    }
    cached_ptr = cov3d;
    cached_N = N;
    return cache;
}

// Device-profiler dump hook (mirrors production render_full_py). render_clean
// deliberately never close()s the MeshDevice (see device_state), and tt-metal does
// not dump device-profiler results at atexit, so without this the per-frame
// device zones never reach the live Tracy stream. ReadMeshDeviceProfilerResults
// is a strict no-op unless the profiler was enabled at init
// (TT_METAL_DEVICE_PROFILER=1). Never changes pixels / PSNR.
void maybe_dump_device_profiler() {
    if (!gsplat_tt::device_state::is_initialized()) {
        return;
    }
    auto dev = gsplat_tt::device_state::get_device();
    if (!dev) {
        return;
    }
    tt::tt_metal::ReadMeshDeviceProfilerResults(*dev);
}

// ── Stage 1: project (means->cam, pfwc, gather) ─────────────────────────────
// Runs the three device MeshWorkloads and leaves the M-compact visible
// Gaussian attributes resident in DRAM (proj_m_* + proj_M). Returns the
// M-only ProjectResult (depths sized M); means_2d/covs_2d/colors stay empty
// because the resident downstream stages read them from the resident buffers.
gsplat_cpu::ProjectResult run_project(const float* means, const float* cov3d,
                                      const float* extrinsics,
                                      const float* intrinsics,
                                      const float* colors,
                                      const float* opacities, float min_opacity,
                                      std::size_t N, int image_height,
                                      int image_width, int max_radius) {
    // 1a. means_cam = R @ means, kept device-resident (no D2H).
    gsplat_tt::transform_means_cam_tt_no_download(means, extrinsics, N);

    // 1b. pfwc -> mean_2d / depth / cov2d(a,b,c) / radii, all resident
    //     (null host outputs => RESIDENT_PROJECT, nothing is read back).
    const std::vector<float>& cov_u = cov3d_unique(cov3d, N);
    gsplat_tt::pfwc_tt(cov_u.data(), extrinsics, intrinsics, N,
                       /*mean_2d=*/nullptr, /*depth=*/nullptr, /*cov2d=*/nullptr,
                       /*radii=*/nullptr);

    // 1c. gather the M visible Gaussians into resident M-compact SoA buffers.
    //     downstream_resident=true: the resident tile_assign/sort/blend read
    //     proj_m_* directly over NoC, so we skip the bulk D2H.
    bool gather_ok = false;
    gsplat_cpu::ProjectResult proj = gsplat_tt::gather_visible_tt(
        colors, opacities, N, image_height, image_width, min_opacity, max_radius,
        /*host_gather=*/false, /*verify=*/false, &worker_pool(), &gather_ok,
        /*timings=*/nullptr, /*downstream_resident=*/true);
    // SINGLE-PATH TT: the clean renderer has no CPU fallback. If the device
    // gather (or its upstream means_cam / pfwc) did not complete on-device,
    // hard-fail instead of silently returning an empty/host result.
    if (!gather_ok) {
        throw std::runtime_error(
            "render_clean: device projection/gather failed; the clean renderer "
            "is single-path TT and has no CPU fallback");
    }
    return proj;
}

py::tuple render_view(
    py::array_t<float, py::array::c_style | py::array::forcecast> means,
    py::array_t<float, py::array::c_style | py::array::forcecast> cov3d,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
    py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
    int image_height, int image_width, int tile_size, float min_opacity,
    float contrib_floor, float mb_contrib_floor, bool cull_disabled,
    float transmittance_threshold, int max_radius, float k_cap,
    bool use_isoellipse, int blend_mode) {
    (void)contrib_floor;
    (void)transmittance_threshold;
    (void)k_cap;
    (void)use_isoellipse;
    (void)blend_mode;  // baked: BLEND_MODE=2 (TT device microblock SFPU blend).

    const auto means_info = means.request();
    const std::size_t N = static_cast<std::size_t>(means_info.shape[0]);
    const float* means_ptr = static_cast<const float*>(means_info.ptr);
    const float* cov3d_ptr = static_cast<const float*>(cov3d.request().ptr);
    const float* opacities_ptr = static_cast<const float*>(opacities.request().ptr);
    const float* colors_ptr = static_cast<const float*>(colors.request().ptr);
    const float* extr_ptr = static_cast<const float*>(extrinsics.request().ptr);
    const float* intr_ptr = static_cast<const float*>(intrinsics.request().ptr);

    const int tiles_x = (image_width + tile_size - 1) / tile_size;
    const int tiles_y = (image_height + tile_size - 1) / tile_size;

    // Output image (H, W, 3), pre-zeroed. The blend writer fully overwrites it.
    py::array_t<float> image({static_cast<py::ssize_t>(image_height),
                              static_cast<py::ssize_t>(image_width),
                              static_cast<py::ssize_t>(3)});
    std::memset(image.mutable_data(), 0,
                static_cast<std::size_t>(image_height) *
                    static_cast<std::size_t>(image_width) * 3 * sizeof(float));

    // One-shot JIT compile of all device programs at scene open.
    gsplat_tt::jit_warmup_ideal_path();

    // Stage 1: project.
    gsplat_cpu::ProjectResult proj =
        run_project(means_ptr, cov3d_ptr, extr_ptr, intr_ptr, colors_ptr,
                    opacities_ptr, min_opacity, N, image_height, image_width,
                    max_radius);
    const std::size_t M = proj.depths.size();

    py::dict stats;
    if (M == 0) {
        stats["num_visible"] = 0;
        stats["num_entries"] = 0;
        return py::make_tuple(image, stats);
    }

    // Stage 2: tile_assign (resident inputs => null host pointers).
    bool ta_ok = false;
    gsplat_cpu::TileAssignResult ta = gsplat_tt::tile_assign_tt(
        /*means_2d=*/nullptr, /*radii=*/nullptr, M, image_height, image_width,
        tile_size, /*covs_2d=*/nullptr, /*opacities=*/nullptr, contrib_floor,
        &ta_ok);
    if (!ta_ok) {
        throw std::runtime_error(
            "render_clean: device tile_assign failed; single-path TT, no CPU "
            "fallback");
    }

    // Stages 3+4+5: sort, then the in-sort continuation runs tile-local L1 cull and
    // the resident microblock blend, writing the final image. Chaining them on
    // one command-queue drain is the production host-free path.
    bool sort_ok = false;
    bool blend_ok = false;
    gsplat_tt::SortBlendContinuation sort_blend;
    sort_blend.image_out = image.mutable_data();
    sort_blend.image_height = image_height;
    sort_blend.image_width = image_width;
    sort_blend.mb_contrib_floor = mb_contrib_floor;
    sort_blend.cull_disabled = cull_disabled;
    sort_blend.blend_ok = &blend_ok;

    gsplat_cpu::SortResult sr = gsplat_tt::sort_and_bin_tt(
        ta.gaussian_ids.data(), ta.tile_ids.data(), proj.depths.data(),
        ta.gaussian_ids.size(), M, tiles_x, tiles_y, &worker_pool(), &sort_ok,
        /*timings=*/nullptr, /*need_host_sorted_ids=*/false, &sort_blend);
    // The sort driver runs the SFPU cull + microblock blend as its on-device
    // continuation. Both must have run on-device; hard-fail otherwise (the CPU
    // sort fallback / any host blend are not a valid result for render_clean).
    if (!sort_ok) {
        throw std::runtime_error(
            "render_clean: device sort failed; single-path TT, no CPU fallback");
    }
    if (!blend_ok) {
        throw std::runtime_error(
            "render_clean: device cull/blend failed; single-path TT, no CPU "
            "fallback");
    }

    std::size_t P_kept = 0;
    for (std::size_t t = 0; 2 * t + 1 < sr.tile_ranges.size(); ++t) {
        const int64_t lo = sr.tile_ranges[2 * t + 0];
        const int64_t hi = sr.tile_ranges[2 * t + 1];
        if (hi > lo) P_kept += static_cast<std::size_t>(hi - lo);
    }

    stats["num_visible"] = static_cast<int64_t>(M);
    stats["num_entries"] = static_cast<int64_t>(P_kept);

    // Push this frame's device-profiler zones into the live Tracy stream (no-op
    // unless TT_METAL_DEVICE_PROFILER=1). See above.
    maybe_dump_device_profiler();
    return py::make_tuple(image, stats);
}

}  // namespace

PYBIND11_MODULE(render_clean, m) {
    m.doc() = "Clean extract of the production Tenstorrent gsplat render pipeline";
    m.def("render_view", &render_view, "Render one view through the clean TT pipeline",
          py::arg("means"), py::arg("cov3d"), py::arg("opacities"),
          py::arg("colors"), py::arg("extrinsics"), py::arg("intrinsics"),
          py::arg("image_height"), py::arg("image_width"), py::arg("tile_size"),
          py::arg("min_opacity"), py::arg("contrib_floor"),
          py::arg("mb_contrib_floor"), py::arg("cull_disabled"),
          py::arg("transmittance_threshold"), py::arg("max_radius"),
          py::arg("k_cap"), py::arg("use_isoellipse"), py::arg("blend_mode") = 2);
    m.def("device_shutdown", []() { gsplat_tt::device_state::shutdown(); });
}
