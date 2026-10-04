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
#include "vis_mode.h"
#include "host_profile.h"
#include "jit_warmup.h"
#include "pfwc.h"
#include "project.h"
#include "sort.h"
#include "stage_timers.h"
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
                                      int image_width, int max_radius,
                                      int tile_size, float mb_contrib_floor,
                                      bool cull_disabled) {
    // 1a+1b. FUSED project(means_cam)+pfwc (iter-133): one device program runs
    //     the world→camera means transform in L1 and then mean_2d / depth /
    //     cov2d(a,b,c) / radii, all resident (null host outputs => nothing read
    //     back). Replaces the former 2-program means_cam→DRAM→pfwc handoff.
    gsplat_tt::stagetimers::Span cov3d_span(
        gsplat_tt::stagetimers::acc().project_cov3d);
    const std::vector<float>& cov_u = cov3d_unique(cov3d, N);
    cov3d_span.stop();
    // Lever 2 (task #99, GSPLAT_TT_SFPU_VIS, render/host/vis_mode.h): pfwc also
    // evaluates the visibility predicate and the tile rectangle on the SFPU.
    // Needs the scene opacities resident first; the packed rectangle limits the
    // grid (vis_tile.h), so larger grids keep the legacy path.
    gsplat_tt::PfwcVisParams vis_params;
    const gsplat_tt::PfwcVisParams* vis = nullptr;
    if (gsplat_tt::sfpu_vis_mode() != 0 && tile_size > 0 &&
        (tile_size & (tile_size - 1)) == 0) {
        const int tx = (image_width + tile_size - 1) / tile_size;
        const int ty = (image_height + tile_size - 1) / tile_size;
        if (tx <= 512 && ty <= 1024 &&
            gsplat_tt::gather_visible_upload_scene(colors, opacities, N)) {
            vis_params.min_opacity = min_opacity;
            vis_params.image_width = static_cast<float>(image_width);
            vis_params.image_height = static_cast<float>(image_height);
            vis_params.max_radius = gsplat_tt::gather_visible_effective_max_radius(
                max_radius, image_height, image_width);
            vis_params.tile_size = tile_size;
            vis_params.tiles_x = tx;
            vis_params.tiles_y = ty;
            static const float edge_tau = [] {
                const char* e = std::getenv("GSPLAT_TT_VIS_EDGE_TAU");
                return (e != nullptr && *e != '\0') ? std::strtof(e, nullptr) : 1.0f / 4096.0f;
            }();
            vis_params.edge_tau = edge_tau;
            // Lever B (task #125, GSPLAT_TT_PFWC_FUSE, default 1 since #122, 0 = kill switch).
            vis_params.fuse = gsplat_tt::pfwc_fuse_mode() == 1;
            // Lever C (task #140, GSPLAT_TT_PRECULL, default 1 since #156, 0 = kill switch; 2 = pixel-centre rect, #157): only with the band cull on.
            if (gsplat_tt::precull_mode() >= 1 && !cull_disabled)
                vis_params.precull_floor = mb_contrib_floor;
            vis = &vis_params;
        }
    }
    gsplat_tt::pfwc_tt(means, cov_u.data(), extrinsics, intrinsics, N,
                       /*mean_2d=*/nullptr, /*depth=*/nullptr, /*cov2d=*/nullptr,
                       /*radii=*/nullptr, /*timings_out=*/nullptr, vis);

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

    gsplat_tt::hostprof::on_view_enter();

    // Per-stage host attribution (stage_timers.h). `head` runs until the first
    // device stage; the fused cull/blend buckets are booked by blend_device.
    namespace st = gsplat_tt::stagetimers;
    st::Span view_span(st::acc().view_total);
    st::Span head_span(st::acc().head);

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

    // Output image (H, W, 3) uint8 = uint8(clip(rgb, 0, 1) * 255), packed on
    // device by the blend writer (task #61), which fully overwrites it.
    py::array_t<uint8_t> image({static_cast<py::ssize_t>(image_height),
                                static_cast<py::ssize_t>(image_width),
                                static_cast<py::ssize_t>(3)});
    const std::size_t image_bytes = static_cast<std::size_t>(image_height) *
                                    static_cast<std::size_t>(image_width) * 3;

    // One-shot JIT compile of all device programs at scene open.
    gsplat_tt::jit_warmup_ideal_path();
    head_span.stop();

    // Stage 1: project.
    gsplat_cpu::ProjectResult proj;
    {
        st::Span s(st::acc().project);
        proj = run_project(means_ptr, cov3d_ptr, extr_ptr, intr_ptr, colors_ptr,
                           opacities_ptr, min_opacity, N, image_height,
                           image_width, max_radius, tile_size, mb_contrib_floor,
                           cull_disabled);
    }
    const std::size_t M = proj.depths.size();

    py::dict stats;
    if (M == 0) {
        std::memset(image.mutable_data(), 0, image_bytes);
        stats["num_visible"] = 0;
        stats["num_entries"] = 0;
        st::acc().views++;
        return py::make_tuple(image, stats);
    }

    // Stage 2: tile_assign (resident inputs => null host pointers).
    bool ta_ok = false;
    gsplat_cpu::TileAssignResult ta;
    {
        st::Span s(st::acc().tile_assign);
        ta = gsplat_tt::tile_assign_tt(
            /*means_2d=*/nullptr, /*radii=*/nullptr, M, image_height,
            image_width, tile_size, /*covs_2d=*/nullptr, /*opacities=*/nullptr,
            contrib_floor, &ta_ok);
    }
    if (!ta_ok) {
        throw std::runtime_error(
            "render_clean: device tile_assign failed; single-path TT, no CPU "
            "fallback");
    }

    // Hand the microblock-cull params to the sort driver via device_state so the
    // fused SFPU cull pass can bake the keep mask into the dense record (these
    // blend args never reach sort_and_bin_tt directly).
    gsplat_tt::device_state::set_bucket_cull_params(mb_contrib_floor,
                                                    cull_disabled);

    // Stages 3+4+5: sort, then the in-sort continuation runs the SFPU cull and
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

    // The whole call is timed into `sort`, then the fused continuation's own
    // buckets (blend_setup/cull/blend/d2h/assemble, booked inside blend_device)
    // are subtracted back out so `sort` is the sort work alone.
    const st::Acc fused_before = st::acc();
    gsplat_cpu::SortResult sr;
    gsplat_tt::SortCallTimings sort_t;
    {
        st::Span s(st::acc().sort);
        sr = gsplat_tt::sort_and_bin_tt(
            ta.gaussian_ids.data(), ta.tile_ids.data(), proj.depths.data(),
            ta.gaussian_ids.size(), M, tiles_x, tiles_y, &worker_pool(),
            &sort_ok,
            &sort_t, /*need_host_sorted_ids=*/false, &sort_blend);
    }
    {
        st::Acc& a = st::acc();
        a.sort -= (a.blend_setup - fused_before.blend_setup) +
                  (a.mat - fused_before.mat) +
                  (a.cull - fused_before.cull) +
                  (a.blend - fused_before.blend) +
                  (a.d2h - fused_before.d2h) +
                  (a.assemble - fused_before.assemble);
        // The sort driver's own leaf spans (SortCallTimings) as sort_* buckets.
        a.sort_pread += sort_t.pread_ms;
        a.sort_bin_count += sort_t.bin_count_ms;
        a.sort_bin_hist_d2h += sort_t.bin_hist_d2h_ms;
        a.sort_bin_layout += sort_t.bin_layout_ms;
        a.sort_upload += sort_t.upload_ms;
        a.sort_bin_emit += sort_t.bin_emit_ms;
        a.sort_kernel += sort_t.kernel_ms;
        a.sort_d2h += sort_t.d2h_ms;
        a.sort_compact += sort_t.compact_ms;
        a.sort_publish_host += sort_t.publish_host_ms;
        a.sort_publish_wait += sort_t.publish_wait_ms;
        a.sort_mat += sort_t.materialize_ms;
    }
    st::Span tail_span(st::acc().tail);
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
    gsplat_tt::hostprof::on_view_return();
    st::acc().views++;
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
    // Per-stage host attribution. Totals in ms accumulated over every
    // render_view since reset_stage_timings(); `views` is the sample count.
    m.def("stage_timings", []() {
        const auto& a = gsplat_tt::stagetimers::acc();
        py::dict d;
        d["views"] = static_cast<int64_t>(a.views);
        d["head"] = a.head;
        d["project"] = a.project;
        d["tile_assign"] = a.tile_assign;
        d["sort"] = a.sort;
        d["blend_setup"] = a.blend_setup;
        d["mat"] = a.mat;
        d["cull"] = a.cull;
        d["blend"] = a.blend;
        d["d2h"] = a.d2h;
        d["assemble"] = a.assemble;
        d["tail"] = a.tail;
        d["view_total"] = a.view_total;
        d["sort_pread"] = a.sort_pread;
        d["sort_bin_count"] = a.sort_bin_count;
        d["sort_bin_hist_d2h"] = a.sort_bin_hist_d2h;
        d["sort_bin_layout"] = a.sort_bin_layout;
        d["sort_upload"] = a.sort_upload;
        d["sort_bin_emit"] = a.sort_bin_emit;
        d["sort_kernel"] = a.sort_kernel;
        d["sort_d2h"] = a.sort_d2h;
        d["sort_compact"] = a.sort_compact;
        d["sort_publish_host"] = a.sort_publish_host;
        d["sort_publish_wait"] = a.sort_publish_wait;
        d["sort_mat"] = a.sort_mat;
        d["project_cov3d"] = a.project_cov3d;
        d["project_pfwc_setup"] = a.project_pfwc_setup;
        d["project_pfwc_rtargs"] = a.project_pfwc_rtargs;
        d["project_pfwc_chunkcull"] = a.project_pfwc_chunkcull;
        d["project_pfwc_enqueue"] = a.project_pfwc_enqueue;
        d["project_pfwc_finish"] = a.project_pfwc_finish;
        d["project_gather_setup"] = a.project_gather_setup;
        d["project_gather_rtargs"] = a.project_gather_rtargs;
        d["project_gather_enqueue"] = a.project_gather_enqueue;
        d["project_gather_wait"] = a.project_gather_wait;
        d["project_gather_result"] = a.project_gather_result;
        d["tile_assign_setup"] = a.tile_assign_setup;
        d["tile_assign_rtargs"] = a.tile_assign_rtargs;
        d["tile_assign_enqueue"] = a.tile_assign_enqueue;
        d["tile_assign_scan_finish"] = a.tile_assign_scan_finish;
        d["tile_assign_p_d2h"] = a.tile_assign_p_d2h;
        d["tile_assign_k2_finish"] = a.tile_assign_k2_finish;
        d["tile_assign_publish"] = a.tile_assign_publish;
        return d;
    });
    m.def("reset_stage_timings", []() { gsplat_tt::stagetimers::reset(); });
}
