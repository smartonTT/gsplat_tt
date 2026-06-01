#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gsplat_cpu/blend.h"
#include "gsplat_cpu/blend_microblock.h"
#include "gsplat_cpu/cull_and_blend.h"
#include "gsplat_cpu/microblock_cull.h"
#include "gsplat_cpu/project.h"
#include "gsplat_cpu/sort.h"
#include "gsplat_cpu/thread_pool.h"
#include "gsplat_cpu/tile_assign.h"

#ifdef GSPLAT_WITH_TT
#include "gsplat_tt/blend.h"
#include "gsplat_tt/device_state.h"
#include "gsplat_tt/gather_visible.h"
#include "gsplat_tt/pfwc.h"
#include "gsplat_tt/project.h"
#include "gsplat_tt/render_blend.h"
#include "gsplat_tt/sort.h"
#include "gsplat_tt/tile_assign.h"

#endif

namespace py = pybind11;

namespace {

// Forward declarations of the shared pool getter (iter-030: collapsed
// from 5 per-stage pools to one global pool; threads stay warm across
// stages -> fewer cv wake/sleep context switches per frame).
gsplat_cpu::ThreadPool& global_pool();

// Stage-specific aliases retained for legacy bindings; all resolve to the
// same shared pool.
inline gsplat_cpu::ThreadPool& global_blend_pool()       { return global_pool(); }
inline gsplat_cpu::ThreadPool& global_cull_pool()        { return global_pool(); }
inline gsplat_cpu::ThreadPool& global_sort_pool()        { return global_pool(); }
inline gsplat_cpu::ThreadPool& global_tile_assign_pool() { return global_pool(); }
inline gsplat_cpu::ThreadPool& global_project_pool()     { return global_pool(); }

py::tuple pack_project_result(const gsplat_cpu::ProjectResult& result, std::size_t N) {
    const std::size_t M = result.depths.size();

    py::array_t<float> means_2d({static_cast<py::ssize_t>(M), static_cast<py::ssize_t>(2)});
    py::array_t<float> covs_2d(
        {static_cast<py::ssize_t>(M), static_cast<py::ssize_t>(2), static_cast<py::ssize_t>(2)});
    py::array_t<float> depths(static_cast<py::ssize_t>(M));
    py::array_t<float> radii({static_cast<py::ssize_t>(M), static_cast<py::ssize_t>(2)});
    py::array_t<bool> valid_mask(static_cast<py::ssize_t>(N));

    if (M > 0) {
        std::memcpy(means_2d.mutable_data(), result.means_2d.data(), M * 2 * sizeof(float));
        std::memcpy(covs_2d.mutable_data(), result.covs_2d.data(), M * 4 * sizeof(float));
        std::memcpy(depths.mutable_data(), result.depths.data(), M * sizeof(float));
        std::memcpy(radii.mutable_data(), result.radii.data(), M * 2 * sizeof(float));
    }

    auto valid_mut = valid_mask.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(N); ++i) {
        valid_mut(i) = result.valid_mask[static_cast<std::size_t>(i)] != 0;
    }

    return py::make_tuple(means_2d, covs_2d, depths, radii, valid_mask);
}

py::tuple project_finalize_py(
    const gsplat_cpu::ProjectPrepared& prep,
    py::array_t<float, py::array::c_style | py::array::forcecast> covs_2d,
    py::object opacities_obj = py::none(),
    float min_opacity = 1.0f / 255.0f) {
    const auto cov_info = covs_2d.request();
    if (cov_info.ndim != 2 || cov_info.shape[1] != 4 ||
        static_cast<std::size_t>(cov_info.shape[0]) != prep.N) {
        throw std::invalid_argument("covs_2d must have shape (N, 4)");
    }

    const float* opacities_ptr = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities;
    if (!opacities_obj.is_none()) {
        opacities = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
            opacities_obj);
        const auto op_info = opacities.request();
        if (op_info.ndim != 1 || static_cast<std::size_t>(op_info.shape[0]) != prep.N) {
            throw std::invalid_argument("opacities must have shape (N,)");
        }
        opacities_ptr = static_cast<const float*>(op_info.ptr);
    }

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project_finalize_parallel(
        prep, static_cast<const float*>(cov_info.ptr), opacities_ptr, min_opacity,
        &global_project_pool());
    return pack_project_result(result, prep.N);
}

gsplat_cpu::ProjectPrepared project_prepare_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
    py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
    int image_height,
    int image_width) {
    const auto means_info = means.request();
    const std::size_t N = static_cast<std::size_t>(means_info.shape[0]);
    return gsplat_cpu::project_prepare(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(scales.request().ptr),
        static_cast<const float*>(rotations.request().ptr),
        static_cast<const float*>(extrinsics.request().ptr),
        static_cast<const float*>(intrinsics.request().ptr),
        N,
        image_height,
        image_width);
}

py::tuple project_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
    py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
    int image_height,
    int image_width,
    py::object opacities_obj = py::none(),
    float min_opacity = 1.0f / 255.0f) {
    const std::size_t N = static_cast<std::size_t>(means.request().shape[0]);

    const float* opacities_ptr = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities;
    if (!opacities_obj.is_none()) {
        opacities = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
            opacities_obj);
        opacities_ptr = static_cast<const float*>(opacities.request().ptr);
    }

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project(
        static_cast<const float*>(means.request().ptr),
        static_cast<const float*>(scales.request().ptr),
        static_cast<const float*>(rotations.request().ptr),
        static_cast<const float*>(extrinsics.request().ptr),
        static_cast<const float*>(intrinsics.request().ptr),
        N,
        image_height,
        image_width,
        opacities_ptr,
        min_opacity);

    return pack_project_result(result, N);
}

py::array_t<float> compute_cov3d_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotations) {
    const auto sc_info = scales.request();
    const std::size_t N = static_cast<std::size_t>(sc_info.shape[0]);
    py::array_t<float> out({static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(3),
                            static_cast<py::ssize_t>(3)});
    if (N > 0) {
        gsplat_cpu::compute_cov3d_batch(
            static_cast<const float*>(sc_info.ptr),
            static_cast<const float*>(rotations.request().ptr),
            N,
            out.mutable_data());
    }
    return out;
}

py::tuple project_full_with_cov3d_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means,
    py::array_t<float, py::array::c_style | py::array::forcecast> cov3d,
    py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
    py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
    int image_height,
    int image_width,
    py::object opacities_obj = py::none(),
    float min_opacity = 1.0f / 255.0f,
    py::object means_cam_obj = py::none()) {
    // iter-022: fused per-Gaussian project_full. Replaces the prior 5-pool.wait()
    // sub-stage chain (geometry -> cov_cam -> cov2d -> valid/radii ->
    // gather count/scatter) with one big parallel pass over N plus the
    // gather count+scatter (3 waits total). cov_cam is held in registers
    // only — never materialised to memory (saves ~21.6 MB of memory
    // traffic per frame at N=601k).
    //
    // amendment-002 tt-005: optional means_cam (N, 3) fp32 lets callers
    // (e.g. TtBackend) supply a device-precomputed R@mean array, skipping
    // the inline host matmul. Translation +t is still applied inside
    // project_full_fused, so the precomp must NOT include translation.
    const auto means_info = means.request();
    const std::size_t N = static_cast<std::size_t>(means_info.shape[0]);

    const float* opacities_ptr = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities;
    if (!opacities_obj.is_none()) {
        opacities = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
            opacities_obj);
        if (static_cast<std::size_t>(opacities.request().shape[0]) != N) {
            throw std::invalid_argument("opacities must have shape (N,)");
        }
        opacities_ptr = static_cast<const float*>(opacities.request().ptr);
    }

    const float* means_cam_ptr = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> means_cam_arr;
    if (!means_cam_obj.is_none()) {
        means_cam_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
            means_cam_obj);
        const auto mc_info = means_cam_arr.request();
        if (mc_info.ndim != 2 || mc_info.shape[1] != 3 ||
            static_cast<std::size_t>(mc_info.shape[0]) != N) {
            throw std::invalid_argument("means_cam must have shape (N, 3)");
        }
        means_cam_ptr = static_cast<const float*>(mc_info.ptr);
    }

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project_full_fused(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(cov3d.request().ptr),
        static_cast<const float*>(extrinsics.request().ptr),
        static_cast<const float*>(intrinsics.request().ptr),
        nullptr,
        opacities_ptr,
        min_opacity,
        N,
        image_height,
        image_width,
        &global_project_pool(),
        0,
        1.0f / 16384.0f,
        3.0f,
        false,
        means_cam_ptr);

    return pack_project_result(result, N);
}

// amendment-002 tt-008a host finisher.
//
// Consumes the (mean_2d, depth, cov_cam_unique) the TT pfwc kernel produced
// on the Tenstorrent device and runs the residual cov2d + radii +
// valid_mask + compact-gather work on host. Matches the math of
// project_full_with_cov3d when given the same input (the device produced
// mean_2d / depth / cov_cam from the same R/t and cov3d).
py::tuple project_finish_with_cov_cam_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> mean_2d_precomp,
    py::array_t<float, py::array::c_style | py::array::forcecast> depth_precomp,
    py::array_t<float, py::array::c_style | py::array::forcecast> cov_cam_precomp,
    py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
    py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
    int image_height,
    int image_width,
    py::object opacities_obj = py::none(),
    float min_opacity = 1.0f / 255.0f) {
    const auto m_info = mean_2d_precomp.request();
    const auto d_info = depth_precomp.request();
    const auto c_info = cov_cam_precomp.request();
    if (m_info.ndim != 2 || m_info.shape[1] != 2) {
        throw std::invalid_argument("mean_2d_precomp must have shape (N, 2)");
    }
    if (d_info.ndim != 1) {
        throw std::invalid_argument("depth_precomp must have shape (N,)");
    }
    if (c_info.ndim != 2 || c_info.shape[1] != 6) {
        throw std::invalid_argument("cov_cam_precomp must have shape (N, 6)");
    }
    const std::size_t N = static_cast<std::size_t>(m_info.shape[0]);
    if (static_cast<std::size_t>(d_info.shape[0]) != N ||
        static_cast<std::size_t>(c_info.shape[0]) != N) {
        throw std::invalid_argument("mean_2d / depth / cov_cam_precomp N mismatch");
    }
    const float* opacities_ptr = nullptr;
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities;
    if (!opacities_obj.is_none()) {
        opacities = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
            opacities_obj);
        if (static_cast<std::size_t>(opacities.request().shape[0]) != N) {
            throw std::invalid_argument("opacities must have shape (N,)");
        }
        opacities_ptr = static_cast<const float*>(opacities.request().ptr);
    }
    const gsplat_cpu::ProjectResult result = gsplat_cpu::project_finish_with_cov_cam(
        static_cast<const float*>(m_info.ptr),
        static_cast<const float*>(d_info.ptr),
        static_cast<const float*>(c_info.ptr),
        static_cast<const float*>(extrinsics.request().ptr),
        static_cast<const float*>(intrinsics.request().ptr),
        opacities_ptr,
        min_opacity,
        N,
        image_height,
        image_width,
        &global_project_pool(),
        /*max_radius=*/0,
        /*contrib_floor_k=*/1.0f / 16384.0f,
        /*k_cap=*/3.0f,
        /*use_isoellipse=*/false);
    return pack_project_result(result, N);
}

py::tuple project_full_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means,
    py::array_t<float, py::array::c_style | py::array::forcecast> scales,
    py::array_t<float, py::array::c_style | py::array::forcecast> rotations,
    py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
    py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
    int image_height,
    int image_width,
    py::object opacities_obj = py::none(),
    float min_opacity = 1.0f / 255.0f) {
    const auto means_info = means.request();
    const std::size_t N = static_cast<std::size_t>(means_info.shape[0]);
    const gsplat_cpu::ProjectPrepared prep = gsplat_cpu::project_prepare(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(scales.request().ptr),
        static_cast<const float*>(rotations.request().ptr),
        static_cast<const float*>(extrinsics.request().ptr),
        static_cast<const float*>(intrinsics.request().ptr),
        N,
        image_height,
        image_width);

    const py::ssize_t n = static_cast<py::ssize_t>(prep.N);
    py::array_t<float> covs_flat({n, static_cast<py::ssize_t>(4)});
    gsplat_cpu::compute_covs_2d(prep, covs_flat.mutable_data(), &global_project_pool());

    return project_finalize_py(prep, covs_flat, opacities_obj, min_opacity);
}

py::tuple pack_tile_assign_result(const gsplat_cpu::TileAssignResult& result, std::size_t M) {
    const std::size_t P = result.gaussian_ids.size();

    py::array_t<int64_t> gaussian_ids(static_cast<py::ssize_t>(P));
    py::array_t<int64_t> tile_ids(static_cast<py::ssize_t>(P));
    py::array_t<int64_t> tiles_per_gaussian(static_cast<py::ssize_t>(M));

    if (P > 0) {
        std::memcpy(gaussian_ids.mutable_data(), result.gaussian_ids.data(), P * sizeof(int64_t));
        std::memcpy(tile_ids.mutable_data(), result.tile_ids.data(), P * sizeof(int64_t));
    }
    if (M > 0) {
        std::memcpy(tiles_per_gaussian.mutable_data(), result.tiles_per_gaussian.data(),
                    M * sizeof(int64_t));
    }

    return py::make_tuple(gaussian_ids, tile_ids, tiles_per_gaussian);
}

py::tuple tile_assign_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> radii,
    int image_height,
    int image_width,
    int tile_size,
    py::object covs_2d_obj = py::none(),
    py::object opacities_obj = py::none(),
    float contrib_floor = 15.0f / 255.0f) {
    const auto means_info = means_2d.request();
    const auto radii_info = radii.request();
    if (means_info.ndim != 2 || means_info.shape[1] != 2) {
        throw std::invalid_argument("means_2d must have shape (M, 2)");
    }
    if (radii_info.ndim != 2 || radii_info.shape[1] != 2 ||
        radii_info.shape[0] != means_info.shape[0]) {
        throw std::invalid_argument("radii must have shape (M, 2)");
    }

    const std::size_t M = static_cast<std::size_t>(means_info.shape[0]);

    const float* covs_ptr = nullptr;
    std::vector<float> covs_flat;
    if (!covs_2d_obj.is_none()) {
        auto covs_2d = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
            covs_2d_obj);
        const auto cov_info = covs_2d.request();
        if (cov_info.ndim == 3 && cov_info.shape[0] == static_cast<py::ssize_t>(M) &&
            cov_info.shape[1] == 2 && cov_info.shape[2] == 2) {
            covs_flat.resize(M * 4);
            const float* src = static_cast<const float*>(cov_info.ptr);
            for (std::size_t i = 0; i < M; ++i) {
                covs_flat[i * 4 + 0] = src[i * 4 + 0];
                covs_flat[i * 4 + 1] = src[i * 4 + 1];
                covs_flat[i * 4 + 2] = src[i * 4 + 1];
                covs_flat[i * 4 + 3] = src[i * 4 + 3];
            }
            covs_ptr = covs_flat.data();
        } else if (cov_info.ndim == 2 && cov_info.shape[0] == static_cast<py::ssize_t>(M) &&
                   cov_info.shape[1] == 4) {
            covs_ptr = static_cast<const float*>(cov_info.ptr);
        } else {
            throw std::invalid_argument("covs_2d must have shape (M, 2, 2) or (M, 4)");
        }
    }

    const float* opacities_ptr = nullptr;
    if (!opacities_obj.is_none()) {
        auto opacities = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(
            opacities_obj);
        const auto op_info = opacities.request();
        if (op_info.ndim != 1 || op_info.shape[0] != static_cast<py::ssize_t>(M)) {
            throw std::invalid_argument("opacities must have shape (M,)");
        }
        opacities_ptr = static_cast<const float*>(op_info.ptr);
    }

    const gsplat_cpu::TileAssignResult result = gsplat_cpu::tile_assign(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(radii_info.ptr),
        M,
        image_height,
        image_width,
        tile_size,
        covs_ptr,
        opacities_ptr,
        contrib_floor,
        &global_tile_assign_pool());

    return pack_tile_assign_result(result, M);
}

py::tuple pack_sort_result(const gsplat_cpu::SortResult& result, int tiles_x, int tiles_y) {
    const std::size_t P = result.sorted_gaussian_ids.size();
    const std::size_t num_tiles = static_cast<std::size_t>(tiles_x * tiles_y);

    py::array_t<int64_t> sorted_gaussian_ids(static_cast<py::ssize_t>(P));
    py::array_t<int64_t> tile_ranges(
        {static_cast<py::ssize_t>(num_tiles), static_cast<py::ssize_t>(2)});

    if (P > 0) {
        std::memcpy(sorted_gaussian_ids.mutable_data(), result.sorted_gaussian_ids.data(),
                    P * sizeof(int64_t));
    }
    if (num_tiles > 0) {
        std::memcpy(tile_ranges.mutable_data(), result.tile_ranges.data(),
                    num_tiles * 2 * sizeof(int64_t));
    }

    return py::make_tuple(sorted_gaussian_ids, tile_ranges);
}

py::tuple sort_py(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> gaussian_ids,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tile_ids,
    py::array_t<float, py::array::c_style | py::array::forcecast> depths,
    int tiles_x,
    int tiles_y) {
    const auto gids_info = gaussian_ids.request();
    const auto tids_info = tile_ids.request();
    const auto depths_info = depths.request();

    if (gids_info.ndim != 1 || tids_info.ndim != 1 ||
        gids_info.shape[0] != tids_info.shape[0]) {
        throw std::invalid_argument("gaussian_ids and tile_ids must be 1-D arrays of equal length");
    }
    if (depths_info.ndim != 1) {
        throw std::invalid_argument("depths must have shape (M,)");
    }

    const std::size_t P = static_cast<std::size_t>(gids_info.shape[0]);
    const std::size_t M = static_cast<std::size_t>(depths_info.shape[0]);

    const gsplat_cpu::SortResult result = gsplat_cpu::sort_and_bin(
        static_cast<const int64_t*>(gids_info.ptr),
        static_cast<const int64_t*>(tids_info.ptr),
        static_cast<const float*>(depths_info.ptr),
        P,
        M,
        tiles_x,
        tiles_y,
        &global_sort_pool());

    return pack_sort_result(result, tiles_x, tiles_y);
}

std::size_t resolve_pool_size() {
    if (const char* env = std::getenv("GSPLAT_TT_NUM_THREADS")) {
        try {
            const long v = std::stol(env);
            if (v > 0) return static_cast<std::size_t>(v);
        } catch (...) {}
    }
    return 0;
}

gsplat_cpu::ThreadPool& global_pool() {
    static gsplat_cpu::ThreadPool pool(resolve_pool_size());
    return pool;
}

py::array_t<float> blend_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> covs_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sorted_gaussian_ids,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tile_ranges,
    int image_height,
    int image_width,
    int tile_size) {
    const auto means_info = means_2d.request();
    const auto covs_info = covs_2d.request();
    const auto colors_info = colors.request();
    const auto opacities_info = opacities.request();
    const auto sgids_info = sorted_gaussian_ids.request();
    const auto ranges_info = tile_ranges.request();

    if (means_info.ndim != 2 || means_info.shape[1] != 2) {
        throw std::invalid_argument("means_2d must have shape (M, 2)");
    }
    if (covs_info.ndim != 2 || covs_info.shape[1] != 4) {
        throw std::invalid_argument("covs_2d must have shape (M, 4)");
    }
    if (colors_info.ndim != 2 || colors_info.shape[1] != 3) {
        throw std::invalid_argument("colors must have shape (M, 3)");
    }
    if (opacities_info.ndim != 1) {
        throw std::invalid_argument("opacities must have shape (M,)");
    }
    if (sgids_info.ndim != 1) {
        throw std::invalid_argument("sorted_gaussian_ids must be 1-D");
    }
    if (ranges_info.ndim != 2 || ranges_info.shape[1] != 2) {
        throw std::invalid_argument("tile_ranges must have shape (num_tiles, 2)");
    }

    const std::size_t M = static_cast<std::size_t>(means_info.shape[0]);
    if (static_cast<std::size_t>(covs_info.shape[0]) != M ||
        static_cast<std::size_t>(colors_info.shape[0]) != M ||
        static_cast<std::size_t>(opacities_info.shape[0]) != M) {
        throw std::invalid_argument("means_2d, covs_2d, colors, opacities must share M");
    }

    const std::size_t P = static_cast<std::size_t>(sgids_info.shape[0]);

    py::array_t<float> image(
        {static_cast<py::ssize_t>(image_height), static_cast<py::ssize_t>(image_width),
         static_cast<py::ssize_t>(3)});

    gsplat_cpu::blend(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(covs_info.ptr),
        static_cast<const float*>(colors_info.ptr),
        static_cast<const float*>(opacities_info.ptr),
        static_cast<const int64_t*>(sgids_info.ptr),
        static_cast<const int64_t*>(ranges_info.ptr),
        M,
        P,
        image_height,
        image_width,
        tile_size,
        image.mutable_data(),
        global_blend_pool());

    return image;
}

py::array_t<float> blend_microblock_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> covs_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> mb_header,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> mb_stream,
    int image_height,
    int image_width,
    int tile_size) {
    const auto means_info = means_2d.request();
    const auto covs_info = covs_2d.request();
    const auto colors_info = colors.request();
    const auto opacities_info = opacities.request();
    const auto header_info = mb_header.request();
    const auto stream_info = mb_stream.request();

    if (means_info.ndim != 2 || means_info.shape[1] != 2) {
        throw std::invalid_argument("means_2d must have shape (M, 2)");
    }
    if (covs_info.ndim != 2 || covs_info.shape[1] != 4) {
        throw std::invalid_argument("covs_2d must have shape (M, 4)");
    }
    if (colors_info.ndim != 2 || colors_info.shape[1] != 3) {
        throw std::invalid_argument("colors must have shape (M, 3)");
    }
    if (opacities_info.ndim != 1) {
        throw std::invalid_argument("opacities must have shape (M,)");
    }
    if (header_info.ndim != 1) {
        throw std::invalid_argument("mb_header must be 1-D flat array");
    }
    if (stream_info.ndim != 1) {
        throw std::invalid_argument("mb_stream must be 1-D");
    }

    const std::size_t M = static_cast<std::size_t>(means_info.shape[0]);
    if (static_cast<std::size_t>(covs_info.shape[0]) != M ||
        static_cast<std::size_t>(colors_info.shape[0]) != M ||
        static_cast<std::size_t>(opacities_info.shape[0]) != M) {
        throw std::invalid_argument("means_2d, covs_2d, colors, opacities must share M");
    }

    const int tiles_x = (image_width + tile_size - 1) / tile_size;
    const int tiles_y = (image_height + tile_size - 1) / tile_size;
    const std::size_t expected_header =
        static_cast<std::size_t>(tiles_x * tiles_y * 32 * 2);
    if (static_cast<std::size_t>(header_info.shape[0]) != expected_header) {
        throw std::invalid_argument("mb_header length must equal num_tiles * 32 * 2");
    }

    const std::size_t L_prime = static_cast<std::size_t>(stream_info.shape[0]);

    const gsplat_cpu::BlendResult result = gsplat_cpu::blend_microblock(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(covs_info.ptr),
        static_cast<const float*>(colors_info.ptr),
        static_cast<const float*>(opacities_info.ptr),
        static_cast<const int64_t*>(header_info.ptr),
        static_cast<const int64_t*>(stream_info.ptr),
        M,
        L_prime,
        image_height,
        image_width,
        tile_size,
        global_blend_pool());

    py::array_t<float> image(
        {static_cast<py::ssize_t>(image_height), static_cast<py::ssize_t>(image_width),
         static_cast<py::ssize_t>(3)});
    if (!result.image.empty()) {
        std::memcpy(image.mutable_data(), result.image.data(),
                    result.image.size() * sizeof(float));
    }
    return image;
}

py::tuple cull_and_blend_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> covs_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sorted_gaussian_ids,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tile_ranges,
    int tiles_x,
    int tiles_y,
    int tile_size,
    int image_height,
    int image_width,
    float mb_contrib_floor);

py::tuple microblock_cull_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> covs_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sorted_gaussian_ids,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tile_ranges,
    int tiles_x,
    int tiles_y,
    int tile_size,
    float mb_contrib_floor) {
    const auto means_info = means_2d.request();
    const auto covs_info = covs_2d.request();
    const auto opacities_info = opacities.request();
    const auto sgids_info = sorted_gaussian_ids.request();
    const auto ranges_info = tile_ranges.request();

    if (means_info.ndim != 2 || means_info.shape[1] != 2) {
        throw std::invalid_argument("means_2d must have shape (M, 2)");
    }
    if (covs_info.ndim != 2 || covs_info.shape[1] != 4) {
        throw std::invalid_argument("covs_2d must have shape (M, 4)");
    }
    if (opacities_info.ndim != 1) {
        throw std::invalid_argument("opacities must have shape (M,)");
    }
    if (sgids_info.ndim != 1) {
        throw std::invalid_argument("sorted_gaussian_ids must be 1-D");
    }
    if (ranges_info.ndim != 2 || ranges_info.shape[1] != 2) {
        throw std::invalid_argument("tile_ranges must have shape (num_tiles, 2)");
    }

    const std::size_t M = static_cast<std::size_t>(means_info.shape[0]);
    if (static_cast<std::size_t>(covs_info.shape[0]) != M ||
        static_cast<std::size_t>(opacities_info.shape[0]) != M) {
        throw std::invalid_argument("means_2d, covs_2d, opacities must share M");
    }

    const std::size_t P = static_cast<std::size_t>(sgids_info.shape[0]);
    const int num_tiles = tiles_x * tiles_y;
    if (ranges_info.shape[0] != num_tiles) {
        throw std::invalid_argument("tile_ranges first dimension must equal tiles_x * tiles_y");
    }

    const gsplat_cpu::MicroblockCullResult result = gsplat_cpu::microblock_cull(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(covs_info.ptr),
        static_cast<const float*>(opacities_info.ptr),
        static_cast<const int64_t*>(sgids_info.ptr),
        static_cast<const int64_t*>(ranges_info.ptr),
        M,
        P,
        tiles_x,
        tiles_y,
        tile_size,
        mb_contrib_floor,
        global_cull_pool());

    py::array_t<int64_t> mb_header(static_cast<py::ssize_t>(result.mb_header.size()));
    py::array_t<int64_t> mb_stream(static_cast<py::ssize_t>(result.mb_stream.size()));
    if (!result.mb_header.empty()) {
        std::memcpy(mb_header.mutable_data(), result.mb_header.data(),
                    result.mb_header.size() * sizeof(int64_t));
    }
    if (!result.mb_stream.empty()) {
        std::memcpy(mb_stream.mutable_data(), result.mb_stream.data(),
                    result.mb_stream.size() * sizeof(int64_t));
    }

    const double drop_pct =
        result.pairs_in == 0
            ? 0.0
            : 100.0 * static_cast<double>(result.pairs_dropped_in_all_mb) /
                  static_cast<double>(result.pairs_in);
    constexpr int kNumMicroblocks = 32;
    const double full_replay =
        static_cast<double>(result.pairs_in) * static_cast<double>(kNumMicroblocks);
    const double work_reduction_pct =
        full_replay == 0.0
            ? 0.0
            : 100.0 * (1.0 - static_cast<double>(result.pairs_out) / full_replay);

    py::dict stats;
    stats["pairs_in"] = result.pairs_in;
    stats["pairs_out"] = result.pairs_out;
    stats["drop_pct"] = drop_pct;
    stats["work_reduction_pct"] = work_reduction_pct;

    return py::make_tuple(mb_header, mb_stream, stats);
}

py::tuple cull_and_blend_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> covs_2d,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sorted_gaussian_ids,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> tile_ranges,
    int tiles_x,
    int tiles_y,
    int tile_size,
    int image_height,
    int image_width,
    float mb_contrib_floor) {
    const auto means_info = means_2d.request();
    const auto covs_info = covs_2d.request();
    const auto colors_info = colors.request();
    const auto opacities_info = opacities.request();
    const auto sg_info = sorted_gaussian_ids.request();
    const auto tr_info = tile_ranges.request();

    if (means_info.ndim != 2 || means_info.shape[1] != 2) {
        throw std::invalid_argument("means_2d must have shape (M, 2)");
    }
    if (covs_info.ndim != 2 || covs_info.shape[1] != 4) {
        throw std::invalid_argument("covs_2d must have shape (M, 4)");
    }
    if (colors_info.ndim != 2 || colors_info.shape[1] != 3) {
        throw std::invalid_argument("colors must have shape (M, 3)");
    }
    if (opacities_info.ndim != 1) {
        throw std::invalid_argument("opacities must have shape (M,)");
    }

    const std::size_t M = static_cast<std::size_t>(means_info.shape[0]);
    const std::size_t P = static_cast<std::size_t>(sg_info.shape[0]);

    const gsplat_cpu::CullAndBlendResult result = gsplat_cpu::cull_and_blend(
        static_cast<const float*>(means_info.ptr),
        static_cast<const float*>(covs_info.ptr),
        static_cast<const float*>(colors_info.ptr),
        static_cast<const float*>(opacities_info.ptr),
        static_cast<const int64_t*>(sg_info.ptr),
        static_cast<const int64_t*>(tr_info.ptr),
        M, P,
        tiles_x, tiles_y, tile_size,
        image_height, image_width,
        mb_contrib_floor,
        global_blend_pool());

    py::array_t<float> image(
        {static_cast<py::ssize_t>(image_height),
         static_cast<py::ssize_t>(image_width),
         static_cast<py::ssize_t>(3)});
    if (!result.image.empty()) {
        std::memcpy(image.mutable_data(), result.image.data(),
                    result.image.size() * sizeof(float));
    }

    py::dict stats;
    stats["pairs_in"] = result.pairs_in;
    stats["pairs_dropped"] = result.pairs_dropped_all_mb;
    stats["pairs_kept_per_mb"] = result.pairs_kept_per_mb;
    return py::make_tuple(image, stats);
}

// Filter colors / opacities by valid_mask. Parallel-filter pattern (chunked
// count + prefix-sum + parallel scatter), bit-identical to numpy
// colors[valid_mask] / opacities[valid_mask] but stays inside C++.
struct VisibleFiltered {
    std::vector<float> colors;     // M*3
    std::vector<float> opacities;  // M
};

VisibleFiltered filter_visible(
    const float* colors_n,   // N*3
    const float* opacities_n,// N
    const uint8_t* valid_mask,// N (0/1)
    std::size_t N,
    std::size_t M,
    gsplat_cpu::ThreadPool& pool) {
    VisibleFiltered out;
    out.colors.assign(M * 3, 0.0f);
    out.opacities.assign(M, 0.0f);
    if (N == 0 || M == 0) return out;

    const std::size_t W = std::max<std::size_t>(1, pool.size());
    const std::size_t chunk_size = std::max<std::size_t>(1024, (N + W - 1) / W);
    const std::size_t num_chunks = (N + chunk_size - 1) / chunk_size;
    std::vector<std::size_t> chunk_counts(num_chunks, 0);

    auto count_chunk = [&](std::size_t c) {
        const std::size_t lo = c * chunk_size;
        const std::size_t hi = std::min(lo + chunk_size, N);
        std::size_t k = 0;
        for (std::size_t i = lo; i < hi; ++i) k += valid_mask[i];
        chunk_counts[c] = k;
    };
    if (W > 1 && num_chunks > 1) {
        for (std::size_t c = 0; c < num_chunks; ++c) {
            pool.submit([c, &count_chunk]() { count_chunk(c); });
        }
        pool.wait();
    } else {
        for (std::size_t c = 0; c < num_chunks; ++c) count_chunk(c);
    }

    std::vector<std::size_t> chunk_offs(num_chunks + 1, 0);
    for (std::size_t c = 0; c < num_chunks; ++c) {
        chunk_offs[c + 1] = chunk_offs[c] + chunk_counts[c];
    }

    auto scatter_chunk = [&](std::size_t c) {
        const std::size_t lo = c * chunk_size;
        const std::size_t hi = std::min(lo + chunk_size, N);
        std::size_t pos = chunk_offs[c];
        for (std::size_t i = lo; i < hi; ++i) {
            if (valid_mask[i]) {
                out.colors[pos * 3 + 0] = colors_n[i * 3 + 0];
                out.colors[pos * 3 + 1] = colors_n[i * 3 + 1];
                out.colors[pos * 3 + 2] = colors_n[i * 3 + 2];
                out.opacities[pos] = opacities_n[i];
                ++pos;
            }
        }
    };
    if (W > 1 && num_chunks > 1) {
        for (std::size_t c = 0; c < num_chunks; ++c) {
            pool.submit([c, &scatter_chunk]() { scatter_chunk(c); });
        }
        pool.wait();
    } else {
        for (std::size_t c = 0; c < num_chunks; ++c) scatter_chunk(c);
    }

    return out;
}

#ifdef GSPLAT_WITH_TT
// amendment-003 M3: run the PROJECT stage on the TT device instead of the host
// CPU project_full_fused (~242ms over 6.13M gaussians). The device kernels
// (transform_means_cam_tt + pfwc_tt) already exist; this wires them into the
// fused loop and produces a ProjectResult IDENTICAL in layout to the CPU path
// so tile_assign/sort/blend are unchanged. Opt-in via GSPLAT_TT_DEVICE_PROJECT=1.
//
// Returns a ProjectResult with empty depths on failure / unsupported config, so
// the caller transparently falls back to the CPU path. The device pfwc kernel
// hardcodes the 3-sigma radius (k=3.0) and no isoellipse, so we only take this
// path when the requested config matches; otherwise we fall back.
static gsplat_cpu::ProjectResult project_via_device(
    const float* means, const float* cov3d, const float* extrinsics,
    const float* intrinsics, const float* colors, const float* opacities,
    float min_opacity, std::size_t N, int image_height, int image_width,
    float k_cap, bool use_isoellipse, int max_radius,
    gsplat_cpu::ThreadPool* pool, int blend_mode) {
    gsplat_cpu::ProjectResult empty;
    if (use_isoellipse || std::fabs(k_cap - 3.0f) > 1e-3f) {
        // pfwc_tt only implements the k=3.0, non-isoellipse radius. Signal
        // fallback rather than silently producing a different cull.
        static bool warned = false;
        if (!warned) {
            std::fprintf(stderr,
                "[DEVICE_PROJECT] config mismatch (k_cap=%.3f isoellipse=%d) -> "
                "CPU fallback\n", k_cap, (int)use_isoellipse);
            warned = true;
        }
        return empty;
    }
    if (!gsplat_tt::project_device_ready() || !gsplat_tt::pfwc_device_ready()) {
        // ready() lazily inits the device on first call; if it can't, fall back.
    }

    // Env-gated sub-component profiling (GSPLAT_TT_PROJECT_TIMING=1). Off by
    // default — pure stderr instrumentation, no effect on the render path.
    const bool proj_timing = [] {
        const char* v = std::getenv("GSPLAT_TT_PROJECT_TIMING");
        return v != nullptr && v[0] == '1';
    }();
    using prof_clock = std::chrono::steady_clock;

    // 1) means_cam = R@means + t, kept device-resident (no D2H).
    gsplat_tt::ProjectCallTimings mc_t;
    const auto t_mc0 = prof_clock::now();
    if (gsplat_tt::transform_means_cam_tt_no_download(means, extrinsics, N, &mc_t) < 0.0)
        return empty;
    const auto t_mc1 = prof_clock::now();

    // 2) cov3d N*9 -> unique N*6 [c00 c01 c02 c11 c12 c22] for pfwc.
    //
    // PERF: cov3d is VIEW-INVARIANT across the 30-view bench (the same scene
    // tensor is reused for every view), yet this SoA repack was a serial O(N)
    // host loop costing a constant ~104 ms/frame at N=6.13M — the dominant
    // *constant* sub-component of project_ms (profiled on bh-30). Cache the
    // result keyed on (cov3d pointer, N) so it runs ONCE per scene instead of
    // once per view. Bit-identical output (same source bytes, same gather
    // order). Mirrors the established pointer-keyed caches in
    // project_device.cpp (means) and pfwc_device.cpp (cov3d upload). Safe for
    // the single-threaded render loop (same pattern as the device-context
    // singletons); the cache vector is only ever read by pfwc_tt, never
    // mutated downstream.
    static std::vector<float> s_cov3d_u_cache;
    static const float* s_cov3d_u_cached_ptr = nullptr;
    static std::size_t s_cov3d_u_cached_N = 0;
    const bool cov3d_cache_hit =
        (s_cov3d_u_cached_ptr == cov3d) && (s_cov3d_u_cached_N == N) &&
        (s_cov3d_u_cache.size() == N * 6);
    if (!cov3d_cache_hit) {
        s_cov3d_u_cache.resize(N * 6);
        float* __restrict cu = s_cov3d_u_cache.data();
        for (std::size_t i = 0; i < N; ++i) {
            cu[i * 6 + 0] = cov3d[i * 9 + 0];
            cu[i * 6 + 1] = cov3d[i * 9 + 1];
            cu[i * 6 + 2] = cov3d[i * 9 + 2];
            cu[i * 6 + 3] = cov3d[i * 9 + 4];
            cu[i * 6 + 4] = cov3d[i * 9 + 5];
            cu[i * 6 + 5] = cov3d[i * 9 + 8];
        }
        s_cov3d_u_cached_ptr = cov3d;
        s_cov3d_u_cached_N = N;
    }
    const std::vector<float>& cov3d_u = s_cov3d_u_cache;
    const auto t_cov1 = prof_clock::now();

    // Residency gates (amendment-003 R1/R2a/R2b). Default (all off) keeps the
    // legacy path: pfwc D2H + host finisher.
    const bool resident_project = [] {
        const char* v = std::getenv("GSPLAT_TT_RESIDENT_PROJECT");
        return v != nullptr && v[0] == '1';
    }();
    const bool resident_gather = [] {
        const char* v = std::getenv("GSPLAT_TT_RESIDENT_GATHER");
        return v != nullptr && v[0] == '1';
    }();
    const bool gather_host = [] {
        const char* v = std::getenv("GSPLAT_TT_GATHER_HOST");
        return v != nullptr && v[0] == '1';
    }();
    const bool gather_verify = [] {
        const char* v = std::getenv("GSPLAT_TT_GATHER_VERIFY");
        return v != nullptr && v[0] == '1';
    }();

    // 3) pfwc on device -> mean_2d, depth, cov2d(a,b,c), radii.
    //    R1 (GSPLAT_TT_RESIDENT_PROJECT=1): call with all four output pointers
    //    NULL so NOTHING is D2H'd — the N-indexed pfwc_* stay resident. The
    //    default path keeps the D2H (non-null outputs).
    gsplat_tt::PfwcCallTimings pf_t;
    std::vector<float> mean_2d, depth, cov2d, radii;
    const auto t_pf0 = prof_clock::now();
    if (resident_project) {
        if (gsplat_tt::pfwc_tt(cov3d_u.data(), extrinsics, intrinsics, N,
                               nullptr, nullptr, nullptr, nullptr, &pf_t) < 0.0)
            return empty;
    } else {
        mean_2d.resize(N * 2);
        depth.resize(N);
        cov2d.resize(N * 3);
        radii.resize(N * 2);
        if (gsplat_tt::pfwc_tt(cov3d_u.data(), extrinsics, intrinsics, N,
                               mean_2d.data(), depth.data(), cov2d.data(),
                               radii.data(), &pf_t) < 0.0)
            return empty;
    }

    // 4) R2a/R2b: gather the visible M-compact outputs into device-resident
    //    DRAM. Returns a layout-identical ProjectResult assembled from the
    //    resident buffers.
    const auto t_pf1 = prof_clock::now();
    if (resident_gather) {
        bool gather_ok = false;
        gsplat_tt::GatherCallTimings g_t;
        const auto t_g0 = prof_clock::now();
        // Only authorize the M-only readback when THIS render's blend is the
        // device blend (blend_mode>=1). The cpu_cpp_mb reference render shares
        // this process + the GSPLAT_TT_DEVICE_PROJECT env but uses blend_mode=0
        // (CPU cull_and_blend), which still consumes the host proj_m_* arrays.
        const bool downstream_resident = (blend_mode >= 1);
        gsplat_cpu::ProjectResult proj = gsplat_tt::gather_visible_tt(
            colors, opacities, N, image_height, image_width, min_opacity,
            max_radius, gather_host, gather_verify, pool, &gather_ok, &g_t,
            downstream_resident);
        const auto t_g1 = prof_clock::now();
        if (proj_timing) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::fprintf(stderr,
                "[PROJECT_TIMING] N=%zu M=%zu | means_cam=%.2f (kern_launch=%.2f "
                "kern_compute=%.2f hit=%d) | cov3d_repack=%.2f (hit=%d) | "
                "pfwc=%.2f (kern_launch=%.2f kern_compute=%.2f hit=%d) | "
                "gather=%.2f (upload=%.2f kernel=%.2f readback=%.2f host=%.2f hit=%d) | "
                "TOTAL_project=%.2f\n",
                N, proj.depths.size(),
                ms(t_mc0, t_mc1), mc_t.launch_ms, mc_t.compute_ms, (int)mc_t.cache_hit,
                ms(t_mc1, t_cov1), (int)cov3d_cache_hit,
                ms(t_pf0, t_pf1), pf_t.launch_ms, pf_t.compute_ms, (int)pf_t.cache_hit,
                ms(t_g0, t_g1), g_t.upload_ms, g_t.kernel_ms, g_t.readback_ms,
                g_t.host_ms, (int)g_t.cache_hit,
                ms(t_mc0, t_g1));
        }
        if (gather_ok) return proj;
        // On gather failure fall through to the host finisher, reading the
        // resident pfwc_* back if R1 skipped the D2H.
    }

    // Legacy / fallback host finisher: valid_mask + M-compact gather (identical
    // math to project_full_fused), then compact colors/opacities in the SAME
    // increasing-i order. When R1 skipped the pfwc D2H, read the resident
    // pfwc_* back here so the host has the per-Gaussian data.
    if (mean_2d.empty()) {
        mean_2d.resize(N * 2);
        depth.resize(N);
        cov2d.resize(N * 3);
        radii.resize(N * 2);
        if (!gsplat_tt::readback_pfwc_resident(
                N, mean_2d.data(), depth.data(), cov2d.data(), radii.data())) {
            return empty;
        }
    }
    gsplat_cpu::ProjectResult proj = gsplat_cpu::project_finish_with_cov2d_radii(
        mean_2d.data(), depth.data(), cov2d.data(), radii.data(), opacities,
        min_opacity, N, image_height, image_width, pool, max_radius);
    const std::size_t M = proj.depths.size();
    proj.colors.resize(M * 3);
    proj.opacities.resize(M);
    std::size_t out = 0;
    for (std::size_t i = 0; i < N; ++i) {
        if (!proj.valid_mask[i]) continue;
        proj.colors[out * 3 + 0] = colors[i * 3 + 0];
        proj.colors[out * 3 + 1] = colors[i * 3 + 1];
        proj.colors[out * 3 + 2] = colors[i * 3 + 2];
        proj.opacities[out] = opacities[i];
        ++out;
    }
    return proj;
}
#endif  // GSPLAT_WITH_TT

// Resident device sort leaves sorted_gaussian_ids empty; kept count is the sum
// of per-tile [start,end) spans in tile_ranges.
static std::size_t sort_kept_entry_count(const gsplat_cpu::SortResult& sr) {
    std::size_t n = 0;
    for (std::size_t t = 0; 2 * t + 1 < sr.tile_ranges.size(); ++t) {
        const int64_t lo = sr.tile_ranges[2 * t + 0];
        const int64_t hi = sr.tile_ranges[2 * t + 1];
        if (hi > lo) n += static_cast<std::size_t>(hi - lo);
    }
    return n;
}

// All-stage fused render. Orchestrates project_full_fused -> filter colors/
// opacities by valid_mask -> tile_assign -> sort_and_bin -> cull_and_blend
// entirely in C++. Eliminates ~4 pybind boundary crossings + ~7 numpy<->C++
// array conversions + the Python torch indexing for colors/opacities filter.
// Returns (image, stats_dict) where stats_dict contains per-stage timings,
// num_visible, num_entries, and microblock cull stats.
py::tuple render_full_py(
    py::array_t<float, py::array::c_style | py::array::forcecast> means,
    py::array_t<float, py::array::c_style | py::array::forcecast> cov3d,
    py::array_t<float, py::array::c_style | py::array::forcecast> opacities,
    py::array_t<float, py::array::c_style | py::array::forcecast> colors,
    py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
    py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
    int image_height,
    int image_width,
    int tile_size,
    float min_opacity,
    float contrib_floor,
    float mb_contrib_floor,
    bool cull_disabled,
    float transmittance_threshold,
    int max_radius,
    float k_cap,
    bool use_isoellipse,
    // amendment-003: stage dispatch selector for the fused C++ render loop.
    // blend_mode 0 = CPU cull_and_blend (default; bit-identical to cpu_cpp_mb).
    // blend_mode 1 = TT device microblock blend (gsplat_tt). Higher stages
    // (tile_assign / sort / project on device) are added the same way.
    int blend_mode = 0) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();

    const auto means_info = means.request();
    const std::size_t N = static_cast<std::size_t>(means_info.shape[0]);

    const float* opacities_ptr = static_cast<const float*>(opacities.request().ptr);
    const float* colors_ptr = static_cast<const float*>(colors.request().ptr);

    // Stage 1: project.
    auto t_p0 = clock::now();
    gsplat_cpu::ProjectResult proj;
#ifdef GSPLAT_WITH_TT
    // M3: opt-in device project (GSPLAT_TT_DEVICE_PROJECT=1). Produces a
    // layout-identical ProjectResult; falls back to CPU on empty result
    // (device unavailable / unsupported config).
    if (const char* dp = std::getenv("GSPLAT_TT_DEVICE_PROJECT");
        dp != nullptr && dp[0] == '1') {
        proj = project_via_device(
            static_cast<const float*>(means_info.ptr),
            static_cast<const float*>(cov3d.request().ptr),
            static_cast<const float*>(extrinsics.request().ptr),
            static_cast<const float*>(intrinsics.request().ptr),
            colors_ptr, opacities_ptr, min_opacity, N, image_height, image_width,
            k_cap, use_isoellipse, max_radius, &global_project_pool(), blend_mode);
    }
#endif
    if (proj.depths.empty()) {
        proj = gsplat_cpu::project_full_fused(
            static_cast<const float*>(means_info.ptr),
            static_cast<const float*>(cov3d.request().ptr),
            static_cast<const float*>(extrinsics.request().ptr),
            static_cast<const float*>(intrinsics.request().ptr),
            colors_ptr,
            opacities_ptr,
            min_opacity,
            N,
            image_height,
            image_width,
            &global_project_pool(),
            max_radius,
            contrib_floor,
            k_cap,
            use_isoellipse);
    }
    const std::size_t M = proj.depths.size();
    auto t_p1 = clock::now();

    // Early-out: zero visible Gaussians -> return all-black image with zero stats.
    if (M == 0) {
        py::array_t<float> image_zero(
            {static_cast<py::ssize_t>(image_height),
             static_cast<py::ssize_t>(image_width),
             static_cast<py::ssize_t>(3)});
        std::memset(image_zero.mutable_data(), 0,
                    static_cast<std::size_t>(image_height) *
                        static_cast<std::size_t>(image_width) * 3 * sizeof(float));
        py::dict stats;
        stats["num_visible"] = 0;
        stats["num_entries"] = 0;
        stats["pairs_in"] = 0;
        stats["pairs_dropped"] = 0;
        stats["pairs_kept_per_mb"] = 0;
        stats["project_ms"] = std::chrono::duration<float, std::milli>(t_p1 - t_p0).count();
        stats["tile_assign_ms"] = 0.0f;
        stats["sort_ms"] = 0.0f;
        stats["blend_ms"] = 0.0f;
        stats["total_ms"] = std::chrono::duration<float, std::milli>(clock::now() - t0).count();
        return py::make_tuple(image_zero, stats);
    }

    // iter-057: colors/opacities compacted during project gather — skip the
    // separate filter_visible scan over all N Gaussians (2 parallel passes
    // + pool.wait that duplicated work already done in the gather).
    const float* vis_colors = proj.colors.empty() ? colors_ptr : proj.colors.data();
    const float* vis_opacities =
        proj.opacities.empty() ? opacities_ptr : proj.opacities.data();

#ifdef GSPLAT_WITH_TT
    // Production TT render (blend_mode>=1 + device/resident env stack): never
    // fall back to host tile_assign/sort — abort so misconfig is obvious.
    const bool tt_host_free_render = (blend_mode >= 1) && [] {
        auto on = [](const char* n) {
            const char* v = std::getenv(n);
            return v != nullptr && v[0] == '1';
        };
        auto on_nz = [](const char* n) {
            const char* v = std::getenv(n);
            return v != nullptr && v[0] != '0' && v[0] != '\0';
        };
        return on("GSPLAT_TT_DEVICE_PROJECT") && on("GSPLAT_TT_RESIDENT_GATHER") &&
               on("GSPLAT_TT_DEVICE_TILE_ASSIGN") && on_nz("GSPLAT_TT_DEVICE_SORT") &&
               on("GSPLAT_TT_RESIDENT_PAIRS") && on("GSPLAT_TT_RESIDENT_BLEND");
    }();
#else
    const bool tt_host_free_render = false;
#endif

    // Stage 2: tile_assign. When cull_disabled, skip the per-pair Mahalanobis
    // cull by passing null cov/opacity pointers — only the AABB-based tile
    // overlap stays. Otherwise the per-pair cull at contrib_floor runs.
    const int tiles_x = (image_width + tile_size - 1) / tile_size;
    const int tiles_y = (image_height + tile_size - 1) / tile_size;
    auto t_ta0 = clock::now();
    gsplat_cpu::TileAssignResult ta;
    bool ta_done = false;
#ifdef GSPLAT_WITH_TT
    // tt-006: opt-in device tile_assign (GSPLAT_TT_DEVICE_TILE_ASSIGN=1).
    // Produces a layout-identical TileAssignResult (same (gid,tid) pair set
    // and gaussian-major order); falls back to CPU on device failure. Mirrors
    // how GSPLAT_TT_DEVICE_PROJECT gates project_via_device above.
    if (const char* dt = std::getenv("GSPLAT_TT_DEVICE_TILE_ASSIGN");
        dt != nullptr && dt[0] == '1') {
        bool device_ok = false;
        const bool resident_ta = [] {
            const char* v = std::getenv("GSPLAT_TT_RESIDENT_TA_IN");
            return v != nullptr && v[0] == '1';
        }();
        gsplat_cpu::TileAssignResult ta_dev = gsplat_tt::tile_assign_tt(
            resident_ta ? nullptr : proj.means_2d.data(),
            resident_ta ? nullptr : proj.radii.data(),
            M,
            image_height,
            image_width,
            tile_size,
            cull_disabled ? nullptr : (resident_ta ? nullptr : proj.covs_2d.data()),
            cull_disabled ? nullptr : (resident_ta ? nullptr : vis_opacities),
            contrib_floor,
            &device_ok);
        if (device_ok) {
            ta = std::move(ta_dev);
            ta_done = true;
        } else if (tt_host_free_render) {
            std::fprintf(stderr,
                "[render_full] FATAL: device tile_assign failed with host-free "
                "env stack\n");
            std::abort();
        }
    }
#endif
    if (!ta_done) {
        if (tt_host_free_render) {
            std::fprintf(stderr,
                "[render_full] FATAL: GSPLAT_TT_DEVICE_TILE_ASSIGN not set or "
                "disabled on host-free path\n");
            std::abort();
        }
        ta = gsplat_cpu::tile_assign(
            proj.means_2d.data(),
            proj.radii.data(),
            M,
            image_height,
            image_width,
            tile_size,
            cull_disabled ? nullptr : proj.covs_2d.data(),
            cull_disabled ? nullptr : vis_opacities,
            contrib_floor,
            &global_tile_assign_pool(),
            /*recompute_tiles_per_gaussian=*/false);
    }
    auto t_ta1 = clock::now();

    // Stage 3: sort + bin.
    auto t_s0 = clock::now();
    gsplat_cpu::SortResult sr;
    bool sort_done = false;
#ifdef GSPLAT_WITH_TT
    // tt-003: opt-in device sort (GSPLAT_TT_DEVICE_SORT>=1). Produces a
    // layout-identical SortResult (byte-identical sorted_gaussian_ids +
    // tile_ranges) and publishes the contiguous outputs resident in
    // device_state; falls back to CPU on device failure. Mirrors how
    // GSPLAT_TT_DEVICE_TILE_ASSIGN gates tile_assign_tt above.
    if (const char* ds = std::getenv("GSPLAT_TT_DEVICE_SORT");
        ds != nullptr && ds[0] != '0' && ds[0] != '\0') {
        bool device_ok = false;
        gsplat_cpu::SortResult sr_dev = gsplat_tt::sort_and_bin_tt(
            ta.gaussian_ids.data(),
            ta.tile_ids.data(),
            proj.depths.data(),
            ta.gaussian_ids.size(),
            M,
            tiles_x,
            tiles_y,
            &global_sort_pool(),
            &device_ok,
            /*timings=*/nullptr,
            /*need_host_sorted_ids=*/blend_mode < 1);
        if (device_ok) {
            sr = std::move(sr_dev);
            sort_done = true;
        } else if (tt_host_free_render) {
            std::fprintf(stderr,
                "[render_full] FATAL: device sort failed with host-free env stack\n");
            std::abort();
        }
    }
#endif
    if (!sort_done) {
        if (tt_host_free_render) {
            std::fprintf(stderr,
                "[render_full] FATAL: GSPLAT_TT_DEVICE_SORT not set on host-free path\n");
            std::abort();
        }
        sr = gsplat_cpu::sort_and_bin(
            ta.gaussian_ids.data(),
            ta.tile_ids.data(),
            proj.depths.data(),
            ta.gaussian_ids.size(),
            M,
            tiles_x,
            tiles_y,
            &global_sort_pool());
    }
    auto t_s1 = clock::now();

#ifdef GSPLAT_WITH_TT
    const std::size_t P_kept = sort_kept_entry_count(sr);
#else
    const std::size_t P_kept = sr.sorted_gaussian_ids.size();
#endif

    // iter-049: pre-allocate + zero the pybind output buffer and thread its
    // data pointer through cull_and_blend. Eliminates a ~3 MB std::vector
    // alloc+zero inside the C++ stage AND the same-sized memcpy on the way
    // out. The kernel writes pixels (and only the kept-microblock pixels)
    // directly into the numpy buffer; remaining pixels keep the up-front
    // zero we just wrote.
    py::array_t<float> image(
        {static_cast<py::ssize_t>(image_height),
         static_cast<py::ssize_t>(image_width),
         static_cast<py::ssize_t>(3)});
    std::memset(image.mutable_data(), 0,
                static_cast<std::size_t>(image_height) *
                    static_cast<std::size_t>(image_width) * 3 * sizeof(float));

    // Stage 4: cull + blend. blend_mode selects the implementation; the
    // microblock cull always runs on CPU and produces the per-(tile,
    // microblock) kept-Gaussian structure. For blend_mode != 0 the TT device
    // kernel consumes that same structure (added in amendment-003 step 3).
    auto t_b0 = clock::now();
    gsplat_cpu::CullAndBlendResult cb;
#ifdef GSPLAT_WITH_TT
    if (blend_mode >= 1) {
        cb = gsplat_tt::render_blend_tt(
            proj.means_2d.data(),
            proj.covs_2d.data(),
            vis_colors,
            vis_opacities,
            sr.sorted_gaussian_ids.empty() ? nullptr : sr.sorted_gaussian_ids.data(),
            sr.tile_ranges.empty() ? nullptr : sr.tile_ranges.data(),
            M,
            P_kept,
            tiles_x,
            tiles_y,
            tile_size,
            image_height,
            image_width,
            mb_contrib_floor,
            global_blend_pool(),
            /*image_out_external=*/image.mutable_data(),
            /*cull_disabled=*/cull_disabled,
            /*transmittance_threshold=*/transmittance_threshold,
            /*blend_mode=*/blend_mode);
    } else
#endif
    {
        cb = gsplat_cpu::cull_and_blend(
            proj.means_2d.data(),
            proj.covs_2d.data(),
            vis_colors,
            vis_opacities,
            sr.sorted_gaussian_ids.data(),
            sr.tile_ranges.data(),
            M,
            P_kept,
            tiles_x,
            tiles_y,
            tile_size,
            image_height,
            image_width,
            mb_contrib_floor,
            global_blend_pool(),
            /*image_out_external=*/image.mutable_data(),
            /*cull_disabled=*/cull_disabled,
            /*transmittance_threshold=*/transmittance_threshold);
    }
    auto t_b1 = clock::now();

    py::dict stats;
    stats["num_visible"] = static_cast<int64_t>(M);
    stats["num_entries"] = static_cast<int64_t>(P_kept);
    stats["pairs_in"] = cb.pairs_in;
    stats["pairs_dropped"] = cb.pairs_dropped_all_mb;
    stats["pairs_kept_per_mb"] = cb.pairs_kept_per_mb;
    stats["project_ms"] = std::chrono::duration<float, std::milli>(t_p1 - t_p0).count();
    stats["tile_assign_ms"] = std::chrono::duration<float, std::milli>(t_ta1 - t_ta0).count();
    stats["sort_ms"] = std::chrono::duration<float, std::milli>(t_s1 - t_s0).count();
    stats["blend_ms"] = std::chrono::duration<float, std::milli>(t_b1 - t_b0).count();
    stats["total_ms"] = std::chrono::duration<float, std::milli>(clock::now() - t0).count();
    return py::make_tuple(image, stats);
}

}  // namespace

PYBIND11_MODULE(_gsplat_cpu, m) {
    m.doc() = "gsplat_cpu — C++ rendering backend";

    py::class_<gsplat_cpu::ProjectPrepared>(m, "ProjectPrepared")
        .def_readonly("N", &gsplat_cpu::ProjectPrepared::N)
        .def_readonly("image_height", &gsplat_cpu::ProjectPrepared::image_height)
        .def_readonly("image_width", &gsplat_cpu::ProjectPrepared::image_width)
        .def_property_readonly(
            "means_2d",
            [](const gsplat_cpu::ProjectPrepared& p) {
                return py::array_t<float>(
                    {static_cast<py::ssize_t>(p.N), static_cast<py::ssize_t>(2)}, p.means_2d.data());
            })
        .def_property_readonly(
            "depths",
            [](const gsplat_cpu::ProjectPrepared& p) {
                return py::array_t<float>(static_cast<py::ssize_t>(p.N), p.depths.data());
            })
        .def_property_readonly(
            "cov_cam",
            [](const gsplat_cpu::ProjectPrepared& p) {
                return py::array_t<float>(
                    {static_cast<py::ssize_t>(p.N), static_cast<py::ssize_t>(9)}, p.cov_cam.data());
            })
        .def_property_readonly(
            "jacobian",
            [](const gsplat_cpu::ProjectPrepared& p) {
                return py::array_t<float>(
                    {static_cast<py::ssize_t>(p.N), static_cast<py::ssize_t>(6)}, p.jacobian.data());
            })
        .def_property_readonly(
            "near_valid",
            [](const gsplat_cpu::ProjectPrepared& p) {
                py::array_t<bool> out(static_cast<py::ssize_t>(p.N));
                auto mut = out.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(p.N); ++i) {
                    mut(i) = p.near_valid[static_cast<std::size_t>(i)] != 0;
                }
                return out;
            });

    m.def("hello", []() { return std::string("hello from gsplat_cpu"); });

    m.def("simd_backend", []() -> std::string {
#if defined(GSPLAT_SCALAR_ONLY)
        return "scalar";
#elif defined(GSPLAT_SIMD_AVX2) && defined(__AVX2__)
        return "avx2";
#elif defined(__ARM_NEON) || defined(__ARM_NEON)
        return "neon";
#else
        return "scalar";
#endif
    });

    m.def("has_tt_support", []() {
#ifdef GSPLAT_WITH_TT
        return true;
#else
        return false;
#endif
    });

#ifdef GSPLAT_WITH_TT
    m.def(
        "blend_microblock_tt",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> packs,
           py::array_t<float, py::array::c_style | py::array::forcecast> offsets,
           py::array_t<float, py::array::c_style | py::array::forcecast> px,
           py::array_t<float, py::array::c_style | py::array::forcecast> py,
           py::array_t<float, py::array::c_style | py::array::forcecast> coeff_table,
           py::array_t<uint32_t, py::array::c_style | py::array::forcecast> mb_header,
           py::array_t<uint32_t, py::array::c_style | py::array::forcecast> mb_stream,
           int image_height,
           int image_width) {
            const auto packs_info = packs.request();
            const auto offsets_info = offsets.request();
            const auto px_info = px.request();
            const auto py_info = py.request();
            const auto coeff_info = coeff_table.request();
            const auto header_info = mb_header.request();
            const auto stream_info = mb_stream.request();

            std::vector<float> packs_vec(
                static_cast<const float*>(packs_info.ptr),
                static_cast<const float*>(packs_info.ptr) + packs_info.size);
            std::vector<float> offsets_vec(
                static_cast<const float*>(offsets_info.ptr),
                static_cast<const float*>(offsets_info.ptr) + offsets_info.size);
            std::vector<float> px_vec(
                static_cast<const float*>(px_info.ptr),
                static_cast<const float*>(px_info.ptr) + px_info.size);
            std::vector<float> py_vec(
                static_cast<const float*>(py_info.ptr),
                static_cast<const float*>(py_info.ptr) + py_info.size);
            std::vector<float> coeff_vec(
                static_cast<const float*>(coeff_info.ptr),
                static_cast<const float*>(coeff_info.ptr) + coeff_info.size);
            std::vector<uint32_t> header_vec(
                static_cast<const uint32_t*>(header_info.ptr),
                static_cast<const uint32_t*>(header_info.ptr) + header_info.size);
            std::vector<uint32_t> stream_vec(
                static_cast<const uint32_t*>(stream_info.ptr),
                static_cast<const uint32_t*>(stream_info.ptr) + stream_info.size);

            std::vector<float> image_out;
            const double kernel_ms = gsplat_tt::blend_from_payload(
                packs_vec,
                offsets_vec,
                px_vec,
                py_vec,
                coeff_vec,
                header_vec,
                stream_vec,
                image_height,
                image_width,
                image_out);

            py::array_t<float> image(
                {static_cast<py::ssize_t>(image_height),
                 static_cast<py::ssize_t>(image_width),
                 static_cast<py::ssize_t>(3)});
            if (!image_out.empty()) {
                std::memcpy(image.mutable_data(), image_out.data(),
                            image_out.size() * sizeof(float));
            }
            return py::make_tuple(image, kernel_ms);
        },
        py::arg("packs"),
        py::arg("offsets"),
        py::arg("px"),
        py::arg("py"),
        py::arg("coeff_table"),
        py::arg("mb_header"),
        py::arg("mb_stream"),
        py::arg("image_height"),
        py::arg("image_width"));

    m.def("tt_device_shutdown", []() {
        gsplat_tt::device_state::leak_stage_contexts_on_exit();
    });

    // amendment-002 tt-005: device transform_means_cam (bounded hotspot port).
    // Input: means (N, 3) fp32 world-space, extrinsics (4, 4) fp32 row-major.
    // Output: (means_cam_np, kernel_ms). means_cam_np is (N, 3) fp32.
    // Returns kernel_ms = -1.0 if device init failed (caller should fall back).
    m.def(
        "transform_means_cam_tt",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> means,
           py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics) {
            const auto means_info = means.request();
            const auto extr_info = extrinsics.request();
            if (means_info.ndim != 2 || means_info.shape[1] != 3) {
                throw std::invalid_argument("means must have shape (N, 3)");
            }
            if (extr_info.size != 16) {
                throw std::invalid_argument("extrinsics must have 16 elements (4x4)");
            }
            const std::size_t N = static_cast<std::size_t>(means_info.shape[0]);
            py::array_t<float> means_cam({static_cast<py::ssize_t>(N), static_cast<py::ssize_t>(3)});
            gsplat_tt::ProjectCallTimings timings;
            const double kernel_ms = gsplat_tt::transform_means_cam_tt(
                static_cast<const float*>(means_info.ptr),
                static_cast<const float*>(extr_info.ptr),
                N,
                means_cam.mutable_data(),
                &timings);
            py::dict d;
            d["total_ms"]    = kernel_ms;
            d["pack_ms"]     = timings.pack_ms;
            d["upload_ms"]   = timings.upload_ms;
            d["launch_ms"]   = timings.launch_ms;
            d["compute_ms"]  = timings.compute_ms;
            d["download_ms"] = timings.download_ms;
            d["unpack_ms"]   = timings.unpack_ms;
            d["cache_hit"]   = timings.cache_hit;
            return py::make_tuple(means_cam, d);
        },
        py::arg("means"),
        py::arg("extrinsics"));

    m.def("tt_project_device_ready", []() { return gsplat_tt::project_device_ready(); });

    // amendment-002 tt-008b: device-resident means_cam — same as
    // transform_means_cam_tt but skips the D2H readback (the means_cam
    // buffers stay live in DRAM and are consumed via NoC by downstream
    // device stages like pfwc_tt). Returns just the timing dict.
    m.def(
        "transform_means_cam_tt_no_download",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> means,
           py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics) {
            const auto means_info = means.request();
            const auto extr_info = extrinsics.request();
            if (means_info.ndim != 2 || means_info.shape[1] != 3) {
                throw std::invalid_argument("means must have shape (N, 3)");
            }
            if (extr_info.size != 16) {
                throw std::invalid_argument("extrinsics must have 16 elements (4x4)");
            }
            const std::size_t N = static_cast<std::size_t>(means_info.shape[0]);
            gsplat_tt::ProjectCallTimings timings;
            const double kernel_ms = gsplat_tt::transform_means_cam_tt_no_download(
                static_cast<const float*>(means_info.ptr),
                static_cast<const float*>(extr_info.ptr),
                N,
                &timings);
            py::dict d;
            d["total_ms"]    = kernel_ms;
            d["pack_ms"]     = timings.pack_ms;
            d["upload_ms"]   = timings.upload_ms;
            d["launch_ms"]   = timings.launch_ms;
            d["compute_ms"]  = timings.compute_ms;
            d["download_ms"] = timings.download_ms;
            d["unpack_ms"]   = timings.unpack_ms;
            d["cache_hit"]   = timings.cache_hit;
            return d;
        },
        py::arg("means"),
        py::arg("extrinsics"));

    // amendment-002 tt-008a: device pfwc — replaces perspective + cov_cam
    // (the heaviest sub-steps of project_full_with_cov3d). Reads means_cam
    // from the device-resident buffers populated by transform_means_cam_tt;
    // writes mean_2d, depth, cov_cam_unique. The host finisher
    // project_finish_with_cov_cam_tt below consumes these.
    //
    // Inputs:
    //   cov3d_unique (N, 6) fp32 — [c00 c01 c02 c11 c12 c22]
    //   extrinsics   (4, 4) fp32 row-major
    //   intrinsics   (3, 3) fp32 row-major
    // Returns: (mean_2d (N, 2), depth (N,), cov_cam (N, 6), timing_dict).
    // tt-008c: device kernel now produces mean_2d, depth, cov2d, radii (full
    // pfwc on-device). When `download=True` (default), all 4 outputs are
    // D2H'd and returned. When `download=False`, outputs remain device-resident
    // (registered in gsplat_tt::device_state as pfwc_m2x/m2y/depth/a/b/c/rx/ry
    // for downstream tt-006 tile_assign to consume via NoC). The Python
    // binding still returns 4 arrays in the no-download case but they're
    // zero-sized placeholders.
    m.def(
        "transform_pfwc_tt",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> cov3d_unique,
           py::array_t<float, py::array::c_style | py::array::forcecast> extrinsics,
           py::array_t<float, py::array::c_style | py::array::forcecast> intrinsics,
           bool download) {
            const auto c_info = cov3d_unique.request();
            const auto e_info = extrinsics.request();
            const auto i_info = intrinsics.request();
            if (c_info.ndim != 2 || c_info.shape[1] != 6) {
                throw std::invalid_argument("cov3d_unique must have shape (N, 6)");
            }
            if (e_info.size != 16) {
                throw std::invalid_argument("extrinsics must have 16 elements (4x4)");
            }
            if (i_info.size != 9) {
                throw std::invalid_argument("intrinsics must have 9 elements (3x3)");
            }
            const std::size_t N = static_cast<std::size_t>(c_info.shape[0]);
            const py::ssize_t Nz = download ? static_cast<py::ssize_t>(N) : 0;
            py::array_t<float> mean_2d({Nz, static_cast<py::ssize_t>(2)});
            py::array_t<float> depth({Nz});
            py::array_t<float> cov2d({Nz, static_cast<py::ssize_t>(3)});
            py::array_t<float> radii({Nz, static_cast<py::ssize_t>(2)});
            float* m2d_ptr = download ? mean_2d.mutable_data() : nullptr;
            float* dep_ptr = download ? depth.mutable_data()   : nullptr;
            float* c2d_ptr = download ? cov2d.mutable_data()   : nullptr;
            float* r_ptr   = download ? radii.mutable_data()   : nullptr;
            gsplat_tt::PfwcCallTimings timings;
            const double kernel_ms = gsplat_tt::pfwc_tt(
                static_cast<const float*>(c_info.ptr),
                static_cast<const float*>(e_info.ptr),
                static_cast<const float*>(i_info.ptr),
                N,
                m2d_ptr,
                dep_ptr,
                c2d_ptr,
                r_ptr,
                &timings);
            py::dict d;
            d["total_ms"]    = kernel_ms;
            d["pack_ms"]     = timings.pack_ms;
            d["upload_ms"]   = timings.upload_ms;
            d["launch_ms"]   = timings.launch_ms;
            d["compute_ms"]  = timings.compute_ms;
            d["download_ms"] = timings.download_ms;
            d["unpack_ms"]   = timings.unpack_ms;
            d["cache_hit"]   = timings.cache_hit;
            return py::make_tuple(mean_2d, depth, cov2d, radii, d);
        },
        py::arg("cov3d_unique"),
        py::arg("extrinsics"),
        py::arg("intrinsics"),
        py::arg("download") = true);

    m.def("tt_pfwc_device_ready", []() { return gsplat_tt::pfwc_device_ready(); });
#endif
    m.def("project_prepare", &project_prepare_py);
    m.def("project_finalize", &project_finalize_py);
    m.def(
        "project",
        &project_py,
        py::arg("means"),
        py::arg("scales"),
        py::arg("rotations"),
        py::arg("extrinsics"),
        py::arg("intrinsics"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("opacities") = py::none(),
        py::arg("min_opacity") = 1.0f / 255.0f);

    m.def(
        "compute_cov3d",
        &compute_cov3d_py,
        py::arg("scales"),
        py::arg("rotations"));

    m.def(
        "project_full_with_cov3d",
        &project_full_with_cov3d_py,
        py::arg("means"),
        py::arg("cov3d"),
        py::arg("extrinsics"),
        py::arg("intrinsics"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("opacities") = py::none(),
        py::arg("min_opacity") = 1.0f / 255.0f,
        py::arg("means_cam") = py::none());

    m.def(
        "project_finish_with_cov_cam",
        &project_finish_with_cov_cam_py,
        py::arg("mean_2d_precomp"),
        py::arg("depth_precomp"),
        py::arg("cov_cam_precomp"),
        py::arg("extrinsics"),
        py::arg("intrinsics"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("opacities") = py::none(),
        py::arg("min_opacity") = 1.0f / 255.0f);

    // amendment-002 tt-008c: cheap finisher for the FULL device pfwc kernel.
    // The kernel now produces mean_2d + depth + cov2d + radii on-device — all
    // the host has to do is the valid_mask check + compact gather.
    m.def(
        "project_finish_with_cov2d_radii",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> mean_2d_precomp,
           py::array_t<float, py::array::c_style | py::array::forcecast> depth_precomp,
           py::array_t<float, py::array::c_style | py::array::forcecast> cov2d_precomp,
           py::array_t<float, py::array::c_style | py::array::forcecast> radii_precomp,
           int image_height,
           int image_width,
           py::object opacities_obj,
           float min_opacity) {
            const auto m_info = mean_2d_precomp.request();
            const auto d_info = depth_precomp.request();
            const auto c_info = cov2d_precomp.request();
            const auto r_info = radii_precomp.request();
            if (m_info.ndim != 2 || m_info.shape[1] != 2) {
                throw std::invalid_argument("mean_2d_precomp must have shape (N, 2)");
            }
            if (d_info.ndim != 1) {
                throw std::invalid_argument("depth_precomp must have shape (N,)");
            }
            if (c_info.ndim != 2 || c_info.shape[1] != 3) {
                throw std::invalid_argument("cov2d_precomp must have shape (N, 3)");
            }
            if (r_info.ndim != 2 || r_info.shape[1] != 2) {
                throw std::invalid_argument("radii_precomp must have shape (N, 2)");
            }
            const std::size_t N = static_cast<std::size_t>(m_info.shape[0]);
            if (static_cast<std::size_t>(d_info.shape[0]) != N ||
                static_cast<std::size_t>(c_info.shape[0]) != N ||
                static_cast<std::size_t>(r_info.shape[0]) != N) {
                throw std::invalid_argument("mean_2d / depth / cov2d / radii N mismatch");
            }
            const float* opacities_ptr = nullptr;
            py::array_t<float, py::array::c_style | py::array::forcecast> opacities;
            if (!opacities_obj.is_none()) {
                opacities = py::cast<
                    py::array_t<float, py::array::c_style | py::array::forcecast>>(opacities_obj);
                if (static_cast<std::size_t>(opacities.request().shape[0]) != N) {
                    throw std::invalid_argument("opacities must have shape (N,)");
                }
                opacities_ptr = static_cast<const float*>(opacities.request().ptr);
            }
            const gsplat_cpu::ProjectResult result = gsplat_cpu::project_finish_with_cov2d_radii(
                static_cast<const float*>(m_info.ptr),
                static_cast<const float*>(d_info.ptr),
                static_cast<const float*>(c_info.ptr),
                static_cast<const float*>(r_info.ptr),
                opacities_ptr,
                min_opacity,
                N,
                image_height,
                image_width,
                &global_project_pool(),
                /*max_radius=*/0);
            return pack_project_result(result, N);
        },
        py::arg("mean_2d_precomp"),
        py::arg("depth_precomp"),
        py::arg("cov2d_precomp"),
        py::arg("radii_precomp"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("opacities") = py::none(),
        py::arg("min_opacity") = 1.0f / 255.0f);

    m.def(
        "project_full",
        &project_full_py,
        py::arg("means"),
        py::arg("scales"),
        py::arg("rotations"),
        py::arg("extrinsics"),
        py::arg("intrinsics"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("opacities") = py::none(),
        py::arg("min_opacity") = 1.0f / 255.0f);

    m.def(
        "tile_assign",
        &tile_assign_py,
        py::arg("means_2d"),
        py::arg("radii"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("tile_size"),
        py::arg("covs_2d") = py::none(),
        py::arg("opacities") = py::none(),
        py::arg("contrib_floor") = 15.0f / 255.0f);

    m.def(
        "sort",
        &sort_py,
        py::arg("gaussian_ids"),
        py::arg("tile_ids"),
        py::arg("depths"),
        py::arg("tiles_x"),
        py::arg("tiles_y"));

    m.def(
        "blend",
        &blend_py,
        py::arg("means_2d"),
        py::arg("covs_2d"),
        py::arg("colors"),
        py::arg("opacities"),
        py::arg("sorted_gaussian_ids"),
        py::arg("tile_ranges"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("tile_size") = 32);

    m.def(
        "blend_microblock",
        &blend_microblock_py,
        py::arg("means_2d"),
        py::arg("covs_2d"),
        py::arg("colors"),
        py::arg("opacities"),
        py::arg("mb_header"),
        py::arg("mb_stream"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("tile_size") = 32);

    m.def(
        "cull_and_blend",
        &cull_and_blend_py,
        py::arg("means_2d"),
        py::arg("covs_2d"),
        py::arg("colors"),
        py::arg("opacities"),
        py::arg("sorted_gaussian_ids"),
        py::arg("tile_ranges"),
        py::arg("tiles_x"),
        py::arg("tiles_y"),
        py::arg("tile_size") = 32,
        py::arg("image_height") = 0,
        py::arg("image_width") = 0,
        py::arg("mb_contrib_floor") = 1.0f / 16384.0f);

    m.def(
        "render_full",
        &render_full_py,
        py::arg("means"),
        py::arg("cov3d"),
        py::arg("opacities"),
        py::arg("colors"),
        py::arg("extrinsics"),
        py::arg("intrinsics"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("tile_size") = 32,
        py::arg("min_opacity") = 1.0f / 255.0f,
        py::arg("contrib_floor") = 1.0f / 16384.0f,
        py::arg("mb_contrib_floor") = 1.0f / 16384.0f,
        py::arg("cull_disabled") = false,
        py::arg("transmittance_threshold") = 1.0f / 255.0f,
        py::arg("max_radius") = 0,
        py::arg("k_cap") = 3.0f,
        py::arg("use_isoellipse") = false,
        py::arg("blend_mode") = 0);

    // amendment-003: fused C++ render loop for the TT backend. Identical to
    // render_full but the C++ owns stage dispatch (no Python per-stage
    // orchestration). blend_mode selects the blend implementation; the TT
    // device path is only available when has_tt_support(). The default
    // blend_mode here is read from the GSPLAT_TT_BLEND_MODE env in the Python
    // backend so device kernels can be toggled without rebuilding.
    m.def(
        "render_full_tt",
        &render_full_py,
        py::arg("means"),
        py::arg("cov3d"),
        py::arg("opacities"),
        py::arg("colors"),
        py::arg("extrinsics"),
        py::arg("intrinsics"),
        py::arg("image_height"),
        py::arg("image_width"),
        py::arg("tile_size") = 32,
        py::arg("min_opacity") = 1.0f / 255.0f,
        py::arg("contrib_floor") = 1.0f / 16384.0f,
        py::arg("mb_contrib_floor") = 1.0f / 16384.0f,
        py::arg("cull_disabled") = false,
        py::arg("transmittance_threshold") = 1.0f / 255.0f,
        py::arg("max_radius") = 0,
        py::arg("k_cap") = 3.0f,
        py::arg("use_isoellipse") = false,
        py::arg("blend_mode") = 0);

    m.def(
        "microblock_cull",
        &microblock_cull_py,
        py::arg("means_2d"),
        py::arg("covs_2d"),
        py::arg("opacities"),
        py::arg("sorted_gaussian_ids"),
        py::arg("tile_ranges"),
        py::arg("tiles_x"),
        py::arg("tiles_y"),
        py::arg("tile_size") = 32,
        py::arg("mb_contrib_floor") = 1.0f / 16384.0f);

    py::class_<gsplat_cpu::ThreadPool>(m, "ThreadPool")
        .def(py::init<std::size_t>(), py::arg("num_threads") = 0)
        .def("size", &gsplat_cpu::ThreadPool::size)
        .def("wait", &gsplat_cpu::ThreadPool::wait);
}
