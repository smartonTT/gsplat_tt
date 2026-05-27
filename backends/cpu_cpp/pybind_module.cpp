#include <cstring>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gsplat_cpu/project.h"
#include "gsplat_cpu/sort.h"
#include "gsplat_cpu/thread_pool.h"
#include "gsplat_cpu/tile_assign.h"

namespace py = pybind11;

namespace {

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

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project_finalize(
        prep, static_cast<const float*>(cov_info.ptr), opacities_ptr, min_opacity);
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
        contrib_floor);

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
        tiles_y);

    return pack_sort_result(result, tiles_x, tiles_y);
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

    py::class_<gsplat_cpu::ThreadPool>(m, "ThreadPool")
        .def(py::init<std::size_t>(), py::arg("num_threads") = 0)
        .def("size", &gsplat_cpu::ThreadPool::size)
        .def("wait", &gsplat_cpu::ThreadPool::wait);
}
