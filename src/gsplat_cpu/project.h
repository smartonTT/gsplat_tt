#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

struct ProjectResult {
    std::vector<float> means_2d;
    std::vector<float> covs_2d;
    std::vector<float> depths;
    std::vector<float> radii;
    std::vector<uint8_t> valid_mask;
};

struct ProjectPrepared {
    std::size_t N{0};
    int image_height{0};
    int image_width{0};
    std::vector<float> means_2d;   // N * 2
    std::vector<float> depths;     // N
    std::vector<float> cov_cam;    // N * 9, row-major 3x3
    std::vector<float> jacobian;   // N * 6, row-major 2x3
    std::vector<uint8_t> near_valid;
};

ProjectPrepared project_prepare(
    const float* means,
    const float* scales,
    const float* rotations,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    int image_height,
    int image_width);

// Geometry only (no per-Gaussian cov3d/cov_cam); used by project_full pybind path.
ProjectPrepared project_prepare_geometry(
    const float* means,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    int image_height,
    int image_width);

ProjectResult project_finalize(
    const ProjectPrepared& prep,
    const float* covs_2d, // N * 4, layout [a, b, b, c]
    const float* opacities,
    float min_opacity);

ProjectResult project(
    const float* means,
    const float* scales,
    const float* rotations,
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    int image_height,
    int image_width,
    const float* opacities,
    float min_opacity);

}  // namespace gsplat_cpu
