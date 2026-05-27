#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

class ThreadPool;

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
    int image_width,
    ThreadPool* pool = nullptr);

// Compute cov3d (= R(q) @ S^2 @ R(q).T) for N Gaussians once per scene. Caller
// caches the (N*9, row-major 3x3) buffer across views; cov3d is view-independent.
// Used by project_prepare_from_cov3d to amortise cov3d build over many camera
// poses (training loop, 30-frame bench).
void compute_cov3d_batch(
    const float* scales,
    const float* rotations,
    std::size_t N,
    float* cov3d_out);  // N * 9

// Same as project_prepare but reuses a precomputed cov3d buffer (skips
// per-Gaussian cov3d math).
ProjectPrepared project_prepare_from_cov3d(
    const float* means,
    const float* cov3d,  // N * 9, row-major 3x3 (use compute_cov3d_batch to fill)
    const float* extrinsics,
    const float* intrinsics,
    std::size_t N,
    int image_height,
    int image_width,
    ThreadPool* pool = nullptr);

// Same as project_finalize but accepts a thread pool for the per-Gaussian
// radius/cull pass.
ProjectResult project_finalize_parallel(
    const ProjectPrepared& prep,
    const float* covs_2d,
    const float* opacities,
    float min_opacity,
    ThreadPool* pool = nullptr);

// Compute covs_2d = J @ cov_cam @ J.T + 0.3*I per Gaussian in pure fp32 C++.
// Pure C++ avoids the python -> torch.bmm round-trip used in the previous
// pybind path (saves ~2-3ms / frame). Matches torch's batched bmm within
// ~5e-4 (same magnitude as the previous covs_2d diff vs numpy reference),
// driven by fp32 accumulation order.
//   prep      output of project_prepare_geometry / project_prepare_from_cov3d
//   covs_2d_out output buffer, size prep.N * 4, layout [a, b, b, c] row-major
//   pool      optional thread pool for the parallel inner loop
void compute_covs_2d(
    const ProjectPrepared& prep,
    float* covs_2d_out,
    ThreadPool* pool = nullptr);

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
