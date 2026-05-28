#include "gsplat_cpu/project.h"

#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include "gsplat_cpu/types.h"

namespace gsplat_cpu {

namespace {

void transform_means_cam(
    const float* means,
    const float* extrinsics,
    const std::size_t N,
    std::vector<float>& means_cam) {
    const float r00 = extrinsics[0];
    const float r01 = extrinsics[1];
    const float r02 = extrinsics[2];
    const float r10 = extrinsics[4];
    const float r11 = extrinsics[5];
    const float r12 = extrinsics[6];
    const float r20 = extrinsics[8];
    const float r21 = extrinsics[9];
    const float r22 = extrinsics[10];
    const float t0 = extrinsics[3];
    const float t1 = extrinsics[7];
    const float t2 = extrinsics[11];

    means_cam.resize(N * 3);

#if defined(__APPLE__)
    const float r[9] = {r00, r01, r02, r10, r11, r12, r20, r21, r22};
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(N), 3, 3, 1.0f, means,
                3, r, 3, 0.0f, means_cam.data(), 3);
    for (std::size_t i = 0; i < N; ++i) {
        means_cam[i * 3 + 0] += t0;
        means_cam[i * 3 + 1] += t1;
        means_cam[i * 3 + 2] += t2;
    }
#else
    for (std::size_t i = 0; i < N; ++i) {
        const float mx = means[i * 3 + 0];
        const float my = means[i * 3 + 1];
        const float mz = means[i * 3 + 2];
        means_cam[i * 3 + 0] = mx * r00 + my * r01 + mz * r02 + t0;
        means_cam[i * 3 + 1] = mx * r10 + my * r11 + mz * r12 + t1;
        means_cam[i * 3 + 2] = mx * r20 + my * r21 + mz * r22 + t2;
    }
#endif
}

ProjectResult gather_visible(
    const ProjectPrepared& prep,
    const float* covs_2d,
    const std::vector<Vec2f>& radii_scratch,
    const std::vector<uint8_t>& valid_mask,
    ThreadPool* pool = nullptr) {
    ProjectResult result;
    result.valid_mask = valid_mask;

    const std::size_t N = prep.N;

    if (pool == nullptr || pool->size() <= 1 || N < 8192) {
        // Serial path: small N or no pool — direct push_back, exactly preserves
        // the original semantics.
        for (std::size_t i = 0; i < N; ++i) {
            if (valid_mask[i] == 0) {
                continue;
            }
            result.means_2d.push_back(prep.means_2d[i * 2 + 0]);
            result.means_2d.push_back(prep.means_2d[i * 2 + 1]);
            result.covs_2d.push_back(covs_2d[i * 4 + 0]);
            result.covs_2d.push_back(covs_2d[i * 4 + 1]);
            result.covs_2d.push_back(covs_2d[i * 4 + 2]);
            result.covs_2d.push_back(covs_2d[i * 4 + 3]);
            result.depths.push_back(prep.depths[i]);
            result.radii.push_back(radii_scratch[i].x);
            result.radii.push_back(radii_scratch[i].y);
        }
        return result;
    }

    // iter-020 parallel filter: contiguous-chunk count -> prefix sum ->
    // parallel scatter into pre-sized output arrays. Preserves the input
    // order (chunks are contiguous index ranges, scatter goes in-order
    // within each chunk).
    const std::size_t W = pool->size();
    const std::size_t chunk = (N + W - 1) / W;
    std::vector<std::size_t> chunk_counts(W, 0);
    std::vector<std::size_t> chunk_starts(W + 1, 0);

    for (std::size_t w = 0; w < W; ++w) {
        pool->submit([w, W, chunk, N, &valid_mask, &chunk_counts]() {
            const std::size_t lo = std::min(w * chunk, N);
            const std::size_t hi = std::min(lo + chunk, N);
            std::size_t cnt = 0;
            for (std::size_t i = lo; i < hi; ++i) {
                cnt += (valid_mask[i] != 0);
            }
            chunk_counts[w] = cnt;
        });
    }
    pool->wait();

    for (std::size_t w = 0; w < W; ++w) {
        chunk_starts[w + 1] = chunk_starts[w] + chunk_counts[w];
    }
    const std::size_t V = chunk_starts[W];  // visible count

    result.means_2d.resize(V * 2);
    result.covs_2d.resize(V * 4);
    result.depths.resize(V);
    result.radii.resize(V * 2);

    for (std::size_t w = 0; w < W; ++w) {
        pool->submit([w, W, chunk, N, &prep, covs_2d, &radii_scratch, &valid_mask,
                      &chunk_starts, &result]() {
            const std::size_t lo = std::min(w * chunk, N);
            const std::size_t hi = std::min(lo + chunk, N);
            std::size_t out = chunk_starts[w];
            for (std::size_t i = lo; i < hi; ++i) {
                if (valid_mask[i] == 0) continue;
                result.means_2d[out * 2 + 0] = prep.means_2d[i * 2 + 0];
                result.means_2d[out * 2 + 1] = prep.means_2d[i * 2 + 1];
                result.covs_2d[out * 4 + 0] = covs_2d[i * 4 + 0];
                result.covs_2d[out * 4 + 1] = covs_2d[i * 4 + 1];
                result.covs_2d[out * 4 + 2] = covs_2d[i * 4 + 2];
                result.covs_2d[out * 4 + 3] = covs_2d[i * 4 + 3];
                result.depths[out] = prep.depths[i];
                result.radii[out * 2 + 0] = radii_scratch[i].x;
                result.radii[out * 2 + 1] = radii_scratch[i].y;
                ++out;
            }
        });
    }
    pool->wait();

    return result;
}

}  // namespace

ProjectPrepared project_prepare(
    const float* means,
    const float* scales,
    const float* rotations,
    const float* extrinsics,
    const float* intrinsics,
    const std::size_t N,
    const int image_height,
    const int image_width) {
    ProjectPrepared prep =
        project_prepare_geometry(means, extrinsics, intrinsics, N, image_height, image_width);
    prep.cov_cam.resize(N * 9);

    Mat3f r_mat;
    r_mat(0, 0) = extrinsics[0];
    r_mat(0, 1) = extrinsics[1];
    r_mat(0, 2) = extrinsics[2];
    r_mat(1, 0) = extrinsics[4];
    r_mat(1, 1) = extrinsics[5];
    r_mat(1, 2) = extrinsics[6];
    r_mat(2, 0) = extrinsics[8];
    r_mat(2, 1) = extrinsics[9];
    r_mat(2, 2) = extrinsics[10];

    for (std::size_t i = 0; i < N; ++i) {
        const Vec3f scale{scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]};
        const Quatf quat{rotations[i * 4 + 0], rotations[i * 4 + 1], rotations[i * 4 + 2],
                         rotations[i * 4 + 3]};

        const Mat3f cov3d = build_covariance_3d(scale, quat);
        const Mat3f cov_cam = mat3_mul(mat3_mul(r_mat, cov3d), mat3_transpose(r_mat));
        std::memcpy(prep.cov_cam.data() + i * 9, cov_cam.m.data(), 9 * sizeof(float));
    }

    return prep;
}

void compute_cov3d_batch(
    const float* scales,
    const float* rotations,
    const std::size_t N,
    float* cov3d_out) {
    for (std::size_t i = 0; i < N; ++i) {
        const Vec3f scale{scales[i * 3 + 0], scales[i * 3 + 1], scales[i * 3 + 2]};
        const Quatf quat{rotations[i * 4 + 0], rotations[i * 4 + 1], rotations[i * 4 + 2],
                         rotations[i * 4 + 3]};
        const Mat3f cov3d = build_covariance_3d(scale, quat);
        std::memcpy(cov3d_out + i * 9, cov3d.m.data(), 9 * sizeof(float));
    }
}

ProjectPrepared project_prepare_from_cov3d(
    const float* means,
    const float* cov3d,
    const float* extrinsics,
    const float* intrinsics,
    const std::size_t N,
    const int image_height,
    const int image_width,
    ThreadPool* pool) {
    ProjectPrepared prep = project_prepare_geometry(means, extrinsics, intrinsics, N,
                                                    image_height, image_width, pool);
    prep.cov_cam.resize(N * 9);

    Mat3f r_mat;
    r_mat(0, 0) = extrinsics[0];
    r_mat(0, 1) = extrinsics[1];
    r_mat(0, 2) = extrinsics[2];
    r_mat(1, 0) = extrinsics[4];
    r_mat(1, 1) = extrinsics[5];
    r_mat(1, 2) = extrinsics[6];
    r_mat(2, 0) = extrinsics[8];
    r_mat(2, 1) = extrinsics[9];
    r_mat(2, 2) = extrinsics[10];

    const Mat3f r_t = mat3_transpose(r_mat);
    auto per_gaussian = [&](std::size_t i) {
        Mat3f c;
        std::memcpy(c.m.data(), cov3d + i * 9, 9 * sizeof(float));
        const Mat3f cov_cam = mat3_mul(mat3_mul(r_mat, c), r_t);
        std::memcpy(prep.cov_cam.data() + i * 9, cov_cam.m.data(), 9 * sizeof(float));
    };
    if (pool != nullptr && pool->size() > 1 && N >= 4096) {
        const std::size_t W = pool->size();
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, W, N, &per_gaussian]() {
                for (std::size_t i = w; i < N; i += W) {
                    per_gaussian(i);
                }
            });
        }
        pool->wait();
    } else {
        for (std::size_t i = 0; i < N; ++i) per_gaussian(i);
    }

    return prep;
}

ProjectPrepared project_prepare_geometry(
    const float* means,
    const float* extrinsics,
    const float* intrinsics,
    const std::size_t N,
    const int image_height,
    const int image_width,
    ThreadPool* pool) {
    const float fx = intrinsics[0];
    const float fy = intrinsics[4];
    const float cx = intrinsics[2];
    const float cy = intrinsics[5];

    constexpr float k_near = 0.2f;

    ProjectPrepared prep;
    prep.N = N;
    prep.image_height = image_height;
    prep.image_width = image_width;
    prep.means_2d.resize(N * 2);
    prep.depths.resize(N);
    prep.jacobian.resize(N * 6);
    prep.near_valid.resize(N);

    std::vector<float> means_cam;
    transform_means_cam(means, extrinsics, N, means_cam);  // Accelerate sgemm — already fast.

    auto per_gaussian = [&](std::size_t i) {
        const float tx = means_cam[i * 3 + 0];
        const float ty = means_cam[i * 3 + 1];
        const float tz = means_cam[i * 3 + 2];

        prep.near_valid[i] = tz > k_near ? 1 : 0;
        prep.depths[i] = tz;

        const float inv_tz = 1.0f / tz;
        prep.means_2d[i * 2 + 0] = fx * tx * inv_tz + cx;
        prep.means_2d[i * 2 + 1] = fy * ty * inv_tz + cy;

        build_jacobian(fx, fy, tx, ty, tz, prep.jacobian.data() + i * 6);
    };

    if (pool != nullptr && pool->size() > 1 && N >= 4096) {
        const std::size_t W = pool->size();
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, W, N, &per_gaussian]() {
                for (std::size_t i = w; i < N; i += W) {
                    per_gaussian(i);
                }
            });
        }
        pool->wait();
    } else {
        for (std::size_t i = 0; i < N; ++i) per_gaussian(i);
    }

    return prep;
}

namespace {

void compute_valid_and_radii(
    const ProjectPrepared& prep,
    const float* covs_2d,
    const float* opacities,
    const float min_opacity,
    const int max_radius,
    std::size_t i,
    std::vector<uint8_t>& valid_mask,
    std::vector<Vec2f>& radii_scratch) {
    bool valid = prep.near_valid[i] != 0;

    const float a = covs_2d[i * 4 + 0];
    const float c = covs_2d[i * 4 + 3];
    const float mean_x = prep.means_2d[i * 2 + 0];
    const float mean_y = prep.means_2d[i * 2 + 1];

    const float k = opacities != nullptr ? opacity_aware_k(opacities[i]) : 3.0f;
    const float rx = std::ceil(k * std::sqrt(std::max(a, 0.0f)));
    const float ry = std::ceil(k * std::sqrt(std::max(c, 0.0f)));
    radii_scratch[i] = Vec2f{rx, ry};

    valid = valid && (mean_x + rx > 0.0f);
    valid = valid && (mean_x - rx < static_cast<float>(prep.image_width));
    valid = valid && (mean_y + ry > 0.0f);
    valid = valid && (mean_y - ry < static_cast<float>(prep.image_height));
    valid = valid && (rx > 0.0f) && (ry > 0.0f);
    valid = valid && (rx <= static_cast<float>(max_radius)) &&
            (ry <= static_cast<float>(max_radius));

    if (opacities != nullptr) {
        valid = valid && (opacities[i] >= min_opacity);
    }

    valid_mask[i] = valid ? 1 : 0;
}

}  // namespace

ProjectResult project_finalize_parallel(
    const ProjectPrepared& prep,
    const float* covs_2d,
    const float* opacities,
    const float min_opacity,
    ThreadPool* pool) {
    const int max_radius = std::min(prep.image_height, prep.image_width) / 2;

    std::vector<uint8_t> valid_mask(prep.N);
    std::vector<Vec2f> radii_scratch(prep.N);

    if (pool != nullptr && pool->size() > 1 && prep.N >= 4096) {
        const std::size_t W = pool->size();
        const std::size_t N = prep.N;
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, W, N, &prep, covs_2d, opacities, min_opacity, max_radius,
                          &valid_mask, &radii_scratch]() {
                for (std::size_t i = w; i < N; i += W) {
                    compute_valid_and_radii(prep, covs_2d, opacities, min_opacity, max_radius,
                                            i, valid_mask, radii_scratch);
                }
            });
        }
        pool->wait();
    } else {
        for (std::size_t i = 0; i < prep.N; ++i) {
            compute_valid_and_radii(prep, covs_2d, opacities, min_opacity, max_radius, i,
                                    valid_mask, radii_scratch);
        }
    }

    return gather_visible(prep, covs_2d, radii_scratch, valid_mask, pool);
}

ProjectResult project_finalize(
    const ProjectPrepared& prep,
    const float* covs_2d,
    const float* opacities,
    const float min_opacity) {
    const int max_radius = std::min(prep.image_height, prep.image_width) / 2;

    std::vector<uint8_t> valid_mask(prep.N);
    std::vector<Vec2f> radii_scratch(prep.N);

    for (std::size_t i = 0; i < prep.N; ++i) {
        bool valid = prep.near_valid[i] != 0;

        const float a = covs_2d[i * 4 + 0];
        const float c = covs_2d[i * 4 + 3];
        const float mean_x = prep.means_2d[i * 2 + 0];
        const float mean_y = prep.means_2d[i * 2 + 1];

        const float k = opacities != nullptr ? opacity_aware_k(opacities[i]) : 3.0f;
        const float rx = std::ceil(k * std::sqrt(std::max(a, 0.0f)));
        const float ry = std::ceil(k * std::sqrt(std::max(c, 0.0f)));
        radii_scratch[i] = Vec2f{rx, ry};

        valid = valid && (mean_x + rx > 0.0f);
        valid = valid && (mean_x - rx < static_cast<float>(prep.image_width));
        valid = valid && (mean_y + ry > 0.0f);
        valid = valid && (mean_y - ry < static_cast<float>(prep.image_height));
        valid = valid && (rx > 0.0f) && (ry > 0.0f);
        valid = valid && (rx <= static_cast<float>(max_radius)) &&
                (ry <= static_cast<float>(max_radius));

        if (opacities != nullptr) {
            valid = valid && (opacities[i] >= min_opacity);
        }

        valid_mask[i] = valid ? 1 : 0;
    }

    return gather_visible(prep, covs_2d, radii_scratch, valid_mask);
}

namespace {

inline void cov2d_for_i(const ProjectPrepared& prep, std::size_t i, float* covs_2d_out) {
    Mat3f cov_cam;
    std::memcpy(cov_cam.m.data(), prep.cov_cam.data() + i * 9, 9 * sizeof(float));
    Cov2f cov2d = project_cov2d(cov_cam, prep.jacobian.data() + i * 6);
    cov2d.a += 0.3f;
    cov2d.c += 0.3f;
    covs_2d_out[i * 4 + 0] = cov2d.a;
    covs_2d_out[i * 4 + 1] = cov2d.b;
    covs_2d_out[i * 4 + 2] = cov2d.b;
    covs_2d_out[i * 4 + 3] = cov2d.c;
}

}  // namespace

void compute_covs_2d(
    const ProjectPrepared& prep,
    float* covs_2d_out,
    ThreadPool* pool) {
    const std::size_t N = prep.N;
    if (pool != nullptr && pool->size() > 1 && N >= 4096) {
        const std::size_t W = pool->size();
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, W, N, &prep, covs_2d_out]() {
                for (std::size_t i = w; i < N; i += W) {
                    cov2d_for_i(prep, i, covs_2d_out);
                }
            });
        }
        pool->wait();
    } else {
        for (std::size_t i = 0; i < N; ++i) {
            cov2d_for_i(prep, i, covs_2d_out);
        }
    }
}

ProjectResult project(
    const float* means,
    const float* scales,
    const float* rotations,
    const float* extrinsics,
    const float* intrinsics,
    const std::size_t N,
    const int image_height,
    const int image_width,
    const float* opacities,
    const float min_opacity) {
    ProjectPrepared prep =
        project_prepare(means, scales, rotations, extrinsics, intrinsics, N, image_height, image_width);

    std::vector<float> covs_2d(N * 4);
    compute_covs_2d(prep, covs_2d.data(), nullptr);

    return project_finalize(prep, covs_2d.data(), opacities, min_opacity);
}

ProjectResult project_full_fused(
    const float* means,
    const float* cov3d,
    const float* extrinsics,
    const float* intrinsics,
    const float* colors,
    const float* opacities,
    const float min_opacity,
    const std::size_t N,
    const int image_height,
    const int image_width,
    ThreadPool* pool,
    const int max_radius_param,
    const float contrib_floor_k,
    const float k_cap,
    const bool use_isoellipse) {
    // Extrinsics (column-3 translation).
    Mat3f r_mat;
    r_mat(0, 0) = extrinsics[0]; r_mat(0, 1) = extrinsics[1]; r_mat(0, 2) = extrinsics[2];
    r_mat(1, 0) = extrinsics[4]; r_mat(1, 1) = extrinsics[5]; r_mat(1, 2) = extrinsics[6];
    r_mat(2, 0) = extrinsics[8]; r_mat(2, 1) = extrinsics[9]; r_mat(2, 2) = extrinsics[10];
    const Mat3f r_t = mat3_transpose(r_mat);

    const float t0 = extrinsics[3], t1 = extrinsics[7], t2 = extrinsics[11];
    const float fx = intrinsics[0], fy = intrinsics[4];
    const float cx = intrinsics[2], cy = intrinsics[5];
    constexpr float k_near = 0.2f;
    const int max_radius = (max_radius_param < 0)
        ? std::numeric_limits<int>::max()
        : ((max_radius_param > 0)
               ? max_radius_param
               : std::min(image_height, image_width) / 2);

    // Single-shot means_cam via Accelerate (already <1ms even at 600k Gaussians).
    std::vector<float> means_cam(N * 3);
#if defined(__APPLE__)
    {
        const float r9[9] = {r_mat(0,0), r_mat(0,1), r_mat(0,2),
                             r_mat(1,0), r_mat(1,1), r_mat(1,2),
                             r_mat(2,0), r_mat(2,1), r_mat(2,2)};
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(N), 3, 3, 1.0f, means, 3, r9, 3, 0.0f,
                    means_cam.data(), 3);
    }
#else
    for (std::size_t i = 0; i < N; ++i) {
        const float mx = means[i*3+0], my = means[i*3+1], mz = means[i*3+2];
        means_cam[i*3+0] = mx*r_mat(0,0) + my*r_mat(0,1) + mz*r_mat(0,2);
        means_cam[i*3+1] = mx*r_mat(1,0) + my*r_mat(1,1) + mz*r_mat(1,2);
        means_cam[i*3+2] = mx*r_mat(2,0) + my*r_mat(2,1) + mz*r_mat(2,2);
    }
#endif

    // Per-Gaussian scratch: means_2d (2), depths (1), cov2d (4), valid (1),
    // radii (2). Final compact arrays are built by the parallel-filter
    // gather below. cov_cam is computed in registers only — never stored.
    std::vector<float> means_2d(N * 2);
    std::vector<float> depths(N);
    std::vector<float> covs_2d(N * 4);
    std::vector<uint8_t> valid_mask(N);
    std::vector<Vec2f> radii_scratch(N);

    auto per_gaussian = [&](std::size_t i) {
        const float tx = means_cam[i*3+0] + t0;
        const float ty = means_cam[i*3+1] + t1;
        const float tz = means_cam[i*3+2] + t2;
        depths[i] = tz;

        // iter-031: short-circuit two cheap per-Gaussian invalidations BEFORE
        // touching cov3d / cov_cam / cov2d. tz <= k_near (near-plane / behind
        // camera) and opacity < min_opacity are independent of the
        // 36-byte cov3d read and the two 3x3 matmuls + Jacobian-by-cov_cam
        // sandwich; skipping them for the 20-30% of Gaussians that fail
        // these cheap checks shaves both the 36-byte cov3d memory read and
        // the cov_cam + cov2d compute. Each invalidated Gaussian also
        // writes 0/sentinel into the scratch arrays so the downstream
        // gather skips them correctly.
        if (tz <= k_near ||
            (opacities != nullptr && opacities[i] < min_opacity)) {
            means_2d[i*2+0] = 0.0f;
            means_2d[i*2+1] = 0.0f;
            covs_2d[i*4+0] = 0.0f;
            covs_2d[i*4+1] = 0.0f;
            covs_2d[i*4+2] = 0.0f;
            covs_2d[i*4+3] = 0.0f;
            radii_scratch[i] = Vec2f{0.0f, 0.0f};
            valid_mask[i] = 0;
            return;
        }

        const float inv_tz = 1.0f / tz;
        const float mean_x = fx * tx * inv_tz + cx;
        const float mean_y = fy * ty * inv_tz + cy;
        means_2d[i*2+0] = mean_x;
        means_2d[i*2+1] = mean_y;

        // Jacobian (kept in registers — never stored to memory).
        const float tz2 = tz * tz;
        const float j00 = fx / tz, j02 = -fx * tx / tz2;
        const float j11 = fy / tz, j12 = -fy * ty / tz2;

        // iter-044: cov_cam = R @ cov3d @ R^T exploiting cov3d symmetry.
        // cov3d is stored as 9 floats but only the upper triangle (6 entries)
        // is independent. Two general 3x3 matmuls cost 54 mul + 36 add and
        // produce a symmetric result; here we directly compute the 6 unique
        // cov_cam entries we will read below, saving ~17% of the per-Gaussian
        // covariance flops (45 mul + 30 add instead of 54 mul + 36 add).
        //
        // Let v_j = cov3d @ R(j,:)^T (a 3-vector). cov_cam(i,j) = R(i,:) . v_j.
        // Only entries used by the cov2d compute are needed:
        //   (0,0), (0,1) = (1,0), (0,2) = (2,0), (1,1), (1,2) = (2,1), (2,2)
        const float c00 = cov3d[i * 9 + 0];
        const float c01 = cov3d[i * 9 + 1];
        const float c02 = cov3d[i * 9 + 2];
        const float c11 = cov3d[i * 9 + 4];
        const float c12 = cov3d[i * 9 + 5];
        const float c22 = cov3d[i * 9 + 8];

        const float r00 = r_mat(0, 0), r01 = r_mat(0, 1), r02 = r_mat(0, 2);
        const float r10 = r_mat(1, 0), r11 = r_mat(1, 1), r12 = r_mat(1, 2);
        const float r20 = r_mat(2, 0), r21 = r_mat(2, 1), r22 = r_mat(2, 2);

        // v_j = cov3d @ R(j,:)^T  (cov3d symmetric — use upper triangle only).
        // v_j[0] = c00*r_j0 + c01*r_j1 + c02*r_j2
        // v_j[1] = c01*r_j0 + c11*r_j1 + c12*r_j2
        // v_j[2] = c02*r_j0 + c12*r_j1 + c22*r_j2
        const float v0_0 = c00 * r00 + c01 * r01 + c02 * r02;
        const float v0_1 = c01 * r00 + c11 * r01 + c12 * r02;
        const float v0_2 = c02 * r00 + c12 * r01 + c22 * r02;

        const float v1_0 = c00 * r10 + c01 * r11 + c02 * r12;
        const float v1_1 = c01 * r10 + c11 * r11 + c12 * r12;
        const float v1_2 = c02 * r10 + c12 * r11 + c22 * r12;

        const float v2_0 = c00 * r20 + c01 * r21 + c02 * r22;
        const float v2_1 = c01 * r20 + c11 * r21 + c12 * r22;
        const float v2_2 = c02 * r20 + c12 * r21 + c22 * r22;

        // cov_cam(i,j) = R(i,:) . v_j  (6 unique entries; symmetric).
        const float cc00 = r00 * v0_0 + r01 * v0_1 + r02 * v0_2;
        const float cc01 = r00 * v1_0 + r01 * v1_1 + r02 * v1_2;
        const float cc02 = r00 * v2_0 + r01 * v2_1 + r02 * v2_2;
        const float cc11 = r10 * v1_0 + r11 * v1_1 + r12 * v1_2;
        const float cc12 = r10 * v2_0 + r11 * v2_1 + r12 * v2_2;
        const float cc22 = r20 * v2_0 + r21 * v2_1 + r22 * v2_2;

        // cov2d = J @ cov_cam @ J^T + 0.3 * I.
        // J is sparse (j01 = j10 = 0); expand by hand for fewer FLOPs.
        // Symmetric cov_cam: (2,0) = cc02, (2,1) = cc12.
        const float m00 = j00 * cc00 + j02 * cc02;
        const float m01 = j00 * cc01 + j02 * cc12;
        const float m02 = j00 * cc02 + j02 * cc22;
        const float m11 = j11 * cc11 + j12 * cc12;
        const float m12 = j11 * cc12 + j12 * cc22;

        const float a = m00 * j00 + m02 * j02 + 0.3f;
        const float b_canonical = m01 * j11 + m02 * j12;
        const float c_diag = m11 * j11 + m12 * j12 + 0.3f;
        covs_2d[i*4+0] = a;
        covs_2d[i*4+1] = b_canonical;
        covs_2d[i*4+2] = b_canonical;
        covs_2d[i*4+3] = c_diag;

        const float k_raw = opacities != nullptr
            ? opacity_aware_k(opacities[i], contrib_floor_k)
            : 3.0f;
        const float k = std::max(apply_k_cap(k_raw, k_cap), 3.0f);
        float rx;
        float ry;
        if (use_isoellipse) {
            float hx;
            float hy;
            isoellipse_aabb_half_extents(a, b_canonical, c_diag, k, hx, hy);
            rx = std::ceil(hx);
            ry = std::ceil(hy);
        } else {
            rx = std::ceil(k * std::sqrt(std::max(a, 0.0f)));
            ry = std::ceil(k * std::sqrt(std::max(c_diag, 0.0f)));
        }
        radii_scratch[i] = Vec2f{rx, ry};

        const bool valid = (mean_x + rx > 0.0f)
                        && (mean_x - rx < static_cast<float>(image_width))
                        && (mean_y + ry > 0.0f)
                        && (mean_y - ry < static_cast<float>(image_height))
                        && (rx > 0.0f) && (ry > 0.0f)
                        && (rx <= static_cast<float>(max_radius))
                        && (ry <= static_cast<float>(max_radius));
        valid_mask[i] = valid ? 1 : 0;
    };

    // iter-052: blocked dispatch (same rationale as iter-051 for gauss_rec).
    // The strided pattern caused false-sharing on means_2d/covs_2d/depths/
    // radii writes — each is contiguous per-Gaussian so adjacent Gaussian
    // writes from different workers shared cache lines. Blocked per-worker
    // chunks give each worker a private contiguous span -> no cross-core
    // line traffic + prefetcher-friendly sequential reads of all input
    // arrays. project_full_fused is the heaviest stage (~2.9 ms / frame)
    // so this is the highest-impact place to apply the pattern.
    if (pool != nullptr && pool->size() > 1 && N >= 4096) {
        const std::size_t W = pool->size();
        const std::size_t chunk_n = (N + W - 1) / W;
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([w, chunk_n, N, &per_gaussian]() {
                const std::size_t lo = w * chunk_n;
                const std::size_t hi = std::min(lo + chunk_n, N);
                for (std::size_t i = lo; i < hi; ++i) per_gaussian(i);
            });
        }
        pool->wait();
    } else {
        for (std::size_t i = 0; i < N; ++i) per_gaussian(i);
    }

    // Parallel-filter gather: count -> prefix -> scatter (iter-020 pattern).
    ProjectResult result;
    result.valid_mask = valid_mask;
    if (pool == nullptr || pool->size() <= 1 || N < 8192) {
        for (std::size_t i = 0; i < N; ++i) {
            if (!valid_mask[i]) continue;
            result.means_2d.push_back(means_2d[i*2+0]);
            result.means_2d.push_back(means_2d[i*2+1]);
            result.covs_2d.push_back(covs_2d[i*4+0]);
            result.covs_2d.push_back(covs_2d[i*4+1]);
            result.covs_2d.push_back(covs_2d[i*4+2]);
            result.covs_2d.push_back(covs_2d[i*4+3]);
            result.depths.push_back(depths[i]);
            result.radii.push_back(radii_scratch[i].x);
            result.radii.push_back(radii_scratch[i].y);
            if (opacities != nullptr) {
                result.opacities.push_back(opacities[i]);
            }
            if (colors != nullptr) {
                result.colors.push_back(colors[i * 3 + 0]);
                result.colors.push_back(colors[i * 3 + 1]);
                result.colors.push_back(colors[i * 3 + 2]);
            }
        }
        return result;
    }

    const std::size_t W = pool->size();
    const std::size_t chunk = (N + W - 1) / W;
    std::vector<std::size_t> chunk_counts(W, 0);
    std::vector<std::size_t> chunk_starts(W + 1, 0);

    for (std::size_t w = 0; w < W; ++w) {
        pool->submit([w, W, chunk, N, &valid_mask, &chunk_counts]() {
            const std::size_t lo = std::min(w * chunk, N);
            const std::size_t hi = std::min(lo + chunk, N);
            std::size_t cnt = 0;
            for (std::size_t i = lo; i < hi; ++i) cnt += (valid_mask[i] != 0);
            chunk_counts[w] = cnt;
        });
    }
    pool->wait();

    for (std::size_t w = 0; w < W; ++w) chunk_starts[w + 1] = chunk_starts[w] + chunk_counts[w];
    const std::size_t V = chunk_starts[W];
    result.means_2d.resize(V * 2);
    result.covs_2d.resize(V * 4);
    result.depths.resize(V);
    result.radii.resize(V * 2);
    if (opacities != nullptr) {
        result.opacities.resize(V);
    }
    if (colors != nullptr) {
        result.colors.resize(V * 3);
    }

    for (std::size_t w = 0; w < W; ++w) {
        pool->submit([w, W, chunk, N, &means_2d, &covs_2d, &depths, &radii_scratch,
                      colors, opacities, &valid_mask, &chunk_starts, &result]() {
            const std::size_t lo = std::min(w * chunk, N);
            const std::size_t hi = std::min(lo + chunk, N);
            std::size_t out = chunk_starts[w];
            for (std::size_t i = lo; i < hi; ++i) {
                if (!valid_mask[i]) continue;
                result.means_2d[out*2+0] = means_2d[i*2+0];
                result.means_2d[out*2+1] = means_2d[i*2+1];
                result.covs_2d[out*4+0] = covs_2d[i*4+0];
                result.covs_2d[out*4+1] = covs_2d[i*4+1];
                result.covs_2d[out*4+2] = covs_2d[i*4+2];
                result.covs_2d[out*4+3] = covs_2d[i*4+3];
                result.depths[out] = depths[i];
                result.radii[out*2+0] = radii_scratch[i].x;
                result.radii[out*2+1] = radii_scratch[i].y;
                if (opacities != nullptr) {
                    result.opacities[out] = opacities[i];
                }
                if (colors != nullptr) {
                    result.colors[out * 3 + 0] = colors[i * 3 + 0];
                    result.colors[out * 3 + 1] = colors[i * 3 + 1];
                    result.colors[out * 3 + 2] = colors[i * 3 + 2];
                }
                ++out;
            }
        });
    }
    pool->wait();

    return result;
}

}  // namespace gsplat_cpu
