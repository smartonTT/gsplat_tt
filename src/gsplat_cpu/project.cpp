#include "gsplat_cpu/project.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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
    const std::vector<uint8_t>& valid_mask) {
    ProjectResult result;
    result.valid_mask = valid_mask;

    for (std::size_t i = 0; i < prep.N; ++i) {
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

    const float r00 = extrinsics[0];
    const float r01 = extrinsics[1];
    const float r02 = extrinsics[2];
    const float r10 = extrinsics[4];
    const float r11 = extrinsics[5];
    const float r12 = extrinsics[6];
    const float r20 = extrinsics[8];
    const float r21 = extrinsics[9];
    const float r22 = extrinsics[10];

    Mat3f r_mat;
    r_mat(0, 0) = r00;
    r_mat(0, 1) = r01;
    r_mat(0, 2) = r02;
    r_mat(1, 0) = r10;
    r_mat(1, 1) = r11;
    r_mat(1, 2) = r12;
    r_mat(2, 0) = r20;
    r_mat(2, 1) = r21;
    r_mat(2, 2) = r22;

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

ProjectPrepared project_prepare_geometry(
    const float* means,
    const float* extrinsics,
    const float* intrinsics,
    const std::size_t N,
    const int image_height,
    const int image_width) {
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
    transform_means_cam(means, extrinsics, N, means_cam);

    for (std::size_t i = 0; i < N; ++i) {
        const float tx = means_cam[i * 3 + 0];
        const float ty = means_cam[i * 3 + 1];
        const float tz = means_cam[i * 3 + 2];

        prep.near_valid[i] = tz > k_near ? 1 : 0;
        prep.depths[i] = tz;

        const float inv_tz = 1.0f / tz;
        prep.means_2d[i * 2 + 0] = fx * tx * inv_tz + cx;
        prep.means_2d[i * 2 + 1] = fy * ty * inv_tz + cy;

        build_jacobian(fx, fy, tx, ty, tz, prep.jacobian.data() + i * 6);
    }

    return prep;
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
    for (std::size_t i = 0; i < N; ++i) {
        Mat3f cov_cam;
        std::memcpy(cov_cam.m.data(), prep.cov_cam.data() + i * 9, 9 * sizeof(float));

        Cov2f cov2d = project_cov2d(cov_cam, prep.jacobian.data() + i * 6);
        cov2d.a += 0.3f;
        cov2d.c += 0.3f;
        covs_2d[i * 4 + 0] = cov2d.a;
        covs_2d[i * 4 + 1] = cov2d.b;
        covs_2d[i * 4 + 2] = cov2d.b;
        covs_2d[i * 4 + 3] = cov2d.c;
    }

    return project_finalize(prep, covs_2d.data(), opacities, min_opacity);
}

}  // namespace gsplat_cpu
