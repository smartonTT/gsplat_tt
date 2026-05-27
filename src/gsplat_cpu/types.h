#pragma once

#include <array>
#include <cmath>

namespace gsplat_cpu {

struct Vec2f {
    float x{0.0f};
    float y{0.0f};
};

struct Vec3f {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};

    Vec3f operator+(const Vec3f& o) const {
        return {x + o.x, y + o.y, z + o.z};
    }

    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
};

struct Quatf {
    float w{1.0f};
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
};

// Row-major 3x3 matrix: m[row * 3 + col].
struct Mat3f {
    std::array<float, 9> m{};

    float& operator()(int row, int col) { return m[static_cast<std::size_t>(row * 3 + col)]; }
    float operator()(int row, int col) const {
        return m[static_cast<std::size_t>(row * 3 + col)];
    }

    static Mat3f identity() {
        Mat3f r;
        r(0, 0) = r(1, 1) = r(2, 2) = 1.0f;
        return r;
    }
};

inline Mat3f mat3_mul(const Mat3f& a, const Mat3f& b) {
    Mat3f r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r(i, j) = a(i, 0) * b(0, j) + a(i, 1) * b(1, j) + a(i, 2) * b(2, j);
        }
    }
    return r;
}

inline Mat3f mat3_transpose(const Mat3f& a) {
    Mat3f r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r(i, j) = a(j, i);
        }
    }
    return r;
}

inline Mat3f quat_to_rotation_matrix(const Quatf& q) {
    const float q0 = q.w;
    const float q1 = q.x;
    const float q2 = q.y;
    const float q3 = q.z;

    Mat3f r;
    r(0, 0) = 1.0f - 2.0f * (q2 * q2 + q3 * q3);
    r(0, 1) = 2.0f * (q1 * q2 - q0 * q3);
    r(0, 2) = 2.0f * (q1 * q3 + q0 * q2);
    r(1, 0) = 2.0f * (q1 * q2 + q0 * q3);
    r(1, 1) = 1.0f - 2.0f * (q1 * q1 + q3 * q3);
    r(1, 2) = 2.0f * (q2 * q3 - q0 * q1);
    r(2, 0) = 2.0f * (q1 * q3 - q0 * q2);
    r(2, 1) = 2.0f * (q2 * q3 + q0 * q1);
    r(2, 2) = 1.0f - 2.0f * (q1 * q1 + q2 * q2);
    return r;
}

inline Mat3f build_covariance_3d(const Vec3f& scales, const Quatf& rot) {
    Mat3f r = quat_to_rotation_matrix(rot);
    // RS = R * diag(scales): column j scaled by scales[j].
    for (int col = 0; col < 3; ++col) {
        const float s = (&scales.x)[col];
        for (int row = 0; row < 3; ++row) {
            r(row, col) *= s;
        }
    }
    return mat3_mul(r, mat3_transpose(r));
}

// 2x2 symmetric covariance stored as [a, b, b, c].
struct Cov2f {
    float a{0.0f};
    float b{0.0f};
    float c{0.0f};
};

// Jacobian J is 2x3 row-major: j[row * 3 + col].
inline Cov2f project_cov2d(const Mat3f& cov_cam, const float* jacobian) {
    // M = J @ cov_cam  (2x3)
    float m00 = jacobian[0] * cov_cam(0, 0) + jacobian[1] * cov_cam(1, 0) +
                jacobian[2] * cov_cam(2, 0);
    float m01 = jacobian[0] * cov_cam(0, 1) + jacobian[1] * cov_cam(1, 1) +
                jacobian[2] * cov_cam(2, 1);
    float m02 = jacobian[0] * cov_cam(0, 2) + jacobian[1] * cov_cam(1, 2) +
                jacobian[2] * cov_cam(2, 2);
    float m10 = jacobian[3] * cov_cam(0, 0) + jacobian[4] * cov_cam(1, 0) +
                jacobian[5] * cov_cam(2, 0);
    float m11 = jacobian[3] * cov_cam(0, 1) + jacobian[4] * cov_cam(1, 1) +
                jacobian[5] * cov_cam(2, 1);
    float m12 = jacobian[3] * cov_cam(0, 2) + jacobian[4] * cov_cam(1, 2) +
                jacobian[5] * cov_cam(2, 2);

    // cov2d = M @ J^T
    Cov2f out;
    out.a = m00 * jacobian[0] + m01 * jacobian[1] + m02 * jacobian[2];
    out.b = m00 * jacobian[3] + m01 * jacobian[4] + m02 * jacobian[5];
    out.c = m10 * jacobian[3] + m11 * jacobian[4] + m12 * jacobian[5];
    return out;
}

inline void build_jacobian(float fx, float fy, float tx, float ty, float tz, float* jacobian) {
    const float tz2 = tz * tz;
    jacobian[0] = fx / tz;
    jacobian[1] = 0.0f;
    jacobian[2] = -fx * tx / tz2;
    jacobian[3] = 0.0f;
    jacobian[4] = fy / tz;
    jacobian[5] = -fy * ty / tz2;
}

// AABB σ-multiplier matching `rasterization.project_gaussians` (numpy).
// Uses contrib_floor = 1/16384, matching the per-microblock cull threshold —
// so the AABB never crops a tile/microblock pair the downstream cull would
// actually keep. The previous floor was 15/255 (≈ 0.0588): with that, every
// Gaussian with ω in [1/255, 15/255] got `k=0` (clamp(arg, 1) → 1 → ln=0),
// then was dropped because `(rx > 0) & (ry > 0)` was false. That silent
// drop is exactly what produced the "head still has holes" silhouette
// thinning at close zoom. The new floor + 3σ cap keeps every Gaussian
// above min_opacity (1/255) and capped at 3σ AABB for perf.
inline float opacity_aware_k(float opacity) {
    const float arg = std::max(opacity * 16384.0f, 1.0f);
    return std::min(std::sqrt(2.0f * std::log(arg)), 3.0f);
}

}  // namespace gsplat_cpu
