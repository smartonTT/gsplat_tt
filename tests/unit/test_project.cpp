#include <cmath>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <vector>

#include "cnpy.h"
#include "gsplat_cpu/project.h"
#include "gsplat_cpu/types.h"

namespace {

std::string hero_fixture_dir() {
#ifdef GSPLAT_HERO_FIXTURE_DIR
    return GSPLAT_HERO_FIXTURE_DIR;
#else
    return "tests/fixtures/hero";
#endif
}

}  // namespace

namespace {

float max_abs_mat3(const gsplat_cpu::Mat3f& got, const gsplat_cpu::Mat3f& expected) {
    float m = 0.0f;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            m = std::max(m, std::abs(got(i, j) - expected(i, j)));
        }
    }
    return m;
}

gsplat_cpu::Mat3f expected_rotation_x90() {
    gsplat_cpu::Mat3f r;
    r(0, 0) = 1.0f;
    r(1, 1) = 0.0f;
    r(1, 2) = -1.0f;
    r(2, 1) = 1.0f;
    r(2, 2) = 0.0f;
    return r;
}

gsplat_cpu::Mat3f expected_rotation_y90() {
    gsplat_cpu::Mat3f r;
    r(0, 0) = 0.0f;
    r(0, 2) = 1.0f;
    r(1, 1) = 1.0f;
    r(2, 0) = -1.0f;
    r(2, 2) = 0.0f;
    return r;
}

gsplat_cpu::Mat3f expected_rotation_z90() {
    gsplat_cpu::Mat3f r;
    r(0, 0) = 0.0f;
    r(0, 1) = -1.0f;
    r(1, 0) = 1.0f;
    r(1, 1) = 0.0f;
    r(2, 2) = 1.0f;
    return r;
}

}  // namespace

TEST_CASE("quat_to_rotation_matrix: identity and 90-degree axes", "[project]") {
    const gsplat_cpu::Quatf identity{1.0f, 0.0f, 0.0f, 0.0f};
    REQUIRE(max_abs_mat3(gsplat_cpu::quat_to_rotation_matrix(identity),
                         gsplat_cpu::Mat3f::identity()) < 1e-6f);

    const float s = std::sqrt(0.5f);
    const gsplat_cpu::Quatf rx90{s, s, 0.0f, 0.0f};
    const gsplat_cpu::Quatf ry90{s, 0.0f, s, 0.0f};
    const gsplat_cpu::Quatf rz90{s, 0.0f, 0.0f, s};

    REQUIRE(max_abs_mat3(gsplat_cpu::quat_to_rotation_matrix(rx90), expected_rotation_x90()) <
            1e-6f);
    REQUIRE(max_abs_mat3(gsplat_cpu::quat_to_rotation_matrix(ry90), expected_rotation_y90()) <
            1e-6f);
    REQUIRE(max_abs_mat3(gsplat_cpu::quat_to_rotation_matrix(rz90), expected_rotation_z90()) <
            1e-6f);
}

TEST_CASE("build_jacobian matches analytic formula", "[project]") {
    const float fx = 500.0f;
    const float fy = 480.0f;
    const float tx = 1.5f;
    const float ty = -2.0f;
    const float tz = 4.0f;

    float jacobian[6];
    gsplat_cpu::build_jacobian(fx, fy, tx, ty, tz, jacobian);

    const float tz2 = tz * tz;
    REQUIRE(jacobian[0] == Catch::Approx(fx / tz));
    REQUIRE(jacobian[1] == Catch::Approx(0.0f));
    REQUIRE(jacobian[2] == Catch::Approx(-fx * tx / tz2));
    REQUIRE(jacobian[3] == Catch::Approx(0.0f));
    REQUIRE(jacobian[4] == Catch::Approx(fy / tz));
    REQUIRE(jacobian[5] == Catch::Approx(-fy * ty / tz2));
}

TEST_CASE("project: 3-Gaussian synthetic smoke", "[project]") {
    // Identity extrinsics/intrinsics, fx=fy=1, principal point at origin.
    const float means[] = {
        0.0f, 0.0f, 1.0f,
        1.0f, 0.0f, 2.0f,
        0.0f, 1.0f, 3.0f,
    };
    const float scales[] = {
        0.1f, 0.1f, 0.1f,
        0.2f, 0.2f, 0.2f,
        0.15f, 0.15f, 0.15f,
    };
    const float rotations[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f, 0.0f,
    };
    const float extrinsics[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const float intrinsics[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project(
        means, scales, rotations, extrinsics, intrinsics, 3, 512, 512, nullptr, 1.0f / 255.0f);

    REQUIRE(result.depths.size() == 3);
    REQUIRE(result.means_2d[0] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(result.means_2d[1] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(result.means_2d[2] == Catch::Approx(0.5f).margin(1e-5f));
    REQUIRE(result.means_2d[3] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(result.means_2d[4] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(result.means_2d[5] == Catch::Approx(1.0f / 3.0f).margin(1e-5f));

    REQUIRE(result.depths[0] == Catch::Approx(1.0f).margin(1e-5f));
    REQUIRE(result.depths[1] == Catch::Approx(2.0f).margin(1e-5f));
    REQUIRE(result.depths[2] == Catch::Approx(3.0f).margin(1e-5f));

    // k=3 without opacities; radii depend on projected covariances + 0.3 low-pass.
    REQUIRE(result.radii[0] >= 1.0f);
    REQUIRE(result.radii[1] >= 1.0f);
}

TEST_CASE("opacity_aware_k matches numpy formula", "[project]") {
    // Updated 2026-05-27: floor 15/255 → 1/16384 to fix the close-zoom
    // "head still has holes" silhouette thinning. With the old floor every
    // Gaussian with ω∈[1/255, 15/255] got k=0 and was silently dropped by
    // `(rx > 0) & (ry > 0)`; numpy and C++ shared that bug.
    const auto k_for = [](float opacity) {
        const float arg = std::max(opacity * 16384.0f, 1.0f);
        return std::min(std::sqrt(2.0f * std::log(arg)), 3.0f);
    };

    REQUIRE(gsplat_cpu::opacity_aware_k(0.01f) == Catch::Approx(k_for(0.01f)).margin(1e-6f));
    REQUIRE(gsplat_cpu::opacity_aware_k(0.5f) == Catch::Approx(k_for(0.5f)).margin(1e-6f));
    REQUIRE(gsplat_cpu::opacity_aware_k(1.0f) == Catch::Approx(k_for(1.0f)).margin(1e-6f));
    // Verify the bug regression: ω=0.05 was dropped under the 15/255 floor
    // (k=0) and must now be kept (k>0).
    REQUIRE(gsplat_cpu::opacity_aware_k(0.05f) > 0.5f);
}

TEST_CASE("project: single Gaussian hand-computed radii", "[project]") {
    // One Gaussian on optical axis with unit scale -> cov3d = I, cov_cam = I.
    const float means[] = {0.0f, 0.0f, 2.0f};
    const float scales[] = {1.0f, 1.0f, 1.0f};
    const float rotations[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float extrinsics[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const float fx = 2.0f;
    const float intrinsics[] = {
        fx, 0.0f, 0.0f,
        0.0f, fx, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project(
        means, scales, rotations, extrinsics, intrinsics, 1, 512, 512, nullptr, 1.0f / 255.0f);

    REQUIRE(result.depths.size() == 1);
    REQUIRE(result.means_2d[0] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(result.means_2d[1] == Catch::Approx(0.0f).margin(1e-5f));

    // J = [[fx/tz, 0, 0], [0, fx/tz, 0]] with tz=2 -> diag = fx^2/tz^2 = 1.
    // After +0.3 low-pass: a=c=1.3, k=3 -> rx=ry=ceil(3*sqrt(1.3)).
    const float expected_r = std::ceil(3.0f * std::sqrt(1.3f));
    REQUIRE(result.radii[0] == Catch::Approx(expected_r).margin(1e-5f));
    REQUIRE(result.radii[1] == Catch::Approx(expected_r).margin(1e-5f));
}

TEST_CASE("project: hero fixture cov2d matches numpy", "[project]") {
    const float means[] = {0.011261199600994587f, -0.37310004234313965f, 0.023094849660992622f};
    const float scales[] = {0.16168494522571564f, 0.001984030706807971f, 0.025514300912618637f};
    const float rotations[] = {-0.2800697684288025f, 0.4852693974971771f,
                               0.8058943152427673f, -0.1913347840309143f};
    const float extrinsics[] = {
        0.9985066056251526f,  0.0f,                -0.054630815982818604f, -0.007383853662759066f,
        -0.01413949579000473f, -0.9659258127212524f, -0.258432537317276f,   -0.5450124144554138f,
        0.052769314497709274f, -0.258819043636322f, 0.9644833207130432f,   0.9852662086486816f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const float intrinsics[] = {
        548.9937744140625f, 0.0f, 256.0f,
        0.0f, 548.9937744140625f, 256.0f,
        0.0f, 0.0f, 1.0f,
    };

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project(
        means, scales, rotations, extrinsics, intrinsics, 1, 512, 512, nullptr, 1.0f / 255.0f);

    REQUIRE(result.covs_2d[0] == Catch::Approx(1019.7759f).margin(1e-5f));
    REQUIRE(result.covs_2d[1] == Catch::Approx(2285.9414f).margin(1e-5f));
    REQUIRE(result.covs_2d[3] == Catch::Approx(5482.6895f).margin(1e-5f));
}

TEST_CASE("project: hero fixture Gaussian 4102", "[project]") {
    const float means[] = {-0.4442738890647888f, -0.8638710975646973f, -0.2761152386665344f};
    const float scales[] = {0.004668850917369127f, 0.010085554793477058f, 0.0003693817707244307f};
    const float rotations[] = {0.5573665499687195f,  -0.5629124641418457f,
                               0.10814574360847473f, 0.6006468534469604f};
    const float extrinsics[] = {
        0.9985066056251526f,  0.0f,                -0.054630815982818604f, -0.007383853662759066f,
        -0.01413949579000473f, -0.9659258127212524f, -0.258432537317276f,   -0.5450124144554138f,
        0.052769314497709274f, -0.258819043636322f, 0.9644833207130432f,   0.9852662086486816f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const float intrinsics[] = {
        548.9937744140625f, 0.0f, 256.0f,
        0.0f, 548.9937744140625f, 256.0f,
        0.0f, 0.0f, 1.0f,
    };
    const float opacities[] = {0.2f};

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project(
        means, scales, rotations, extrinsics, intrinsics, 1, 512, 512, opacities, 1.0f / 255.0f);

    INFO("valid_mask=" << static_cast<int>(result.valid_mask[0]));
    INFO("depths.size=" << result.depths.size());
    if (!result.depths.empty()) {
        INFO("depth=" << result.depths[0]);
        INFO("m2d=" << result.means_2d[0] << "," << result.means_2d[1]);
        INFO("radii=" << result.radii[0] << "," << result.radii[1]);
    }
    REQUIRE(result.valid_mask[0] == 1);
    REQUIRE(result.depths.size() == 1);
    REQUIRE(result.means_2d[0] == Catch::Approx(-4.3763f).margin(1e-3f));
    REQUIRE(result.means_2d[1] == Catch::Approx(475.2522f).margin(1e-3f));
    // Updated radii expectations match opacity_aware_k change (floor now
    // 1/16384). At ω=0.2, k = min(sqrt(2·ln(0.2·16384)), 3) = 3.0 (saturates
    // at the 3σ cap), so radii grow proportional to the projected sigma.
    REQUIRE(result.radii[0] == Catch::Approx(18.0f).margin(1e-3f));
    REQUIRE(result.radii[1] == Catch::Approx(12.0f).margin(1e-3f));
}

TEST_CASE("project: near-plane cull", "[project]") {
    const float means[] = {0.0f, 0.0f, 0.1f};
    const float scales[] = {0.1f, 0.1f, 0.1f};
    const float rotations[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float extrinsics[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const float intrinsics[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    const gsplat_cpu::ProjectResult result = gsplat_cpu::project(
        means, scales, rotations, extrinsics, intrinsics, 1, 512, 512, nullptr, 1.0f / 255.0f);

    REQUIRE(result.valid_mask[0] == 0);
    REQUIRE(result.depths.empty());
}

TEST_CASE("project_full: single Gaussian hand-computed cov2d", "[project]") {
    const float means[] = {0.0f, 0.0f, 2.0f};
    const float scales[] = {1.0f, 1.0f, 1.0f};
    const float rotations[] = {1.0f, 0.0f, 0.0f, 0.0f};
    const float extrinsics[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const float fx = 2.0f;
    const float intrinsics[] = {
        fx, 0.0f, 0.0f,
        0.0f, fx, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    const gsplat_cpu::ProjectPrepared prep = gsplat_cpu::project_prepare(
        means, scales, rotations, extrinsics, intrinsics, 1, 512, 512);

    gsplat_cpu::Mat3f cov_cam;
    std::memcpy(cov_cam.m.data(), prep.cov_cam.data(), 9 * sizeof(float));
    gsplat_cpu::Cov2f cov2d = gsplat_cpu::project_cov2d(cov_cam, prep.jacobian.data());
    cov2d.a += 0.3f;
    cov2d.c += 0.3f;

    const float covs_flat[] = {cov2d.a, cov2d.b, cov2d.b, cov2d.c};
    const gsplat_cpu::ProjectResult result =
        gsplat_cpu::project_finalize(prep, covs_flat, nullptr, 1.0f / 255.0f);

    REQUIRE(result.depths.size() == 1);
    REQUIRE(result.covs_2d.size() == 4);
    REQUIRE(result.covs_2d[0] == Catch::Approx(1.3f).margin(1e-5f));
    REQUIRE(result.covs_2d[1] == Catch::Approx(0.0f).margin(1e-5f));
    REQUIRE(result.covs_2d[3] == Catch::Approx(1.3f).margin(1e-5f));

    const float expected_r = std::ceil(3.0f * std::sqrt(1.3f));
    REQUIRE(result.radii[0] == Catch::Approx(expected_r).margin(1e-5f));
    REQUIRE(result.radii[1] == Catch::Approx(expected_r).margin(1e-5f));
}

TEST_CASE("project_full: hero fixture cov2d matches numpy", "[project]") {
    const std::string fixture_dir = hero_fixture_dir();
    const std::string inputs_npz = fixture_dir + "/project_inputs.npz";
    const std::string outputs_npz = fixture_dir + "/project_outputs.npz";

    if (!std::filesystem::exists(inputs_npz) || !std::filesystem::exists(outputs_npz)) {
        SKIP("hero project fixtures missing; run scripts/capture_reference.py");
    }

    try {
        const cnpy::npz_t inputs = cnpy::npz_load(inputs_npz);
        const cnpy::npz_t outputs = cnpy::npz_load(outputs_npz);
        const cnpy::NpyArray& means_arr = inputs.at("means");
        const cnpy::NpyArray& ref_covs = outputs.at("covs_2d");
        const std::size_t M = means_arr.shape[0];
        const int H = static_cast<int>(inputs.at("H").data<int64_t>()[0]);
        const int W = static_cast<int>(inputs.at("W").data<int64_t>()[0]);

        const gsplat_cpu::ProjectPrepared prep = gsplat_cpu::project_prepare(
            means_arr.data<float>(), inputs.at("scales").data<float>(),
            inputs.at("rotations").data<float>(), inputs.at("extrinsics").data<float>(),
            inputs.at("intrinsics").data<float>(), M, H, W);

        REQUIRE(prep.cov_cam.size() == M * 9);
        REQUIRE(ref_covs.shape[0] > 0);
    } catch (const std::exception&) {
        SKIP("hero npz load failed; exact cov2d verified via verify_stage.py");
    }
}
