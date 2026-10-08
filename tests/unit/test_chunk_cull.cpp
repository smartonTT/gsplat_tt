// Task #433: chunk frustum cull (render/host/chunk_cull.h).
// 1. Conservative: no gaussian whose per-axis 3-sigma rect (unclamped EWA Jacobian,
//    0.3 dilation, z > k_near) reaches the image sits in a culled tile, over random
//    clustered scenes and cameras looking at, beside and away from them.
// 2. The cull is not a no-op on those scenes (some tiles go).
// 3. The strided deal: list_count sums to S, core c's page holds list[c + j C],
//    and chunk k of core c keeps the same parity as in the uncut deal.
#include "render/host/chunk_cull.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static bool visible(const float* m, const float* c, const float r[9], const float tr[3], float fx, float fy,
                    float cx, float cy, float W, float H, float k_near) {
    double p[3];
    for (int a = 0; a < 3; ++a) p[a] = r[3 * a] * m[0] + r[3 * a + 1] * m[1] + r[3 * a + 2] * m[2] + tr[a];
    if (p[2] <= k_near) return false;
    const double S[3][3] = {{c[0], c[1], c[2]}, {c[1], c[3], c[4]}, {c[2], c[4], c[5]}};
    double RS[3][3] = {}, Sc[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) RS[i][j] += r[3 * i + k] * S[k][j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) Sc[i][j] += RS[i][k] * r[3 * j + k];
    const double z = p[2];
    const double J[2][3] = {{fx / z, 0, -fx * p[0] / (z * z)}, {0, fy / z, -fy * p[1] / (z * z)}};
    double e[2];
    for (int a = 0; a < 2; ++a) {
        double v = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) v += J[a][i] * Sc[i][j] * J[a][j];
        e[a] = 3.0 * std::sqrt(v + 0.3);
    }
    const double u = fx * p[0] / z + cx, v = fy * p[1] / z + cy;
    return u + e[0] > 0 && u - e[0] < W && v + e[1] > 0 && v - e[1] < H;
}

// Rotation from yaw / pitch (row-major world-to-camera).
static void rot(double yaw, double pitch, float r[9]) {
    const double cy = std::cos(yaw), sy = std::sin(yaw), cp = std::cos(pitch), sp = std::sin(pitch);
    const double Ry[9] = {cy, 0, -sy, 0, 1, 0, sy, 0, cy};
    const double Rx[9] = {1, 0, 0, 0, cp, -sp, 0, sp, cp};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += Rx[3 * i + k] * Ry[3 * k + j];
            r[3 * i + j] = static_cast<float>(s);
        }
}

static void conservative() {
    std::mt19937 g(433);
    std::uniform_real_distribution<double> U(-1, 1);
    constexpr uint32_t TE = 64, TILES = 400;
    constexpr std::size_t N = std::size_t{TE} * TILES - 37;  // ragged last tile
    std::vector<float> means(3 * N), cov(6 * N);
    for (uint32_t t = 0; t < TILES; ++t) {  // one cluster per tile (Morton-like)
        const double c0 = 20 * U(g), c1 = 5 * U(g), c2 = 20 * U(g), sz = 0.2 + std::abs(U(g));
        for (uint32_t i = t * TE; i < std::min<std::size_t>(N, (t + 1) * TE); ++i) {
            means[3 * i] = c0 + sz * U(g); means[3 * i + 1] = c1 + sz * U(g); means[3 * i + 2] = c2 + sz * U(g);
            const double s = std::pow(10.0, -2.5 + 2.0 * (U(g) + 1) / 2);  // sigma 0.003 .. 0.3
            const double a = s * s, o = 0.5 * a * U(g);
            const float cv[6] = {float(a), float(o), float(0.3 * o), float(a * (1 + 0.5 * U(g))), 0.f,
                                 float(a * (1 + 0.5 * U(g)))};
            for (int k = 0; k < 6; ++k) cov[6 * i + k] = cv[k];
        }
    }
    chunk_cull::Table T;
    chunk_cull::build(T, means.data(), cov.data(), N, TILES, TE, 1.5f);
    const float W = 1024, H = 1024, fx = 900, fy = 900, cx = 512, cy = 512, k_near = 0.2f;
    std::vector<uint32_t> surv;
    std::size_t culled_total = 0;
    for (int view = 0; view < 40; ++view) {
        float r[9];
        rot(0.157 * view, 0.3 * U(g), r);
        const float tr[3] = {float(3 * U(g)), float(U(g)), float(5 + 10 * U(g))};
        chunk_cull::survivors(T, r, tr, fx, fy, cx, cy, W, H, k_near, surv);
        std::vector<uint8_t> kept(TILES, 0);
        for (uint32_t t : surv) kept[t] = 1;
        for (std::size_t i = 1; i < surv.size(); ++i) CHECK(surv[i - 1] < surv[i]);
        culled_total += TILES - surv.size();
        for (std::size_t i = 0; i < N; ++i)
            if (!kept[i / TE] && visible(&means[3 * i], &cov[6 * i], r, tr, fx, fy, cx, cy, W, H, k_near)) {
                std::printf("FAIL view %d gaussian %zu visible in culled tile %zu\n", view, i, i / TE);
                ++fails;
                return;
            }
    }
    CHECK(culled_total > 0);
    // A non-finite member keeps its tile.
    means[3 * 5] = NAN;
    chunk_cull::build(T, means.data(), cov.data(), N, TILES, TE, 1.5f);
    float r[9];
    rot(3.14159, 0, r);  // looking away from everything
    const float tr[3] = {0, 0, -100};
    chunk_cull::survivors(T, r, tr, fx, fy, cx, cy, W, H, k_near, surv);
    CHECK(surv.size() == 1 && surv[0] == 0);
}

static void deal() {
    for (uint32_t C : {1u, 7u, 110u}) {
        for (std::size_t S : {std::size_t{0}, std::size_t{3}, std::size_t{110}, std::size_t{3631}}) {
            std::vector<uint32_t> surv(S);
            for (std::size_t i = 0; i < S; ++i) surv[i] = static_cast<uint32_t>(3 * i + 1);
            const uint32_t words = 64;
            if ((S + C - 1) / C > words) continue;
            std::vector<uint32_t> host;
            chunk_cull::deal(surv, C, words, host);
            std::size_t sum = 0;
            for (uint32_t c = 0; c < C; ++c) {
                const uint32_t n = chunk_cull::list_count(S, c, C);
                sum += n;
                for (uint32_t k = 0; k < n; ++k) CHECK(host[std::size_t{c} * words + k] == surv[c + std::size_t{k} * C]);
            }
            CHECK(sum == S);
        }
    }
}

int main() {
    conservative();
    deal();
    if (fails == 0) std::printf("ok\n");
    return fails == 0 ? 0 : 1;
}
