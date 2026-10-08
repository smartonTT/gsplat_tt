// Task #408: microblock-shape dispatch model (CPU only).
//
// For one projected view (project.py output) and each microblock shape, model
// the device blend's (record, microblock) dispatch stream:
//   * per tile, records in depth order (tile lists from the PRECULL rect);
//   * cull mask per record: keep block iff the ellipse m2 <= t, t = 2 ln(op /
//     floor) + 0.05 (kThrMargin), meets the block's pixel-centre box (the
//     band_batch SFPU cull and the box-min cull are the same test; here fp64);
//   * T early stop as alpha_blend_compute_mb.cpp: live mask rebuilt before
//     record g_seen when g_seen % 512 == 0 (g_seen > 0, per tile), bit m set
//     iff some pixel of block m has T >= eps (eps = 1/256); mask &= live;
//   * per pixel: power = A dx^2 + B dx dy + C dy^2, w = exp(min(power, 0)),
//     alpha = min(op w, 0.99), alpha < floor -> 0, RGB += alpha T c, T *= 1 - alpha
//     (fp32, no FMA contraction; same code for every shape).
// Shapes are ROWSxCOLS (4x8 = 4 rows by 8 columns, today's layout, bit m =
// (m / (32/cols)) row-band, (m % (32/cols)) column group). Pair ops count the
// device jump-walk unit: bits (2J, 2J+1), i.e. horizontally adjacent blocks
// (vertically adjacent for 32x1).
//
// Output: one line per shape with counts, plus image differences vs shape 0.
//   mbsim VIEW.bin [--threads N] [--period 512] [--eps 0.00390625] [--dump PREFIX]
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

struct Shape { int rows, cols; };
static const Shape kShapes[] = {{4, 8}, {8, 4}, {2, 16}, {16, 2}, {1, 32}, {32, 1}};
constexpr int NS = sizeof(kShapes) / sizeof(kShapes[0]);

struct Stats {
    uint64_t rec = 0, rec_cull = 0, rec_live = 0, disp = 0, disp_nocut = 0, pairs = 0,
             lanes_live = 0, disp_dead = 0, readbacks = 0;
    void add(const Stats& o) {
        rec += o.rec; rec_cull += o.rec_cull; rec_live += o.rec_live; disp += o.disp;
        disp_nocut += o.disp_nocut; pairs += o.pairs; lanes_live += o.lanes_live;
        disp_dead += o.disp_dead; readbacks += o.readbacks;
    }
};

static inline double boxmin_m2(double ca, double cb, double cc, double ulo, double uhi,
                               double vlo, double vhi) {
    const double uc = std::min(std::max(0.0, ulo), uhi);
    const double vc = std::min(std::max(0.0, vlo), vhi);
    const double vs = std::min(std::max(-cb * uc / cc, vlo), vhi);
    const double us = std::min(std::max(-cb * vc / ca, ulo), uhi);
    const double qv = ca * uc * uc + 2.0 * cb * uc * vs + cc * vs * vs;
    const double qh = ca * us * us + 2.0 * cb * us * vc + cc * vc * vc;
    return std::min(qv, qh);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: mbsim VIEW.bin [--threads N] [--period P] [--eps E] [--dump PREFIX]\n"); return 2; }
    int nthreads = (int)std::thread::hardware_concurrency();
    uint32_t period = 512;
    float eps = 0.00390625f;
    std::string dump;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--threads")) nthreads = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--period")) period = (uint32_t)std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--eps")) eps = (float)std::atof(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--dump")) dump = argv[i + 1];
    }
    FILE* fh = std::fopen(argv[1], "rb");
    if (!fh) { std::perror(argv[1]); return 1; }
    int32_t hdr[3]; float floor_f;
    if (std::fread(hdr, 4, 3, fh) != 3 || std::fread(&floor_f, 4, 1, fh) != 1) return 1;
    const int n = hdr[0], W = hdr[1], H = hdr[2];
    std::vector<float> g((size_t)n * 12);
    if (std::fread(g.data(), 4, g.size(), fh) != g.size()) return 1;
    std::fclose(fh);
    const int TX = W / 32, TY = H / 32, NT = TX * TY;

    // Tile lists (depth order, ties by index).
    std::vector<uint32_t> cnt(NT + 1, 0);
    auto rng = [&](int i, int& x0, int& x1, int& y0, int& y1) {
        const float* r = &g[(size_t)i * 12];
        const int xs = (int)r[10], ys = (int)r[11];
        x0 = xs % 1024; x1 = xs / 1024; y0 = ys % 1024; y1 = ys / 1024;
    };
    for (int i = 0; i < n; ++i) {
        int x0, x1, y0, y1; rng(i, x0, x1, y0, y1);
        for (int ty = y0; ty <= y1; ++ty) for (int tx = x0; tx <= x1; ++tx) cnt[ty * TX + tx + 1]++;
    }
    for (int t = 0; t < NT; ++t) cnt[t + 1] += cnt[t];
    std::vector<uint32_t> lst(cnt[NT]);
    {
        std::vector<uint32_t> pos(cnt.begin(), cnt.end() - 1);
        for (int i = 0; i < n; ++i) {
            int x0, x1, y0, y1; rng(i, x0, x1, y0, y1);
            for (int ty = y0; ty <= y1; ++ty) for (int tx = x0; tx <= x1; ++tx) lst[pos[ty * TX + tx]++] = (uint32_t)i;
        }
    }

    std::vector<float> img((size_t)NS * W * H * 3, 0.f), timg((size_t)NS * W * H, 0.f);
    std::vector<Stats> st((size_t)nthreads * NS);
    std::atomic<int> next{0};
    uint32_t eps_bits; std::memcpy(&eps_bits, &eps, 4);

    // Pass 1: depth-sort every tile list. Pass 2: one work item per (tile,
    // shape), biggest tiles first so the long tiles do not finish last.
    auto sort_pass = [&]() {
        for (;;) {
            const int t = next.fetch_add(1);
            if (t >= NT) break;
            uint32_t* L = &lst[cnt[t]];
            std::stable_sort(L, L + (cnt[t + 1] - cnt[t]),
                             [&](uint32_t a, uint32_t b) { return g[(size_t)a * 12 + 9] < g[(size_t)b * 12 + 9]; });
        }
    };
    {
        std::vector<std::thread> th;
        for (int i = 0; i < nthreads; ++i) th.emplace_back(sort_pass);
        for (auto& x : th) x.join();
    }
    std::vector<int> items;
    for (int t = 0; t < NT; ++t) for (int s = 0; s < NS; ++s) items.push_back(t * NS + s);
    std::stable_sort(items.begin(), items.end(), [&](int a, int b) {
        return cnt[a / NS + 1] - cnt[a / NS] > cnt[b / NS + 1] - cnt[b / NS];
    });
    next = 0;
    auto work = [&](int tid) {
        std::vector<uint32_t> masks;
        for (;;) {
            const int it = next.fetch_add(1);
            if (it >= (int)items.size()) break;
            const int t = items[it] / NS, s = items[it] % NS;
            const int tx = t % TX, ty = t / TX;
            const uint32_t* L = &lst[cnt[t]];
            const uint32_t nl = cnt[t + 1] - cnt[t];
            masks.assign(nl, 0u);
            {
                const int h = kShapes[s].rows, w = kShapes[s].cols, nbx = 32 / w;
                for (uint32_t k = 0; k < nl; ++k) {
                    const float* r = &g[(size_t)L[k] * 12];
                    const double u0 = (double)r[0] - 32.0 * tx, v0 = (double)r[1] - 32.0 * ty;
                    const double ca = -2.0 * r[2], cb = -(double)r[3], cc = -2.0 * r[4];
                    const double thr = 2.0 * std::log((double)r[5] / (double)floor_f) + 0.05;
                    if (thr < 0.0) continue;
                    uint32_t m = 0;
                    for (int b = 0; b < 32; ++b) {
                        const int bx = b % nbx, by = b / nbx;
                        const double ulo = bx * w + 0.5 - u0, uhi = bx * w + w - 0.5 - u0;
                        const double vlo = by * h + 0.5 - v0, vhi = by * h + h - 0.5 - v0;
                        if (boxmin_m2(ca, cb, cc, ulo, uhi, vlo, vhi) <= thr) m |= 1u << b;
                    }
                    masks[k] = m;
                }
            }
            {
                Stats& S = st[(size_t)tid * NS + s];
                const int h = kShapes[s].rows, w = kShapes[s].cols, nbx = 32 / w;
                // per block: pixel tile-local raster indices
                int pix[32][32];
                for (int b = 0; b < 32; ++b)
                    for (int q = 0; q < 32; ++q) {
                        const int c = (b % nbx) * w + q % w, rr = (b / nbx) * h + q / w;
                        pix[b][q] = rr * 32 + c;
                    }
                float T[1024], Rc[1024], Gc[1024], Bc[1024];
                for (int p = 0; p < 1024; ++p) { T[p] = 1.f; Rc[p] = Gc[p] = Bc[p] = 0.f; }
                uint32_t live = 0xFFFFFFFFu, g_seen = 0;
                for (uint32_t k = 0; k < nl; ++k) {
                    if (period != 0u && g_seen != 0u && g_seen % period == 0u) {
                        ++S.readbacks;
                        live = 0;
                        for (int b = 0; b < 32; ++b) {
                            bool any = false;
                            for (int q = 0; q < 32 && !any; ++q) any = T[pix[b][q]] >= eps;
                            if (any) live |= 1u << b;
                        }
                    }
                    ++g_seen;
                    ++S.rec;
                    const uint32_t m0 = masks[k];
                    if (m0) ++S.rec_cull;
                    S.disp_nocut += (uint32_t)__builtin_popcount(m0);
                    const uint32_t m = m0 & live;
                    if (!m) continue;
                    ++S.rec_live;
                    S.disp += (uint32_t)__builtin_popcount(m);
                    for (uint32_t pm = m; pm; ) {
                        const uint32_t b = (uint32_t)__builtin_ctz(pm) & ~1u;
                        ++S.pairs;
                        pm &= ~(3u << b);
                    }
                    const float* r = &g[(size_t)L[k] * 12];
                    const float mx = r[0] - 32.f * tx, my = r[1] - 32.f * ty;
                    const float A = r[2], B = r[3], C = r[4], op = r[5], cr = r[6], cg = r[7], cbv = r[8];
                    for (uint32_t pm = m; pm; pm &= pm - 1) {
                        const int b = __builtin_ctz(pm);
                        int nlive = 0;
                        for (int q = 0; q < 32; ++q) {
                            const int p = pix[b][q];
                            const float dx = (float)(p & 31) + 0.5f - mx;
                            const float dy = (float)(p >> 5) + 0.5f - my;
                            float power = A * (dx * dx);
                            power = power + B * (dx * dy);
                            power = power + C * (dy * dy);
                            power = std::min(power, 0.f);
                            float alpha = op * std::exp(power);
                            alpha = std::min(alpha, 0.99f);
                            if (alpha < floor_f) alpha = 0.f;
                            else ++nlive;
                            const float at = alpha * T[p];
                            Rc[p] = Rc[p] + at * cr;
                            Gc[p] = Gc[p] + at * cg;
                            Bc[p] = Bc[p] + at * cbv;
                            T[p] = T[p] * (1.f - alpha);
                        }
                        S.lanes_live += nlive;
                        if (!nlive) ++S.disp_dead;
                    }
                }
                for (int p = 0; p < 1024; ++p) {
                    const size_t X = (size_t)tx * 32 + (p & 31), Y = (size_t)ty * 32 + (p >> 5);
                    const size_t o = ((size_t)s * H + Y) * W + X;
                    img[o * 3] = Rc[p]; img[o * 3 + 1] = Gc[p]; img[o * 3 + 2] = Bc[p];
                    timg[o] = T[p];
                }
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nthreads; ++i) th.emplace_back(work, i);
    for (auto& x : th) x.join();

    const size_t NP = (size_t)W * H;
    std::printf("view n=%d tile_recs=%u\n", n, cnt[NT]);
    for (int s = 0; s < NS; ++s) {
        Stats S;
        for (int i = 0; i < nthreads; ++i) S.add(st[(size_t)i * NS + s]);
        uint64_t dbits = 0, du8 = 0, dtbits = 0;
        double se = 0, maxd = 0;
        for (size_t i = 0; i < NP * 3; ++i) {
            const float a = img[(size_t)s * NP * 3 + i], b = img[i];
            uint32_t ab, bb; std::memcpy(&ab, &a, 4); std::memcpy(&bb, &b, 4);
            if (ab != bb) ++dbits;
            auto q8 = [](float v) { return (int)std::lround(std::min(std::max(v, 0.f), 1.f) * 255.f); };
            if (q8(a) != q8(b)) ++du8;
            const double d = (double)a - b; se += d * d; maxd = std::max(maxd, std::fabs(d));
        }
        for (size_t i = 0; i < NP; ++i) if (timg[(size_t)s * NP + i] != timg[i]) ++dtbits;
        const double psnr = se > 0 ? 10.0 * std::log10(1.0 / (se / (NP * 3))) : 999.0;
        std::printf("shape %dx%d rec %llu rec_cull %llu rec_live %llu disp %llu disp_nocut %llu pairs %llu "
                    "lanes_live %llu disp_dead %llu readbacks %llu ch_diff_bits %llu ch_diff_u8 %llu "
                    "px_T_diff %llu maxabs %.3g psnr_vs_4x8 %.2f\n",
                    kShapes[s].rows, kShapes[s].cols, (unsigned long long)S.rec, (unsigned long long)S.rec_cull,
                    (unsigned long long)S.rec_live, (unsigned long long)S.disp, (unsigned long long)S.disp_nocut,
                    (unsigned long long)S.pairs, (unsigned long long)S.lanes_live, (unsigned long long)S.disp_dead,
                    (unsigned long long)S.readbacks, (unsigned long long)dbits, (unsigned long long)du8,
                    (unsigned long long)dtbits, maxd, psnr);
    }
    if (!dump.empty()) {
        for (int s = 0; s < NS; ++s) {
            const std::string fn = dump + "_" + std::to_string(kShapes[s].rows) + "x" + std::to_string(kShapes[s].cols) + ".f32";
            FILE* o = std::fopen(fn.c_str(), "wb");
            std::fwrite(&img[(size_t)s * NP * 3], 4, NP * 3, o);
            std::fclose(o);
        }
    }
    return 0;
}
