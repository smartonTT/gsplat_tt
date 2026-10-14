// Task #690 (PLAN-single-chip.md T7): device training loss, standalone bench.
//
//   loss_bench <in_dir> <out_dir> [iters=20]
//
// <in_dir> holds image.f32 and target.f32, (H, W, 3) float32 row-major, and hw.txt
// ("H W"), written by opt/t690/loss_check.py. The bench pads to 32x32 fp32 tiles
// (planar, channel-major, the device layout), runs the L1 step once to check, writes
// <out_dir>/grad.f32 (H, W, 3) and loss.txt, then runs `iters` back-to-back launches and
// prints one line:
//   [LOSS] mode=l1 H= W= tiles= cores= loss= dev_ms_med= dev_ms_min= iters=
// The python side compares grad.f32 and loss against gsplat/train_loss.py and the oracle.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t TS = 32, TILE_BYTES = TS * TS * 4;

// Element (r, c) of a 32x32 tile -> index in the 4-face fp32 tile layout.
inline uint32_t tidx(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

struct LossLayout {
    uint32_t H, W, nh, nw;
    uint32_t tiles() const { return 3 * nh * nw; }
};

std::vector<float> lb_tilize(const std::vector<float>& hwc, const LossLayout& L) {
    std::vector<float> out(size_t(L.tiles()) * TS * TS, 0.0f);
    for (uint32_t ch = 0; ch < 3; ch++)
        for (uint32_t r = 0; r < L.H; r++)
            for (uint32_t c = 0; c < L.W; c++) {
                const size_t t = (size_t(ch) * L.nh + r / TS) * L.nw + c / TS;
                out[t * TS * TS + tidx(r % TS, c % TS)] = hwc[(size_t(r) * L.W + c) * 3 + ch];
            }
    return out;
}

std::vector<float> lb_untilize(const std::vector<float>& tl, const LossLayout& L) {
    std::vector<float> out(size_t(L.H) * L.W * 3);
    for (uint32_t ch = 0; ch < 3; ch++)
        for (uint32_t r = 0; r < L.H; r++)
            for (uint32_t c = 0; c < L.W; c++) {
                const size_t t = (size_t(ch) * L.nh + r / TS) * L.nw + c / TS;
                out[(size_t(r) * L.W + c) * 3 + ch] = tl[t * TS * TS + tidx(r % TS, c % TS)];
            }
    return out;
}

std::vector<float> read_f32(const std::string& p, size_t n) {
    std::vector<float> v(n);
    std::ifstream f(p, std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(v.data()), n * 4)) {
        std::fprintf(stderr, "[LOSS] ERROR cannot read %zu floats from %s\n", n, p.c_str());
        std::exit(2);
    }
    return v;
}

std::shared_ptr<distributed::MeshBuffer> make_dram(distributed::MeshDevice* dev, uint64_t bytes) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{.page_size = TILE_BYTES, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

void add_cb(Program& p, const CoreRangeSet& cores, uint32_t id, uint32_t pages) {
    CircularBufferConfig c(pages * TILE_BYTES, {{id, DataFormat::Float32}});
    c.set_page_size(id, TILE_BYTES);
    CreateCircularBuffer(p, cores, c);
}

// hi part of a split3 operand: HI_MASK of filt_compute.cpp. LOSS_HIBITS explicit mantissa
// bits (default 9; 10 = exact TF32, the run-1971 setting).
uint32_t HI_MASK_U = 0xFFFFC000u;
inline float trunc_tf32(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u &= HI_MASK_U;
    std::memcpy(&v, &u, 4);
    return v;
}

// 12 constant band tiles for filt_compute.cpp: Rh[-1,0,1], Rl[...], RTh[...], RTl[...].
// R[k][a][b] = g[32 k + a - b + 5], g the normalized 11-tap Gaussian (sigma 1.5), as
// gsplat/train_loss.py band_tiles(); h = fp32 & HI_MASK_U, l = (fp32 - h) & HI_MASK_U.
// LOSS_RDIAG=<c> (#706 precision diagnostic): R[0] = c I, R[+-1] = 0, so G * x = c^2 x exactly.
// LOSS_RDIAG=-<c>: an 11-tap box with every tap = c (exact products, 11-term sums: tests accumulation).
// LOSS_RDIAG=t<preset> (#714): an 11-tap separable filter with chosen taps; the taps the device
// gets (h + l) go to <out>/taps.txt and loss_check.py uses them as the fp64 reference.
//   tg9 / tg5: Gaussian truncated to 9 / 5 explicit mantissa bits (Rl = 0, exact single products)
//   tpw: power-of-two taps of mixed size (exact products, mixed-magnitude sums)
//   tgl: only the Gaussian's lo parts (Rh = 0): the zh Rl products alone
std::vector<double> TAPS_USED;
std::vector<float> band_consts() {
    const char* rd = std::getenv("LOSS_RDIAG");
    std::string preset = (rd && rd[0] == 't') ? std::string(rd + 1) : std::string();
    preset = preset.substr(0, preset.find("_q"));
    if (!preset.empty()) rd = nullptr;
    const double diag = rd ? std::atof(rd) : 0.0;
    double g[11], sum = 0.0;
    for (int i = 0; i < 11; i++) sum += g[i] = std::exp(-double((i - 5) * (i - 5)) / (2.0 * 1.5 * 1.5));
    auto mask = [](float v, int bits) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        u &= ~((1u << (23 - bits)) - 1u);
        std::memcpy(&v, &u, 4);
        return v;
    };
    static const int pw_exp[11] = {10, 7, 5, 3, 2, 2, 2, 3, 5, 7, 10};
    float th[11], tl[11];
    for (int i = 0; i < 11; i++) {
        const float r = float(g[i] / sum);
        th[i] = trunc_tf32(r);
        tl[i] = trunc_tf32(r - th[i]);
        if (preset == "g9") th[i] = mask(r, 9), tl[i] = 0.0f;
        if (preset == "g5") th[i] = mask(r, 5), tl[i] = 0.0f;
        if (preset == "pw") th[i] = float(std::ldexp(1.0, -pw_exp[i])), tl[i] = 0.0f;
        if (preset == "gl") th[i] = 0.0f;
    }
    TAPS_USED.assign(11, 0.0);
    for (int i = 0; i < 11; i++) TAPS_USED[i] = double(th[i]) + double(tl[i]);
    std::vector<float> out(12 * TS * TS, 0.0f);
    for (int k = -1; k <= 1; k++)
        for (uint32_t a = 0; a < TS; a++)
            for (uint32_t b = 0; b < TS; b++) {
                const int off = int(TS) * k + int(a) - int(b) + 5;
                float r = (off >= 0 && off < 11) ? float(g[off] / sum) : 0.0f;
                if (rd && diag >= 0.0) r = (k == 0 && a == b) ? float(diag) : 0.0f;
                if (rd && diag < 0.0) r = (off >= 0 && off < 11) ? float(-diag) : 0.0f;
                float h = trunc_tf32(r), l = trunc_tf32(r - h);
                if (!rd) h = (off >= 0 && off < 11) ? th[off] : 0.0f, l = (off >= 0 && off < 11) ? tl[off] : 0.0f;
                const size_t t = size_t(k + 1);
                out[(0 + t) * TS * TS + tidx(a, b)] = h;
                out[(3 + t) * TS * TS + tidx(a, b)] = l;
                out[(6 + t) * TS * TS + tidx(b, a)] = h;  // R[k]^T
                out[(9 + t) * TS * TS + tidx(b, a)] = l;
            }
    return out;
}

double time_b2b(distributed::MeshCommandQueue& cq, distributed::MeshWorkload& wl, int iters,
                std::vector<double>& ms) {
    for (int i = 0; i < iters; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, wl, false);
        distributed::Finish(cq);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    const auto b0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);
    std::sort(ms.begin(), ms.end());
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count() /
           std::max(1, iters);
}

// LOSS_MODE=filt (task #706, D-SSIM step 2): G * x per channel plane, split3 banded matmuls,
// one strip (channel, tile column) per core. Writes <out>/filt.f32 (H, W, 3).
int run_filt(distributed::MeshDevice* dev, const std::vector<float>& x, const LossLayout& L,
             const std::string& out, int iters) {
    auto& cq = dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    const uint32_t ncores = grid.x * grid.y, T = L.tiles(), S = 3 * L.nw, NT = S * L.nh;
    auto bx = make_dram(dev, uint64_t(T) * TILE_BYTES), bo = make_dram(dev, uint64_t(T) * TILE_BYTES);
    auto br = make_dram(dev, 12ull * TILE_BYTES);
    // LOSS_RDIAG=t<preset>_q<n> (#714): input quantised to multiples of 2^-n (fixed point), to test
    // whether the FPU sums fixed-point products exactly. loss_check.py reads quant.txt.
    const char* rq = std::getenv("LOSS_RDIAG");
    const char* qp = rq ? std::strstr(rq, "_q") : nullptr;
    std::vector<float> xq(x);
    if (qp) {
        const int qn = std::atoi(qp + 2);
        for (auto& v : xq) v = float(std::floor(double(v) * std::ldexp(1.0, qn)) * std::ldexp(1.0, -qn));
        std::ofstream(out + "/quant.txt") << qn << "\n";
    }
    distributed::EnqueueWriteMeshBuffer(cq, bx, xq, true);
    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    add_cb(prog, cores, 0, 6);   // X (u2d)
    add_cb(prog, cores, 2, 12);  // band constants
    add_cb(prog, cores, 4, 3);   // XH
    add_cb(prog, cores, 5, 3);   // XL
    for (uint32_t q = 0; q < 3; q++) {
        add_cb(prog, cores, 6 + q, 1);  // row-pass slot q, hi
        add_cb(prog, cores, 9 + q, 1);  // row-pass slot q, lo
    }
    add_cb(prog, cores, 16, 2);  // out
    std::vector<uint32_t> ct_r, ct_w;
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_r);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_r);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_w);
    const std::string kd = LOSS_KERNEL_DIR;
    auto kr = CreateKernel(prog, kd + "reader_filt.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                              .noc = NOC::RISCV_1_default, .compile_args = ct_r});
    auto kw = CreateKernel(prog, kd + "writer_filt.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                              .noc = NOC::RISCV_0_default, .compile_args = ct_w});
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[0] = UnpackToDestMode::UnpackToDestFp32;
    const char* hb = std::getenv("LOSS_HIBITS");
    const int hibits = hb ? std::atoi(hb) : 9;
    HI_MASK_U = ~((1u << (23 - hibits)) - 1u);
    auto kc = CreateKernel(prog, kd + "filt_compute.cpp", cores,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .math_approx_mode = false,
                                         .defines = {{"HI_MASK", std::to_string(int32_t(HI_MASK_U))}}});
    distributed::EnqueueWriteMeshBuffer(cq, br, band_consts(), true);
    for (uint32_t cy = 0, k = 0; cy < grid.y; cy++)
        for (uint32_t cx = 0; cx < grid.x; cx++, k++) {
            // balanced: contiguous runs of output tiles (strip-major), not whole strips
            const uint32_t t0 = uint32_t(uint64_t(NT) * k / ncores), t1 = uint32_t(uint64_t(NT) * (k + 1) / ncores);
            const CoreCoord c{cx, cy};
            SetRuntimeArgs(prog, kr, c, {uint32_t(bx->address()), uint32_t(br->address()), t0, t1, L.nh, L.nw});
            SetRuntimeArgs(prog, kw, c, {uint32_t(bo->address()), t0, t1, L.nh, L.nw});
            SetRuntimeArgs(prog, kc, c, {t0, t1, L.nh});
        }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(prog));
    std::fprintf(stderr, "[LOSS] stage=filt_program strips=%u cores=%u\n", S, ncores);
    distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);
    std::fprintf(stderr, "[LOSS] stage=launch1_done\n");
    std::vector<float> o;
    distributed::EnqueueReadMeshBuffer(cq, o, bo, true);
    {
        const auto oh = lb_untilize(o, L);
        std::ofstream f(out + "/filt.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(oh.data()), oh.size() * 4);
    }
    if (std::getenv("LOSS_RDIAG") && std::getenv("LOSS_RDIAG")[0] == 't') {
        std::ofstream f(out + "/taps.txt");
        f.precision(17);
        for (double v : TAPS_USED) f << v << "\n";
    }
    std::vector<double> ms;
    const double b2b = time_b2b(cq, wl, iters, ms);
    std::printf("[LOSS] mode=filt hibits=%d H=%u W=%u tiles=%u strips=%u cores=%u b2b_ms=%.4f launch_ms_med=%.4f "
                "launch_ms_min=%.4f iters=%d\n",
                hibits, L.H, L.W, T, S, ncores, b2b, ms.empty() ? 0.0 : ms[ms.size() / 2], ms.empty() ? 0.0 : ms[0],
                iters);
    return 0;
}

// LOSS_MODE=sfilt (task #720): the same 11-tap Gaussian on the SFPU in fp32 FMAs
// (sfilt_compute.cpp), on P arbitrary padded planes: <in>/planes.txt ("P Hp Wp") and
// <in>/planes.f32 (P, Hp, Wp) float32, written by opt/t720/sfilt_check.py. Writes
// <out>/planes_out.f32 (same shape) unless LOSS_NOOUT=1, and <out>/sf_taps.txt (fp32 taps).
int run_sfilt(distributed::MeshDevice* dev, const std::string& in, const std::string& out, int iters) {
    uint32_t P = 0, Hp = 0, Wp = 0;
    {
        std::ifstream f(in + "/planes.txt");
        if (!(f >> P >> Hp >> Wp) || Hp % TS || Wp % TS) {
            std::fprintf(stderr, "[LOSS] ERROR bad %s/planes.txt\n", in.c_str());
            return 2;
        }
    }
    const uint32_t nh = Hp / TS, nw = Wp / TS, NT = P * nh * nw, S = P * nw;
    const auto pl = read_f32(in + "/planes.f32", size_t(P) * Hp * Wp);
    std::vector<float> x(size_t(NT) * TS * TS);
    for (uint32_t p = 0; p < P; p++)
        for (uint32_t r = 0; r < Hp; r++)
            for (uint32_t c = 0; c < Wp; c++)
                x[((size_t(p) * nh + r / TS) * nw + c / TS) * TS * TS + tidx(r % TS, c % TS)] =
                    pl[(size_t(p) * Hp + r) * Wp + c];
    auto& cq = dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    const uint32_t ncores = grid.x * grid.y;
    auto bx = make_dram(dev, uint64_t(NT) * TILE_BYTES), bo = make_dram(dev, uint64_t(NT) * TILE_BYTES);
    distributed::EnqueueWriteMeshBuffer(cq, bx, x, true);
    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    add_cb(prog, cores, 0, 6);  // X (u2d)
    for (uint32_t q = 0; q < 3; q++) add_cb(prog, cores, 6 + q, 1);  // H-pass slot q (u2d)
    add_cb(prog, cores, 16, 2);  // out
    std::vector<uint32_t> ct_r, ct_w;
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_r);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_r);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_w);
    const std::string kd = LOSS_KERNEL_DIR;
    auto kr = CreateKernel(prog, kd + "reader_filt.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                              .noc = NOC::RISCV_1_default, .compile_args = ct_r});
    auto kw = CreateKernel(prog, kd + "writer_filt.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                              .noc = NOC::RISCV_0_default, .compile_args = ct_w});
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    for (uint32_t cb : {0u, 6u, 7u, 8u}) u2d[cb] = UnpackToDestMode::UnpackToDestFp32;
    // fp32 taps g0 (offset 5) .. g5 (centre), as hex-float literals (exact).
    double g[11], sum = 0.0;
    for (int i = 0; i < 11; i++) sum += g[i] = std::exp(-double((i - 5) * (i - 5)) / (2.0 * 1.5 * 1.5));
    std::map<std::string, std::string> defs;
    std::ofstream tf(out + "/sf_taps.txt");
    tf.precision(17);
    for (int i = 0; i < 6; i++) {
        const float v = float(g[i] / sum);
        char b[48];
        std::snprintf(b, sizeof b, "%af", double(v));
        defs["SF_G" + std::to_string(i)] = b;
        tf << double(v) << "\n";
    }
    // LOSS_SF_CMODE: 1 = taps g3..g5 hoisted into LRegs (default), 0 = reloaded per row (same op order).
    const char* cm = std::getenv("LOSS_SF_CMODE");
    defs["SF_CMODE"] = (cm && cm[0] == '0') ? "0" : "1";
    std::fprintf(stderr, "[LOSS] sfilt SF_CMODE=%s\n", defs["SF_CMODE"].c_str());
    auto kc = CreateKernel(prog, kd + "sfilt_compute.cpp", cores,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .math_approx_mode = false, .defines = defs});
    for (uint32_t cy = 0, k = 0; cy < grid.y; cy++)
        for (uint32_t cx = 0; cx < grid.x; cx++, k++) {
            const uint32_t t0 = uint32_t(uint64_t(NT) * k / ncores), t1 = uint32_t(uint64_t(NT) * (k + 1) / ncores);
            const CoreCoord c{cx, cy};
            SetRuntimeArgs(prog, kr, c, {uint32_t(bx->address()), 0u, t0, t1, nh, nw});
            SetRuntimeArgs(prog, kw, c, {uint32_t(bo->address()), t0, t1, nh, nw});
            SetRuntimeArgs(prog, kc, c, {t0, t1, nh});
        }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(prog));
    std::fprintf(stderr, "[LOSS] stage=sfilt_program planes=%u strips=%u cores=%u\n", P, S, ncores);
    distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);
    std::fprintf(stderr, "[LOSS] stage=launch1_done\n");
    const char* no = std::getenv("LOSS_NOOUT");
    if (!(no && no[0] == '1')) {
        std::vector<float> o;
        distributed::EnqueueReadMeshBuffer(cq, o, bo, true);
        std::vector<float> oh(size_t(P) * Hp * Wp);
        for (uint32_t p = 0; p < P; p++)
            for (uint32_t r = 0; r < Hp; r++)
                for (uint32_t c = 0; c < Wp; c++)
                    oh[(size_t(p) * Hp + r) * Wp + c] =
                        o[((size_t(p) * nh + r / TS) * nw + c / TS) * TS * TS + tidx(r % TS, c % TS)];
        std::ofstream f(out + "/planes_out.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(oh.data()), oh.size() * 4);
    }
    std::vector<double> ms;
    const double b2b = time_b2b(cq, wl, iters, ms);
    std::printf("[LOSS] mode=sfilt P=%u Hp=%u Wp=%u tiles=%u strips=%u cores=%u b2b_ms=%.4f launch_ms_med=%.4f "
                "launch_ms_min=%.4f iters=%d\n",
                P, Hp, Wp, NT, S, ncores, b2b, ms.empty() ? 0.0 : ms[ms.size() / 2], ms.empty() ? 0.0 : ms[0],
                iters);
    return 0;
}

// LOSS_MODE=ssim (task #736): fused L1 + D-SSIM loss and dL/dimage on the SFPU fp32 filter of
// #720. Two programs back to back (ssim_fwd_compute.cpp: filtered maps, SSIM, loss partials, d
// maps; ssim_bwd_compute.cpp: filtered d maps -> dL/dx). <in>: hw.txt, image.f32, target.f32
// (H, W, 3) and consts.f32 (2, 3, Hp, Wp): mu_y, vy (per-view constants), written by
// opt/t736/ssim_check.py prep. Writes <out>/grad.f32 (H, W, 3) and <out>/loss.txt.
std::string hexf(double v) {
    char b[48];
    std::snprintf(b, sizeof b, "%af", double(float(v)));
    return b;
}

int run_ssim(distributed::MeshDevice* dev, const std::string& in, const std::string& out, int iters) {
    LossLayout L{};
    {
        std::ifstream f(in + "/hw.txt");
        if (!(f >> L.H >> L.W)) {
            std::fprintf(stderr, "[LOSS] ERROR bad %s/hw.txt\n", in.c_str());
            return 2;
        }
    }
    L.nh = (L.H + TS - 1) / TS;
    L.nw = (L.W + TS - 1) / TS;
    const uint32_t nh = L.nh, nw = L.nw, NP = nh * nw, NT = 3 * NP, Hp = nh * TS, Wp = nw * TS;
    const size_t npx = size_t(L.H) * L.W * 3;
    const double n = double(npx), lam = 0.2;
    const auto x = lb_tilize(read_f32(in + "/image.f32", npx), L);
    const auto y = lb_tilize(read_f32(in + "/target.f32", npx), L);
    const auto kc = read_f32(in + "/consts.f32", size_t(6) * Hp * Wp);
    std::vector<float> k(size_t(7) * NP * TS * TS, 0.0f);
    for (uint32_t p = 0; p < 7; p++)
        for (uint32_t r = 0; r < Hp; r++)
            for (uint32_t c = 0; c < Wp; c++)
                k[((size_t(p) * nh + r / TS) * nw + c / TS) * TS * TS + tidx(r % TS, c % TS)] =
                    p < 6 ? kc[(size_t(p) * Hp + r) * Wp + c] : (r < L.H && c < L.W ? float(-lam / n) : 0.0f);
    auto& cq = dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    const uint32_t ncores = grid.x * grid.y;
    auto bx = make_dram(dev, uint64_t(NT) * TILE_BYTES), by = make_dram(dev, uint64_t(NT) * TILE_BYTES);
    auto bk = make_dram(dev, uint64_t(7) * NP * TILE_BYTES), bd = make_dram(dev, uint64_t(9) * NP * TILE_BYTES);
    auto bg = make_dram(dev, uint64_t(NT) * TILE_BYTES), bs = make_dram(dev, uint64_t(ncores) * TILE_BYTES);
    distributed::EnqueueWriteMeshBuffer(cq, bx, x, true);
    distributed::EnqueueWriteMeshBuffer(cq, by, y, true);
    distributed::EnqueueWriteMeshBuffer(cq, bk, k, true);

    double g[11], sum = 0.0;
    for (int i = 0; i < 11; i++) sum += g[i] = std::exp(-double((i - 5) * (i - 5)) / (2.0 * 1.5 * 1.5));
    std::map<std::string, std::string> defs;
    for (int i = 0; i < 6; i++) defs["SF_G" + std::to_string(i)] = hexf(g[i] / sum);
    defs["SS_C1"] = hexf(1e-4);
    defs["SS_C2"] = hexf(9e-4);
    defs["SS_WL1"] = hexf((1.0 - lam) / n);
    if (const char* a = std::getenv("LOSS_SSIM_ABL"); a && a[0] && std::string(a) != "0") {
        defs["SSIM_ABL"] = a;  // #743 timing ablation only (garbage results)
        std::fprintf(stderr, "[LOSS] SSIM_ABL=%s (timing ablation, results invalid)\n", a);
    }
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const std::string kd = LOSS_KERNEL_DIR;
    // #743 LOSS_SSIM_RD (default 1): reader_ssim.cpp SSIM_RD (fewer DRAM reads, same data); 0 = t736 reader.
    const char* rde = std::getenv("LOSS_SSIM_RD");
    const bool rd = !(rde && rde[0] == '0');
    // #743 LOSS_SSIM_CBD: reader input CBs hold this many rows / output tiles (default 2).
    const char* cbe = std::getenv("LOSS_SSIM_CBD");
    const uint32_t cbd = cbe && cbe[0] ? uint32_t(std::max(2, std::atoi(cbe))) : 2u;
    // #753 LOSS_SSIM_K (default 4, #758): each core walks strips of K adjacent tile columns (shared
    // neighbour tiles in the H pass); work is split in units of one tile row of a K-column group.
    const char* ke = std::getenv("LOSS_SSIM_K");
    const uint32_t K = ke && ke[0] ? uint32_t(std::clamp(std::atoi(ke), 1, 8)) : 4u;
    defs["SSIM_K"] = std::to_string(K);
    // #753 LOSS_SSIM_VF (default 1, #758; 0 = off): each V pass in one dest acquire (no CB_M / CB_F / CB_PQ round trips).
    const char* vfe = std::getenv("LOSS_SSIM_VF");
    const bool vf = !(vfe && vfe[0] == '0');
    if (vf) defs["SSIM_VF"] = "1";
    const uint32_t ng = (nw + K - 1) / K, NU = 3 * ng * nh;
    std::vector<uint32_t> ub(ncores + 1, NU);  // core c: units [ub[c], ub[c + 1]), split by tile count
    for (uint32_t u = 0, c = 0, acc = 0; u < NU && c <= ncores; u++) {
        while (c <= ncores && acc >= uint32_t(uint64_t(NT) * c / ncores)) ub[c++] = u;
        acc += std::min(K, nw - (u / nh % ng) * K);
    }
    ub[0] = 0;
    std::fprintf(stderr, "[LOSS] ssim reader SSIM_RD=%d cbd=%u K=%u VF=%d units=%u\n", int(rd), cbd, K, int(vf), NU);
    float wmf = float(-lam / n);
    uint32_t wm_bits;
    std::memcpy(&wm_bits, &wmf, 4);
    auto mk = [&](bool bwd) {
        Program prog = CreateProgram();
        std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
        std::vector<std::pair<uint32_t, uint32_t>> cbs;
        if (bwd) {
            cbs = {{0, 3 * (K + 2) * cbd}, {1, 2 * K * cbd}, {16, 2}, {17, 2}};
        } else {
            cbs = {{0, (K + 2) * cbd}, {1, (K + 2) * cbd}, {2, 2 * K * cbd}, {3, 3 * K * cbd}, {16, 6}, {17, 2}, {18, 4}, {19, 1}, {20, 1}};
            if (rd) cbs.push_back({4, 1});  // reader's constant wm tile (never pushed)
        }
        for (uint32_t q = 6; q < 15; q++) cbs.push_back({q, K});
        for (auto [id, pages] : cbs) {
            add_cb(prog, cores, id, pages);
            if (id != 16 && id != 20) u2d[id] = UnpackToDestMode::UnpackToDestFp32;
        }
        std::vector<uint32_t> ct_r, ct_w;
        for (int i = 0; i < 3; i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct_r);
        for (int i = 0; i < (bwd ? 1 : 2); i++) TensorAccessorArgs::create_dram_interleaved().append_to(ct_w);
        std::map<std::string, std::string> rdefs{{"SSIM_K", std::to_string(K)}};
        if (bwd) rdefs["SSIM_BWD"] = "1";
        if (rd) rdefs["SSIM_RD"] = "1";
        std::map<std::string, std::string> wdefs{{"SSIM_K", std::to_string(K)}};
        if (const char* a = std::getenv("LOSS_SSIM_DMABL"); a && a[0] && std::string(a) != "0") {
            rdefs["SSIM_DMABL"] = wdefs["SSIM_DMABL"] = a;  // #743 timing ablation only
            if (!bwd) std::fprintf(stderr, "[LOSS] SSIM_DMABL=%s (timing ablation, results invalid)\n", a);
        }
        auto kr = CreateKernel(prog, kd + "reader_ssim.cpp", cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                                  .noc = NOC::RISCV_1_default, .compile_args = ct_r,
                                                  .defines = rdefs});
        auto kw = CreateKernel(prog, kd + (bwd ? "writer_filt.cpp" : "writer_ssim_fwd.cpp"), cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                                  .noc = NOC::RISCV_0_default, .compile_args = ct_w,
                                                  .defines = wdefs});
        auto kcmp = CreateKernel(prog, kd + (bwd ? "ssim_bwd_compute.cpp" : "ssim_fwd_compute.cpp"), cores,
                                 ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                               .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                               .math_approx_mode = false, .defines = defs});
        for (uint32_t cy = 0, c = 0; cy < grid.y; cy++)
            for (uint32_t cx = 0; cx < grid.x; cx++, c++) {
                const uint32_t t0 = ub[c], t1 = ub[c + 1];  // work units (#753)
                const CoreCoord cc{cx, cy};
                if (bwd) {
                    SetRuntimeArgs(prog, kr, cc, {uint32_t(bd->address()), uint32_t(bx->address()),
                                                  uint32_t(by->address()), t0, t1, nh, nw});
                    SetRuntimeArgs(prog, kw, cc, {uint32_t(bg->address()), t0, t1, nh, nw});
                } else {
                    SetRuntimeArgs(prog, kr, cc, {uint32_t(bx->address()), uint32_t(by->address()),
                                                  uint32_t(bk->address()), t0, t1, nh, nw, wm_bits,
                                                  L.H / TS, L.W / TS});
                    SetRuntimeArgs(prog, kw, cc, {uint32_t(bd->address()), uint32_t(bs->address()), t0, t1, nh, nw, c});
                }
                SetRuntimeArgs(prog, kcmp, cc, {t0, t1, nh, nw});
            }
        distributed::MeshWorkload wl;
        wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(prog));
        return wl;
    };
    distributed::MeshWorkload wf = mk(false), wb = mk(true);
    std::fprintf(stderr, "[LOSS] stage=ssim_programs H=%u W=%u tiles=%u cores=%u\n", L.H, L.W, NT, ncores);
    distributed::EnqueueMeshWorkload(cq, wf, false);
    distributed::EnqueueMeshWorkload(cq, wb, false);
    distributed::Finish(cq);
    std::fprintf(stderr, "[LOSS] stage=launch1_done\n");
    std::vector<float> o, s;
    distributed::EnqueueReadMeshBuffer(cq, o, bg, true);
    distributed::EnqueueReadMeshBuffer(cq, s, bs, true);
    double loss = lam;
    for (float v : s) loss += v;
    {
        const auto gh = lb_untilize(o, L);
        std::ofstream f(out + "/grad.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(gh.data()), gh.size() * 4);
        std::ofstream lf(out + "/loss.txt");
        lf.precision(17);
        lf << loss << "\n";
    }
    auto pair_b2b = [&](std::vector<double>& ms) {
        for (int i = 0; i < iters; i++) {
            const auto t0 = std::chrono::steady_clock::now();
            distributed::EnqueueMeshWorkload(cq, wf, false);
            distributed::EnqueueMeshWorkload(cq, wb, false);
            distributed::Finish(cq);
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        const auto b0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) {
            distributed::EnqueueMeshWorkload(cq, wf, false);
            distributed::EnqueueMeshWorkload(cq, wb, false);
        }
        distributed::Finish(cq);
        std::sort(ms.begin(), ms.end());
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count() /
               std::max(1, iters);
    };
    std::vector<double> ms, msf;
    const double b2b = pair_b2b(ms);
    const double fwd = time_b2b(cq, wf, iters, msf);
    std::printf("[LOSS] mode=ssim H=%u W=%u tiles=%u cores=%u loss=%.9g b2b_ms=%.4f fwd_b2b_ms=%.4f "
                "launch_ms_med=%.4f launch_ms_min=%.4f iters=%d\n",
                L.H, L.W, NT, ncores, loss, b2b, fwd, ms.empty() ? 0.0 : ms[ms.size() / 2],
                ms.empty() ? 0.0 : ms[0], iters);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: loss_bench <in_dir> <out_dir> [iters]\n");
        return 2;
    }
    const std::string in = argv[1], out = argv[2];
    const int iters = argc > 3 ? std::atoi(argv[3]) : 20;
    if (const char* m = std::getenv("LOSS_MODE"); m && std::string(m) == "ssim") {
        auto dev = distributed::MeshDevice::create_unit_mesh(0);
        const int rc = run_ssim(dev.get(), in, out, iters);
        dev->close();
        return rc;
    }
    if (const char* m = std::getenv("LOSS_MODE"); m && std::string(m) == "sfilt") {
        auto dev = distributed::MeshDevice::create_unit_mesh(0);
        const int rc = run_sfilt(dev.get(), in, out, iters);
        dev->close();
        return rc;
    }
    LossLayout L{};
    {
        std::ifstream f(in + "/hw.txt");
        if (!(f >> L.H >> L.W)) {
            std::fprintf(stderr, "[LOSS] ERROR bad %s/hw.txt\n", in.c_str());
            return 2;
        }
    }
    L.nh = (L.H + TS - 1) / TS;
    L.nw = (L.W + TS - 1) / TS;
    const size_t npx = size_t(L.H) * L.W * 3;
    const auto x = lb_tilize(read_f32(in + "/image.f32", npx), L);
    const auto y = lb_tilize(read_f32(in + "/target.f32", npx), L);

    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    if (const char* m = std::getenv("LOSS_MODE"); m && std::string(m) == "filt") {
        const int rc = run_filt(dev.get(), x, L, out, iters);
        dev->close();
        return rc;
    }
    auto& cq = dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    const uint32_t ncores = grid.x * grid.y;
    const uint32_t T = L.tiles();
    const uint64_t bytes = uint64_t(T) * TILE_BYTES;
    auto bx = make_dram(dev.get(), bytes), by = make_dram(dev.get(), bytes), bg = make_dram(dev.get(), bytes);
    auto bp = make_dram(dev.get(), uint64_t(ncores) * TILE_BYTES);
    distributed::EnqueueWriteMeshBuffer(cq, bx, x, true);
    distributed::EnqueueWriteMeshBuffer(cq, by, y, true);

    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    add_cb(prog, cores, 0, 2);
    add_cb(prog, cores, 1, 2);
    add_cb(prog, cores, 3, 2);
    add_cb(prog, cores, 16, 2);
    add_cb(prog, cores, 17, 1);
    std::vector<uint32_t> ct_r, ct_w;
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_r);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_r);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_w);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct_w);
    const std::string kd = LOSS_KERNEL_DIR;
    auto kr = CreateKernel(prog, kd + "reader_loss_l1.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                              .noc = NOC::RISCV_1_default, .compile_args = ct_r});
    auto kw = CreateKernel(prog, kd + "writer_loss_l1.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                              .noc = NOC::RISCV_0_default, .compile_args = ct_w});
    // fp32 operands must reach DEST unrounded (the default unpack path rounds to TF32).
    // LOSS_U2D=0 keeps the default path (diagnostic A/B for the #690 launch hang).
    const char* u2d_env = std::getenv("LOSS_U2D");
    const bool use_u2d = !(u2d_env && u2d_env[0] == '0');
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    if (use_u2d)
        for (uint32_t cb : {0u, 1u, 3u}) u2d[cb] = UnpackToDestMode::UnpackToDestFp32;
    std::fprintf(stderr, "[LOSS] stage=program u2d=%d tiles=%u cores=%u\n", int(use_u2d), T, ncores);
    auto kc = CreateKernel(prog, kd + "loss_l1_compute.cpp", cores,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .math_approx_mode = false});
    const float w = 0.8f / float(npx);
    uint32_t wb;
    std::memcpy(&wb, &w, 4);
    for (uint32_t cy = 0, k = 0; cy < grid.y; cy++)
        for (uint32_t cx = 0; cx < grid.x; cx++, k++) {
            const uint32_t t0 = uint32_t(uint64_t(T) * k / ncores), t1 = uint32_t(uint64_t(T) * (k + 1) / ncores);
            const CoreCoord c{cx, cy};
            SetRuntimeArgs(prog, kr, c, {uint32_t(bx->address()), uint32_t(by->address()), t0, t1});
            SetRuntimeArgs(prog, kw, c, {uint32_t(bg->address()), uint32_t(bp->address()), t0, t1, k});
            SetRuntimeArgs(prog, kc, c, {t1 - t0, wb});
        }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(prog));

    auto launch = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, wl, false);
        distributed::Finish(cq);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    launch();  // first launch: JIT compile + check
    std::fprintf(stderr, "[LOSS] stage=launch1_done\n");
    std::vector<float> g, part;
    distributed::EnqueueReadMeshBuffer(cq, g, bg, true);
    distributed::EnqueueReadMeshBuffer(cq, part, bp, true);
    double sum = 0.0;
    for (float v : part) sum += v;
    const double loss = 0.8 * sum / double(npx);
    {
        const auto gh = lb_untilize(g, L);
        std::ofstream f(out + "/grad.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(gh.data()), gh.size() * 4);
        char lb[64];
        std::snprintf(lb, sizeof lb, "%.12g\n", loss);
        std::ofstream(out + "/loss.txt") << lb;
    }
    // Back-to-back: `iters` launches enqueued without waiting, one Finish (the trainer's
    // pattern); plus per-launch wall times for the spread.
    std::fprintf(stderr, "[LOSS] stage=outputs_written\n");
    std::vector<double> ms;
    for (int i = 0; i < iters; i++) ms.push_back(launch());
    std::fprintf(stderr, "[LOSS] stage=timed_launches_done\n");
    const auto b0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);
    const double b2b = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count() /
                       std::max(1, iters);
    std::sort(ms.begin(), ms.end());
    std::printf("[LOSS] mode=l1 H=%u W=%u tiles=%u cores=%u loss=%.9g b2b_ms=%.4f launch_ms_med=%.4f "
                "launch_ms_min=%.4f iters=%d\n",
                L.H, L.W, T, ncores, loss, b2b, ms.empty() ? 0.0 : ms[ms.size() / 2], ms.empty() ? 0.0 : ms[0],
                iters);
    dev->close();
    return 0;
}
