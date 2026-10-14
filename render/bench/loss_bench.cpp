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

inline float trunc_tf32(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u &= 0xFFFFE000u;
    std::memcpy(&v, &u, 4);
    return v;
}

// 12 constant band tiles for filt_compute.cpp: Rh[-1,0,1], Rl[...], RTh[...], RTl[...].
// R[k][a][b] = g[32 k + a - b + 5], g the normalized 11-tap Gaussian (sigma 1.5), as
// gsplat/train_loss.py band_tiles(); h = fp32 truncated to TF32, l = TF32(fp32 - h).
std::vector<float> band_consts() {
    double g[11], sum = 0.0;
    for (int i = 0; i < 11; i++) sum += g[i] = std::exp(-double((i - 5) * (i - 5)) / (2.0 * 1.5 * 1.5));
    std::vector<float> out(12 * TS * TS, 0.0f);
    for (int k = -1; k <= 1; k++)
        for (uint32_t a = 0; a < TS; a++)
            for (uint32_t b = 0; b < TS; b++) {
                const int off = int(TS) * k + int(a) - int(b) + 5;
                const float r = (off >= 0 && off < 11) ? float(g[off] / sum) : 0.0f;
                const float h = trunc_tf32(r), l = trunc_tf32(r - h);
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
    const uint32_t ncores = grid.x * grid.y, T = L.tiles(), S = 3 * L.nw;
    auto bx = make_dram(dev, uint64_t(T) * TILE_BYTES), bo = make_dram(dev, uint64_t(T) * TILE_BYTES);
    auto br = make_dram(dev, 12ull * TILE_BYTES);
    distributed::EnqueueWriteMeshBuffer(cq, bx, x, true);
    distributed::EnqueueWriteMeshBuffer(cq, br, band_consts(), true);
    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    add_cb(prog, cores, 0, 6);   // X (u2d)
    add_cb(prog, cores, 2, 12);  // band constants
    add_cb(prog, cores, 4, 3);   // XH
    add_cb(prog, cores, 5, 3);   // XL
    add_cb(prog, cores, 6, 4);   // row pass, hi (3-tile window + 1)
    add_cb(prog, cores, 7, 4);   // row pass, lo
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
    auto kc = CreateKernel(prog, kd + "filt_compute.cpp", cores,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .math_approx_mode = false});
    for (uint32_t cy = 0, k = 0; cy < grid.y; cy++)
        for (uint32_t cx = 0; cx < grid.x; cx++, k++) {
            const uint32_t s0 = uint32_t(uint64_t(S) * k / ncores), s1 = uint32_t(uint64_t(S) * (k + 1) / ncores);
            const CoreCoord c{cx, cy};
            SetRuntimeArgs(prog, kr, c, {uint32_t(bx->address()), uint32_t(br->address()), s0, s1, L.nh, L.nw});
            SetRuntimeArgs(prog, kw, c, {uint32_t(bo->address()), s0, s1, L.nh, L.nw});
            SetRuntimeArgs(prog, kc, c, {s1 - s0, L.nh});
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
    std::vector<double> ms;
    const double b2b = time_b2b(cq, wl, iters, ms);
    std::printf("[LOSS] mode=filt H=%u W=%u tiles=%u strips=%u cores=%u b2b_ms=%.4f launch_ms_med=%.4f "
                "launch_ms_min=%.4f iters=%d\n",
                L.H, L.W, T, S, ncores, b2b, ms.empty() ? 0.0 : ms[ms.size() / 2], ms.empty() ? 0.0 : ms[0],
                iters);
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
