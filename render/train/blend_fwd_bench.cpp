// T0 blend forward device harness (#716). Runs the production blend compute kernel
// (render/kernels/compute/alpha_blend_compute_mb.cpp) with the host's training-mode defines
// (GSPLAT_TT_BLEND_TRAIN=1: BLEND_TRAIN 1, BLEND_SCHED 0, BLEND_T_PERIOD 0u) between
// render/train/kernels/{fwd_reader,fwd_writer} on the prep dir of
// `python3 -m gsplat.blend_forward_dev prep`, one screen tile per core, and writes the fp32
// R, G, B, T train buffer (4 tilized tiles per screen tile) to train.f32 plus timing.json
// for `blend_forward_dev check`.
//   blend_fwd_bench <prep_dir> [reps]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {
constexpr uint32_t SMALL = 64, TILE_B = 4096, TILE_BF16 = 2048, TRAIN_PAGE = 4 * TILE_B, BULK_REC_SLOT = 4096;
constexpr uint32_t CB_XRAMP = 0, CB_YRAMP = 1, CB_T_RB = 2, CB_MB_COUNTS = 3, CB_CORE_TILES = 7, CB_DA_RING = 10,
                   CB_BUCKET_BULK = 12, CB_COLOR_OUT = 16, CB_TRAIN_OUT = 17;

template <class T>
std::vector<T> load(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    const size_t n = static_cast<size_t>(f.tellg());
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}

uint64_t meta_int(const std::string& js, const std::string& key) {
    const auto p = js.find("\"" + key + "\"");
    if (p == std::string::npos) { std::fprintf(stderr, "meta.json: no %s\n", key.c_str()); std::exit(2); }
    return std::strtoull(js.c_str() + js.find(':', p) + 1, nullptr, 10);
}

std::shared_ptr<distributed::MeshBuffer> make_dram(distributed::MeshDevice* dev, uint64_t bytes, uint32_t page) {
    distributed::ReplicatedBufferConfig rc{.size = std::max<uint64_t>(bytes, page)};
    distributed::DeviceLocalBufferConfig lc{.page_size = page, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

void add_cb(Program& prog, const CoreRangeSet& cores, uint32_t idx, uint32_t page, uint32_t n, DataFormat fmt) {
    CircularBufferConfig c(page * n, {{idx, fmt}});
    c.set_page_size(idx, page);
    CreateCircularBuffer(prog, cores, c);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <prep_dir> [reps]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    std::string js;
    { std::ifstream f(dir + "/meta.json"); std::stringstream s; s << f.rdbuf(); js = s.str(); }
    const uint32_t n_tiles = meta_int(js, "n_tiles"), n_pages = meta_int(js, "n_rec_pages");
    auto tiles = load<uint32_t>(dir + "/tiles.u32");
    auto recs = load<uint32_t>(dir + "/recs.u32");
    auto ramps = load<float>(dir + "/ramps.f32");
    if (tiles.size() != size_t(n_tiles) * 16 || recs.size() > size_t(n_pages) * 16 || ramps.size() != 2 * 1024) {
        std::fprintf(stderr, "prep file sizes do not match meta.json\n");
        return 2;
    }
    recs.resize(size_t(n_pages) * 16, 0u);
    for (uint32_t t = 0; t < n_tiles; t++) {
        if (tiles[t * 16 + 2] > 2 * BULK_REC_SLOT) {
            std::fprintf(stderr, "tile %u: %u records > one bulk slot (%u)\n", t, tiles[t * 16 + 2], 2 * BULK_REC_SLOT);
            return 2;
        }
    }

    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    auto* cq = &dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    if (n_tiles > grid.x * grid.y) { std::fprintf(stderr, "more tiles than cores\n"); return 2; }
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;

    auto b_tiles = make_dram(dev.get(), uint64_t(n_tiles) * SMALL, SMALL);
    auto b_recs = make_dram(dev.get(), uint64_t(n_pages) * SMALL, SMALL);
    auto b_ramps = make_dram(dev.get(), 2ull * TILE_B, TILE_B);
    auto b_out = make_dram(dev.get(), uint64_t(n_tiles) * TRAIN_PAGE, TRAIN_PAGE);
    distributed::EnqueueWriteMeshBuffer(*cq, b_tiles, tiles, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_recs, recs, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_ramps, ramps, true);
    {
        // Poison the output so a tile the device never wrote cannot pass.
        std::vector<uint32_t> nan(size_t(n_tiles) * TRAIN_PAGE / 4, 0x7fc00000u);
        distributed::EnqueueWriteMeshBuffer(*cq, b_out, nan, true);
    }

    // The production blend CB set (render/host/blend_device.cpp, unfused, BLEND_CB_BASE 0).
    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    add_cb(prog, cores, CB_XRAMP, TILE_B, 2, DataFormat::Float32);
    add_cb(prog, cores, CB_YRAMP, TILE_B, 2, DataFormat::Float32);
    add_cb(prog, cores, CB_T_RB, TILE_BF16, 2, DataFormat::Float16_b);
    add_cb(prog, cores, CB_MB_COUNTS, 128, 64, DataFormat::UInt32);
    add_cb(prog, cores, CB_CORE_TILES, SMALL, 1, DataFormat::UInt32);
    add_cb(prog, cores, CB_DA_RING, SMALL, 1 + 64, DataFormat::UInt32);
    add_cb(prog, cores, CB_BUCKET_BULK, SMALL, BULK_REC_SLOT, DataFormat::Float32);
    add_cb(prog, cores, CB_COLOR_OUT, TILE_BF16, 6, DataFormat::Float16_b);
    add_cb(prog, cores, CB_TRAIN_OUT, TILE_B, 4, DataFormat::Float32);

    std::vector<uint32_t> rct, wct;
    for (int i = 0; i < 3; i++) TensorAccessorArgs::create_dram_interleaved().append_to(rct);
    TensorAccessorArgs::create_dram_interleaved().append_to(wct);
    const std::string kd = BWD_KERNEL_DIR;
    auto kr = CreateKernel(prog, kd + "fwd_reader.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                              .noc = NOC::RISCV_1_default, .compile_args = rct});
    auto kw = CreateKernel(prog, kd + "fwd_writer.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                              .noc = NOC::RISCV_0_default, .compile_args = wct});
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_XRAMP] = UnpackToDestMode::UnpackToDestFp32;
    u2d[CB_YRAMP] = UnpackToDestMode::UnpackToDestFp32;
    // blend_device.cpp's defaults with GSPLAT_TT_BLEND_TRAIN=1 (no env overrides; unfused).
    std::map<std::string, std::string> cdef = {
        {"BLEND_T_EPS", "0.00390625f"}, {"BLEND_T_PERIOD", "0u"}, {"BLEND_COEF_DEST", "1"},
        {"BLEND_ABL", "0"}, {"BLEND_FPU_QF_ABL", "0"}, {"BLEND_QF_PROBE", "0"}, {"BLEND_QF_PAD", "0"},
        {"BLEND_JUMP_WALK", "1"}, {"BLEND_SCHED", "0"}, {"BLEND_SFPU_UNORM", "1"},
        {"BLEND_CONST_HOIST", "1"}, {"BLEND_RAW_STAGE", "0"}, {"BLEND_FAST_TRED", "1"},
        {"BLEND_EXP1", "1"}, {"BLEND_SWAP_FILL", "1"}, {"BLEND_DECODE_AHEAD", "2"}, {"BLEND_TRAIN", "1"},
        {"BLEND_PIXEL_FLOOR", "1"},
    };
    // render/run.py's default contrib_floor (1/255); the train math applies its own 1/255 rule.
    const float floor_f = 1.0f / 255.0f;
    uint32_t floor_bits;
    std::memcpy(&floor_bits, &floor_f, 4);
    auto kc = CreateKernel(prog, FWD_COMPUTE_KERNEL, cores,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi3, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .math_approx_mode = false, .defines = cdef});
    for (uint32_t y = 0; y < grid.y; y++)
        for (uint32_t x = 0; x < grid.x; x++) {
            const uint32_t lin = y * grid.x + x;
            const uint32_t ta = std::min(lin, n_tiles), nt = lin < n_tiles ? 1u : 0u;
            SetRuntimeArgs(prog, kr, CoreCoord{x, y},
                           {uint32_t(b_tiles->address()), uint32_t(b_recs->address()), uint32_t(b_ramps->address()), ta, nt});
            SetRuntimeArgs(prog, kw, CoreCoord{x, y}, {uint32_t(b_out->address()), ta, nt});
            SetRuntimeArgs(prog, kc, CoreCoord{x, y}, {0u, floor_bits});  // eps 0: compile-time default
        }

    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(prog));
    std::vector<double> ms;
    for (int i = 0; i < std::max(reps, 1); i++) {
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(*cq, wl, false);
        distributed::Finish(*cq);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }

    std::vector<float> out;
    distributed::EnqueueReadMeshBuffer(*cq, out, b_out, true);
    out.resize(size_t(n_tiles) * TRAIN_PAGE / 4);
    {
        std::ofstream f(dir + "/train.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(out.data()), std::streamsize(out.size() * 4));
    }
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    char buf[384];
    std::snprintf(buf, sizeof buf,
                  "{\"n_tiles\": %u, \"n_rec_pages\": %u, \"device_ms_first\": %.4f, \"device_ms_min\": %.4f, "
                  "\"device_ms_med\": %.4f, \"reps\": %zu}",
                  n_tiles, n_pages, ms[0], s[0], s[s.size() / 2], ms.size());
    std::ofstream(dir + "/timing.json") << buf << "\n";
    std::printf("[FWD] %s\n", buf);
    dev->close();
    return 0;
}
