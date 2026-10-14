// T1 blend backward device bench (#693). Runs render/train/kernels/{bwd_reader,
// blend_bwd_compute,bwd_writer} on the prep dir of `python3 -m gsplat.blend_backward_dev prep`,
// reads back the slots the device finished (v2 #704: lane reduce + finish on device, 9 floats
// per 64 B slot page) and writes slots.f32 (16 floats per slot, first 9 used) plus timing.json
// for `blend_backward_dev check`.
//   blend_bwd_bench <prep_dir> [n_cores] [reps]
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
constexpr uint32_t SMALL = 64, TILE_B = 4096, N_PLANES = 5, REC_WORDS = 16;
constexpr uint32_t CB_HDR = 0, CB_PLANE = 1, CB_RAMP = 2, CB_REC = 3, CB_OUT = 16, CB_WS = 17;

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

void add_cb(Program& prog, const CoreRangeSet& cores, uint32_t idx, uint32_t page, uint32_t n) {
    CircularBufferConfig c(page * n, {{idx, DataFormat::Float32}});
    c.set_page_size(idx, page);
    CreateCircularBuffer(prog, cores, c);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <prep_dir> [n_cores] [reps]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    std::string js;
    { std::ifstream f(dir + "/meta.json"); std::stringstream s; s << f.rdbuf(); js = s.str(); }
    const uint32_t n_tiles = meta_int(js, "n_tiles"), n_recs = meta_int(js, "n_recs"),
                   n_slots = meta_int(js, "n_slots");
    auto tiles = load<uint32_t>(dir + "/tiles.u32");
    auto recs = load<uint32_t>(dir + "/recs.u32");
    auto planes = load<float>(dir + "/planes.f32");
    auto ramps = load<float>(dir + "/ramps.f32");
    if (tiles.size() != size_t(n_tiles) * 16 || recs.size() != size_t(n_recs) * REC_WORDS ||
        planes.size() != size_t(n_tiles) * N_PLANES * 1024 || ramps.size() != 2 * 1024) {
        std::fprintf(stderr, "prep file sizes do not match meta.json\n");
        return 2;
    }

    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    auto* cq = &dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    uint32_t want = argc > 2 ? std::atoi(argv[2]) : grid.x * grid.y;
    want = std::max<uint32_t>(1, std::min<uint32_t>({want, grid.x * grid.y, std::max<uint32_t>(n_tiles, 1)}));
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

    // Contiguous tile blocks balanced by pair count (+1 per tile for the plane load).
    std::vector<uint32_t> t_beg(want + 1, n_tiles);
    {
        uint64_t total = 0;
        for (uint32_t t = 0; t < n_tiles; t++) total += tiles[t * 16 + 2] + 1;
        uint64_t acc = 0;
        uint32_t c = 0;
        t_beg[0] = 0;
        for (uint32_t t = 0; t < n_tiles && c + 1 < want; t++) {
            acc += tiles[t * 16 + 2] + 1;
            if (acc * want >= total * (c + 1)) t_beg[++c] = t + 1;
        }
        for (uint32_t k = c + 1; k <= want; k++) t_beg[k] = n_tiles;
    }

    auto b_tiles = make_dram(dev.get(), uint64_t(n_tiles) * SMALL, SMALL);
    auto b_recs = make_dram(dev.get(), uint64_t(n_recs) * SMALL, SMALL);
    auto b_planes = make_dram(dev.get(), uint64_t(n_tiles) * N_PLANES * TILE_B, TILE_B);
    auto b_ramps = make_dram(dev.get(), 2ull * TILE_B, TILE_B);
    auto b_out = make_dram(dev.get(), uint64_t(n_slots) * SMALL, SMALL);  // v2: 64 B slot pages
    distributed::EnqueueWriteMeshBuffer(*cq, b_tiles, tiles, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_recs, recs, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_planes, planes, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_ramps, ramps, true);

    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    add_cb(prog, cores, CB_HDR, SMALL, 2);
    add_cb(prog, cores, CB_PLANE, TILE_B, N_PLANES);
    add_cb(prog, cores, CB_RAMP, TILE_B, 2);
    add_cb(prog, cores, CB_REC, SMALL, 4);
    add_cb(prog, cores, CB_OUT, TILE_B, 2);
    add_cb(prog, cores, CB_WS, TILE_B, 1);  // writer scratch: header, 16 recs, 16 staging pages

    std::vector<uint32_t> rct, wct;
    for (int i = 0; i < 4; i++) TensorAccessorArgs::create_dram_interleaved().append_to(rct);
    for (int i = 0; i < 3; i++) TensorAccessorArgs::create_dram_interleaved().append_to(wct);
    const std::string kd = BWD_KERNEL_DIR;
    auto kr = CreateKernel(prog, kd + "bwd_reader.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                              .noc = NOC::RISCV_1_default, .compile_args = rct});
    auto kw = CreateKernel(prog, kd + "bwd_writer.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                              .noc = NOC::RISCV_0_default, .compile_args = wct});
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_PLANE] = UnpackToDestMode::UnpackToDestFp32;
    u2d[CB_RAMP] = UnpackToDestMode::UnpackToDestFp32;
    std::map<std::string, std::string> cdef = {
        {"BWD_AMAX_BITS", std::to_string(meta_int(js, "alpha_max_bits")) + "u"},
        {"BWD_AMIN_BITS", std::to_string(meta_int(js, "alpha_min_bits")) + "u"},
        {"BWD_TMIN_BITS", std::to_string(meta_int(js, "t_min_bits")) + "u"},
    };
    auto kc = CreateKernel(prog, kd + "blend_bwd_compute.cpp", cores,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi3, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .math_approx_mode = false, .defines = cdef});
    uint32_t max_core_recs = 0;
    for (uint32_t y = 0; y < grid.y; y++)
        for (uint32_t x = 0; x < grid.x; x++) {
            const uint32_t lin = y * grid.x + x;
            const uint32_t ta = lin < want ? t_beg[lin] : n_tiles, tb = lin < want ? t_beg[lin + 1] : n_tiles;
            const uint32_t nt = tb - ta;
            const uint32_t r0 = nt ? tiles[ta * 16 + 1] : 0;
            uint32_t nr = 0;
            for (uint32_t t = ta; t < tb; t++) nr += tiles[t * 16 + 2];
            max_core_recs = std::max(max_core_recs, nr);
            SetRuntimeArgs(prog, kr, CoreCoord{x, y},
                           {uint32_t(b_tiles->address()), uint32_t(b_recs->address()),
                            uint32_t(b_planes->address()), uint32_t(b_ramps->address()), ta, nt});
            SetRuntimeArgs(prog, kw, CoreCoord{x, y},
                           {uint32_t(b_out->address()), uint32_t(b_tiles->address()), uint32_t(b_recs->address()), ta, nt});
            (void)r0;
            SetRuntimeArgs(prog, kc, CoreCoord{x, y}, {nt});
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
    const auto h0 = std::chrono::steady_clock::now();
    // v2: the device writes finished slots (9 floats in a 64 B page); no host reduce.
    std::vector<float> slots = std::move(out);
    slots.resize(size_t(n_slots) * 16, 0.0f);
    const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - h0).count();
    {
        std::ofstream f(dir + "/slots.f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(slots.data()), std::streamsize(slots.size() * 4));
    }
    std::vector<double> s = ms;
    std::sort(s.begin(), s.end());
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "{\"n_tiles\": %u, \"n_recs\": %u, \"n_slots\": %u, \"cores\": %u, \"max_core_recs\": %u, "
                  "\"device_ms_first\": %.4f, \"device_ms_min\": %.4f, \"device_ms_med\": %.4f, "
                  "\"host_reduce_ms\": %.4f, \"reps\": %zu, \"version\": \"v2\"}",
                  n_tiles, n_recs, n_slots, want, max_core_recs, ms[0], s[0], s[s.size() / 2], host_ms, ms.size());
    std::ofstream(dir + "/timing.json") << buf << "\n";
    std::printf("[BWD] %s\n", buf);
    dev->close();
    return 0;
}
