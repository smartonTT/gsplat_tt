// T2 param pass device bench (#697). Runs render/train/kernels/{t2_reader,t2_compute,t2_writer}
// on the input dir of `python3 -m gsplat.t2_dev prep-*`: one run on fresh inputs, outputs read
// back (params_out.f32, m_out.u16, v_out.u16, pfwc.f32, grads.f32 with "grads") for
// `t2_dev check`, then `reps` timed runs (in place, values drift; timing only) -> timing.json.
//   t2_bench <in_dir> <out_dir> [n_cores] [reps] [grads]
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
constexpr uint32_t P_PAGE = 4096, MV_PAGE = 2048, META_PAGE = 512, VIS_PAGE = 128, F_PAGE = 256,
                   SLOT_PAGE = 2048, GPB = 16, NCONST = 31, A_ENTRIES = 2, A_TILE = 2048;
constexpr uint32_t CB_HDR = 0, CB_P = 1, CB_M = 2, CB_V = 3, CB_K = 4, CB_WH = 5, CB_WS = 6, CB_A = 7;
constexpr uint32_t CB_PO = 16, CB_MO = 17, CB_VO = 18, CB_FO = 19, CB_GO = 20;

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

void add_cb(Program& prog, const CoreRangeSet& cores, uint32_t idx, uint32_t page, uint32_t n,
            DataFormat fmt = DataFormat::Float32) {
    CircularBufferConfig c(page * n, {{idx, fmt}});
    c.set_page_size(idx, page);
    CreateCircularBuffer(prog, cores, c);
}


std::vector<float> meta_consts(const std::string& js) {
    std::vector<float> c;
    const char* p = js.c_str() + js.find('[', js.find("\"consts\"")) + 1;
    while (c.size() < NCONST) {
        char* e;
        c.push_back(std::strtof(p, &e));
        p = e;
        while (*p == ',' || *p == ' ' || *p == '\n') p++;
    }
    return c;
}

void save(const std::string& p, const std::vector<uint32_t>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <in_dir> <out_dir> [n_cores] [reps] [grads]\n", argv[0]); return 2; }
    const std::string dir = argv[1], odir = argv[2];
    std::string js;
    { std::ifstream f(dir + "/meta.json"); std::stringstream s; s << f.rdbuf(); js = s.str(); }
    const uint32_t ng = meta_int(js, "n_groups"), nsp = meta_int(js, "n_slot_pages");
    const uint32_t nb = ng / GPB;
    const auto cst = meta_consts(js);
    auto params = load<uint32_t>(dir + "/params.f32");
    auto m = load<uint32_t>(dir + "/m.u16");
    auto v = load<uint32_t>(dir + "/v.u16");
    auto meta = load<uint32_t>(dir + "/meta.u32");
    auto slots = load<uint32_t>(dir + "/slots.f32");
    auto vis = load<uint32_t>(dir + "/vis.u32");
    if (params.size() != size_t(ng) * 1024 || m.size() != size_t(ng) * 512 || meta.size() != size_t(ng) * 128 ||
        slots.size() != size_t(nsp) * 1024 || vis.size() != size_t(nb) * 32) {
        std::fprintf(stderr, "input sizes do not match meta.json\n");
        return 2;
    }
    const bool grads = argc > 5 && std::string(argv[5]) == "grads";
    // Slot page q -> DRAM pages 3q..3q+2: bf16 pieces hi, mid, lo with hi + mid + lo == x exactly
    // (each piece the bf16 truncation of what the earlier ones left). The matmul puts the slots in
    // SrcA, which keeps only 8 explicit mantissa bits even at HiFi4 (tech_reports/matrix_engine), so
    // a fp32 or tf32 page loses ~2^-9 (#713 runs 1-2 failed the T0 oracle); bf16 pieces are exact.
    // The T1 producer emits the pieces; here the host splits.
    std::vector<uint32_t> slots3(slots.size() * 3 / 2);
    auto* s3 = reinterpret_cast<uint16_t*>(slots3.data());
    for (size_t q = 0; q < nsp; q++)
        for (size_t i = 0; i < 1024; i++) {
            float r;
            std::memcpy(&r, &slots[q * 1024 + i], 4);
            for (size_t k = 0; k < 3; k++) {
                uint32_t b;
                std::memcpy(&b, &r, 4);
                b &= 0xFFFF0000u;
                float h;
                std::memcpy(&h, &b, 4);
                s3[(q * 3 + k) * 1024 + i] = uint16_t(b >> 16);
                r -= h;
            }
        }
    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    auto* cq = &dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    uint32_t want = argc > 3 ? std::atoi(argv[3]) : grid.x * grid.y;
    want = std::max<uint32_t>(1, std::min<uint32_t>(want, grid.x * grid.y));
    const int reps = argc > 4 ? std::atoi(argv[4]) : 10;

    auto b_p = make_dram(dev.get(), uint64_t(ng) * P_PAGE, P_PAGE);
    auto b_m = make_dram(dev.get(), uint64_t(ng) * MV_PAGE, MV_PAGE);
    auto b_v = make_dram(dev.get(), uint64_t(ng) * MV_PAGE, MV_PAGE);
    auto b_meta = make_dram(dev.get(), uint64_t(ng) * META_PAGE, META_PAGE);
    auto b_s = make_dram(dev.get(), 3ull * nsp * SLOT_PAGE, SLOT_PAGE);
    auto b_vis = make_dram(dev.get(), uint64_t(nb) * VIS_PAGE, VIS_PAGE);
    auto b_f = make_dram(dev.get(), 13ull * ng * F_PAGE, F_PAGE);
    auto b_g = make_dram(dev.get(), uint64_t(ng) * P_PAGE, P_PAGE);
    distributed::EnqueueWriteMeshBuffer(*cq, b_p, params, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_m, m, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_v, v, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_meta, meta, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_s, slots3, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_vis, vis, true);
    std::vector<uint32_t> zf(size_t(13) * ng * 64, 0u);
    distributed::EnqueueWriteMeshBuffer(*cq, b_f, zf, true);

    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    add_cb(prog, cores, CB_HDR, 64, 2);
    add_cb(prog, cores, CB_P, P_PAGE, 2);
    add_cb(prog, cores, CB_M, MV_PAGE, 2, DataFormat::Float16_b);
    add_cb(prog, cores, CB_V, MV_PAGE, 2, DataFormat::Float16_b);
    add_cb(prog, cores, CB_K, SLOT_PAGE, 6, DataFormat::Float16_b);
    add_cb(prog, cores, CB_WH, 64, 2);
    add_cb(prog, cores, CB_WS, VIS_PAGE + META_PAGE + A_ENTRIES * 64 * 2, 1);
    add_cb(prog, cores, CB_A, A_TILE, 4 * A_ENTRIES, DataFormat::Float16_b);
    add_cb(prog, cores, CB_PO, P_PAGE, 2);
    add_cb(prog, cores, CB_MO, MV_PAGE, 2, DataFormat::Float16_b);
    add_cb(prog, cores, CB_VO, MV_PAGE, 2, DataFormat::Float16_b);
    add_cb(prog, cores, CB_FO, P_PAGE, 2);
    add_cb(prog, cores, CB_GO, P_PAGE, 2);

    std::vector<uint32_t> rct, wct;
    for (int i = 0; i < 6; i++) TensorAccessorArgs::create_dram_interleaved().append_to(rct);
    for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(wct);
    std::map<std::string, std::string> rdef = {{"T2_A_ENTRIES", std::to_string(A_ENTRIES) + "u"}}, wdef;
    if (grads) wdef["T2_GRADS_OUT"] = "1";
    const std::string kd = T2_KERNEL_DIR;
    auto kr = CreateKernel(prog, kd + "t2_reader.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                              .noc = NOC::RISCV_1_default, .compile_args = rct, .defines = rdef});
    auto kw = CreateKernel(prog, kd + "t2_writer.cpp", cores,
                           DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                              .noc = NOC::RISCV_0_default, .compile_args = wct, .defines = wdef});
    std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
    u2d[CB_P] = UnpackToDestMode::UnpackToDestFp32;
    auto kc = CreateKernel(prog, kd + "t2_compute.cpp", cores,
                           ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true,
                                         .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                         .math_approx_mode = false, .defines = wdef});
    // Blocks round-robin over the used cores; groups processed = groups with any visible row.
    uint32_t max_groups = 0, tot_groups = 0;
    for (uint32_t y = 0; y < grid.y; y++)
        for (uint32_t x = 0; x < grid.x; x++) {
            const uint32_t lin = y * grid.x + x;
            uint32_t n = 0;
            if (lin < want)
                for (uint32_t b = lin; b < nb; b += want)
                    for (uint32_t gi = 0; gi < GPB; gi++) n += (vis[b * 32 + 2 * gi] | vis[b * 32 + 2 * gi + 1]) != 0;
            max_groups = std::max(max_groups, n);
            tot_groups += n;
            SetRuntimeArgs(prog, kr, CoreCoord{x, y},
                           {uint32_t(b_p->address()), uint32_t(b_m->address()), uint32_t(b_v->address()),
                            uint32_t(b_s->address()), uint32_t(b_meta->address()), uint32_t(b_vis->address()),
                            lin < want ? lin : nb, want, nb});
            SetRuntimeArgs(prog, kw, CoreCoord{x, y},
                           {uint32_t(b_p->address()), uint32_t(b_m->address()), uint32_t(b_v->address()),
                            uint32_t(b_f->address()), uint32_t(b_g->address()), n, ng});
            std::vector<uint32_t> ca = {n};
            for (float f : cst) { uint32_t u; std::memcpy(&u, &f, 4); ca.push_back(u); }
            SetRuntimeArgs(prog, kc, CoreCoord{x, y}, ca);
        }

    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(prog));
    std::vector<double> ms;
    for (int i = 0; i < 1 + std::max(reps, 0); i++) {
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(*cq, wl, false);
        distributed::Finish(*cq);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        if (i == 0) {
            std::vector<uint32_t> o;
            distributed::EnqueueReadMeshBuffer(*cq, o, b_p, true);
            save(odir + "/params_out.f32", o);
            distributed::EnqueueReadMeshBuffer(*cq, o, b_m, true);
            save(odir + "/m_out.u16", o);
            distributed::EnqueueReadMeshBuffer(*cq, o, b_v, true);
            save(odir + "/v_out.u16", o);
            distributed::EnqueueReadMeshBuffer(*cq, o, b_f, true);
            save(odir + "/pfwc.f32", o);
            if (grads) {
                distributed::EnqueueReadMeshBuffer(*cq, o, b_g, true);
                save(odir + "/grads.f32", o);
            }
        }
    }
    std::vector<double> s(ms.begin() + (ms.size() > 1 ? 1 : 0), ms.end());
    std::sort(s.begin(), s.end());
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "{\"n_groups\": %u, \"groups_processed\": %u, \"cores\": %u, \"max_core_groups\": %u, "
                  "\"slot_pages\": %u, \"device_ms_first\": %.4f, \"device_ms_min\": %.4f, \"device_ms_med\": %.4f, "
                  "\"reps\": %zu, \"grads_out\": %s}",
                  ng, tot_groups, want, max_groups, nsp, ms[0], s[0], s[s.size() / 2], s.size(), grads ? "true" : "false");
    std::ofstream(odir + "/timing.json") << buf << "\n";
    std::printf("[T2] %s\n", buf);
    dev->close();
    return 0;
}
