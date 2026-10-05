// T-A: hardware-ceiling RISC / NoC microbenchmark driver.
//
// Runs each probe kernel in render/bench/kernels/ on one core and on the whole
// compute grid, collects per-(core, RISC) wall-clock tick counts from a DRAM
// results buffer and prints one `[MB]` line per run:
//
//   [MB] probe=<name> scope=<1core|all> cores=<k> risc=<b|n|b+n> n=<ops/risc>
//        tpo_med=<ticks/op median over cores> tpo_min= tpo_max= [extra fields]
//
// Ticks are the RISC-V debug wall clock; `[MB] clock` measures its rate against
// the host steady clock (difference of two long spins, so launch overhead
// cancels). See docs/hw-ceilings.md for the results and their interpretation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

constexpr uint32_t N = 1u << 20;
constexpr uint32_t RES_BYTES = 64;
constexpr uint32_t RES_WORDS = RES_BYTES / 4;
constexpr uint32_t MAGIC = 0x4D42A5A5u;
constexpr uint32_t SCRATCH_BYTES = 64 * 1024;
constexpr uint32_t SCAT_PAGE = 1024, SCAT_PAGES = 4096;  // 4 MiB scatter target
constexpr uint32_t BW_PAGE = 8192, BW_PAGES = 8192;      // 64 MiB bandwidth buffer

enum Risc : uint32_t { BRISC = 0, NCRISC = 1 };

struct Env {
    std::shared_ptr<distributed::MeshDevice> dev;
    distributed::MeshCommandQueue* cq = nullptr;
    CoreCoord grid;
    std::shared_ptr<distributed::MeshBuffer> res, scat, bw;
};

struct Spec {
    std::string name;
    std::string kfile;
    std::map<std::string, std::string> defines;
    std::vector<std::pair<Risc, uint32_t>> movers;  // (risc, n)
    uint32_t mode = 0;
    int buf = 0;  // 0 none, 1 scatter, 2 bandwidth
};

struct Sample {
    uint32_t ops;
    uint64_t total, a, b;
};

std::shared_ptr<distributed::MeshBuffer> make_dram(distributed::MeshDevice* dev, uint64_t bytes, uint32_t page) {
    distributed::ReplicatedBufferConfig rc{.size = bytes};
    distributed::DeviceLocalBufferConfig lc{.page_size = page, .buffer_type = BufferType::DRAM};
    return distributed::MeshBuffer::create(rc, lc, dev);
}

double med(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// Runs one probe on the given core rectangle; returns (per slot) samples and the
// host wall time of enqueue + finish.
std::vector<Sample> run(Env& e, const Spec& s, CoreCoord hi, double* host_s = nullptr) {
    const uint32_t nslots = e.grid.x * e.grid.y * 2;
    std::vector<uint32_t> zero(nslots * RES_WORDS, 0);
    distributed::EnqueueWriteMeshBuffer(*e.cq, e.res, zero, true);

    Program prog = CreateProgram();
    const CoreRangeSet cores(CoreRange({0, 0}, hi));
    {
        CircularBufferConfig c(SCRATCH_BYTES, {{0, DataFormat::Float32}});
        c.set_page_size(0, SCRATCH_BYTES);
        CreateCircularBuffer(prog, cores, c);
        CircularBufferConfig r(2 * RES_BYTES, {{1, DataFormat::Float32}});
        r.set_page_size(1, 2 * RES_BYTES);
        CreateCircularBuffer(prog, cores, r);
    }
    std::vector<uint32_t> ct;
    TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    TensorAccessorArgs::create_dram_interleaved().append_to(ct);
    const auto& data = s.buf == 2 ? e.bw : e.scat;
    const uint32_t npages = s.buf == 2 ? BW_PAGES : SCAT_PAGES;

    for (auto [risc, n] : s.movers) {
        auto k = CreateKernel(prog, std::string(MB_KERNEL_DIR) + s.kfile, cores,
                              DataMovementConfig{
                                  .processor = risc == BRISC ? DataMovementProcessor::RISCV_0
                                                             : DataMovementProcessor::RISCV_1,
                                  .noc = risc == BRISC ? NOC::RISCV_0_default : NOC::RISCV_1_default,
                                  .compile_args = ct,
                                  .defines = s.defines,
                              });
        // Two movers on one core split the scratch CB in halves.
        const uint32_t off = s.movers.size() > 1 ? risc * (SCRATCH_BYTES / 2) : 0;
        for (uint32_t y = 0; y <= hi.y; y++)
            for (uint32_t x = 0; x <= hi.x; x++) {
                const uint32_t lin = y * e.grid.x + x;
                SetRuntimeArgs(prog, k, CoreCoord{x, y},
                               {static_cast<uint32_t>(e.res->address()), lin * 2 + risc, n, off,
                                static_cast<uint32_t>(data->address()), npages, lin, s.mode});
            }
    }
    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(e.dev->shape()), std::move(prog));
    const auto t0 = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(*e.cq, wl, false);
    distributed::Finish(*e.cq);
    const auto t1 = std::chrono::steady_clock::now();
    if (host_s) *host_s = std::chrono::duration<double>(t1 - t0).count();

    std::vector<uint32_t> raw;
    distributed::EnqueueReadMeshBuffer(*e.cq, raw, e.res, true);
    std::vector<Sample> out;
    for (uint32_t y = 0; y <= hi.y; y++)
        for (uint32_t x = 0; x <= hi.x; x++)
            for (auto [risc, n] : s.movers) {
                const uint32_t* r = &raw[((y * e.grid.x + x) * 2 + risc) * RES_WORDS];
                if (r[0] != MAGIC) {
                    std::printf("[MB] ERROR probe=%s core=(%u,%u) risc=%u missing result\n", s.name.c_str(), x, y,
                                risc);
                    continue;
                }
                auto u64 = [&](int i) { return static_cast<uint64_t>(r[i]) | (static_cast<uint64_t>(r[i + 1]) << 32); };
                out.push_back({r[1], u64(2), u64(4), u64(6)});
            }
    return out;
}

std::string movers_str(const Spec& s) {
    std::string m;
    for (auto [r, n] : s.movers) m += std::string(m.empty() ? "" : "+") + (r == BRISC ? "b" : "n");
    return m;
}

}  // namespace

int main() {
    Env e;
    e.dev = distributed::MeshDevice::create_unit_mesh(0);
    e.cq = &e.dev->mesh_command_queue();
    e.grid = e.dev->compute_with_storage_grid_size();
    const uint32_t ncores = e.grid.x * e.grid.y;
    e.res = make_dram(e.dev.get(), uint64_t(ncores) * 2 * RES_BYTES, RES_BYTES);
    e.scat = make_dram(e.dev.get(), uint64_t(SCAT_PAGE) * SCAT_PAGES, SCAT_PAGE);
    e.bw = make_dram(e.dev.get(), uint64_t(BW_PAGE) * BW_PAGES, BW_PAGE);
    {
        std::vector<uint32_t> fill(size_t(BW_PAGE) * BW_PAGES / 4, 0x3F800000u);
        distributed::EnqueueWriteMeshBuffer(*e.cq, e.bw, fill, true);
    }
    std::printf("[MB] grid=%ux%u cores=%u dram_channels=%u\n", e.grid.x, e.grid.y, ncores,
                static_cast<uint32_t>(e.dev->num_dram_channels()));

    // Clock: two spins of 2^28 and 2^30 ticks; the host-time difference over the
    // tick difference cancels launch/finish overhead.
    const CoreCoord one{0, 0}, all{e.grid.x - 1, e.grid.y - 1};
    Spec spin{"p0_clock_spin", "p0_clock.cpp", {}, {{NCRISC, 1u << 20}}, 1};
    double h_short = 0, h_long = 0;
    run(e, spin, one, &h_short);  // JIT warm
    auto s_short = run(e, spin, one, &h_short);
    spin.movers = {{NCRISC, 1u << 22}};
    auto s_long = run(e, spin, one, &h_long);
    const double dticks = double(s_long[0].total) - double(s_short[0].total);
    const double mhz = dticks / (h_long - h_short) / 1e6;
    std::printf("[MB] clock tick_mhz=%.1f (ticks %llu vs %llu, host %.4f s vs %.4f s)\n", mhz,
                (unsigned long long)s_long[0].total, (unsigned long long)s_short[0].total, h_long, h_short);

    const std::vector<Spec> specs = {
        {"p0_loop_overhead", "p0_clock.cpp", {}, {{NCRISC, N}}},
        {"p1_l1_store_volatile", "p1_l1_store_volatile.cpp", {}, {{NCRISC, N}}},
        {"p1_l1_store_volatile", "p1_l1_store_volatile.cpp", {}, {{BRISC, N}}},
        {"p2_l1_store_block_memcpy", "p2_l1_store_block.cpp", {{"MB_P2", "0"}}, {{NCRISC, N}}},
        {"p2_l1_store_block_typed", "p2_l1_store_block.cpp", {{"MB_P2", "1"}}, {{NCRISC, N}}},
        {"p2_l1_store_block_aligned", "p2_l1_store_block.cpp", {{"MB_P2", "2"}}, {{NCRISC, N}}},
        {"p3_l1_load", "p3_l1_load.cpp", {}, {{NCRISC, N}}},
        {"p4_noc_write_issue_pre", "p4_noc_write_issue.cpp", {{"MB_ACC", "0"}}, {{NCRISC, N}}, 0, 1},
        {"p4_noc_write_issue_acc", "p4_noc_write_issue.cpp", {{"MB_ACC", "1"}}, {{NCRISC, N}}, 0, 1},
        {"p5_noc_read_rt_d1", "p5_noc_read_rt.cpp", {{"MB_DEPTH", "1"}}, {{NCRISC, N}}, 0, 1},
        {"p5_noc_read_rt_d8", "p5_noc_read_rt.cpp", {{"MB_DEPTH", "8"}}, {{NCRISC, N}}, 0, 1},
        {"p6_dual_mover", "p6_dual_mover.cpp", {{"MB_READ", "0"}}, {{NCRISC, N}}, 0, 1},
        {"p6_dual_mover", "p6_dual_mover.cpp", {{"MB_READ", "0"}}, {{BRISC, N}}, 0, 1},
        {"p6_dual_mover", "p6_dual_mover.cpp", {{"MB_READ", "0"}}, {{BRISC, N / 2}, {NCRISC, N / 2}}, 0, 1},
        {"p6_dual_mover_rd", "p6_dual_mover.cpp", {{"MB_READ", "1"}}, {{NCRISC, N / 4}}, 0, 1},
        {"p6_dual_mover_rd", "p6_dual_mover.cpp", {{"MB_READ", "1"}}, {{BRISC, N / 8}, {NCRISC, N / 8}}, 0, 1},
        {"p7_dram_bw_read", "p7_dram_bw.cpp", {}, {{NCRISC, 4096}}, 0, 2},
        {"p7_dram_bw_write", "p7_dram_bw.cpp", {}, {{NCRISC, 4096}}, 1, 2},
        {"p7_dram_bw_read", "p7_dram_bw.cpp", {}, {{BRISC, 2048}, {NCRISC, 2048}}, 0, 2},
        {"p8_barrier_empty", "p8_barrier.cpp", {}, {{NCRISC, N}}},
        {"p9_fsub_i2f", "p9_scalar_math.cpp", {{"MB_OP", "0"}}, {{NCRISC, N / 4}}},
        {"p9_fmul", "p9_scalar_math.cpp", {{"MB_OP", "1"}}, {{NCRISC, N / 4}}},
        {"p9_udiv", "p9_scalar_math.cpp", {{"MB_OP", "2"}}, {{NCRISC, N / 4}}},
    };

    for (const auto& s : specs) {
        for (int scope = 0; scope < 2; scope++) {
            const CoreCoord hi = scope == 0 ? one : all;
            run(e, s, hi);  // JIT compile + warm, discarded
            double host_s = 0;
            const auto v = run(e, s, hi, &host_s);
            if (v.empty()) continue;
            std::vector<double> tpo, a, b, tot;
            uint64_t tmax = 0, bytes = 0;
            for (const auto& x : v) {
                tpo.push_back(double(x.total) / x.ops);
                a.push_back(double(x.a) / x.ops);
                b.push_back(double(x.b) / x.ops);
                tot.push_back(double(x.total));
                tmax = std::max(tmax, x.total);
                bytes += uint64_t(x.ops) * BW_PAGE;
            }
            const uint32_t ncore = (hi.x + 1) * (hi.y + 1);
            std::printf("[MB] probe=%s scope=%s cores=%u risc=%s n=%u tpo_med=%.3f tpo_min=%.3f tpo_max=%.3f "
                        "a_med=%.3f b_med=%.3f makespan_ticks=%llu makespan_ms=%.3f host_ms=%.3f",
                        s.name.c_str(), scope == 0 ? "1core" : "all", ncore, movers_str(s).c_str(),
                        s.movers[0].second, med(tpo), *std::min_element(tpo.begin(), tpo.end()),
                        *std::max_element(tpo.begin(), tpo.end()), med(a), med(b), (unsigned long long)tmax,
                        tmax / (mhz * 1e3), host_s * 1e3);
            if (s.buf == 2)
                std::printf(" per_core_GBps_med=%.2f agg_GBps=%.1f", BW_PAGE / med(tpo) * mhz * 1e6 / 1e9,
                            double(bytes) / (tmax / (mhz * 1e6)) / 1e9);
            std::printf("\n");
            std::fflush(stdout);
        }
    }
    e.dev->close();
    return 0;
}
