// On-device training step chain (#748): T1 blend backward -> T2 param pass + sparse Adam on one
// device, sharing DRAM buffers. T1 (bwd_reader, blend_bwd_compute, bwd_writer with
// BWD_SLOT_PAGE=4096) writes its finished slots straight into T2's slot-page layout (64 slots of
// 64 B per 4 KB page), and T2 (t697 gather reduce: t2g_reader, t2g_compute, t2_writer) reads that
// same buffer and updates params / m / v in place. No host re-pack between T1 and T2.
// Forward and loss are not on device in this chain yet: their outputs come from the prep dir
// (python3 -m gsplat.train_step_dev prep <t0.npz> <dir>).
//   train_step <dir> <out_dir> [n_cores] [reps] [grads]
// <dir>/t1: blend_backward_dev prep files; <dir>/t2: t2_dev prep-dump files (slots.f32 unused).
// Writes slots.f32 (16 floats per slot), params_out.f32, m_out.u16, v_out.u16, pfwc.f32,
// [grads.f32] from the first step, and timing.json (per stage and chained, over reps steps).
// #784 env knobs: T2_SPLIT (default 1 since #792; 0 = the old one-program T2) runs T2 as the
// heavy-row split (gsplat/t2_split.py, docs/t784, docs/t792):
// program A sums rows with k > T2_KCAP (default 8 since #792, was 16) in heavy groups of T2_HL (16) lanes, LPT over
// the cores, into the hsum buffer; program B runs the light groups, one contiguous range per core
// balanced by planes + T2_OVH (2) per active group, heavy lanes taking hsum as plane 0.
// T2_G2D=1 skips the projection backward + Adam, so pfwc.f32 holds the per-row slot sums (g2d).
// SH mode (#751/#756, when <dir>/t2/sh_rest.f32 exists): T1 -> t2sh pre (aux page per group:
// view dir, clamp keep, colour Jacobian) -> T2 (T2_SH: dir term in dL/dmeans, full-colour mask
// on dL/dsh0, grads always out) -> t2sh post (bands 1-15 grads + sparse Adam, rest pages in
// place). Also writes sh_rest_out.f32, sh_m_out.u16, sh_v_out.u16, [sh_grads.f32].
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
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
constexpr uint32_t P_PAGE = 4096, MV_PAGE = 2048, META_PAGE = 512, VIS_PAGE = 128, F_PAGE = 256,
                   SLOT_PAGE = 4096, SLOTS_PER_PAGE = 64, GPB = 16, NCONST = 31, STAGE_PAGES = 64,
                   NSH = 10;

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

std::string slurp(const std::string& p) {
    std::ifstream f(p);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

uint64_t meta_int(const std::string& js, const std::string& key) {
    const auto p = js.find("\"" + key + "\"");
    if (p == std::string::npos) { std::fprintf(stderr, "meta.json: no %s\n", key.c_str()); std::exit(2); }
    return std::strtoull(js.c_str() + js.find(':', p) + 1, nullptr, 10);
}

std::vector<float> meta_consts(const std::string& js, const std::string& key = "consts", uint32_t n = NCONST) {
    std::vector<float> c;
    const auto k = js.find("\"" + key + "\"");
    if (k == std::string::npos) { std::fprintf(stderr, "meta.json: no %s\n", key.c_str()); std::exit(2); }
    const char* p = js.c_str() + js.find('[', k) + 1;
    while (c.size() < n) {
        char* e;
        c.push_back(std::strtof(p, &e));
        p = e;
        while (*p == ',' || *p == ' ' || *p == '\n') p++;
    }
    return c;
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

void save(const std::string& p, const std::vector<uint32_t>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

struct Stat {
    double first, min, med;
};
Stat stat(std::vector<double> v) {
    Stat s{v.empty() ? 0.0 : v[0], 0.0, 0.0};
    if (v.size() > 1) v.erase(v.begin());  // first step pays program compile / load
    std::sort(v.begin(), v.end());
    if (!v.empty()) { s.min = v[0]; s.med = v[v.size() / 2]; }
    return s;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <dir> <out_dir> [n_cores] [reps] [grads]\n", argv[0]); return 2; }
    const std::string d1 = std::string(argv[1]) + "/t1", d2 = std::string(argv[1]) + "/t2", odir = argv[2];
    const std::string j1 = slurp(d1 + "/meta.json"), j2 = slurp(d2 + "/meta.json");
    const uint32_t n_tiles = meta_int(j1, "n_tiles"), n_recs = meta_int(j1, "n_recs"), n_slots = meta_int(j1, "n_slots");
    const uint32_t ng = meta_int(j2, "n_groups"), nsp = meta_int(j2, "n_slot_pages"), nb = ng / GPB;
    if (meta_int(j2, "n_slots") != n_slots || uint64_t(nsp) * SLOTS_PER_PAGE < n_slots) {
        std::fprintf(stderr, "T1 n_slots %u does not match T2 (%llu slots, %u pages)\n", n_slots,
                     (unsigned long long)meta_int(j2, "n_slots"), nsp);
        return 2;
    }
    auto tiles = load<uint32_t>(d1 + "/tiles.u32");
    auto recs = load<uint32_t>(d1 + "/recs.u32");
    auto planes = load<float>(d1 + "/planes.f32");
    auto ramps = load<float>(d1 + "/ramps.f32");
    auto params = load<uint32_t>(d2 + "/params.f32");
    auto m = load<uint32_t>(d2 + "/m.u16");
    auto v = load<uint32_t>(d2 + "/v.u16");
    auto meta = load<uint32_t>(d2 + "/meta.u32");
    auto vis = load<uint32_t>(d2 + "/vis.u32");
    if (tiles.size() != size_t(n_tiles) * 16 || recs.size() != size_t(n_recs) * REC_WORDS ||
        planes.size() != size_t(n_tiles) * N_PLANES * 1024 || ramps.size() != 2 * 1024 ||
        params.size() != size_t(ng) * 1024 || m.size() != size_t(ng) * 512 || meta.size() != size_t(ng) * 128 ||
        vis.size() != size_t(nb) * 32) {
        std::fprintf(stderr, "prep file sizes do not match meta.json\n");
        return 2;
    }
    const auto cst = meta_consts(j2);
    const bool grads = argc > 5 && std::string(argv[5]) == "grads";
    const bool sh = std::ifstream(d2 + "/sh_rest.f32").good();
    std::vector<uint32_t> sh_rest, sh_m, sh_v;
    std::vector<float> shc;
    if (sh) {
        sh_rest = load<uint32_t>(d2 + "/sh_rest.f32");
        sh_m = load<uint32_t>(d2 + "/sh_m.u16");
        sh_v = load<uint32_t>(d2 + "/sh_v.u16");
        shc = meta_consts(j2, "sh_consts", NSH);
        if (sh_rest.size() != size_t(ng) * 3 * 1024 || sh_m.size() != size_t(ng) * 3 * 512 ||
            sh_v.size() != size_t(ng) * 3 * 512) {
            std::fprintf(stderr, "SH rest file sizes do not match meta.json\n");
            return 2;
        }
    }
    auto envu = [](const char* k, uint32_t d) { const char* e = std::getenv(k); return e && *e ? uint32_t(std::atoi(e)) : d; };
    const bool split = envu("T2_SPLIT", 1) != 0, g2d_only = envu("T2_G2D", 0) != 0;
    const uint32_t kcap = envu("T2_KCAP", 8), HL = envu("T2_HL", 16), ovh = envu("T2_OVH", 2);
    if (split && (HL == 0 || HL > 64 || 64 % HL != 0)) { std::fprintf(stderr, "T2_HL must divide 64\n"); return 2; }

    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    auto* cq = &dev->mesh_command_queue();
    const CoreCoord grid = dev->compute_with_storage_grid_size();
    const uint32_t ncore = grid.x * grid.y;
    uint32_t want = argc > 3 && std::atoi(argv[3]) > 0 ? std::atoi(argv[3]) : ncore;
    want = std::max<uint32_t>(1, std::min<uint32_t>(want, ncore));
    const uint32_t want1 = std::max<uint32_t>(1, std::min<uint32_t>(want, std::max<uint32_t>(n_tiles, 1)));
    const int reps = argc > 4 && std::atoi(argv[4]) > 0 ? std::atoi(argv[4]) : 10;
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    // Shared buffers: T1 writes b_s, T2 reads it; T2 updates b_p / b_m / b_v in place.
    auto b_tiles = make_dram(dev.get(), uint64_t(n_tiles) * SMALL, SMALL);
    auto b_recs = make_dram(dev.get(), uint64_t(n_recs) * SMALL, SMALL);
    auto b_planes = make_dram(dev.get(), uint64_t(n_tiles) * N_PLANES * TILE_B, TILE_B);
    auto b_ramps = make_dram(dev.get(), 2ull * TILE_B, TILE_B);
    auto b_s = make_dram(dev.get(), uint64_t(nsp) * SLOT_PAGE, SLOT_PAGE);
    auto b_p = make_dram(dev.get(), uint64_t(ng) * P_PAGE, P_PAGE);
    auto b_m = make_dram(dev.get(), uint64_t(ng) * MV_PAGE, MV_PAGE);
    auto b_v = make_dram(dev.get(), uint64_t(ng) * MV_PAGE, MV_PAGE);
    auto b_meta = make_dram(dev.get(), uint64_t(ng) * META_PAGE, META_PAGE);
    auto b_vis = make_dram(dev.get(), uint64_t(nb) * VIS_PAGE, VIS_PAGE);
    auto b_f = make_dram(dev.get(), 13ull * ng * F_PAGE, F_PAGE);
    auto b_g = make_dram(dev.get(), uint64_t(ng) * P_PAGE, P_PAGE);
    // SH mode: rest bands (3 pages per group), their Adam state, pre-pass aux page, rest grads.
    const uint64_t nsh = sh ? 3ull * ng : 1;
    auto b_r = make_dram(dev.get(), nsh * P_PAGE, P_PAGE);
    auto b_rm = make_dram(dev.get(), nsh * MV_PAGE, MV_PAGE);
    auto b_rv = make_dram(dev.get(), nsh * MV_PAGE, MV_PAGE);
    auto b_x = make_dram(dev.get(), (sh ? uint64_t(ng) : 1) * P_PAGE, P_PAGE);
    auto b_rg = make_dram(dev.get(), nsh * P_PAGE, P_PAGE);
    if (sh) {
        distributed::EnqueueWriteMeshBuffer(*cq, b_r, sh_rest, true);
        distributed::EnqueueWriteMeshBuffer(*cq, b_rm, sh_m, true);
        distributed::EnqueueWriteMeshBuffer(*cq, b_rv, sh_v, true);
    }
    distributed::EnqueueWriteMeshBuffer(*cq, b_tiles, tiles, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_recs, recs, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_planes, planes, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_ramps, ramps, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_p, params, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_m, m, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_v, v, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_meta, meta, true);
    distributed::EnqueueWriteMeshBuffer(*cq, b_vis, vis, true);
    {
        // NaN poison: a slot T1 does not write shows up as NaN in T2's grads and in slots.f32.
        std::vector<uint32_t> poison(size_t(nsp) * 1024, 0x7fc00000u);
        distributed::EnqueueWriteMeshBuffer(*cq, b_s, poison, true);
        std::vector<uint32_t> zf(size_t(13) * ng * 64, 0u);
        distributed::EnqueueWriteMeshBuffer(*cq, b_f, zf, true);
    }

    // ---- T1 program (blend_bwd_bench layout, stats off, slot pages in T2 layout)
    Program p1 = CreateProgram();
    {
        enum : uint32_t { CB_HDR = 0, CB_PLANE = 1, CB_RAMP = 2, CB_REC = 3, CB_OUT = 16, CB_WS = 17 };
        add_cb(p1, cores, CB_HDR, SMALL, 2);
        add_cb(p1, cores, CB_PLANE, TILE_B, N_PLANES);
        add_cb(p1, cores, CB_RAMP, TILE_B, 2);
        add_cb(p1, cores, CB_REC, SMALL, 4);
        add_cb(p1, cores, CB_OUT, TILE_B, 2);
        add_cb(p1, cores, CB_WS, TILE_B, 1);
        std::vector<uint32_t> rct, wct;
        for (int i = 0; i < 4; i++) TensorAccessorArgs::create_dram_interleaved().append_to(rct);
        for (int i = 0; i < 4; i++) TensorAccessorArgs::create_dram_interleaved().append_to(wct);
        const std::string kd = TRAIN_KERNEL_DIR;
        auto kr = CreateKernel(p1, kd + "bwd_reader.cpp", cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                                  .noc = NOC::RISCV_1_default, .compile_args = rct});
        auto kw = CreateKernel(p1, kd + "bwd_writer.cpp", cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                                  .noc = NOC::RISCV_0_default, .compile_args = wct,
                                                  .defines = {{"BWD_SLOT_PAGE", std::to_string(SLOT_PAGE) + "u"}}});
        std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
        u2d[CB_PLANE] = UnpackToDestMode::UnpackToDestFp32;
        u2d[CB_RAMP] = UnpackToDestMode::UnpackToDestFp32;
        std::map<std::string, std::string> cdef = {
            {"BWD_AMAX_BITS", std::to_string(meta_int(j1, "alpha_max_bits")) + "u"},
            {"BWD_AMIN_BITS", std::to_string(meta_int(j1, "alpha_min_bits")) + "u"},
            {"BWD_TMIN_BITS", std::to_string(meta_int(j1, "t_min_bits")) + "u"},
        };
        auto kc = CreateKernel(p1, kd + "blend_bwd_compute.cpp", cores,
                               ComputeConfig{.math_fidelity = MathFidelity::HiFi3, .fp32_dest_acc_en = true,
                                             .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                             .math_approx_mode = false, .defines = cdef});
        // Contiguous tile blocks balanced by pair count (+1 per tile for the plane load).
        std::vector<uint32_t> t_beg(want1 + 1, n_tiles);
        uint64_t total = 0, acc = 0;
        for (uint32_t t = 0; t < n_tiles; t++) total += tiles[t * 16 + 2] + 1;
        uint32_t c = 0;
        t_beg[0] = 0;
        for (uint32_t t = 0; t < n_tiles && c + 1 < want1; t++) {
            acc += tiles[t * 16 + 2] + 1;
            if (acc * want1 >= total * (c + 1)) t_beg[++c] = t + 1;
        }
        for (uint32_t y = 0; y < grid.y; y++)
            for (uint32_t x = 0; x < grid.x; x++) {
                const uint32_t lin = y * grid.x + x;
                const uint32_t ta = lin < want1 ? t_beg[lin] : n_tiles, tb = lin < want1 ? t_beg[lin + 1] : n_tiles;
                SetRuntimeArgs(p1, kr, CoreCoord{x, y},
                               {uint32_t(b_tiles->address()), uint32_t(b_recs->address()),
                                uint32_t(b_planes->address()), uint32_t(b_ramps->address()), ta, tb - ta});
                SetRuntimeArgs(p1, kw, CoreCoord{x, y},
                               {uint32_t(b_s->address()), uint32_t(b_tiles->address()), uint32_t(b_recs->address()),
                                ta, tb - ta, 0u, lin});
                SetRuntimeArgs(p1, kc, CoreCoord{x, y}, {tb - ta});
            }
    }

    // ---- #784 split plan (host, from this view's counts; nothing kept between views)
    const auto tp0 = std::chrono::steady_clock::now();
    constexpr uint32_t HFLAG = 0x80000000u;
    std::vector<uint32_t> heavy, hplanes, metab, htab;
    std::vector<std::vector<uint32_t>> core_h(want);
    std::vector<uint32_t> core_g(want + 1, ng), core_n(want, 0);
    uint64_t light_planes = 0, heavy_planes = 0, a_max = 0, b_max = 0;
    if (split) {
        auto active = [&](uint32_t g) { return (vis[(g / GPB) * 32 + 2 * (g % GPB)] | vis[(g / GPB) * 32 + 2 * (g % GPB) + 1]) != 0; };
        for (uint32_t g = 0; g < ng; g++)
            if (active(g))
                for (uint32_t i = 0; i < 64; i++)
                    if (meta[g * 128 + 64 + i] > kcap) heavy.push_back(g * 64 + i);
        auto cnt = [&](uint32_t r) { return meta[(r / 64) * 128 + 64 + r % 64]; };
        std::stable_sort(heavy.begin(), heavy.end(), [&](uint32_t a, uint32_t b) { return cnt(a) > cnt(b); });
        const uint32_t nhg = (uint32_t(heavy.size()) + HL - 1) / HL;
        metab = meta;
        htab.assign(size_t(std::max<uint32_t>(nhg, 1)) * 128, 0u);
        for (uint32_t h = 0; h < heavy.size(); h++) {
            const uint32_t r = heavy[h], j = h / HL, i = h % HL;
            htab[j * 128 + i] = meta[(r / 64) * 128 + r % 64];
            htab[j * 128 + 64 + i] = cnt(r);
            metab[(r / 64) * 128 + 64 + r % 64] = HFLAG | h;
            heavy_planes += i == 0 ? cnt(r) : 0;
        }
        // Phase A: LPT, largest group first (groups are already in descending order), ties to
        // the lower core.
        std::vector<uint64_t> load(want, 0);
        for (uint32_t j = 0; j < nhg; j++) {
            const uint32_t c = uint32_t(std::min_element(load.begin(), load.end()) - load.begin());
            core_h[c].push_back(j);
            load[c] += htab[j * 128 + 64];
            if (core_h[c].size() > 300) { std::fprintf(stderr, "too many heavy groups on core %u\n", c); return 2; }
        }
        a_max = *std::max_element(load.begin(), load.end());
        // Phase B: light planes per active group (a heavy lane feeds plane 0 only), one contiguous
        // range per core, smallest capacity that fits (binary search on the greedy fill).
        std::vector<uint64_t> w(ng, 0);
        uint64_t wsum = 0, wmax = 0;
        for (uint32_t g = 0; g < ng; g++) {
            if (!active(g)) continue;
            uint32_t k = 1;
            for (uint32_t i = 0; i < 64; i++) {
                const uint32_t c = metab[g * 128 + 64 + i];
                if (!(c & HFLAG)) k = std::max(k, c);
            }
            light_planes += k;
            w[g] = k + ovh;
            wsum += w[g];
            wmax = std::max(wmax, w[g]);
        }
        auto fill = [&](uint64_t cap, std::vector<uint32_t>* b) {
            uint32_t c = 0;
            uint64_t acc = 0;
            if (b) (*b)[0] = 0;
            for (uint32_t g = 0; g < ng; g++) {
                if (acc + w[g] > cap) {
                    if (++c >= want) return false;
                    if (b) (*b)[c] = g;
                    acc = 0;
                }
                acc += w[g];
            }
            if (b)
                for (uint32_t q = c + 1; q <= want; q++) (*b)[q] = ng;
            return true;
        };
        uint64_t lo = std::max<uint64_t>(wmax, (wsum + want - 1) / want), hi = wsum;
        while (lo < hi) {
            const uint64_t mid = (lo + hi) / 2;
            if (fill(mid, nullptr)) hi = mid; else lo = mid + 1;
        }
        fill(lo, &core_g);
        for (uint32_t c = 0; c < want; c++) {
            uint64_t acc = 0;
            for (uint32_t g = core_g[c]; g < core_g[c + 1]; g++) {
                acc += w[g];
                core_n[c] += w[g] != 0;
            }
            b_max = std::max(b_max, acc);
        }
    }
    const double plan_ms = ms_since(tp0);
    const uint32_t nhrec = split ? std::max<uint32_t>(1, (uint32_t(heavy.size()) + HL - 1) / HL * HL) : 1;
    auto b_metab = make_dram(dev.get(), uint64_t(split ? ng : 1) * META_PAGE, META_PAGE);
    auto b_htab = make_dram(dev.get(), uint64_t(split ? htab.size() / 128 : 1) * META_PAGE, META_PAGE);
    auto b_hs = make_dram(dev.get(), uint64_t((nhrec + 63) / 64) * SLOT_PAGE, SLOT_PAGE);
    const auto tu0 = std::chrono::steady_clock::now();
    if (split) {
        distributed::EnqueueWriteMeshBuffer(*cq, b_metab, metab, true);
        distributed::EnqueueWriteMeshBuffer(*cq, b_htab, htab, true);
    }
    const double split_upload_ms = ms_since(tu0);
    if (split)
        std::printf("[SPLIT] kcap %u hl %u ovh %u heavy_rows %zu heavy_groups %zu heavy_planes %llu light_planes %llu "
                    "a_max_planes %llu b_max_weight %llu plan_ms %.3f upload_ms %.3f\n",
                    kcap, HL, ovh, heavy.size(), htab.size() / 128, (unsigned long long)heavy_planes,
                    (unsigned long long)light_planes, (unsigned long long)a_max, (unsigned long long)b_max, plan_ms,
                    split_upload_ms);

    // ---- T2 program (t697 gather reduce + SFPU fwd/bwd + sparse Adam)
    Program p2 = CreateProgram();
    uint32_t tot_groups = 0;
    {
        enum : uint32_t { CB_HDR = 0, CB_P = 1, CB_M = 2, CB_V = 3, CB_K = 4, CB_WH = 5, CB_WS = 6 };
        enum : uint32_t { CB_PO = 16, CB_MO = 17, CB_VO = 18, CB_FO = 19, CB_GO = 20 };
        add_cb(p2, cores, CB_HDR, 64, 2);
        add_cb(p2, cores, CB_P, P_PAGE, 2);
        add_cb(p2, cores, CB_M, MV_PAGE, 2, DataFormat::Float16_b);
        add_cb(p2, cores, CB_V, MV_PAGE, 2, DataFormat::Float16_b);
        add_cb(p2, cores, CB_K, P_PAGE, 2);
        add_cb(p2, cores, CB_WH, 64, 2);
        add_cb(p2, cores, CB_WS, VIS_PAGE + META_PAGE + STAGE_PAGES * SLOT_PAGE + (split ? 64 * 64 : 0), 1);
        add_cb(p2, cores, CB_PO, P_PAGE, 2);
        add_cb(p2, cores, CB_MO, MV_PAGE, 2, DataFormat::Float16_b);
        add_cb(p2, cores, CB_VO, MV_PAGE, 2, DataFormat::Float16_b);
        add_cb(p2, cores, CB_FO, P_PAGE, 2);
        add_cb(p2, cores, CB_GO, P_PAGE, 2);
        std::vector<uint32_t> rct, wct;
        for (int i = 0; i < 6 + int(split) + int(sh); i++) TensorAccessorArgs::create_dram_interleaved().append_to(rct);
        for (int i = 0; i < 5; i++) TensorAccessorArgs::create_dram_interleaved().append_to(wct);
        std::map<std::string, std::string> rdef = {{"T2_STAGE_PAGES", std::to_string(STAGE_PAGES) + "u"}}, wdef;
        // SH mode: the post pass reads T2's grads page, so they always go out.
        if (grads || sh) wdef["T2_GRADS_OUT"] = "1";
        if (split) rdef["T2_SPLIT"] = "1";
        if (g2d_only) wdef["T2_G2D_ONLY"] = "1";
        auto cdef = wdef;
        if (sh) {
            rdef["T2_SH"] = "1";
            cdef["T2_SH"] = "1";
            add_cb(p2, cores, 7, P_PAGE, 2);  // CB_AUX
        }
        const std::string kd = TRAIN_KERNEL_DIR;
        auto kr = CreateKernel(p2, kd + "t2g_reader.cpp", cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                                  .noc = NOC::RISCV_1_default, .compile_args = rct, .defines = rdef});
        auto kw = CreateKernel(p2, kd + "t2_writer.cpp", cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                                  .noc = NOC::RISCV_0_default, .compile_args = wct, .defines = wdef});
        std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
        u2d[CB_P] = UnpackToDestMode::UnpackToDestFp32;
        u2d[CB_K] = UnpackToDestMode::UnpackToDestFp32;
        if (sh) u2d[7] = UnpackToDestMode::UnpackToDestFp32;
        auto kc = CreateKernel(p2, kd + "t2g_compute.cpp", cores,
                               ComputeConfig{.math_fidelity = MathFidelity::HiFi3, .fp32_dest_acc_en = true,
                                             .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                             .math_approx_mode = false, .defines = cdef});
        for (uint32_t y = 0; y < grid.y; y++)
            for (uint32_t x = 0; x < grid.x; x++) {
                const uint32_t lin = y * grid.x + x;
                uint32_t n = 0;
                if (split) {
                    n = lin < want ? core_n[lin] : 0;
                    std::vector<uint32_t> ra = {uint32_t(b_p->address()), uint32_t(b_m->address()), uint32_t(b_v->address()),
                                                uint32_t(b_s->address()), uint32_t(b_metab->address()), uint32_t(b_vis->address()),
                                                uint32_t(b_hs->address()), lin < want ? core_g[lin] : ng,
                                                lin < want ? core_g[lin + 1] : ng};
                    if (sh) ra.push_back(uint32_t(b_x->address()));
                    SetRuntimeArgs(p2, kr, CoreCoord{x, y}, ra);
                } else {
                if (lin < want)
                    for (uint32_t b = lin; b < nb; b += want)
                        for (uint32_t gi = 0; gi < GPB; gi++) n += (vis[b * 32 + 2 * gi] | vis[b * 32 + 2 * gi + 1]) != 0;
                std::vector<uint32_t> ra = {uint32_t(b_p->address()), uint32_t(b_m->address()), uint32_t(b_v->address()),
                                            uint32_t(b_s->address()), uint32_t(b_meta->address()), uint32_t(b_vis->address()),
                                            lin < want ? lin : nb, want, nb};
                if (sh) ra.push_back(uint32_t(b_x->address()));
                SetRuntimeArgs(p2, kr, CoreCoord{x, y}, ra);
                }
                tot_groups += n;
                SetRuntimeArgs(p2, kw, CoreCoord{x, y},
                               {uint32_t(b_p->address()), uint32_t(b_m->address()), uint32_t(b_v->address()),
                                uint32_t(b_f->address()), uint32_t(b_g->address()), n, ng});
                std::vector<uint32_t> ca = {n};
                for (float f : cst) { uint32_t u; std::memcpy(&u, &f, 4); ca.push_back(u); }
                SetRuntimeArgs(p2, kc, CoreCoord{x, y}, ca);
            }
    }

    // ---- #784 phase A program: heavy groups -> hsum records
    Program pa = CreateProgram();
    if (split) {
        enum : uint32_t { CB_HDR = 0, CB_K = 4, CB_WH = 5, CB_WS = 6, CB_HR = 7, CB_FO = 19 };
        add_cb(pa, cores, CB_HDR, 64, 2);
        add_cb(pa, cores, CB_K, P_PAGE, 2);
        add_cb(pa, cores, CB_WH, 64, 2);
        add_cb(pa, cores, CB_WS, VIS_PAGE + META_PAGE + STAGE_PAGES * SLOT_PAGE, 1);
        add_cb(pa, cores, CB_HR, P_PAGE, 1);
        add_cb(pa, cores, CB_FO, P_PAGE, 2);
        std::vector<uint32_t> rct, wct;
        for (int i = 0; i < 2; i++) TensorAccessorArgs::create_dram_interleaved().append_to(rct);
        TensorAccessorArgs::create_dram_interleaved().append_to(wct);
        const std::map<std::string, std::string> def = {{"T2_STAGE_PAGES", std::to_string(STAGE_PAGES) + "u"},
                                                        {"T2_HL", std::to_string(HL) + "u"},
                                                        {"T2_PHASE_A", "1"}};
        const std::string kd = TRAIN_KERNEL_DIR;
        auto kr = CreateKernel(pa, kd + "t2g_reader.cpp", cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                                  .noc = NOC::RISCV_1_default, .compile_args = rct, .defines = def});
        auto kw = CreateKernel(pa, kd + "t2h_writer.cpp", cores,
                               DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                                  .noc = NOC::RISCV_0_default, .compile_args = wct, .defines = def});
        std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
        u2d[CB_K] = UnpackToDestMode::UnpackToDestFp32;
        auto kc = CreateKernel(pa, kd + "t2g_compute.cpp", cores,
                               ComputeConfig{.math_fidelity = MathFidelity::HiFi3, .fp32_dest_acc_en = true,
                                             .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                             .math_approx_mode = false, .defines = def});
        for (uint32_t y = 0; y < grid.y; y++)
            for (uint32_t x = 0; x < grid.x; x++) {
                const uint32_t lin = y * grid.x + x;
                const uint32_t n = lin < want ? uint32_t(core_h[lin].size()) : 0;
                std::vector<uint32_t> ra = {uint32_t(b_s->address()), uint32_t(b_htab->address()), n};
                for (uint32_t q = 0; q < n; q++) ra.push_back(core_h[lin][q]);
                SetRuntimeArgs(pa, kr, CoreCoord{x, y}, ra);
                SetRuntimeArgs(pa, kw, CoreCoord{x, y}, {uint32_t(b_hs->address()), n});
                std::vector<uint32_t> ca = {n};
                for (float f : cst) { uint32_t u; std::memcpy(&u, &f, 4); ca.push_back(u); }
                SetRuntimeArgs(pa, kc, CoreCoord{x, y}, ca);
            }
    }


    // ---- t2sh pre / post programs (SH mode only): same groups and block walk as T2.
    Program p3 = CreateProgram(), p4 = CreateProgram();
    if (sh) {
        enum : uint32_t { CB_A = 1, CB_B = 2, CB_M = 3, CB_V = 4, CB_WH = 5, CB_WS = 6, CB_G = 7 };
        enum : uint32_t { CB_PO = 16, CB_MO = 17, CB_VO = 18, CB_GO = 20 };
        const std::string kd = TRAIN_KERNEL_DIR;
        auto build = [&](Program& pr, bool post) {
            add_cb(pr, cores, CB_A, P_PAGE, 2);
            add_cb(pr, cores, CB_B, P_PAGE, post ? 2 : 6);
            add_cb(pr, cores, CB_WH, 64, 2);
            add_cb(pr, cores, CB_WS, VIS_PAGE, 1);
            add_cb(pr, cores, CB_PO, P_PAGE, 2);
            std::vector<UnpackToDestMode> u2d(64, UnpackToDestMode::Default);
            u2d[CB_A] = UnpackToDestMode::UnpackToDestFp32;
            u2d[CB_B] = UnpackToDestMode::UnpackToDestFp32;
            std::map<std::string, std::string> def;
            if (post) {
                add_cb(pr, cores, CB_G, P_PAGE, 2);
                add_cb(pr, cores, CB_M, MV_PAGE, 2, DataFormat::Float16_b);
                add_cb(pr, cores, CB_V, MV_PAGE, 2, DataFormat::Float16_b);
                add_cb(pr, cores, CB_MO, MV_PAGE, 2, DataFormat::Float16_b);
                add_cb(pr, cores, CB_VO, MV_PAGE, 2, DataFormat::Float16_b);
                add_cb(pr, cores, CB_GO, P_PAGE, 2);
                u2d[CB_G] = UnpackToDestMode::UnpackToDestFp32;
                def["T2SH_POST"] = "1";
            }
            std::vector<uint32_t> rct, wct;
            for (int i = 0; i < (post ? 6 : 3); i++) TensorAccessorArgs::create_dram_interleaved().append_to(rct);
            for (int i = 0; i < (post ? 4 : 1); i++) TensorAccessorArgs::create_dram_interleaved().append_to(wct);
            auto kr = CreateKernel(pr, kd + "t2sh_reader.cpp", cores,
                                   DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                                      .noc = NOC::RISCV_1_default, .compile_args = rct, .defines = def});
            auto kw = CreateKernel(pr, kd + "t2sh_writer.cpp", cores,
                                   DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                                      .noc = NOC::RISCV_0_default, .compile_args = wct, .defines = def});
            auto kc = CreateKernel(pr, kd + "t2sh_compute.cpp", cores,
                                   ComputeConfig{.math_fidelity = MathFidelity::HiFi3, .fp32_dest_acc_en = true,
                                                 .dst_full_sync_en = true, .unpack_to_dest_mode = u2d,
                                                 .math_approx_mode = false, .defines = def});
            for (uint32_t y = 0; y < grid.y; y++)
                for (uint32_t x = 0; x < grid.x; x++) {
                    const uint32_t lin = y * grid.x + x;
                    uint32_t n = 0;
                    if (lin < want)
                        for (uint32_t b = lin; b < nb; b += want)
                            for (uint32_t gi = 0; gi < GPB; gi++) n += (vis[b * 32 + 2 * gi] | vis[b * 32 + 2 * gi + 1]) != 0;
                    const uint32_t b0 = lin < want ? lin : nb;
                    std::vector<uint32_t> ra, wa;
                    if (post) {
                        ra = {uint32_t(b_x->address()), uint32_t(b_g->address()), uint32_t(b_r->address()),
                              uint32_t(b_rm->address()), uint32_t(b_rv->address()), uint32_t(b_vis->address()), b0, want, nb};
                        wa = {uint32_t(b_r->address()), uint32_t(b_rm->address()), uint32_t(b_rv->address()),
                              uint32_t(b_rg->address()), n};
                    } else {
                        ra = {uint32_t(b_p->address()), uint32_t(b_r->address()), uint32_t(b_vis->address()), b0, want, nb};
                        wa = {uint32_t(b_x->address()), n};
                    }
                    SetRuntimeArgs(pr, kr, CoreCoord{x, y}, ra);
                    SetRuntimeArgs(pr, kw, CoreCoord{x, y}, wa);
                    std::vector<uint32_t> ca = {n};
                    for (float f : shc) { uint32_t u; std::memcpy(&u, &f, 4); ca.push_back(u); }
                    SetRuntimeArgs(pr, kc, CoreCoord{x, y}, ca);
                }
        };
        build(p3, false);
        build(p4, true);
    }

    distributed::MeshWorkload w1, w2, wa, w3, w4;
    w1.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(p1));
    w2.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(p2));
    if (split) wa.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(pa));
    if (sh) {
        w3.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(p3));
        w4.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(p4));
    }

    // Step 0 (the checked one): T1 -> [pre] -> T2 -> [post] with a Finish between, so each stage
    // is timed alone.
    std::vector<double> t1ms, t2ms, prems, postms, chain;
    auto timed = [&](distributed::MeshWorkload& w, std::vector<double>& acc) {
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(*cq, w, false);
        distributed::Finish(*cq);
        acc.push_back(ms_since(t0));
    };
    auto staged = [&]() {
        timed(w1, t1ms);
        if (sh) timed(w3, prems);
        const auto t0 = std::chrono::steady_clock::now();
        if (split) distributed::EnqueueMeshWorkload(*cq, wa, false);
        distributed::EnqueueMeshWorkload(*cq, w2, false);
        distributed::Finish(*cq);
        t2ms.push_back(ms_since(t0));
        if (sh) timed(w4, postms);
    };
    staged();
    {
        std::vector<uint32_t> o;
        distributed::EnqueueReadMeshBuffer(*cq, o, b_s, true);
        save(odir + "/slots.f32", o);
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
        if (split) {  // #784: the host plan, for the check against gsplat.t2_split.device_plan
            save(odir + "/split_heavy.u32", heavy);
            save(odir + "/split_bounds.u32", core_g);
            distributed::EnqueueReadMeshBuffer(*cq, o, b_hs, true);
            save(odir + "/hsum.f32", o);
        }
        if (sh) {
            distributed::EnqueueReadMeshBuffer(*cq, o, b_r, true);
            save(odir + "/sh_rest_out.f32", o);
            distributed::EnqueueReadMeshBuffer(*cq, o, b_rm, true);
            save(odir + "/sh_m_out.u16", o);
            distributed::EnqueueReadMeshBuffer(*cq, o, b_rv, true);
            save(odir + "/sh_v_out.u16", o);
            if (grads) {
                distributed::EnqueueReadMeshBuffer(*cq, o, b_rg, true);
                save(odir + "/sh_grads.f32", o);
            }
        }
    }
    // Sustained steps on the same buffers (params / m / v keep updating; same view inputs).
    for (int i = 0; i < reps; i++) staged();
    for (int i = 0; i < reps; i++) {
        const auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(*cq, w1, false);
        if (sh) distributed::EnqueueMeshWorkload(*cq, w3, false);
        if (split) distributed::EnqueueMeshWorkload(*cq, wa, false);
        distributed::EnqueueMeshWorkload(*cq, w2, false);
        if (sh) distributed::EnqueueMeshWorkload(*cq, w4, false);
        distributed::Finish(*cq);
        chain.push_back(ms_since(t0));
    }
    // #784: each split phase alone (A only re-sums the same slots; B is a normal T2 step).
    std::vector<double> tams, tbms;
    for (int i = 0; split && i < reps; i++) {
        auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(*cq, wa, false);
        distributed::Finish(*cq);
        tams.push_back(ms_since(t0));
        t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(*cq, w2, false);
        distributed::Finish(*cq);
        tbms.push_back(ms_since(t0));
    }
    const Stat s1 = stat(t1ms), s2 = stat(t2ms), sc = stat(chain), sa = stat(tams), sb = stat(tbms);
    const Stat s3 = stat(prems), s4 = stat(postms);
    char sbuf[640];
    std::snprintf(sbuf, sizeof sbuf,
                  ", \"split\": %s, \"g2d_only\": %s, \"kcap\": %u, \"hl\": %u, \"ovh\": %u, \"heavy_rows\": %zu, "
                  "\"heavy_planes\": %llu, \"light_planes\": %llu, \"a_max_planes\": %llu, \"b_max_weight\": %llu, "
                  "\"plan_ms\": %.3f, \"split_upload_ms\": %.3f, \"ta_ms_min\": %.4f, \"ta_ms_med\": %.4f, "
                  "\"tb_ms_min\": %.4f, \"tb_ms_med\": %.4f",
                  split ? "true" : "false", g2d_only ? "true" : "false", kcap, HL, ovh, heavy.size(),
                  (unsigned long long)heavy_planes, (unsigned long long)light_planes, (unsigned long long)a_max,
                  (unsigned long long)b_max, plan_ms, split_upload_ms, sa.min, sa.med, sb.min, sb.med);
    char buf[2048];
    std::snprintf(buf, sizeof buf,
                  "{\"n_tiles\": %u, \"n_recs\": %u, \"n_slots\": %u, \"n_groups\": %u, \"groups_processed\": %u, "
                  "\"cores\": %u, \"t1_cores\": %u, \"reps\": %d, \"grads_out\": %s, "
                  "\"t1_ms_first\": %.4f, \"t1_ms_min\": %.4f, \"t1_ms_med\": %.4f, "
                  "\"t2_ms_first\": %.4f, \"t2_ms_min\": %.4f, \"t2_ms_med\": %.4f, "
                  "\"sh_rest\": %s, \"t2sh_pre_ms_first\": %.4f, \"t2sh_pre_ms_min\": %.4f, \"t2sh_pre_ms_med\": %.4f, "
                  "\"t2sh_post_ms_first\": %.4f, \"t2sh_post_ms_min\": %.4f, \"t2sh_post_ms_med\": %.4f, "
                  "\"chain_ms_min\": %.4f, \"chain_ms_med\": %.4f, \"host_repack_ms\": 0%s}",
                  n_tiles, n_recs, n_slots, ng, tot_groups, want, want1, reps, grads ? "true" : "false",
                  s1.first, s1.min, s1.med, s2.first, s2.min, s2.med, sh ? "true" : "false", s3.first, s3.min,
                  s3.med, s4.first, s4.min, s4.med, sc.min, sc.med, sbuf);
    std::ofstream(odir + "/timing.json") << buf << "\n";
    std::printf("[STEP] %s\n", buf);
    dev->close();
    return 0;
}
