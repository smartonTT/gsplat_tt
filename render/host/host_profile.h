#pragma once
//
// host_profile.h — env-gated (GSPLAT_TT_HOST_PROFILE=1) host-side timing of the
// INTER-FRAME host gap, re-introduced for the iter-136 diagnostic (the iter-120
// GSPLAT_TT_HOST_PROFILE chrono was reverted; this is the focused successor).
//
// The clean renderer drives one in-order command queue: every device op is
// enqueued, then a blocking Finish / blocking read re-imposes ordering. So the
// DEVICE is idle precisely in the window between "view i's blend finished on
// device" and "view i+1's first compute program (pfwc) is enqueued". iter-135
// Tracy measured that idle window at a steady-state ~32 ms/view. This profiler
// decomposes THAT window into named host activities, sampled per view boundary:
//
//   gap(i->i+1) = d2h_xfer(i)    final image D2H readback (device idle, transfer)
//               + unpack(i)      bf16 tiles -> fp32 image host CPU repack
//               + tail(i)        post-blend C++ (P_kept loop) + pybind return build
//               + py_gap         pybind return + Python loop + c2w_to_w2c + marshal
//               + head(i+1)      render_view head: array.request + memset + caches
//               + pfwc_disp(i+1) pfwc SetRuntimeArgs(110*3) + EnqueueMeshWorkload
//
// Everything is a strict no-op (one cached getenv read) unless the flag is set;
// it never changes pixels/PSNR. The marks are absolute high-resolution
// timestamps; boundaries are computed + printed (HPGAP ...) at pfwc-enqueue time
// of the FOLLOWING view, when both the previous view (fully complete) and the
// current view's pfwc dispatch are known. The per-component summary (median over
// interior boundaries, warmup excluded) is computed offline by parsing the HPGAP
// lines from the run log.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gsplat_tt::hostprof {

using clk = std::chrono::high_resolution_clock;
using tp = clk::time_point;

inline double ms(tp a, tp b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

inline bool enabled() {
    static const bool e = [] {
        const char* v = std::getenv("GSPLAT_TT_HOST_PROFILE");
        return v && v[0] && v[0] != '0';
    }();
    return e;
}

struct Marks {
    tp t_enter{};            // render_view entry
    tp t_pfwc_disp0{};       // pfwc SetRuntimeArgs loop start (dispatch begin)
    tp t_pfwc_enq{};         // pfwc EnqueueMeshWorkload returned (device starts pfwc)
    tp t_blend_dev_done{};   // host_finish_blend Finish returned (blend done on device)
    tp t_blend_readback{};   // final image D2H read returned
    tp t_blend_unpack{};     // tiles_to_image (bf16->fp32) done
    tp t_return{};           // render_view return
};

struct Sample {
    double gap, d2h, unpack, tail, py, head, pfwc_disp;
};

struct State {
    Marks cur;
    Marks prev;
    bool have_prev = false;
    int view = 0;
    std::vector<Sample> samples;  // one per interior boundary
};

inline State& state() {
    static State s;
    return s;
}

// render_view entry: rotate cur into a fresh view.
inline void on_view_enter() {
    if (!enabled()) return;
    auto& s = state();
    s.cur = Marks{};
    s.cur.t_enter = clk::now();
}

// pfwc records its dispatch window. After the enqueue we know view i+1's
// pfwc-start, so close the boundary against the previous (complete) view.
inline void on_pfwc_dispatch_start() {
    if (!enabled()) return;
    state().cur.t_pfwc_disp0 = clk::now();
}

inline void on_pfwc_enqueued() {
    if (!enabled()) return;
    auto& s = state();
    s.cur.t_pfwc_enq = clk::now();
    if (!s.have_prev) return;  // warmup->view0 boundary skipped (huge JIT/alloc)
    const Marks& p = s.prev;
    const Marks& c = s.cur;
    Sample smp;
    smp.d2h = ms(p.t_blend_dev_done, p.t_blend_readback);
    smp.unpack = ms(p.t_blend_readback, p.t_blend_unpack);
    smp.tail = ms(p.t_blend_unpack, p.t_return);
    smp.py = ms(p.t_return, c.t_enter);
    smp.head = ms(c.t_enter, c.t_pfwc_disp0);
    smp.pfwc_disp = ms(c.t_pfwc_disp0, c.t_pfwc_enq);
    smp.gap = ms(p.t_blend_dev_done, c.t_pfwc_enq);
    s.samples.push_back(smp);
    std::printf(
        "HPGAP b=%d gap=%.3f d2h=%.3f unpack=%.3f tail=%.3f py=%.3f head=%.3f "
        "pfwc_disp=%.3f\n",
        s.view, smp.gap, smp.d2h, smp.unpack, smp.tail, smp.py, smp.head,
        smp.pfwc_disp);
    std::fflush(stdout);
}

inline void on_blend_device_done() {
    if (!enabled()) return;
    state().cur.t_blend_dev_done = clk::now();
}

inline void on_blend_readback_done() {
    if (!enabled()) return;
    state().cur.t_blend_readback = clk::now();
}

inline void on_blend_unpack_done() {
    if (!enabled()) return;
    state().cur.t_blend_unpack = clk::now();
}

// render_view return: freeze cur as prev for the next boundary.
inline void on_view_return() {
    if (!enabled()) return;
    auto& s = state();
    s.cur.t_return = clk::now();
    s.prev = s.cur;
    s.have_prev = true;
    s.view++;
}

}  // namespace gsplat_tt::hostprof
