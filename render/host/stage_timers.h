#pragma once
//
// stage_timers.h — always-on, per-stage host wall-clock attribution for the
// clean render pipeline.
//
// Before this, the bench only emitted `TTW_TIMING ms_view=` / `blend=` (both the
// same avg frame time) plus the [SORT] line, so ~135 ms of the ~173 ms/view
// bicycle frame was unattributed. These counters split render_view into the
// stages the pipeline actually executes:
//
//   head         render_view entry -> stage 1 (py buffer requests, image memset)
//   project      fused means_cam+pfwc, then gather_visible                (stage 1)
//   tile_assign  device tile_assign                                       (stage 2)
//   sort         sort_and_bin_tt MINUS its fused cull/blend continuation  (stage 3)
//   blend_setup  resident-blend SetRuntimeArgs pre-pass (SetupRuntimeArgsOnly)
//   mat          sort_subchunk_mat launch + Finish; GSPLAT_TT_SPLIT_BLEND=1 only
//   cull         SFPU microblock cull pass                                (stage 4)
//   blend        blend enqueue + Finish (the device blend window)         (stage 5)
//
// Booking of the device windows (task #90): by default sort_subchunk_mat, the
// SFPU cull and the blend run back to back on one in-order CQ with ONE Finish
// at the blend readback, so `blend` holds the device time of all three
// programs, `cull` is only its enqueue, `mat` is 0 and `sort_mat` is the mat
// enqueue. GSPLAT_TT_SPLIT_BLEND=1 (diagnostic) adds a Finish after mat and
// after cull so `mat`, `cull` and `blend` each hold one program (plus ~0.1 ms
// of extra drain/launch latency per added Finish); the image is unchanged.
//   d2h          final image bf16 readback
//   assemble     bf16 microblock tiles -> fp32 HWC image (host CPU)
//   tail         P_kept scan + stats dict + pybind return build
//   view_total   render_view entry -> return
//
// head+project+tile_assign+sort+blend_setup+cull+blend+d2h+assemble+tail
// must reconstruct view_total; view_total vs the Python-side avg_frame_ms
// leaves only the pybind/marshal residual.
//
// Cost: two steady_clock::now() per span, ~50 spans per view -> well under a
// microsecond on a 173 ms frame. Unconditional (no env gate) so the numbers are
// always available, and it never touches pixels.

#include <chrono>
#include <cstdint>

namespace gsplat_tt::stagetimers {

using clk = std::chrono::steady_clock;

// Milliseconds accumulated over every render_view since the last reset().
struct Acc {
    double head = 0.0;
    double project = 0.0;
    double tile_assign = 0.0;
    double sort = 0.0;
    double blend_setup = 0.0;
    double mat = 0.0;
    double cull = 0.0;
    double blend = 0.0;
    double d2h = 0.0;
    double assemble = 0.0;
    double tail = 0.0;
    // Task #379: host time of the next view's pfwc enqueued inside this view's
    // blend (GSPLAT_TT_XVIEW_OVERLAP); not part of sort/blend.
    double xview = 0.0;
    double view_total = 0.0;
    // Sub-buckets of `sort`, booked by render.cpp from SortCallTimings. They
    // partition the sort span; sort - sum(sort_*) is the unattributed rest.
    double sort_pread = 0.0;
    double sort_bin_count = 0.0;
    double sort_bin_hist_d2h = 0.0;
    double sort_bin_layout = 0.0;
    double sort_upload = 0.0;
    double sort_bin_emit = 0.0;
    double sort_kernel = 0.0;
    double sort_d2h = 0.0;
    double sort_compact = 0.0;
    double sort_publish_host = 0.0;
    double sort_publish_wait = 0.0;
    double sort_mat = 0.0;
    // Task #483: the host steps the leaves above left out (one-launch path).
    //   pre          P read -> emit: buffer grow checks + device_state registers
    //   log          the per-frame [SORT] stderr line
    //   cont_prep    blend host prep before its rtargs: buffer lookups, out ring,
    //                parked-mat fallback (inside the fused continuation)
    //   cont_rtargs  blend per-core runtime-arg build + SetRuntimeArgs
    //   cont_other   continuation wall minus its booked buckets and the two above
    double sort_pre = 0.0;
    double sort_log = 0.0;
    double sort_cont_prep = 0.0;
    double sort_cont_rtargs = 0.0;
    double sort_cont_other = 0.0;
    // Sub-buckets of `project` (render.cpp run_project + pfwc_device +
    // gather_visible_device). Disjoint; project - sum(project_*) is the rest.
    //   cov3d           scene cov3d 9->6 repack (cached after the first view)
    //   pfwc_setup      context + buffer checks, cached H2D, cc_scales, split
    //   pfwc_rtargs     per-core runtime-arg build + SetRuntimeArgs
    //   pfwc_enqueue    EnqueueMeshWorkload
    //   pfwc_finish     Finish (pfwc device time not hidden by the enqueue)
    //   gather_setup    context, cached scene H2D, output/count buffer checks
    //   gather_rtargs   count + scan + scatter arg build + SetRuntimeArgs
    //   gather_enqueue  the 3 EnqueueMeshWorkload (count, scan, scatter)
    //   gather_wait     blocking 1-page M read = drain of count+scan+scatter
    //   gather_result   host ProjectResult build (depths sized M)
    double project_cov3d = 0.0;
    double project_pfwc_setup = 0.0;
    double project_pfwc_rtargs = 0.0;
    double project_pfwc_chunkcull = 0.0;  // task #433 host chunk cull + list upload
    double project_pfwc_enqueue = 0.0;
    double project_pfwc_finish = 0.0;
    double project_gather_setup = 0.0;
    double project_gather_rtargs = 0.0;
    double project_gather_enqueue = 0.0;
    double project_gather_wait = 0.0;
    double project_gather_result = 0.0;
    // Sub-buckets of `tile_assign` (tile_assign_device). Disjoint.
    //   setup        context, resident-buffer lookups, buffer grow checks
    //   rtargs       K1 + scan1 + scan_bases + scan2 + K2 SetRuntimeArgs
    //   enqueue      the 5 EnqueueMeshWorkload
    //   scan_finish  Finish after scan_bases (drains K1 + scan1 + scan_bases)
    //   p_d2h        blocking 64 B read of the pair count P
    //   k2_finish    Finish after K2 (drains scan2 + K2)
    //   publish      keep-mask fill check + resident-pair registration
    double tile_assign_setup = 0.0;
    double tile_assign_rtargs = 0.0;
    double tile_assign_enqueue = 0.0;
    double tile_assign_scan_finish = 0.0;
    double tile_assign_p_d2h = 0.0;
    double tile_assign_k2_finish = 0.0;
    double tile_assign_publish = 0.0;
    std::uint64_t views = 0;
};

// Single process-wide accumulator. Defined out-of-line in stage_timers.cpp
// (part of render_tt) so the pybind module and the device drivers — which are
// compiled with different -fvisibility — share exactly one instance.
Acc& acc();
void reset();

// GSPLAT_TT_SPLIT_BLEND=1: Finish after sort_subchunk_mat and after the cull so
// the mat / cull / blend buckets each time one program (default off).
bool split_blend();

// Scoped span: adds its lifetime (ms) into `sink` at stop()/destruction.
class Span {
  public:
    explicit Span(double& sink) : sink_(&sink), t0_(clk::now()) {}
    Span(const Span&) = delete;
    Span& operator=(const Span&) = delete;
    ~Span() { stop(); }

    // Closes the span early. Returns the elapsed ms (0 if already closed).
    double stop() {
        if (sink_ == nullptr) return 0.0;
        const double ms =
            std::chrono::duration<double, std::milli>(clk::now() - t0_).count();
        *sink_ += ms;
        sink_ = nullptr;
        return ms;
    }

  private:
    double* sink_;
    clk::time_point t0_;
};

}  // namespace gsplat_tt::stagetimers
