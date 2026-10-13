// Task #643 (L-A', GSPLAT_TT_SORT_IN_FOLD, default off): the next view's early
// one-launch sort is enqueued in the blend's after-hook, on CQ0 right behind
// the mat+blend that ran that view's K2 (B4a fold), instead of at the start of
// that view. The next tile_assign_fused_k2 then skips the sort enqueue and has
// CQ1 wait on the event recorded before the sort (not behind it) to read
// proj_M. The clamp path (proj_M[2] != 0) is unchanged: drain, regrow, rerun.
//
// Device-free so tests/unit/test_fold_sort_t643.cpp can check it on the host.
#pragma once

#include <cstddef>

namespace gsplat_tt::fold_sort {

// The after-hook enqueues the next view's sort only when the blend just
// enqueued ran that view's whole K2 (fold_ran), the K2 was set up for the
// early sort (want_early), no B4a verify rerun of the K2 is pending (it would
// rewrite the pairs under the sort) and CQ1 exists (proj_M is read there).
inline bool hook_enqueues(bool enabled, bool fold_ran, bool want_early, bool verify_pending,
                          bool have_cq1) {
    return enabled && fold_ran && want_early && !verify_pending && have_cq1;
}

// The next fused K2 uses the hook's sort only if it also skips its own K2
// (the fold held for this view) and the hook did enqueue it.
inline bool use_hook_sort(bool fold_skip, bool hook_sorted) { return fold_skip && hook_sorted; }

// Task #552 rule: a queued mat+blend may still read the sort's buffers, so
// growing any of them from the hook needs a Finish first.
inline bool must_drain(std::size_t cap, std::size_t need, bool exists) {
    return exists && cap < need;
}

}  // namespace gsplat_tt::fold_sort
