// Task #374 (GSPLAT_TT_OUT_ZEROCOPY): the ring of pinned output image buffers.
//
// The blend writer writes each frame into one slot; render_view hands the
// caller that slot's lease (the host memory, no copy). A slot whose lease the
// caller still holds is never written again, so a returned image stays valid
// for as long as the caller keeps it. A caller that keeps one frame while it
// renders the next (bench, viewer) cycles through two slots.
//
// acquire() picks the slot for the next frame: the first free slot after the
// last one written; else a new slot while under `cap`; else it replaces the
// slot after the last one written (the caller keeps the old lease; only the
// ring's reference and the slot's device mapping, `extra`, are dropped).
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace gsplat_tt {

template <class Lease, class Extra>
class OutRing {
public:
    struct Slot {
        std::shared_ptr<Lease> lease;
        Extra extra{};
    };

    explicit OutRing(std::size_t cap = 2) : cap_(cap < 1 ? 1 : cap) {}

    // make(Slot&) fills a fresh slot (lease + extra). Returns the slot index.
    template <class Make>
    std::size_t acquire(Make&& make) {
        const std::size_t n = slots_.size();
        for (std::size_t k = 1; k <= n; k++) {
            const std::size_t i = (last_ + k) % n;
            if (slots_[i].lease && slots_[i].lease.use_count() == 1) return last_ = i;
        }
        if (n < cap_) {
            slots_.emplace_back();
            make(slots_.back());
            grows_++;
            return last_ = n;
        }
        const std::size_t i = (last_ + 1) % n;
        slots_[i] = Slot{};  // drop the old device mapping before making the new one
        make(slots_[i]);
        replaced_++;
        return last_ = i;
    }

    Slot& at(std::size_t i) { return slots_[i]; }
    std::size_t last() const { return last_; }
    std::size_t size() const { return slots_.size(); }
    std::size_t grows() const { return grows_; }
    std::size_t replaced() const { return replaced_; }
    // Leases the caller holds stay valid; only the ring's references go.
    void clear() {
        slots_.clear();
        last_ = 0;
    }

private:
    std::vector<Slot> slots_;
    std::size_t cap_;
    std::size_t last_ = 0;
    std::size_t grows_ = 0;
    std::size_t replaced_ = 0;
};

}  // namespace gsplat_tt
