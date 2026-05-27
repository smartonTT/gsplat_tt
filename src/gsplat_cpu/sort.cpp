#include "gsplat_cpu/sort.h"

#include "gsplat_cpu/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <vector>

namespace gsplat_cpu {

namespace {

// Per-worker scratch for the per-tile radix sort. One worker reuses these
// buffers across every tile it processes. Sized lazily to the largest tile's
// entry count it sees. Avoids ~768 std::vector heap allocations per frame
// (3 per tile * 256 tiles).
struct SortScratch {
    std::vector<uint32_t> tmp_keys;
    std::vector<int64_t>  tmp_ids;
};

// LSD radix sort on uint32 keys, stable, 4 passes of 8 bits each. Sorts the
// (key, id) pair in place over [base, base + n). The caller owns the
// data buffers; we just swap the contents twice per pass (in/out -> out/in).
// Stability is required to match numpy's stable radix-argsort tie-break.
inline void radix_sort_tile(
    uint32_t* keys, int64_t* ids, std::size_t n, SortScratch& sc) {
    if (n <= 1) return;
    // Tiny-tile fallback: a stable insertion sort is faster than 4 passes
    // of 256-bucket histograms when n is small. Threshold tuned to favor
    // the heavy-tile case (where radix wins decisively).
    if (n <= 16) {
        for (std::size_t i = 1; i < n; ++i) {
            const uint32_t k = keys[i];
            const int64_t  v = ids[i];
            std::size_t j = i;
            while (j > 0 && keys[j - 1] > k) {
                keys[j] = keys[j - 1];
                ids[j]  = ids[j - 1];
                --j;
            }
            keys[j] = k;
            ids[j]  = v;
        }
        return;
    }

    if (sc.tmp_keys.size() < n) sc.tmp_keys.resize(n);
    if (sc.tmp_ids.size()  < n) sc.tmp_ids.resize(n);

    uint32_t* in_k  = keys;
    int64_t*  in_v  = ids;
    uint32_t* out_k = sc.tmp_keys.data();
    int64_t*  out_v = sc.tmp_ids.data();

    uint32_t counts[256];
    uint32_t offsets[256];

    for (int byte_idx = 0; byte_idx < 4; ++byte_idx) {
        const int shift = byte_idx * 8;
        for (int i = 0; i < 256; ++i) counts[i] = 0;

        for (std::size_t i = 0; i < n; ++i) {
            ++counts[(in_k[i] >> shift) & 0xFF];
        }
        uint32_t sum = 0;
        for (int i = 0; i < 256; ++i) {
            offsets[i] = sum;
            sum += counts[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            const uint8_t b = static_cast<uint8_t>((in_k[i] >> shift) & 0xFF);
            const uint32_t pos = offsets[b]++;
            out_k[pos] = in_k[i];
            out_v[pos] = in_v[i];
        }
        std::swap(in_k, out_k);
        std::swap(in_v, out_v);
    }

    // After 4 passes (even), in_k/in_v are the originals: data is in keys/ids.
    // No final copy needed.
}

}  // namespace

SortResult sort_and_bin(
    const int64_t* gaussian_ids,
    const int64_t* tile_ids,
    const float* depths,
    const std::size_t P,
    const std::size_t M,
    const int tiles_x,
    const int tiles_y,
    ThreadPool* pool) {
    (void)M;

    const int num_tiles = tiles_x * tiles_y;
    SortResult result;
    result.tile_ranges.assign(static_cast<std::size_t>(num_tiles) * 2, 0);

    if (P == 0) {
        return result;
    }

    const std::size_t W_count = (pool != nullptr) ? std::max<std::size_t>(1, pool->size()) : 1;

    // iter-043: parallel per-chunk count -> per-(chunk, tile) starts ->
    // parallel scatter into globally-shared packed_keys / packed_ids.
    // Replaces the two prior serial 294k-iteration loops (Pass 1a count
    // and Pass 2 scatter) that were single-thread on ~0.3 ms wall each
    // and gated the radix-sort parallelism behind them.
    //
    // Stability is preserved: chunks tile [0, P) in contiguous ranges,
    // so within each tile the per-(chunk, tile) sub-slots scatter in
    // input order and the cross-chunk concatenation maintains the global
    // input order (chunk 0's pairs end up before chunk 1's for any tile).
    //
    // Chunk size is chosen so each worker gets exactly one chunk: keeps
    // the W * num_tiles per-(chunk, tile) table at ~20 KB (10 * 256 * 8 B)
    // - small enough to live in L1 of the prefix-sum driver.
    const std::size_t chunk_count_sz = std::max<std::size_t>(1024, (P + W_count - 1) / W_count);
    const std::size_t num_chunks = (P + chunk_count_sz - 1) / chunk_count_sz;
    std::vector<std::vector<int64_t>> chunk_counts(num_chunks,
        std::vector<int64_t>(static_cast<std::size_t>(num_tiles), 0));

    auto count_chunk = [&](std::size_t c) {
        const std::size_t lo = c * chunk_count_sz;
        const std::size_t hi = std::min(lo + chunk_count_sz, P);
        int64_t* local = chunk_counts[c].data();
        for (std::size_t i = lo; i < hi; ++i) {
            ++local[static_cast<std::size_t>(tile_ids[i])];
        }
    };
    if (pool != nullptr && W_count > 1 && num_chunks > 1) {
        for (std::size_t c = 0; c < num_chunks; ++c) {
            pool->submit([c, &count_chunk]() { count_chunk(c); });
        }
        pool->wait();
    } else {
        for (std::size_t c = 0; c < num_chunks; ++c) count_chunk(c);
    }

    // Pass 1b: global per-tile totals + prefix sum into starts.
    std::vector<int64_t> counts(static_cast<std::size_t>(num_tiles), 0);
    for (std::size_t c = 0; c < num_chunks; ++c) {
        const int64_t* local = chunk_counts[c].data();
        for (int t = 0; t < num_tiles; ++t) {
            counts[static_cast<std::size_t>(t)] += local[static_cast<std::size_t>(t)];
        }
    }
    std::vector<int64_t> starts(static_cast<std::size_t>(num_tiles) + 1, 0);
    for (int t = 0; t < num_tiles; ++t) {
        const int64_t c = counts[static_cast<std::size_t>(t)];
        starts[static_cast<std::size_t>(t) + 1] =
            starts[static_cast<std::size_t>(t)] + c;
        if (c > 0) {
            result.tile_ranges[static_cast<std::size_t>(t) * 2 + 0] =
                starts[static_cast<std::size_t>(t)];
            result.tile_ranges[static_cast<std::size_t>(t) * 2 + 1] =
                starts[static_cast<std::size_t>(t) + 1];
        }
    }

    // Pass 1c: per-(chunk, tile) start offset = global tile_start[t] +
    // sum of earlier chunks' contributions to that tile. Stable order
    // means chunk 0 writes first, then chunk 1, etc.
    std::vector<std::vector<int64_t>> chunk_starts(num_chunks,
        std::vector<int64_t>(static_cast<std::size_t>(num_tiles), 0));
    for (int t = 0; t < num_tiles; ++t) {
        int64_t cur = starts[static_cast<std::size_t>(t)];
        for (std::size_t c = 0; c < num_chunks; ++c) {
            chunk_starts[c][static_cast<std::size_t>(t)] = cur;
            cur += chunk_counts[c][static_cast<std::size_t>(t)];
        }
    }

    // Pass 2: parallel scatter. Each chunk writes into its disjoint
    // per-(chunk, tile) slots in input order.
    std::vector<uint32_t> packed_keys(P);
    std::vector<int64_t> packed_ids(P);

    auto scatter_chunk = [&](std::size_t c) {
        const std::size_t lo = c * chunk_count_sz;
        const std::size_t hi = std::min(lo + chunk_count_sz, P);
        int64_t* local_starts = chunk_starts[c].data();
        for (std::size_t i = lo; i < hi; ++i) {
            const std::size_t t = static_cast<std::size_t>(tile_ids[i]);
            const int64_t g = gaussian_ids[i];
            const uint32_t db = std::bit_cast<uint32_t>(depths[static_cast<std::size_t>(g)]);
            const std::size_t pos = static_cast<std::size_t>(local_starts[t]++);
            packed_keys[pos] = db;
            packed_ids[pos] = g;
        }
    };
    if (pool != nullptr && W_count > 1 && num_chunks > 1) {
        for (std::size_t c = 0; c < num_chunks; ++c) {
            pool->submit([c, &scatter_chunk]() { scatter_chunk(c); });
        }
        pool->wait();
    } else {
        for (std::size_t c = 0; c < num_chunks; ++c) scatter_chunk(c);
    }

    // Pass 3: per-tile STABLE LSD radix sort by depth_bits.
    // - LSD 8-bit radix, 4 passes for uint32 depth_bits: ~4n work per tile
    //   vs std::stable_sort's ~n log n on indirect index permutation. On the
    //   stitch hero scene's largest tiles (~11k entries) this is roughly
    //   3-4x faster than the iter-023 stable_sort indirect-permutation path.
    // - Each worker reuses its SortScratch (tmp_keys + tmp_ids) across all
    //   tiles it processes; lazy resize means at most one heap grow per
    //   worker per frame.
    // - Dispatch via LPT atomic counter (heaviest tile first), so workers
    //   never starve waiting on a long-tail giant tile.
    const std::size_t W = (pool != nullptr) ? std::max<std::size_t>(1, pool->size()) : 1;
    std::vector<SortScratch> scratches(W);

    // LPT order: sort tile indices by descending entry count so heavy tiles
    // get dispatched first.
    std::vector<int> tile_order(static_cast<std::size_t>(num_tiles));
    for (int i = 0; i < num_tiles; ++i) tile_order[static_cast<std::size_t>(i)] = i;
    std::sort(tile_order.begin(), tile_order.end(), [&](int a, int b) {
        return counts[static_cast<std::size_t>(a)] > counts[static_cast<std::size_t>(b)];
    });

    auto sort_one_tile = [&](int t, SortScratch& sc) {
        const std::size_t lo = static_cast<std::size_t>(starts[static_cast<std::size_t>(t)]);
        const std::size_t hi = static_cast<std::size_t>(starts[static_cast<std::size_t>(t) + 1]);
        const std::size_t n = hi - lo;
        if (n <= 1) return;
        radix_sort_tile(packed_keys.data() + lo, packed_ids.data() + lo, n, sc);
    };

    if (pool != nullptr && num_tiles > 1) {
        // Collapse 256 individual pool.submit() calls down to W tasks pulling
        // from a shared atomic counter (LPT scheduling). Two wins over per-tile
        // submit: mutex contention on the task queue drops from ~256 to W,
        // and workers race to grab the heaviest tiles first.
        std::atomic<int> next_idx{0};
        for (std::size_t w = 0; w < W; ++w) {
            pool->submit([&, w]() {
                SortScratch& sc = scratches[w];
                for (;;) {
                    const int idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= num_tiles) return;
                    const int t = tile_order[static_cast<std::size_t>(idx)];
                    sort_one_tile(t, sc);
                }
            });
        }
        pool->wait();
    } else {
        SortScratch& sc = scratches[0];
        for (int t = 0; t < num_tiles; ++t) sort_one_tile(t, sc);
    }

    // Pass 4: move packed_ids into result.sorted_gaussian_ids (no copy).
    result.sorted_gaussian_ids = std::move(packed_ids);

    return result;
}

}  // namespace gsplat_cpu
