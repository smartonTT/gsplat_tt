#!/usr/bin/env python3
"""t183 step 2: replay blend dynamic tile claiming.

Model of today's reader (reader_alpha_blend_mb_devcull.cpp): one shared claim counter; the
reader claims a tile, then for each subchunk reserves one of 2 bulk-ring slots, reads it and
pushes it to the TRISCs. A slot frees when the TRISCs finish the subchunk two back. Today the
claim comes right after the previous push ('early'), so a core holds up to 3 tiles (computing,
in ring, claimed and waiting for a slot). 'late' = reserve the slot first, then claim.

A tile = list of subchunk compute costs (us). Claim order = list of tile indices.
"""
import heapq

CLAIM_US = 3.5   # claim + meta reads between tiles (measured gap 3-4 us)
READ_US = 20.0   # subchunk read after the slot is reserved


def replay(tiles, order, n_cores, policy='early', start=None, read_us=READ_US):
    start = start or [0.0] * n_cores
    comp_free = list(start)          # TRISC free time per core
    fins = [[] for _ in range(n_cores)]  # per-core subchunk finish times
    reader = list(start)             # reader time per core (ready to claim)
    seq = [[] for _ in range(n_cores)]
    def slot_free(c, k):  # time the ring slot for this core's k-th subchunk frees
        return fins[c][k - 2] if k >= 2 else 0.0
    # next claim time per core
    def next_claim(c):
        if policy == 'early':
            return reader[c] + CLAIM_US
        return max(reader[c], slot_free(c, len(fins[c]))) + CLAIM_US
    h = [(next_claim(c), c) for c in range(n_cores)]
    heapq.heapify(h)
    for t in order:
        tc, c = heapq.heappop(h)
        seq[c].append(t)
        r = tc
        for cost in tiles[t]:
            k = len(fins[c])
            r = max(r, slot_free(c, k)) + read_us      # reserve, read, push
            s = max(comp_free[c], r)
            comp_free[c] = s + cost
            fins[c].append(comp_free[c])
        reader[c] = r
        heapq.heappush(h, (next_claim(c), c))
    return comp_free, seq


def stats(ends):
    m = sum(ends) / len(ends)
    return max(ends), m, max(ends) - m


def rank_interleave(lists):
    """Claim order of today's kernel: rank 0 of every core's LPT list (core order), rank 1..."""
    out = []; r = 0
    while True:
        row = [l[r] for l in lists if len(l) > r]
        if not row: return out
        out += row; r += 1


def build_lpt(cost, n_cores):
    """Host build_lpt: tiles by cost desc, each to the least-loaded core (ties: lowest idx)."""
    ids = sorted((i for i in range(len(cost)) if cost[i] > 0), key=lambda i: (-cost[i], -i))
    h = [(0, c) for c in range(n_cores)]; lists = [[] for _ in range(n_cores)]
    for i in ids:
        load, c = heapq.heappop(h); lists[c].append(i); heapq.heappush(h, (load + cost[i], c))
    return lists
