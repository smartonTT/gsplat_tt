#!/usr/bin/env python3
"""Host model of the #196 bulk blendrec read (sort_bin_onelaunch.cpp issue_brec
+ the fast loop's slot stepping). Simulates an interleaved buffer (page p in
bank p % nb at bank offset p // nb), the per-bank run reads into a ring half,
and checks that the stepped (jr, ji) slot of every g holds page g. Also covers
non-monotonic g (reset) and g outside the run (blocking-read fallback)."""
import random

HALF = 256


def run_batch(nb, gs):
    sf = HALF // nb
    cap = sf * nb
    ga, gb = gs[0], gs[-1]
    n = gb - ga + 1
    if not (gb >= ga and n <= cap):
        return None  # per-g fallback
    half = [None] * HALF
    q, rem = divmod(n, nb)
    for r in range(min(n, nb)):
        c = q + (1 if r < rem else 0)
        # one read: c pages from page ga + r, consecutive in that bank
        p = ga + r
        for i in range(c):
            half[r * sf + i] = p + i * nb
    run_bytes, wrap = sf, cap - 1  # in pages
    jl = jr = 0
    rp = 0
    g_c = None
    for g in gs:
        if g == g_c:
            continue
        g_c = g
        jg = g - ga
        if 0 <= jg < n:
            if jg < jl:
                jl = jr = 0
                rp = 0
            while jl != jg:
                rp += run_bytes
                jr += 1
                if jr == nb:
                    jr = 0
                    rp -= wrap
                jl += 1
            assert half[rp] == g, (nb, g, ga, rp, half[rp])
        # else: blocking read of page g (always correct)
    return n


random.seed(1)
for nb in (7, 8):
    ok = fb = 0
    for trial in range(20000):
        base = random.randrange(0, 1 << 20)
        m = random.randrange(1, 129)
        gs, g = [], base
        for _ in range(m):
            g += random.choice([0, 0, 1, 1, 1, 2, 3]) if random.random() > 0.02 else random.randrange(0, 300)
            gs.append(g)
        if random.random() < 0.05:  # non-monotonic inside
            i, j = random.randrange(m), random.randrange(m)
            gs[i], gs[j] = gs[j], gs[i]
        r = run_batch(nb, gs)
        if r is None:
            fb += 1
        else:
            ok += 1
    print(f"nb={nb}: {ok} bulk batches checked, {fb} fell back")
