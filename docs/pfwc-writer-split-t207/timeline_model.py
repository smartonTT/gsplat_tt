#!/usr/bin/env python3
"""t207: per-core chunk timeline of the pfwc writer, single vs split (model, not measured).

Per chunk (t197 means, 54 chunks per core): classify 13.4 us, records 23.3 us
(gamma, CV 0.5); compute emits chunk k at (k + 1) * P. Single writer: chunks in
order. Split: chunk k on RISC k % 2, waits for its previous own chunk, the
PREFIX chain (chunk k - 1 classified) and ends no earlier than chunk k - 1 (OPEN).
NCRISC pays extra us per chunk for the reader polls. Heavy core: the slowest core
(sets the makespan), classify x1.1, records x1.2 / x1.4.
"""
import random
import statistics


def sim(P, n=54, cv=0.5, single=False, seed=0, cs=1.0, rsc=1.0, ncr_extra=0.0, open_wait=True):
    rng = random.Random(seed)
    c = [13.4 * cs * rng.gammavariate(100, 0.01) for _ in range(n)]
    k_ = 1 / (cv * cv)
    r = [23.3 * rsc * rng.gammavariate(k_, 1 / k_) for _ in range(n)]
    ready = [P * (k + 1) for k in range(n)]
    if single:
        t = 0.0
        for k in range(n):
            t = max(t, ready[k]) + c[k] + r[k]
        return t
    fin, end, cls, pfx = {}, [0.0] * n, [0.0] * n, [0.0] * n
    for k in range(n):
        extra = ncr_extra if k % 2 else 0.0
        start = max(ready[k], fin.get(k - 2, 0.0))
        cls[k] = start + c[k]
        pfx[k] = cls[k] if k == 0 else max(cls[k], pfx[k - 1])
        rec_start = max(cls[k], 0.0 if k == 0 else pfx[k - 1])
        end[k] = rec_start + r[k] + extra
        if open_wait and k:  # head merge waits for chunk k - 1 (OPEN)
            end[k] = max(end[k], end[k - 1])
        fin[k] = end[k]
    return max(end)


for label, cs, rsc in (("mean core", 1.0, 1.0), ("heavy core, records x1.2", 1.1, 1.2),
                       ("heavy core, records x1.4", 1.1, 1.4)):
    for P, what in ((46, "today's compute"), (20, "with SFPU cov2d/conic/radii fusion")):
        runs = [(sim(P, single=True, seed=s, cs=cs, rsc=rsc),
                 sim(P, seed=s, cs=cs, rsc=rsc, ncr_extra=2.0)) for s in range(400)]
        one = statistics.mean(x[0] for x in runs) / 1000
        two = statistics.mean(x[1] for x in runs) / 1000
        print(f"{label:26s} P={P:2d} us: single {one:.3f} ms, split {two:.3f} ms, saving {one - two:.3f} ms"
              f" (compute alone {54 * P / 1000:.3f}; {what})")

# Merging the shared page later (no OPEN wait at the chunk end) changes nothing:
for label, cs, rsc in (("mean core", 1.0, 1.0), ("heavy core, records x1.4", 1.1, 1.4)):
    a = statistics.mean(sim(20, seed=s, cs=cs, rsc=rsc, ncr_extra=2.0) for s in range(400)) / 1000
    b = statistics.mean(sim(20, seed=s, cs=cs, rsc=rsc, ncr_extra=2.0, open_wait=False)
                        for s in range(400)) / 1000
    print(f"{label:26s} P=20 us: split {a:.3f} ms, split without the OPEN wait {b:.3f} ms")
