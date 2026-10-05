#!/usr/bin/env python3
"""t200: model of the sort_ol emit pack moved to the 3 idle TRISCs with
tile-owned cursors (no device). Inputs are t196 Tracy counters
(docs/emit-imbalance-t196/out/tracy-on-emit_parts.txt, views 0:10, traced)."""
F = 1.35e9                      # RISC clock, Hz
# t196 bulk-on, mean mover, ms/view
zone, pro, rdw, brec, pairs = 2.162, 0.010, 0.001, 0.037, 0.002
proc, wfl, wiss, drain, wbar = 2.027, 0.018, 0.108, 0.083, 0.001
pack = proc - wfl - wiss        # 1.901 ms
nrec = 11888                    # records per mover per view
c_mov = pack * 1e-3 * F / nrec  # 216 cycles per record on a mover
nrun = 1488                     # 256 B runs per mover per view
nb = 93                         # batches per mover per view

def trisc_emit(alpha, c_list=10, imb=0.04, hs_cyc=300, stall_ms=0.02):
    """Emit zone (ms/view, mean core) when the 3 TRISCs own tiles t%3 and
    pack both movers' records; movers build per-TRISC record lists and do
    all NoC work. alpha = TRISC/mover cycles for the same per-record work."""
    rec_core = 2 * nrec
    c_t = alpha * c_mov + c_list                  # cursor RMW + pack + list load
    trisc = rec_core / 3 * c_t / F * 1e3 * (1 + imb)
    trisc += 2 * nb * hs_cyc / F * 1e3 + stall_ms  # batch handshakes, flush waits
    # mover side (parallel): list build ~45 cyc/rec, brec/pair issue, run writes
    mover = nrec * 45 / F * 1e3 + brec + pairs + wiss + wfl + 2 * nb * hs_cyc / F * 1e3
    body = max(trisc, mover)
    return pro + rdw + body + drain + wbar, trisc, mover

print(f"t196 mean mover: zone {zone:.3f} ms, pack {pack:.3f} ms, {c_mov:.0f} cyc/record")
print(f"{'case':34s} {'alpha':>5s} {'imb':>5s} {'TRISC':>6s} {'mover':>6s} {'zone':>6s} {'saving':>7s}")
cases = [
    ("optimistic (cursors in TRISC local mem)", 0.92, 0.03, 0.01),
    ("nominal (same cost per record)", 1.00, 0.04, 0.02),
    ("TRISC 10% slower", 1.10, 0.05, 0.04),
    ("TRISC 20% slower", 1.20, 0.06, 0.06),
    ("TRISC 30% slower", 1.30, 0.08, 0.08),
]
for name, a, imb, st in cases:
    z, t, m = trisc_emit(a, imb=imb, stall_ms=st)
    print(f"{name:34s} {a:5.2f} {imb:5.2f} {t:6.3f} {m:6.3f} {z:6.3f} {zone - z:7.3f}")
# break-even alpha for a 0.3 ms saving (nominal overheads)
lo, hi = 0.8, 2.0
for _ in range(40):
    mid = (lo + hi) / 2
    if zone - trisc_emit(mid)[0] > 0.3: lo = mid
    else: hi = mid
print(f"alpha at which the saving falls to 0.3 ms/view: {lo:.2f}")
# t165 design (mover keeps scan + cursor RMW, adds a mailbox word): mover-bound
print("t165-style (mover keeps scan+cursor, TRISC only stores): mover stays ~"
      f"{pack:.2f} ms -> saving ~0 (measured +0.056 ms/view frame, +0.02 emit)")
