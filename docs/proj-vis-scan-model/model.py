#!/usr/bin/env python3
"""No-device cycle model for task #179: single-core proj_vis_scan vs a multi-core
scan fused into the gather_vis_scatter program.

Inputs: docs/t159 (scan zone 0.396 ms/view, gaps), docs/hw-ceilings.md (p100a RISC
and NoC probe costs), render/kernels/dataflow/vis_tile.h (scan_slots loops).
All cycle costs are per RISC at 1.35 GHz. Run: python3 model.py
"""
import math

GHZ = 1.35
N_GAUSS = 6_131_954          # bicycle (docs: 6.13 M)
TILES = math.ceil(N_GAUSS / 1024)  # counts entries, 5989
C = 110                      # cores (11 x 10)
S = 2 * C                    # slots (core, mover)
TPC = math.ceil(TILES / C)   # tiles per core in the strided deal, 55

# hw-ceilings.md (p100a, measured)
LOAD_DEP = 8.26      # dependent L1 load-to-use
LOAD_IND = 4.07      # independent L1 load
WR_ISSUE = 77.7      # noc_async_write + accessor + barrier share
RD_RT = 476          # one 64 B DRAM read, exposed round trip
RD_PIPE = 105        # per 64 B read, 8 outstanding, 1 core

def us(cyc):
    return cyc / GHZ / 1000.0

# ---- 1. single-core scan_slots, today's code -----------------------------------
# Per-tile instruction estimates read off vis_tile.h (volatile loads, rv32 64-bit math):
pass_total = LOAD_DEP + 6            # weight(t): load vis, test, 2 adds, 64-bit add, loop
pass_cut = (LOAD_DEP + 8             # weight(c + k*C): mul, add, load, test, adds, 64-bit add
            + 16                     # while test: prefix*S and s*total as 64-bit muls + compare
            + 4)                     # nested loop bookkeeping
pass_bases = (2 * LOAD_IND + LOAD_DEP + 9)  # vis + pairs loads, m/pr adds, last, ++k vs count(c)
per_tile = pass_total + pass_cut + pass_bases
io = (47 * 1024 / 32) / 1.0 + RD_RT + (S + 3) * WR_ISSUE   # 47 KB in (~32 B/cyc), 223 page writes
serial_cyc = TILES * per_tile + io
print(f"tiles={TILES} cores={C} slots={S} tiles/core={TPC}")
print(f"serial scan model: {per_tile:.0f} cyc/tile, io {io/1e3:.1f} kcyc -> "
      f"{us(serial_cyc):.0f} us  (measured t159 396 us, t121/t146 436-445 us)")

# ---- 2. fused multi-core scan in the scatter program ---------------------------
# Per core (BRISC; NCRISC waits on a local semaphore as in sort_bin_onelaunch.cpp):
#  a) read the core's 55 count entries (64 B windows, 8 outstanding; chip cap 0.54 G reads/s)
#  b) core sums (vis, pairs, weight, last non-empty position)
#  c) gather 16 B records on core 0 + release (multicast; sort_ol unicast as pessimistic)
#  d) launch skew / slowest-core wait
#  e) each mover: total, cut walk over core prefixes, tile walk inside 1-2 cores
#     (on-demand 64 B reads of that core's entries), bases, is_last
def fused(case):
    if case == "opt":
        a = max(TPC * RD_PIPE, TILES / 0.54e9 * 1.35e9 * 1.0)   # chip read cap
        c = 1.0 * 1350 + 2.0 * 1350                            # arrive ~1 us + mcast ~2 us
        d = 3.0 * 1350
        e = 4.0 * 1350
    elif case == "mid":
        a = 15.0 * 1350                                         # 48 KB per core, 5.3 MB total
        c = 1.0 * 1350 + 4.0 * 1350
        d = 4.0 * 1350
        e = 6.0 * 1350
    else:  # pessimistic
        a = 20.0 * 1350
        c = 1.5 * 1350 + (C - 1) * 80                           # sort_ol-style 109 unicast incs
        d = 5.0 * 1350
        e = 8.0 * 1350 + 2 * TPC * RD_PIPE                      # plus on-demand reads
    b = TPC * (2 * LOAD_IND + 12)
    return a + b + c + d + e

# Program boundary removed (one fewer program in the chain). Traced gap before the
# scan is 35-48 us (t159/t121/t146 gaps.txt), an upper bound; untraced assumed 10-40 us.
cases = {"opt": (445, 40), "mid": (400, 25), "pess": (396, 10)}
for k, (scan_us, boundary_us) in cases.items():
    pro = us(fused(k))
    save = scan_us + boundary_us - pro
    print(f"{k:5s}: scan {scan_us} us + boundary {boundary_us} us - fused prologue {pro:.0f} us"
          f" = saving {save/1000:.2f} ms/view")

# ---- 3. alternative: keep one core, rewrite the loop (no sync) -----------------
# non-volatile loads, 32-bit thresholds (one divu per slot), pointer-stride walk,
# cut and bases fused into one pass; total pass stays.
fast_tile = (LOAD_IND + 3) + (LOAD_IND * 2 + 8)
fast_cyc = TILES * fast_tile + S * 20.4 + io
print(f"fast serial scan: {fast_tile:.0f} cyc/tile -> {us(fast_cyc):.0f} us, "
      f"saving {(396 - us(fast_cyc))/1000:.2f}-{(445 - us(fast_cyc))/1000:.2f} ms/view")
