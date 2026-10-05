"""t215: model two follow-ups to t202's tile-owned TRISC emit (no device).

Inputs: docs/emit-trisc-own-t200/out/tracy-town-{town_parts,emit_parts,zones}.txt
(656a1fa-era rev 626de7b, Tracy views 0:10, GSPLAT_TT_OL_EMIT_PROF=1, yyzo-bh-07 p100a).
All times ms/view, mean core unless noted. Clock 1350 MHz.
"""
F = 1350e3  # cycles per ms

# Measured (tracy-town-*).
REC_T = 7925          # records per TRISC per view (7927/7914/7934)
CYC_T = 178.8         # TRISC net cycles per owned record
TRISC_PROC = 1.113    # TRISC process() incl. fl waits
ZONE_MEAN, ZONE_MAX = 1.228, 1.328   # sort_ol_emit zone, mean mover / busiest
# Mover busy parts: prologue, list build + read issue, queue service, drain, barriers.
MOVER_BUSY = 0.023 + 0.468 + 0.291 + 0.097 + 0.001 + 0.001
REC_M = 11888         # records per mover stream per view

pack_t = REC_T * CYC_T / F
over = ZONE_MEAN - TRISC_PROC          # head + tail around the TRISC loop
lane_t0 = over + pack_t + (TRISC_PROC - pack_t)
print(f"TRISC net pack {pack_t:.3f} ms, overhead around it {over:.3f} ms, "
      f"mover busy {MOVER_BUSY:.3f} ms, mover idle {ZONE_MEAN - MOVER_BUSY:.3f} ms")
scale = ZONE_MAX / ZONE_MEAN

def trisc_lane(cyc, x):
    """TRISC critical path with x records per mover taken by the movers."""
    wait = TRISC_PROC - pack_t
    return over + wait + (3 * REC_T - 2 * x) / 3 * cyc / F

def mover_lane(cm, x):
    return MOVER_BUSY + x * cm / F

def best(cyc, cm):
    lo, hi = 0.0, float(REC_M)
    for _ in range(60):
        x = (lo + hi) / 2
        if mover_lane(cm, x) < trisc_lane(cyc, x):
            lo = x
        else:
            hi = x
    return x, max(mover_lane(cm, x), trisc_lane(cyc, x))

base = trisc_lane(CYC_T, 0)
print(f"model zone at x=0: {base:.3f} (measured {ZONE_MEAN})\n")
print("A: cursors in TRISC local memory")
print(" saved cyc/rec   saving mean   saving busiest")
for d in (12, 17, 25, 35):
    s = REC_T * d / F
    print(f"   {d:3d}            {s:.3f}         {s * scale:.3f}")
print("\nB: movers as 4th/5th owners (mover extra cycles per owned record cm)")
print(" TRISC cyc  cm    x/mover  zone   saving mean  saving busiest")
for cyc, tag in ((CYC_T, "B"), (CYC_T - 17, "A+B")):
    for cm in (130, 170, 200):
        x, z = best(cyc, cm)
        s = base - z
        print(f" {cyc:6.1f}  {cm:4d}  {x:7.0f}  {z:.3f}   {s:.3f}        {s * scale:.3f}   {tag}")
