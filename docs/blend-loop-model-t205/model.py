#!/usr/bin/env python3
"""t205: cycle model of the blend records loop on TRISC1. No device.

The RISC pushes Tensix instructions into its thread's instruction FIFO
(depth d). The SFPU takes one instruction per cycle from the FIFO head. A
REPLAY entry plays back L recorded instructions from one push. Walk, staging
math and the record scan are RISC-only cycles: they only move the RISC clock.
When pushing and issuing both run at 1 per cycle (today), the FIFO never
fills, so every RISC-only cycle is an SFPU bubble. A REPLAY gives the RISC a
lead, so its RISC-only work can overlap SFPU issue.

H4 (added after stalls.py): the SFPU itself loses one cycle on each of the 11
back-to-back MAD -> use pairs per microblock; F removes them by reordering, A2
is F plus two replays per body and a tail addressed relative to RWC D.

Inputs: t172/t189 view-0 counts per core. Output: out/model.txt.
Run: python3 model.py > out/model.txt
"""
import random

MB, CALLS, LIVE, RECS = 59857, 46946, 21323, 26325
MB_PER_LIVE = MB / LIVE                    # 2.807
PAIR_MB_FRAC = 0.431
P_PAIR = PAIR_MB_FRAC * MB_PER_LIVE / 2    # 0.605 pair calls per live record
P_SINGLE = MB_PER_LIVE - 2 * P_PAIR        # 1.597 single calls per live record
MS_PER_CYC = LIVE / 1.35e9 * 1e3           # ms/view per cycle per live record
MEASURED = 366.0                           # 231 dispatch + 108 staging + 27 scan

SCAN = 22.0 * RECS / LIVE                  # 27.2 RISC cycles per live record
STAGE_PUSH, STAGE_RISC = 31, 77            # 108 cycles, 31 Tensix ops (t189/t146)
MID, TAIL = 41, 17                         # single body: 2 x/y loads + 41 + 17 = 60

# Hypotheses for the 14.5 cycles/microblock t189 could not attribute.
# name: (walk cycles per call, push cost, SFPU stall cycles per microblock that
#        are MAD latency (stalls.py counts 11) and that are other hazards)
HYP = {
    "H1 stall is RISC (jumps, load-use)": (31.4, 1.0, 0.0, 0.0),
    "H2 stall is SFPU hazards": (13.0, 1.0, 11.0, 3.5),
    "H3 stall is push cost": (13.0, 1.0 + 14.5 * MB_PER_LIVE / 191.0, 0.0, 0.0),
    # t205: the ISA says SFPMAD/MUL/ADD take two cycles and the compiled bodies
    # have 11 back-to-back MAD -> use pairs per microblock (stalls.py); the
    # other 3.5 cycles/microblock are put on the walk (1.275 microblocks/call).
    "H4 stall is MAD latency (counted), rest RISC": (13.0 + 3.5 * MB / CALLS, 1.0, 11.0, 0.0),
}


def mad_removed(design):
    """MAD-latency stalls a design removes per (single, pair) body."""
    if design.startswith("A2"):
        return 11, 22            # raw TTI bodies, pair = two stall-free singles
    if design.startswith("F2"):
        return 11, 22            # pair = two stall-free singles in one call
    if design.startswith("F"):
        return 11, 18            # hand schedule; pair keeps ~4 (LREG pressure)
    if "B-" in design:
        return 4, 8              # LM delayed stores: store-after-MAD stalls go
    return 0, 0

I, M = ("i", 1.0), ("i", 0.0)       # one pushed op; record marker (no SFPU work)


def P(n):
    return ("p", float(n))           # REPLAY of n recorded ops (one push)


def bodies(design, rep):
    """(single body, pair body) entry lists for one design."""
    pair_d0 = [I] * 38 + [M] + [I] * 66 + [P(5)]   # 109 ops from 106 pushes
    cut = 0                                        # tail ops removed per microblock
    if "B-min" in design:
        cut = 4                                    # LM load + delayed store: R, G, B, T
    if "+col" in design:
        cut += 3                                   # cr/cg/cb kept in LREGs per record
    if "B-full" in design:
        cut = 11                                   # tail 17 -> 6 (needs srcC substitution)
    if design.startswith("F2"):
        return [I] * 60, [I] * 120                 # pair = two singles, one call
    if design.startswith("D0") or not design.startswith("A"):
        return [I] * (60 - cut), [I] * 38 + [M] + [I] * (66 - 2 * cut) + [P(5)]
    if design.startswith("A-late"):                # replay is the last push of the body
        single = [I] * (60 - rep - cut) + [P(rep)]
    elif design.startswith("A2"):                  # 13 replayed at D=0, 3 INCRWC, 19-op tail replay
        single = [I] * 2 + [P(13)] + [I] * 27 + [I] * 3 + [P(19 - cut)]
    else:                                          # A: x/y loads, replay, rest of middle, tail
        single = [I] * 2 + [P(rep)] + [I] * (MID - rep) + [I] * (TAIL - cut)
    return single, single * 2                      # pair = two singles, one call


def with_stalls(body, stall):
    """Spread `stall` SFPU cycles evenly over the body's issued ops."""
    ops = sum(v for k, v in body if k in ("i", "p"))
    f = 1.0 + stall / ops
    return [(k, v * f) if k in ("i", "p") else (k, v) for k, v in body]


def stream(design, rep, walk, mad=0.0, other=0.0, seed=1, n=800):
    rnd = random.Random(seed)
    single, pair = bodies(design, rep)
    rs, rp = mad_removed(design) if mad else (0, 0)
    single = with_stalls(single, mad - rs + other)
    pair = with_stalls(pair, 2 * mad - rp + 2 * other)
    extra = 3 if ("B-full" in design or "+col" in design) else 0   # color preload
    out = []
    for _ in range(n):
        out.append(("r", SCAN))
        for _ in range(STAGE_PUSH + extra):
            out.append(("r", STAGE_RISC / (STAGE_PUSH + extra)))
            out.append(("s", 1.0))                 # staging op: no hazard factor
        calls = ["s"] * (1 + (rnd.random() < P_SINGLE - 1)) + ["p"] * (rnd.random() < P_PAIR)
        rnd.shuffle(calls)
        for c in calls:
            out.append(("r", walk))
            out.extend(single if c == "s" else pair)
    return out, n


def run(entries, d, push_cost, mult, block_replay=False):
    tr, fin = 0.0, 0.0
    leave = []                                     # time each FIFO entry leaves
    for kind, v in entries:
        if kind == "r":
            tr += v
            continue
        k = len(leave)
        t_push = tr if k < d else max(tr, leave[k - d])
        work = v if kind == "s" else v * mult
        start = max(t_push + 1.0, fin)
        fin = start + work
        leave.append(fin if (block_replay and kind == "p") else start)
        tr = t_push + push_cost
    return max(tr, fin)


def sfpu_bound(design, rep, mad, other):
    single, pair = bodies(design, rep)
    rs, rp = mad_removed(design) if mad else (0, 0)
    w = lambda b: sum(v for k, v in b if k in ("i", "p"))
    extra = 3 if ("B-full" in design or "+col" in design) else 0
    return (P_SINGLE * (w(single) + mad - rs + other)
            + P_PAIR * (w(pair) + 2 * mad - rp + 2 * other) + STAGE_PUSH + extra)


DESIGNS = [
    ("D0 today", 0),
    ("B-min: LM load+delayed store", 0),
    ("B-min+col: and colors in LREGs", 0),
    ("B-full: LM mad (not md5-safe?)", 0),
    ("A: replay 24 (slots 8-31)", 24),
    ("A: replay 32 (no compiler replay)", 32),
    ("A+B-min+col: replay 32", 32),
    ("A2: replay 13 + RWC tail replay 19", 0),
    ("A2+B-min", 0),
    ("F: stall-free hand schedule", 0),
    ("F2: F, pair = two singles", 0),
    ("A-late: replay 32 at body end (ref)", 32),
]


def main():
    print("cycles per live record (view 0, per core); dms = ms/view saved vs D0 at the")
    print(f"same hypothesis and FIFO depth; 1 cycle = {MS_PER_CYC:.4f} ms/view; measured today {MEASURED:.0f}")
    for hname, (walk, pc, mad, other) in HYP.items():
        print(f"\n== {hname}: walk {walk:.1f}/call, push {pc:.3f}, "
              f"SFPU stall/microblock {mad:.1f} MAD + {other:.1f} other")
        depths = [2, 4, 8, 16, 32]
        print(f"{'design':38s} " + " ".join(f"d={d:<2d} cyc  dms" for d in depths)
              + "  | blocking replay d=8 | SFPU bound")
        base = {}
        for name, rep in DESIGNS:
            ent, n = stream(name, rep, walk, mad, other)
            row = []
            for d in depths:
                c = run(ent, d, pc, 1.0) / n
                if name == "D0 today":
                    base[d] = c
                row.append(f"{c:6.1f} {(base[d] - c) * MS_PER_CYC:+5.2f}")
            cb = run(ent, 8, pc, 1.0, block_replay=True) / n
            if name == "D0 today":
                base["b"] = cb
            print(f"{name:38s} " + " ".join(row)
                  + f"  | {cb:6.1f} {(base['b'] - cb) * MS_PER_CYC:+5.2f}"
                  + f" | {sfpu_bound(name, rep, mad, other):6.1f}")


if __name__ == "__main__":
    main()
