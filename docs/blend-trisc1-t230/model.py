#!/usr/bin/env python3
"""t230: post-A2 cycle model of the blend records loop on TRISC1. No device.

Reuses the t205 discrete-event model (RISC pushes into a depth-d instruction FIFO,
the SFPU issues one op per cycle, a REPLAY push plays back L recorded ops, RISC-only
cycles move only the RISC clock). Step 1 fits the hypotheses and FIFO depths to
the two #229 measurements (F: +0.022 untraced / +0.063 traced; A2: -1.015 / -0.968
ms/view). Step 2 runs the candidate cuts on top of A2 at the surviving fits.
Run: python3 model.py > out/model.txt
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blend-loop-model-t205"))
import model as m5  # noqa: E402

I, P = m5.I, m5.P
MEAS_F, MEAS_A2 = (0.022, 0.063), (-1.015, -0.968)   # ms/view vs level 0 (untraced, traced)
DEPTHS = [2, 4, 8, 16, 32]

# Post-A2 bodies. A2 single: x/y loads, replay front 13, 27 pushed middle (5 of them
# sfpnop), INCRWC steps (2.66 mean over m = 0..31), replay tail 18 + SETRWC.
INCR = (0 * 1 + 1 * 7 + 2 * 7 + 3 * 7 + 4 * 7 + 5 * 3) / 32.0


def a2_single(mid=27, incr=INCR, tail=19, front=13, pre=2, mop=0):
    body = [I] * pre + [P(front)] + [I] * mid + [("i", incr)] + [P(tail)]
    if mop:                                   # one MOP push expands to `mop` middle ops
        body = [I] * pre + [P(front)] + [I] * (mid - mop) + [("p", float(mop))] + [("i", incr)] + [P(tail)]
    return body


def stream(single, pair, walk, scan, stage_push, stage_risc, pipe=0.0, seed=1, n=800):
    """Same record mix as t205. pipe = RISC-only cycles of the next record that a
    software-pipelined loop moves behind each body's front replay (capped by the
    work there is)."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        rec_risc = scan + stage_risc
        calls = ["s"] * (1 + (rnd.random() < m5.P_SINGLE - 1)) + ["p"] * (rnd.random() < m5.P_PAIR)
        rnd.shuffle(calls)
        hidden = 0.0
        if pipe:
            nfront = sum(1 if c == "s" else 2 for c in calls)
            hidden = min(rec_risc, pipe * nfront)
        left = rec_risc - hidden
        frac_scan = scan / rec_risc if rec_risc else 0.0
        out.append(("r", left * frac_scan))
        for _ in range(stage_push):
            out.append(("r", left * (1.0 - frac_scan) / stage_push))
            out.append(("s", 1.0))
        per_front = hidden / max(1, sum(1 if c == "s" else 2 for c in calls))
        for c in calls:
            out.append(("r", walk))
            body = single if c == "s" else single + single if pair is None else pair
            for k, v in body:
                out.append((k, v))
                if per_front and k == "p" and v == 13.0:   # RISC work after the front replay
                    out.append(("r", per_front))
    return out, n


def cyc(ent, n, d, pc):
    return m5.run(ent, d, pc, 1.0) / n


def t205_stream(design, walk):
    ent, n = m5.stream(design, 0, walk)
    return ent, n


def fit():
    print("== Step 1: fit to #229 (dms = model ms/view vs level 0; measured F +0.022/+0.063, A2 -1.015/-0.968)")
    print(f"{'hypothesis':44s} " + " ".join(f"d={d:<2d} F    A2  " for d in DEPTHS))
    fits = []
    for hname, (walk, pc, mad, other) in m5.HYP.items():
        row = []
        for d in DEPTHS:
            res = {}
            for name in ("D0 today", "F: stall-free hand schedule", "A2: replay 13 + RWC tail replay 19"):
                ent, n = m5.stream(name, 0, walk, mad, other)
                res[name[:2]] = m5.run(ent, d, pc, 1.0) / n
            dF = (res["F:"] - res["D0"]) * m5.MS_PER_CYC
            dA = (res["A2"] - res["D0"]) * m5.MS_PER_CYC
            row.append(f"{dF:+.2f} {dA:+.2f}")
            err = abs(dF - MEAS_F[1]) + abs(dA - MEAS_A2[1])
            fits.append((err, hname, d, res["A2"]))
        print(f"{hname:44s} " + "  ".join(row))
    fits.sort()
    print("\nbest fits (|dF - 0.063| + |dA2 + 0.968|):")
    for err, h, d, a2 in fits[:6]:
        print(f"  err {err:.2f}  {h}  d={d}  A2 cycles/live record {a2:.1f}")
    return [(h, d) for err, h, d, _ in fits if err <= 0.25]


def designs():
    s = a2_single()
    return [
        # name, single, pair (None = two singles), scan, stage_push, stage_risc, pipe
        ("A2 (tip)", s, None, m5.SCAN, 31, 77, 0.0),
        ("M1 middle nops filled (-3 ops)", a2_single(mid=24), None, m5.SCAN, 31, 77, 0.0),
        ("M2 MOP holds 7 middle ops (1 push)", a2_single(mop=7), None, m5.SCAN, 31, 77, 0.0),
        ("M3 shared-load pair pushed (F pair 109)", s, [I] * 106 + [P(3)], m5.SCAN, 31, 77, 0.0),
        ("S1 RAW_STAGE knob (-17 RISC/record)", s, None, m5.SCAN, 31, 60, 0.0),
        ("S2a decode-ahead, staging words only", s, None, m5.SCAN, 31, 20, 0.0),
        ("S2a' same, slow L1 loads (35 RISC)", s, None, m5.SCAN, 31, 35, 0.0),
        ("S2 decode-ahead + live-only list (scan 8)", s, None, 8.0, 31, 20, 0.0),
        ("S3 pipelined staging behind front replay", s, None, m5.SCAN, 31, 77, 13.0),
        ("S4 = S2 + S3", s, None, 8.0, 31, 20, 13.0),
        ("S5 bound: all scan/stage RISC work gone", s, None, 0.0, 31, 0, 0.0),
    ]


def main():
    keep = fit()
    hyp = m5.HYP
    print("\n== Step 2: candidates on top of A2, at every fit within 0.25 ms of both measurements")
    print("cycles per live record; dms = ms/view saved vs A2 (positive = faster); 1 cycle = "
          f"{m5.MS_PER_CYC:.4f} ms/view")
    for h, d in keep:
        walk, pc, mad, other = hyp[h]
        print(f"\n-- {h}, d={d} (walk {walk:.1f}/call, push {pc:.3f})")
        base = None
        for name, single, pair, scan, sp, sr, pipe in designs():
            ent, n = stream(single, pair, walk, scan, sp, sr, pipe)
            c = cyc(ent, n, d, pc)
            base = c if base is None else base
            sfpu = m5.MB_PER_LIVE * sum(v for k, v in single if k in ("i", "p")) + sp
            print(f"   {name:44s} {c:6.1f}  dms {(base - c) * m5.MS_PER_CYC:+.2f}   SFPU-bound {sfpu:6.1f}")


if __name__ == "__main__":
    main()
