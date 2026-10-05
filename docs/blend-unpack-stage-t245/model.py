#!/usr/bin/env python3
"""t245: unpacker-staged blend coefficients. Cycle model on top of iter 198 (S2). No device.

Step 1 re-fits the t230 stream model to the t231 measurements (ms/view vs the A2 tip):
RAW_STAGE -0.206, DECODE_AHEAD=1 (S2a) -0.357, DECODE_AHEAD=2 (S2) -0.707. Free
parameters per surviving t230 fit: the RISC staging cost per live record left after
RAW (sr_raw) and after decode-ahead (sr_da), and the scan cost left at knob 2 (scan2).
Step 2 runs the staging designs of this note on top of the fitted S2 point.
Step 3 checks the TRISC0 producer load of each design against the TRISC1 pace.
Run: python3 model.py > out/model.txt
"""
import importlib.util
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("m30", os.path.join(HERE, "..", "blend-trisc1-t230", "model.py"))
m30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m30)  # loads t205 as m30.m5

m5 = m30.m5
I, P = m5.I, m5.P
MEAS = {"RAW": 0.206, "S2a": 0.357, "S2": 0.707}     # ms/view saved vs A2 tip (t231)
GATE = 0.3                                          # ms/view, task gate


def stream(single, walk, scan, stage, stage_risc, seed=1, n=800):
    """t230 record mix. `stage` = list of staging entries per live record (("s", 1) is a
    pushed SFPU op, ("p", k) one push that plays k fixed ops: MOP or replay). The RISC
    staging cycles are spread evenly in front of the staging pushes, like t230."""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        calls = ["s"] * (1 + (rnd.random() < m5.P_SINGLE - 1)) + ["p"] * (rnd.random() < m5.P_PAIR)
        rnd.shuffle(calls)
        out.append(("r", scan))
        k = max(1, len(stage))
        for e in stage:
            out.append(("r", stage_risc / k))
            out.append(e)
        if not stage and stage_risc:
            out.append(("r", stage_risc))
        for c in calls:
            out.append(("r", walk))
            for e in (single if c == "s" else single + single):
                out.append(e)
    return out, n


def cyc(single, walk, scan, stage, sr, d, pc):
    ent, n = stream(single, walk, scan, stage, sr)
    return m5.run(ent, d, pc, 1.0) / n


def solve(f, lo, hi, target, it=40):
    """f decreasing in x (cycles); find x with f(x) = target, None if out of range."""
    flo, fhi = f(lo), f(hi)
    if not (min(flo, fhi) <= target <= max(flo, fhi)):
        return None
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if (f(mid) - target) * (flo - target) > 0:
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


S31 = [("s", 1.0)] * 31


def refit(h, d):
    walk, pc, _, _ = m5.HYP[h]
    s = m30.a2_single()
    base = cyc(s, walk, m5.SCAN, S31, 77, d, pc)
    ms = lambda c: (base - c) * m5.MS_PER_CYC  # noqa: E731
    sr_raw = solve(lambda x: ms(cyc(s, walk, m5.SCAN, S31, x, d, pc)), 0.0, 77.0, MEAS["RAW"])
    sr_da = solve(lambda x: ms(cyc(s, walk, m5.SCAN, S31, x, d, pc)), 0.0, 77.0, MEAS["S2a"])
    scan2 = None
    if sr_da is not None:
        scan2 = solve(lambda x: ms(cyc(s, walk, x, S31, sr_da, d, pc)), 0.0, m5.SCAN, MEAS["S2"])
    return dict(walk=walk, pc=pc, base=base, sr_raw=sr_raw, sr_da=sr_da, scan2=scan2)


def designs(sr_da):
    """name, staging entries, staging RISC cycles per live record, extra RISC (exposed wait)."""
    sem = [("s", 1.0)] * 2                     # SEMWAIT + SEMPOST pushed by TRISC1
    # U2 copy ops: 5 raw fp32 x (SFPLOAD + SFPSTORE), 4 UNORM x (SFPLOAD, SFPADDI -2^23,
    # SFPMUL 1/65535, SFPSTORE), 1 INV load = 27. RISC left: ring poll, mask lw, counter.
    u2 = [("s", 1.0)] * 27
    return [
        ("S2 tip (iter 198, fitted)", S31, sr_da, 0.0),
        ("U2  staging area in DEST, 27 copy ops pushed, RISC 12", sem + u2, 12.0, 0.0),
        ("U2  same, RISC 20 (slow mask/ring loads)", sem + u2, 20.0, 0.0),
        ("U2  same, RISC 30 (ring poll as costly as today)", sem + u2, 30.0, 0.0),
        ("U2m 27 copy ops in one MOP push, RISC 12", sem + [("p", 27.0)], 12.0, 0.0),
        ("U3  unpack straight into slot 6, exposed latency 10", sem + [("s", 1.0)], 6.0, 10.0),
        ("U3  same, exposed latency 40", sem + [("s", 1.0)], 6.0, 40.0),
        ("U1  ideal bound: 2 handshake ops, RISC 4", sem, 4.0, 0.0),
        ("X   keep ring loads, 31 staging ops in one push", [("p", 31.0)], sr_da, 0.0),
    ]


def trisc0_load():
    """Producer instructions per live record on TRISC0 (t231 put_rec ~55 today)."""
    today = 55
    rows = [
        ("S2 today (14 SFPLOADI words + mask into the ring)", today),
        # U2: 9 values as even/odd column pairs, 16-word rows = 8 sw per value; the
        # unpacker repeats each row into 4 DEST rows (zero source y-stride); a UNORM q is
        # sent as 0x4B000000|q (= 2^23 + q), 1 OR each; ~12 for UNPACR, cfg, semaphores.
        ("U2 9 values x 8 words + 4 ORs + unpack/sem", today - 14 + 72 + 4 + 12),
        ("U2 if rows cannot be repeated (4x the stores)", today - 14 + 4 * 72 + 4 + 12),
        ("U3 with UNORM fully decoded on TRISC0 (soft RNE mul)", today - 14 + 72 + 4 * 30 + 12),
        ("dead-record walk on TRISC0, per live record (t205 SCAN)", 27),
    ]
    return rows


def main():
    print("== Step 1: re-fit to t231 (ms/view saved vs A2: RAW 0.206, S2a 0.357, S2 0.707)")
    print("t230 assumed sr 77 -> 60 (RAW), 77 -> 20 (S2a), scan 27.2 -> 8 (S2)")
    print(f"{'fit':38s} {'A2 cyc':>7s} {'sr_raw':>7s} {'sr_da':>7s} {'scan2':>7s} {'S2 cyc':>7s}")
    fits = []
    for h, d in KEEP:
        r = refit(h, d)
        s2 = None
        if r["sr_da"] is not None and r["scan2"] is not None:
            s2 = cyc(m30.a2_single(), r["walk"], r["scan2"], S31, r["sr_da"], d, r["pc"])
            fits.append((h, d, r, s2))
        f = lambda v: "   n/a" if v is None else f"{v:7.1f}"  # noqa: E731
        print(f"{h[:30] + ' d=' + str(d):38s} {r['base']:7.1f} {f(r['sr_raw'])} {f(r['sr_da'])} "
              f"{f(r['scan2'])} {f(s2)}")

    print("\n== Step 2: designs on top of the fitted S2 point")
    print("cycles per live record; dms = ms/view saved vs S2 tip; gate 0.30")
    best = {}
    for h, d, r, s2 in fits:
        print(f"\n-- {h}, d={d}  (sr_da {r['sr_da']:.1f}, scan2 {r['scan2']:.1f})")
        single = m30.a2_single()
        base = None
        for name, stage, sr, wait in designs(r["sr_da"]):
            c = cyc(single, r["walk"], r["scan2"] + wait, stage, sr, d, r["pc"])
            base = c if base is None else base
            sfpu = m5.MB_PER_LIVE * sum(v for k, v in single if k in ("i", "p")) + sum(v for _, v in stage)
            dms = (base - c) * m5.MS_PER_CYC
            best.setdefault(name, []).append(dms)
            print(f"   {name:58s} {c:6.1f}  dms {dms:+.2f}   SFPU-bound {sfpu:6.1f}")

    print("\n== Summary over fits (min .. max dms vs S2 tip)")
    for name, v in best.items():
        print(f"   {name:58s} {min(v):+.2f} .. {max(v):+.2f}")

    print("\n== Step 3: TRISC0 producer load per live record (instructions, ~1 cycle each)")
    print("TRISC1 pace at S2 is ~255 cycles per live record (t231); TRISC0 also walks dead records")
    for name, n in trisc0_load():
        print(f"   {name:58s} {n:5d}")


KEEP = [("H3 stall is push cost", 16), ("H1 stall is RISC (jumps, load-use)", 32),
        ("H4 stall is MAD latency (counted), rest RISC", 32), ("H3 stall is push cost", 8)]

if __name__ == "__main__":
    main()
