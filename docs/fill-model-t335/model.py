#!/usr/bin/env python3
"""Task #335: no-device model of (1) finer fill jobs and (2) big-tile split on
the MATCULL_TRISC_FILL mat phase, from the t319 Tracy counters
(opt/profiler/t319-fz/percore.csv: per core, mean over 29 views, ms).

Frame = model mat gain x CONV (#306: 0.216 ms measured mat stage / 0.56 ms model
~ 0.4; its frame/model was 0.158 / 0.56 ~ 0.28, shown as the low case).
All numbers are per-core means over views; the per-view makespan (2.834 ms mean
of per-view max) is higher than the max of per-core means (2.675 ms) because
different cores are critical in different views. Gains are measured on the
per-core-mean makespan.
"""
import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "opt/profiler/t319-fz/percore.csv"
CLK_MHZ = 1350.0
CHUNK = 512          # records per fill job in variant (1)
JOB_OVH_CYC = 300    # extra mover cost per extra job (post + done poll), cycles
CONV = 0.40          # #306 model -> measured conversion (spec)
CONV_LO = 0.28       # #306 model -> frame

rows = [{k: (v if k == "core" else float(v)) for k, v in r.items()} for r in csv.DictReader(open(CSV))]
N = len(rows)
base_ms = max(r["mat_end"] for r in rows)


def chunk_ms(r, n=CHUNK):
    """Fill latency of one n-record job on this core (TRISC0 fill + one patch batch)."""
    return (n * r["t0_fill_cyc_rec"] + 128 * r["t2_patch_cyc_rec"]) / CLK_MHZ / 1000.0


def pure_big(r):
    # NCRISC carries only the big-tile item (one job of <= 8192 records per view).
    return r["nc_big"] > 0 and r["nc_jobs"] < 1.05


def mover_extra_jobs_ms(recs, jobs):
    extra = max(recs / CHUNK - jobs, 0.0)
    return extra * JOB_OVH_CYC / CLK_MHZ / 1000.0


def fine_fill(r, resid_chunks, big_streamable):
    """Variant (1): 512-record jobs, per-mover queues, streamed emit.
    A mover's done-wait shrinks to resid_chunks x one chunk fill (its last chunk
    plus queueing behind the other mover's chunk). The big-tile item's slab is
    complete only after its last gather pass (chunk-outer loop), so on pure-big
    cores its final wait stays unless the gather is made rank-ordered."""
    c = chunk_ms(r)
    nc_res = min(r["nc_dw"], resid_chunks * c)
    if pure_big(r) and not big_streamable:
        nc_res = r["nc_dw"]
    br_res = min(r["br_dw"], resid_chunks * c)
    nc = r["nc_busy"] - r["nc_dw"] + nc_res + mover_extra_jobs_ms(r["nc_recs"], r["nc_jobs"])
    br = r["br_busy"] - r["br_dw"] + br_res + mover_extra_jobs_ms(r["br_recs"], r["br_jobs"])
    return nc, br


def makespan(pairs):
    return max(max(nc, br) for nc, br in pairs)


def split_big(per_core, dup, lpt_imb, tail_fn):
    """Variant (2): the big-tile item is split over both movers of its core
    (dup = extra work fraction: duplicated key read, per-pass sync of a shared
    radix), and the host LPT moves BRISC's other items off that core.
    Fluid model: makespan = max(global mean slot work + LPT imbalance + tail,
    the floor of each pure-big core = (big*(1+dup) + its other work) / 2 + tail).
    per_core: list of (row, nc_work_no_wait, br_work_no_wait)."""
    tot = 0.0
    floors = []
    for r, nc, br in per_core:
        B = r["nc_big"]
        tot += nc + br + dup * B
        if B > 0:
            other_nc = nc - B
            # the core's floor: its two halves of the big item plus whatever
            # work cannot leave the core (on pure-big cores: NCRISC's leftover)
            floors.append(((B * (1 + dup)) / 2 + max(other_nc, 0.0) / 2 + tail_fn(r, True), r["core"]))
    mean = tot / (2 * len(per_core))
    tail = max(tail_fn(r, False) for r, _, _ in per_core)
    fl = max(floors)
    return max(mean + lpt_imb + tail, fl[0]), mean, fl


def main():
    out = []
    p = out.append
    p(f"t319 percore.csv: {N} cores, clock {CLK_MHZ:.0f} MHz, one 512-record fill = "
      f"{st.mean(chunk_ms(r) for r in rows) * 1000:.1f} us mean")
    p(f"baseline mat makespan (max of per-core means) {base_ms:.3f} ms; "
      f"mean core end {st.mean(r['mat_end'] for r in rows):.3f}")
    p(f"pure-big cores (NCRISC holds only the >8192-record item): "
      f"{', '.join(r['core'] for r in rows if pure_big(r))}")
    nc0 = [r["nc_busy"] - r["nc_dw"] for r in rows]
    br0 = [r["br_busy"] - r["br_dw"] for r in rows]
    p(f"mover work without done-wait: NCRISC mean {st.mean(nc0):.3f} max {max(nc0):.3f}; "
      f"BRISC mean {st.mean(br0):.3f} max {max(br0):.3f}; mean of both {st.mean(nc0 + br0):.3f}")
    brw = [r["br_busy"] - r["br_dw"] for r in rows]
    imb_emp = max(brw) - st.mean(brw)
    p(f"LPT imbalance seen on BRISC slots (no big items): max-mean {imb_emp:.3f} ms")
    p("")

    def line(name, ms):
        g = base_ms - ms
        p(f"{name:58s} mat {ms:.3f}  gain {g:+.3f}  frame x{CONV}: {g * CONV:+.3f}  x{CONV_LO}: {g * CONV_LO:+.3f}")
        return g

    res = {}
    p("(0) bound from t319: every done-wait removed, nothing else")
    res["bound_nowait"] = line("  zero done-wait", makespan(list(zip(nc0, br0))))
    p("")
    p("(1) finer fill jobs (512 recs), per-mover queues, streamed emit")
    for rc, bs, tag in [(1.0, False, "opt"), (2.0, False, "real"), (1.0, True, "opt+rank-ordered big gather"),
                        (2.0, True, "real+rank-ordered big gather")]:
        pairs = [fine_fill(r, rc, bs) for r in rows]
        ms = makespan(pairs)
        crit = max(rows, key=lambda r: max(fine_fill(r, rc, bs)))
        res[f"fill_{tag}"] = line(f"  resid {rc:.0f} chunk(s), {tag} [crit {crit['core']}]", ms)
    p("")
    p("(2) big-tile item split over both movers of its core + host LPT rebalance (fills as today)")
    cores_now = [(r, r["nc_busy"] - r["nc_dw"], r["br_busy"] - r["br_dw"]) for r in rows]

    def tail_now(r, big):
        # today's exposed waits: the big item's whole-slab fill stays exposed on
        # its core (both halves posted together: TRISC fills all 8192 records);
        # elsewhere the mean done-wait stays (it moves with the items).
        if big:
            return r["nc_dw"] if pure_big(r) else max(r["nc_dw"], r["br_dw"])
        return 0.0

    mean_dw = st.mean([max(r["nc_dw"], r["br_dw"]) for r in rows])
    for dup, imb, tag in [(0.0, imb_emp, "ideal split"), (0.15, 0.10, "real split")]:
        ms, mean, fl = split_big(cores_now, dup, imb, tail_now)
        ms = max(ms, mean + imb + mean_dw)
        res[f"split_{tag}"] = line(f"  {tag}: dup {dup:.2f}, LPT imb {imb:.3f}, waits as today "
                                   f"[floor {fl[1]} {fl[0]:.3f}]", ms)
    p("")
    p("(1)+(2) combined")
    for rc, dup, imb, bs, tag in [(1.0, 0.0, imb_emp, True, "ceiling: 1 chunk, ideal split, rank-ordered gather"),
                                  (2.0, 0.15, 0.10, True, "real: 2 chunks, dup 0.15, rank-ordered gather"),
                                  (2.0, 0.15, 0.10, False, "real: 2 chunks, dup 0.15, gather as today")]:
        def tail(r, big, rc=rc, bs=bs):
            c = chunk_ms(r)
            if big and not bs:
                # the two halves' slabs complete only at the end of the gather:
                # the TRISC fills 8192 records after it
                return r["nc_dw"] if pure_big(r) else rc * c
            return rc * c
        ms, mean, fl = split_big(cores_now, dup, imb, tail)
        res[f"comb_{tag}"] = line(f"  {tag} [floor {fl[1]} {fl[0]:.3f}]", ms)
    p("")
    p("(C) absolute ceiling: perfect balance, zero waits, zero split cost")
    res["ceiling"] = line("  mean mover work", st.mean(nc0 + br0))
    p("")
    best = max(v for k, v in res.items() if k.startswith("comb_real"))
    verdict = "BUILD" if best * CONV >= 0.30 else "STOP"
    p(f"gate: best realistic combined frame gain {best * CONV:.3f} ms (x{CONV}) vs 0.30 ms -> {verdict}")
    txt = "\n".join(out)
    print(txt)
    (Path(__file__).parent / "out.txt").write_text(txt + "\n")


if __name__ == "__main__":
    sys.exit(main())
