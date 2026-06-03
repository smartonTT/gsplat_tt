#!/usr/bin/env python3
"""Parse tt-npe per-window raw output (npe_raw.json, produced on the device host
by npe_chunk_sim.py) into a compact, canvas-ready summary (npe_summary.json).

tt-npe simulated the production render's NoC-event trace in 60 equal-cycle
windows (see npe_chunk_sim.py for why the trace is chunked). For each window it
reports DRAM bandwidth utilization, NoC link utilization, and congestion impact.
This script rolls those up into headline peak/average figures, a downsampled
time-series (<= ~60 points) for plotting, and a one-line takeaway classifying
the render as bandwidth-saturated, NoC-congested, or latency/serialization-bound.

Definitions (from tt-npe DeviceStats):
  DRAM BW UTIL   = dram_bw_util            (% of peak DRAM bandwidth used)
  NOC UTIL       = overall_avg_link_util   (mean % of NoC link bandwidth used)
  NOC peak link  = overall_max_link_util   (busiest single NoC link, %)
  congestion     = 100*(estimated - cong_free)/cong_free  (extra cycles from
                   NoC contention, % of the congestion-free runtime)

Usage: parse_npe.py [npe_raw.json] [npe_summary.json]
"""
import json
import os
import sys

RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "npe_raw.json")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(__file__), "npe_summary.json")

# Classification thresholds (heuristics for the one-line takeaway).
DRAM_SATURATED_PCT = 80.0     # >= this peak DRAM BW util => bandwidth-bound
NOC_SATURATED_PCT = 80.0      # >= this peak NoC link util => NoC-bound
CONGESTED_PCT = 10.0          # >= this peak congestion impact => congested


def avg(xs):
    return (sum(xs) / len(xs)) if xs else 0.0


def main():
    with open(RAW) as f:
        raw = json.load(f)

    wins = raw["windows"]
    dram = [float(w.get("dram_bw_util", 0.0)) for w in wins]
    noc_avg = [float(w.get("noc_avg_link_util", 0.0)) for w in wins]
    noc_max = [float(w.get("noc_max_link_util", 0.0)) for w in wins]
    cong = [float(w.get("cong_impact_pct", 0.0)) for w in wins]
    # "Active" = windows that actually moved data (had transfer events).
    active = [i for i, w in enumerate(wins) if w.get("n_events", 0) > 0
              and not w.get("idle")]
    dram_active = [dram[i] for i in active]
    noc_active = [noc_avg[i] for i in active]

    peak_dram = max(dram) if dram else 0.0
    peak_noc = max(noc_avg) if noc_avg else 0.0
    peak_noc_link = max(noc_max) if noc_max else 0.0
    peak_cong = max(cong) if cong else 0.0
    peak_dram_win = dram.index(peak_dram) if dram else -1

    # Compact time-series (already <= 60 windows; pass through, rounded).
    series = [{
        "i": w["index"],
        "t_ms": w.get("start_ms", 0.0),
        "dram": round(dram[k], 2),
        "noc_avg": round(noc_avg[k], 2),
        "noc_max": round(noc_max[k], 2),
        "cong": round(cong[k], 3),
        "nev": w.get("n_events", 0),
    } for k, w in enumerate(wins)]

    # Total NoC traffic for context.
    bbt = raw.get("bytes_by_type", {})
    read_bytes = sum(v for k, v in bbt.items() if k.startswith("READ"))
    write_bytes = sum(v for k, v in bbt.items() if k.startswith("WRITE"))
    total_bytes = read_bytes + write_bytes

    # One-line takeaway.
    if peak_dram >= DRAM_SATURATED_PCT:
        verdict = "dram-bandwidth-saturated"
        takeaway = (f"Blend/render is DRAM-bandwidth-bound: peak DRAM BW util "
                    f"{peak_dram:.0f}% (window {peak_dram_win}).")
    elif peak_noc_link >= NOC_SATURATED_PCT:
        verdict = "noc-bandwidth-saturated"
        takeaway = (f"NoC link-bandwidth-bound: peak single-link util "
                    f"{peak_noc_link:.0f}%.")
    elif peak_cong >= CONGESTED_PCT:
        verdict = "noc-congested"
        takeaway = (f"NoC-congestion-bound: peak congestion adds "
                    f"{peak_cong:.0f}% extra cycles.")
    else:
        idle_frac = 1.0 - (len(active) / len(wins)) if wins else 0.0
        verdict = "latency-serialized"
        takeaway = (
            f"Latency/serialization-bound, NOT bandwidth-saturated or "
            f"congested: peak DRAM BW util only {peak_dram:.0f}% and peak NoC "
            f"link {peak_noc_link:.0f}%, congestion ~{peak_cong:.1f}%, with "
            f"{idle_frac*100:.0f}% of the timeline idle between sparse bursts.")

    summary = {
        "tool": raw.get("tool", "tt-npe"),
        "device": raw.get("device"),
        "cong_model": raw.get("cong_model"),
        "freq_mhz": raw.get("freq_mhz"),
        "trace": raw.get("trace"),
        "span_ms": raw.get("span_ms"),
        "span_cycles": raw.get("span_cycles"),
        "n_events_total": raw.get("n_events_total"),
        "n_windows": raw.get("n_windows"),
        "n_active_windows": len(active),
        "headline": {
            "dram_bw_util_peak_pct": round(peak_dram, 2),
            "dram_bw_util_avg_pct": round(avg(dram), 2),
            "dram_bw_util_avg_active_pct": round(avg(dram_active), 2),
            "noc_util_peak_pct": round(peak_noc, 2),
            "noc_util_avg_pct": round(avg(noc_avg), 2),
            "noc_util_avg_active_pct": round(avg(noc_active), 2),
            "noc_link_peak_pct": round(peak_noc_link, 2),
            "congestion_peak_pct": round(peak_cong, 3),
            "congestion_avg_pct": round(avg(cong), 3),
            "peak_dram_window": peak_dram_win,
            "verdict": verdict,
        },
        "traffic": {
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "total_bytes": total_bytes,
            "read_mb": round(read_bytes / 1e6, 2),
            "write_mb": round(write_bytes / 1e6, 2),
            "total_mb": round(total_bytes / 1e6, 2),
        },
        "takeaway": takeaway,
        "series_fields": ["i", "t_ms", "dram", "noc_avg", "noc_max", "cong",
                          "nev"],
        "series": series,
    }

    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2)

    h = summary["headline"]
    print(f"device={summary['device']} span={summary['span_ms']}ms "
          f"events={summary['n_events_total']} "
          f"windows={summary['n_windows']} active={summary['n_active_windows']}")
    print(f"DRAM BW UTIL  peak={h['dram_bw_util_peak_pct']}%  "
          f"avg={h['dram_bw_util_avg_pct']}%  "
          f"avg_active={h['dram_bw_util_avg_active_pct']}%")
    print(f"NOC UTIL      peak={h['noc_util_peak_pct']}%  "
          f"avg={h['noc_util_avg_pct']}%  peak_link={h['noc_link_peak_pct']}%")
    print(f"congestion    peak={h['congestion_peak_pct']}%  "
          f"avg={h['congestion_avg_pct']}%   verdict={h['verdict']}")
    print(f"traffic       read={summary['traffic']['read_mb']}MB  "
          f"write={summary['traffic']['write_mb']}MB")
    print("takeaway:", summary["takeaway"])
    print("wrote", OUT, f"({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
