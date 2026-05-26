#!/usr/bin/env python3
"""Parse a tt-metal device profiler CSV and classify the bound class.

The profiler emits raw ZONE_START / ZONE_END events into
`generated/profiler/.logs/profile_log_device.csv` when the binary was built
with `-DENABLE_TRACY=ON` and run with `TT_METAL_DEVICE_PROFILER=1`. This
script aggregates per-RISC durations across cores+frames, prints
medians/p99, and emits a JSON summary suitable for decide_and_log.

Bound-class classification follows tt-buddy interpretation.md:
  reader-bound       BRISC ≳ TRISC1
  compute-bound      TRISC1 (math) dominates
  writer-bound       NCRISC dominates
  under-parallelized low CORE COUNT + high per-core duration
  NOC-stall          BRISC + NCRISC high, TRISC low
  host-dominated     (FW - KERNEL) / kernel_ms > 40% (host kernel timer)

  overhead_ratio = (host_kernel_ms - device_fw_max_ms) / host_kernel_ms
                   > 40% => dispatch/sync/CB-flush bound
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CHIP_FREQ_HZ_BLACKHOLE = 1337e6


def parse_profile(path: Path) -> dict[tuple[str, str], list[int]]:
    """Return {(risc, zone_name): [duration_cycles, ...]}."""
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        raise SystemExit(f"{path}: too short, no rows after header")

    opens: dict[tuple, int] = {}
    zones: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ln in lines[2:]:
        parts = [c.strip() for c in ln.split(",", 14)]
        if len(parts) < 12:
            continue
        try:
            cx = int(parts[1]); cy = int(parts[2])
            risc = parts[3]
            t_cycles = int(parts[5])
            run_id = int(parts[7])
            zone = parts[10]
            ztype = parts[11]
        except (ValueError, IndexError):
            continue
        key = (cx, cy, risc, zone, run_id)
        if ztype == "ZONE_START":
            opens[key] = t_cycles
        elif ztype == "ZONE_END":
            st = opens.pop(key, None)
            if st is not None:
                zones[(risc, zone)].append(t_cycles - st)
    return zones


def summarize(zones, freq_hz=CHIP_FREQ_HZ_BLACKHOLE):
    out = {}
    for (risc, zone), durs_cycles in sorted(zones.items()):
        ns = sorted(d / freq_hz * 1e9 for d in durs_cycles)
        n = len(ns)
        out[f"{risc}:{zone}"] = {
            "n": n,
            "median_ns": ns[n // 2] if n else 0.0,
            "p99_ns": ns[min(n - 1, int(0.99 * n))] if n else 0.0,
            "max_ns": ns[-1] if n else 0.0,
        }
    return out


def classify(summary, host_kernel_ms_median: float | None):
    kernel_keys = [k for k in summary if k.endswith(":BRISC-KERNEL")
                   or k.endswith(":NCRISC-KERNEL")
                   or k.endswith(":TRISC-KERNEL")]
    by_risc = {}
    for k in kernel_keys:
        risc = k.split(":")[0]
        by_risc[risc] = summary[k]["median_ns"] / 1e6  # ms

    fw_max_ms = max(
        (summary[k]["median_ns"] / 1e6 for k in summary if "-FW" in k),
        default=0.0,
    )
    tags = {"fw_max_ms": fw_max_ms, "per_risc_ms": by_risc}

    if host_kernel_ms_median:
        tags["host_kernel_ms"] = host_kernel_ms_median
        tags["overhead_ratio"] = (host_kernel_ms_median - fw_max_ms) / host_kernel_ms_median

    if not by_risc:
        tags["bound_class"] = "unknown"
        return tags

    brisc = by_risc.get("BRISC", 0)
    ncrisc = by_risc.get("NCRISC", 0)
    trisc1 = by_risc.get("TRISC_1", 0)
    others = [by_risc.get(r, 0) for r in ("BRISC", "NCRISC", "TRISC_0", "TRISC_2")]
    spread_pct = (max(by_risc.values()) - min(by_risc.values())) / max(by_risc.values()) if by_risc else 0

    if tags.get("overhead_ratio", 0) > 0.40:
        tags["bound_class"] = "host-dominated (dispatch/transfer)"
    elif spread_pct < 0.05:
        tags["bound_class"] = "fully-synchronized (CB or barrier-bound)"
    elif trisc1 > max(brisc, ncrisc) * 1.20:
        tags["bound_class"] = "compute-bound (TRISC1 math)"
    elif brisc > trisc1 * 1.20:
        tags["bound_class"] = "reader-bound (BRISC)"
    elif ncrisc > trisc1 * 1.20:
        tags["bound_class"] = "writer-bound (NCRISC)"
    else:
        tags["bound_class"] = "mixed"

    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path,
                    help="Path to profile_log_device.csv")
    ap.add_argument("--host-kernel-ms", type=float, default=None,
                    help="Host-side kernel_ms median (from metrics.json)")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Write classification JSON here (default stdout only)")
    args = ap.parse_args()

    zones = parse_profile(args.csv)
    summary = summarize(zones)
    tags = classify(summary, args.host_kernel_ms)

    out = {"per_zone": summary, "classification": tags}
    js = json.dumps(out, indent=2)
    if args.out_json:
        args.out_json.write_text(js)
    print(js)


if __name__ == "__main__":
    main()
