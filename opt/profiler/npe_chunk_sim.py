#!/usr/bin/env python3
"""Run tt-npe (Tenstorrent NoC Performance Estimator) over a tt-metal NoC-event
trace and emit a compact per-window raw JSON (npe_raw.json).

WHY CHUNKING: gsplat runs one fused *resident* device pipeline, so the whole
render is a single NoC-trace "op" (run_host_id 0, op_name "") that spans ~285M
device cycles. tt-npe's engine has a hard MAX_CYCLE_LIMIT of 50M cycles per
simulation, so one run over the whole trace aborts with
EXCEEDED_SIM_CYCLE_LIMIT. We slice the trace into N equal-cycle windows
(default 60), simulate each independently with tt-npe's congestion model, and
record per-window DeviceStats. The per-window series is exactly the coarse
NoC/DRAM-utilization + congestion timeline the dashboard wants.
createWorkloadFromJSON normalizes each window file's timestamps to its own min,
so each window's relative span (~span/N << 50M) stays under the limit.

WHY SUBPROCESS-PER-WINDOW: tt_npe_pybind's engine keeps global/static state and
segfaults if runNPE() is called more than once in a single process. Each window
is therefore simulated in its own short-lived worker subprocess (this same
script invoked with --worker), which loads only its small chunk file.

This runs ON THE DEVICE HOST (needs tt_npe_pybind on PYTHONPATH) but is pure
simulation/analysis -- it does NOT open the TT device. The Mac-side parse_npe.py
turns npe_raw.json into the canvas-ready npe_summary.json.

Usage:
  PYTHONPATH=<tt-npe>/install/lib:<tt-npe>/install/bin \
    python3 npe_chunk_sim.py TRACE.json OUT_raw.json [--device P100] \
      [--windows 60] [--freq-mhz 1350] [--cycles-per-timestep 32]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import orjson

# NoC-event "type" values that carry transferred bytes (vs. barriers / zone
# markers, which frame the trace but move no data).
DATA_TYPES = {"WRITE_", "WRITE_MULTICAST", "READ", "READ_WITH_STATE",
              "WRITE_WITH_STATE"}


def worker(chunk_path, device, cong_model, cyc_per_ts):
    """Simulate one chunk file with tt-npe; print a single JSON stats line.
    Runs in an isolated subprocess (one runNPE per process)."""
    import tt_npe_pybind as npe
    cfg = npe.Config()
    cfg.device_name = device
    cfg.congestion_model_name = cong_model
    cfg.workload_is_noc_trace = True
    cfg.cycles_per_timestep = cyc_per_ts
    cfg.emit_timeline_file = False
    cfg.infer_injection_rate_from_src = True
    wl = npe.createWorkloadFromJSON(chunk_path, device, True)
    if wl is None:
        print(json.dumps({"error": "workload_create_failed"}))
        return
    api = npe.InitAPI(cfg)
    if api is None:
        print(json.dumps({"error": "init_api_failed"}))
        return
    res = api.runNPE(wl)
    if not isinstance(res, npe.Stats):
        print(json.dumps({"error": str(res)}))
        return
    pds = res.per_device_stats           # dict; key 0 = device, -1 = aggregate
    ds = pds.get(0) if hasattr(pds, "get") else pds[0]
    est = ds.estimated_cycles
    cfree = ds.estimated_cong_free_cycles
    print(json.dumps({
        "dram_bw_util": ds.dram_bw_util,
        "noc_avg_link_util": ds.overall_avg_link_util,
        "noc_max_link_util": ds.overall_max_link_util,
        "noc0_avg_link_util": ds.overall_avg_noc0_link_util,
        "noc1_avg_link_util": ds.overall_avg_noc1_link_util,
        "estimated_cycles": est,
        "cong_free_cycles": cfree,
        # extra cycles induced by NoC contention, as % of congestion-free time
        "cong_impact_pct": (100.0 * (est - cfree) / cfree) if cfree else 0.0,
        "completed": bool(ds.completed),
    }))


def run_worker_subprocess(chunk_path, device, cong_model, cyc_per_ts):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", chunk_path,
           "--device", device, "--cong-model", cong_model,
           "--cycles-per-timestep", str(cyc_per_ts)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    out = (p.stdout or "").strip().splitlines()
    if p.returncode != 0:
        return {"error": f"worker_rc={p.returncode}",
                "stderr": (p.stderr or "")[-200:]}
    for line in reversed(out):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"error": "no_json_output", "stderr": (p.stderr or "")[-200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--device", default="P100")
    ap.add_argument("--windows", type=int, default=60)
    ap.add_argument("--freq-mhz", type=float, default=1350.0)
    ap.add_argument("--cong-model", default="fast")
    ap.add_argument("--cycles-per-timestep", type=int, default=32)
    ap.add_argument("--worker", action="store_true",
                    help="internal: simulate a single chunk file (arg=trace)")
    args = ap.parse_args()

    if args.worker:
        worker(args.trace, args.device, args.cong_model,
               args.cycles_per_timestep)
        return

    data = orjson.loads(open(args.trace, "rb").read())
    ts = [e["timestamp"] for e in data if "timestamp" in e]
    tmin, tmax = min(ts), max(ts)
    span = tmax - tmin
    n = args.windows
    ns_per_cycle = 1000.0 / args.freq_mhz if args.freq_mhz else 0.0

    bytes_by_type, count_by_type = {}, {}
    for e in data:
        t = e.get("type")
        if t is None:
            continue
        count_by_type[t] = count_by_type.get(t, 0) + 1
        if "num_bytes" in e:
            bytes_by_type[t] = bytes_by_type.get(t, 0) + e["num_bytes"]

    # Build each window as a tight equal-cycle slice [w_start, w_end), then
    # REPAIR zone balance so tt-npe's ingest accepts it. A slice can clip a core
    # mid-bracket (a ZONE_START in an earlier window, or ZONE_END in a later
    # one), which the ingest rejects ("Zones are not correctly structured") or
    # even crashes on. For each core we wrap the slice in synthetic
    # <PROC>-KERNEL ZONE_START/END events (prepended at w_start / appended at
    # w_end) so the slice is self-balanced and properly nested, WITHOUT
    # stretching the window's cycle span -- giving honest per-window
    # (instantaneous) NoC/DRAM utilization rather than smearing a long resident
    # kernel across the whole timeline.
    def repair(core, evs, w_start, w_end):
        proc, sx, sy = core
        zname = f"{proc}-KERNEL"
        depth, mind = 0, 0
        for e in evs:
            ph = e.get("zone_phase")
            if ph == "ZONE_START":
                depth += 1
            elif ph == "ZONE_END":
                depth -= 1
                mind = min(mind, depth)
        pre = -mind                  # synthetic STARTs to keep depth >= 0
        post = depth + pre           # synthetic ENDs to return to 0

        def z(phase, t):
            return {"proc": proc, "sx": sx, "sy": sy, "zone": zname,
                    "zone_phase": phase, "timestamp": t, "run_host_id": 0,
                    "src_device_id": 0}
        return ([z("ZONE_START", w_start)] * pre + evs
                + [z("ZONE_END", w_end)] * post)

    tmpdir = tempfile.mkdtemp(prefix="npe_chunks_")
    windows = []
    for i in range(n):
        w_start = tmin + (span * i) // n
        w_end = tmin + (span * (i + 1)) // n
        if i == n - 1:
            w_end = tmax + 1
        slice_ev = [e for e in data
                    if w_start <= e.get("timestamp", -1) < w_end]
        per_core = {}
        for e in slice_ev:
            per_core.setdefault((e.get("proc"), e.get("sx"), e.get("sy")),
                                []).append(e)
        ev = []
        for core, evs in per_core.items():
            ev.extend(repair(core, evs, w_start, w_end))
        ev.sort(key=lambda e: e.get("timestamp", 0))
        rec = {
            "index": i,
            "start_cycle": w_start - tmin,
            "end_cycle": w_end - tmin,
            "start_ms": round((w_start - tmin) * ns_per_cycle / 1e6, 4),
            "n_events": len(slice_ev),
        }
        if not any(e.get("type") in DATA_TYPES for e in ev):
            rec.update({"idle": True, "dram_bw_util": 0.0,
                        "noc_avg_link_util": 0.0, "noc_max_link_util": 0.0,
                        "cong_impact_pct": 0.0})
        else:
            cp = os.path.join(tmpdir, f"chunk_{i:03d}.json")
            with open(cp, "wb") as f:
                f.write(orjson.dumps(ev))
            stats = run_worker_subprocess(cp, args.device, args.cong_model,
                                          args.cycles_per_timestep)
            try:
                os.unlink(cp)
            except OSError:
                pass
            rec.update(stats)
        windows.append(rec)
        print(f"window {i:>2}/{n} events={len(ev):>7} "
              f"dram_bw={rec.get('dram_bw_util', 0.0):6.2f}% "
              f"noc_avg={rec.get('noc_avg_link_util', 0.0):6.2f}% "
              f"cong={rec.get('cong_impact_pct', 0.0):6.2f}%"
              + (f"  ERR={rec['error']}" if "error" in rec else ""),
              flush=True)

    try:
        os.rmdir(tmpdir)
    except OSError:
        pass

    out = {
        "tool": "tt-npe",
        "device": args.device,
        "cong_model": args.cong_model,
        "cycles_per_timestep": args.cycles_per_timestep,
        "freq_mhz": args.freq_mhz,
        "ns_per_cycle": round(ns_per_cycle, 6),
        "trace": os.path.abspath(args.trace),
        "n_events_total": len(data),
        "span_cycles": span,
        "span_ms": round(span * ns_per_cycle / 1e6, 4),
        "n_windows": n,
        "bytes_by_type": bytes_by_type,
        "count_by_type": count_by_type,
        "windows": windows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
