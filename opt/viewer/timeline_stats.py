"""Summarize a GSPLAT_VIEWER_TIMELINE file (task #587): the live viewer's per-frame
render/send times from FastRenderer.

Per arm: rendered and sent FPS over the moving frames, render interval, render_fn call
ms, the C++ total (dev), the host gap between one render_fn return and the next call
(the time only a queued pfwc keeps the device busy), frames replaced unsent (drop), and
the sender's paced (backlogged) share and encode ms.
  python3 opt/viewer/timeline_stats.py TIMELINE.jsonl [--skip-s 2]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys


def pct(xs, f):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(f * len(s)))]


def bursts(recs, min_frames=500):
    """Runs of back-to-back moving frames (one per sweep), in time order."""
    r = [x for x in recs if x.get("k") == "r" and x.get("a") in ("move", "rerender")]
    if not r:
        return []
    out, cur = [], [r[0]]
    for a, b in zip(r, r[1:]):
        if b["te"] - a["t1"] > 0.25:
            out.append(cur)
            cur = []
        cur.append(b)
    out.append(cur)
    return [b for b in out if len(b) >= min_frames]


def summarize(burst, sends, skip_s=2.0):
    """One burst minus its first skip_s."""
    r = burst
    s = sends
    t_start = r[0]["t0"] + skip_s
    r = [x for x in r if x["t0"] >= t_start]
    if len(r) < 3:
        return {"frames": len(r)}
    t_end = r[-1]["t1"]
    span = t_end - r[0]["t0"]
    keys = {x["t0"] for x in r}
    s = [x for x in s if x["t0"] in keys]
    ms = lambda xs: [1000.0 * v for v in xs]
    interval = ms([b["t0"] - a["t0"] for a, b in zip(r, r[1:])])
    call = ms([x["t1"] - x["t0"] for x in r])
    gap = ms([b["te"] - a["t1"] for a, b in zip(r, r[1:])])
    pre = ms([x["t0"] - x["te"] for x in r])
    dev = [x["dev"] for x in r if x.get("dev") is not None]
    enc = ms([x["ts"] - x["te"] for x in s])
    wait = ms([x["te"] - x["tt"] for x in s])
    out = dict(
        frames=len(r), span_s=round(span, 2),
        rendered_fps=round((len(r) - 1) / span, 2),
        sent_fps=round(len(s) / span, 2),
        dropped=sum(1 for x in r if x.get("drop")),
        hinted=sum(1 for x in r if x.get("h")),
        interval_ms_mean=round(statistics.mean(interval), 3),
        interval_ms_p50=round(pct(interval, 0.5), 3), interval_ms_p90=round(pct(interval, 0.9), 3),
        interval_ms_p99=round(pct(interval, 0.99), 3),
        call_ms_mean=round(statistics.mean(call), 3), call_ms_p50=round(pct(call, 0.5), 3),
        dev_ms_mean=round(statistics.mean(dev), 3) if dev else None,
        gap_ms_mean=round(statistics.mean(gap), 3), gap_ms_p50=round(pct(gap, 0.5), 3),
        gap_ms_p90=round(pct(gap, 0.9), 3), gap_ms_p99=round(pct(gap, 0.99), 3),
        pre_ms_mean=round(statistics.mean(pre), 3),
        backlogged_share=round(sum(1 for x in s if x.get("bl")) / max(1, len(s)), 3),
        pace_wait_ms_mean=round(statistics.mean(wait), 3) if wait else None,
        encode_ms_mean=round(statistics.mean(enc), 3) if enc else None,
    )
    # Per-view: a cycled sweep paced near 2x the render rate renders an aliased subset of
    # its views, different per arm; the unweighted mean over views compares like with like.
    per = {}
    for x in r:
        if x.get("v") is not None:
            per.setdefault(x["v"], []).append(1000.0 * (x["t1"] - x["t0"]))
    if per:
        out["views"] = len(per)
        out["call_ms_view_mean"] = round(statistics.mean(statistics.mean(c) for c in per.values()), 3)
        out["per_view_call_ms"] = {k: [len(c), round(statistics.mean(c), 3)] for k, c in sorted(per.items())}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--skip-s", type=float, default=2.0)
    a = ap.parse_args()
    with open(a.path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    sends = [x for x in recs if x.get("k") == "s"]
    for i, b in enumerate(bursts(recs)):
        print(f"TL burst={i} " + json.dumps(summarize(b, sends, a.skip_s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
