"""opt/viewer/timeline_stats.py (task #587): bursts and per-burst summary of a viewer timeline.

Pure Python; runs under pytest or directly:
  python3 tests/unit/test_viewer_timeline_stats.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "opt" / "viewer"))

import timeline_stats as ts  # noqa: E402


def _burst(t, n, period=0.008, call=0.0075):
    r, s = [], []
    for _ in range(n):
        te = t
        t0 = te + 0.0001
        t1 = t0 + call
        r.append(dict(k="r", te=te, t0=t0, t1=t1, a="move", h=True, drop=False, dev=7.0))
        s.append(dict(k="s", t0=t0, tt=t1, te=t1 + 0.0001, ts=t1 + 0.0032, bl=False))
        t += period
    return r, s, t


def test_bursts_split_on_idle_gap():
    r1, s1, t = _burst(0.0, 600)
    r2, s2, _ = _burst(t + 1.0, 700)
    assert [len(b) for b in ts.bursts(r1 + r2 + s1 + s2)] == [600, 700]
    assert ts.bursts(r1[:10]) == []


def test_summarize_rates_and_gap():
    r, s, _ = _burst(0.0, 1000)
    out = ts.summarize(r, s, skip_s=2.0)
    assert out["frames"] == 750
    assert abs(out["rendered_fps"] - 125.0) < 0.5
    assert abs(out["sent_fps"] - 125.0) < 0.5
    assert abs(out["call_ms_mean"] - 7.5) < 1e-6
    assert abs(out["gap_ms_mean"] - 0.4) < 1e-6
    assert abs(out["encode_ms_mean"] - 3.1) < 1e-6
    assert out["dev_ms_mean"] == 7.0 and out["dropped"] == 0


def test_per_view_mean_is_unweighted():
    # View "a" costs 6 ms and is rendered 3x as often as view "b" at 10 ms: the frame mean
    # leans to "a" (7 ms), the per-view mean does not (8 ms).
    r, s, _ = _burst(0.0, 1000)
    for i, x in enumerate(r):
        slow = i % 4 == 3
        x["v"] = "b" if slow else "a"
        x["t1"] = x["t0"] + (0.010 if slow else 0.006)
    out = ts.summarize(r, s, skip_s=0.0)
    assert out["views"] == 2
    assert abs(out["call_ms_mean"] - 7.0) < 1e-6
    assert abs(out["call_ms_view_mean"] - 8.0) < 1e-6
    assert out["per_view_call_ms"]["b"] == [250, 10.0]


if __name__ == "__main__":
    test_bursts_split_on_idle_gap()
    test_summarize_rates_and_gap()
    test_per_view_mean_is_unweighted()
    print("ok")
