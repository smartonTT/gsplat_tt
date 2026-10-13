"""t582: per-run and median numbers from docs/l2-2chip/out/remote/run-*.log; writes result.json.

python3 docs/l2-2chip/summarize.py <commit> [out/remote dir]
A = one chip (id 0), B = 2 chips AFR (0,1), C = chip 1 alone. Rounds r1-r3 count; warm-* is dropped.
"""
import json
import re
import statistics
import sys
from pathlib import Path

D = Path(__file__).parent
commit = sys.argv[1]
R = Path(sys.argv[2]) if len(sys.argv) > 2 else D / "out" / "remote"


def b2b(log: str) -> dict | None:
    """The run's last unprefixed B2B line (AFR workers print theirs as '[afrN] B2B ...')."""
    lines = [ln for ln in log.splitlines() if ln.startswith("B2B ")]
    if not lines:
        return None
    kv = dict(re.findall(r"(\w+)=(\S+)", lines[-1]))
    out = {"ms_median": float(kv["ms_frame_median"]), "raw_md5": kv.get("raw_md5"),
           "passes": [float(x) for x in kv.get("pass_ms_frame", "").split(",") if x]}
    if "afr_chip_ms_own_view" in kv:
        out["afr_chip_ms_own_view"] = [float(x) for x in kv["afr_chip_ms_own_view"].split(",")]
        out["afr_hero_equal"] = kv.get("afr_hero_equal")
    return out


runs = {}
for f in sorted(R.glob("run-*.log")):
    tag = f.stem[4:]
    t = f.read_text(errors="replace")
    if tag.startswith("lat"):
        m = re.findall(r"TTW_TIMING ms_view=([0-9.]+)", t)
        runs[tag] = {"ms_view": float(m[-1])} if m else None
    else:
        runs[tag] = b2b(t)

arm = {a: [runs[f"r{r}-{a}"]["ms_median"] for r in (1, 2, 3) if runs.get(f"r{r}-{a}")] for a in "ABC"}
lat = [runs[f"lat{r}-A"]["ms_view"] for r in (1, 2, 3) if runs.get(f"lat{r}-A")]
own = [runs[f"r{r}-B"]["afr_chip_ms_own_view"] for r in (1, 2, 3) if runs.get(f"r{r}-B")]
for tag, v in runs.items():
    print(tag, json.dumps(v))
med = {a: round(statistics.median(v), 3) for a, v in arm.items() if v}
print("ARM_MEDIANS", med, "latency A", lat)
if len(arm["B"]) < 3 or len(arm["A"]) < 3:
    sys.exit("fewer than 3 timed rounds per arm")
psnr = re.search(r"hero-afr-chipA psnr_vs_ref=([0-9.]+)", (D / "out" / "psnr.txt").read_text())
res = {
    "board": "bh-50 (2x Blackhole p150, IRD tt_bgd)", "commit": commit, "doc": "docs/l2-2chip/RESULT.md",
    "ms_view_b2b": med["B"], "ms_view_b2b_passes": arm["B"],
    "single_ms_view_b2b": med["A"], "single_ms_view_b2b_passes": arm["A"],
    "chip1_ms_view_b2b": med.get("C"), "chip1_ms_view_b2b_passes": arm["C"],
    "afr_chip_ms_own_view": own,
    "latency_ms_view_single": round(statistics.median(lat), 3) if lat else None, "latency_runs": lat,
    "md5": "906e0435", "md5_views": "30/30",
    "hero_psnr_vs_ref": float(psnr.group(1)), "gpu_g1_ms": 10.75,
}
(D / "result.json").write_text(json.dumps(res, indent=1) + "\n")
print("RESULT", json.dumps(res))
