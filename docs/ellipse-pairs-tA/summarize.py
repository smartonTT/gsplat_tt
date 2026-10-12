"""Summarize out/views.jsonl (count.py output + #528 device counts) into the RESULT tables."""
import json
import statistics as st

SLOPE = 2.23   # ms per million pairs, k2 + sort + mat+blend (#528 regressions)
SLOPE_NOBLEND = 0.255 + 0.455  # k2 + sort only
rs = [json.loads(l) for l in open("docs/ellipse-pairs-tA/out/views.jsonl")]


def mmm(xs, f="{:.3f}"):
    return f"{f.format(st.median(xs))} | {f.format(min(xs))} | {f.format(max(xs))}"


print("| view | M model / dev | P pc model / dev | P exact | P exact_m | P row_dev | removed exact_m | share | bound ms |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rs:
    d = r["P_pc"] - r["P_exact_m"]
    print(f"| {r['view']} | {r['M']} / {r['M_dev']} | {r['P_pc']} / {r['P_dev']} | {r['P_exact']} | "
          f"{r['P_exact_m']} | {r['P_row_dev']} | {d} | {d / r['P_pc']:.4f} | {SLOPE * d / 1e6:.3f} |")
print()
cal = [r["P_pc"] / r["P_dev"] - 1 for r in rs]
print(f"model P_pc vs device P: median {st.median(cal):+.4f}, min {min(cal):+.4f}, max {max(cal):+.4f}")
calm = [r["M"] / r["M_dev"] - 1 for r in rs]
print(f"model M vs device M: median {st.median(calm):+.4f}, min {min(calm):+.4f}, max {max(calm):+.4f}")
print()
print("| arm | removed pairs (M) med / min / max | share of P med / min / max | bound 2.23 ms/M med / min / max | k2+sort only 0.71 ms/M med |")
print("|---|---|---|---|---|")
for k in ("exact", "exact_m", "row_dev"):
    d = [(r["P_pc"] - r["P_" + k]) / 1e6 for r in rs]
    s = [(r["P_pc"] - r["P_" + k]) / r["P_pc"] for r in rs]
    b = [SLOPE * x for x in d]
    print(f"| {k} | {mmm(d)} | {mmm(s, '{:.4f}')} | {mmm(b)} | {SLOPE_NOBLEND * st.median(d):.3f} |")
tot = sum(r["P_pc"] for r in rs)
for k in ("exact", "exact_m", "row_dev"):
    print(f"30-view total {k}: {(tot - sum(r['P_' + k] for r in rs)) / tot:.4f} of pc pairs; "
          f"mean bound {SLOPE * (tot - sum(r['P_' + k] for r in rs)) / 30 / 1e6:.3f} ms/view")
nk = [r["rm_notok"] / (r["P_pc"] - r["P_exact_m"]) for r in rs]
print(f"share of exact_m removals on not-ok gaussians: median {st.median(nk):.3f}")
pm = [r["pairs_multi"] / r["P_pc"] for r in rs]
rows = [r["rows_multi"] / r["pairs_multi"] for r in rs]
print(f"pairs on multi-tile gaussians: median {st.median(pm):.3f} of P; rows per such pair {st.median(rows):.3f}")
print(f"max tile pc {max(r['maxtile_pc'] for r in rs)} -> exact_m {max(r['maxtile_exact_m'] for r in rs)}")
