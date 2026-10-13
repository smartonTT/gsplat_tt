#!/usr/bin/env python3
"""t579: per-RISC code+rodata (PT_LOAD bytes, 16 B aligned) of the per-view programs and of a
single persistent program holding all of them, from elfscan.py output.  sizes.py out/elfscan-<c>.jsonl"""
import json
import sys

PROGRAMS = {  # kernel name -> program (production per-view programs at best-iter-226, B4A fold on)
    "writer_pfwc_split": "pfwc", "project_pfwc_compute": "pfwc",
    "sort_bin_onelaunch": "sort_ol", "sort_ol_town_compute": "sort_ol",
    "matblend_brisc": "matblend+K2", "matblend_ncrisc": "matblend+K2", "matblend_compute": "matblend+K2",
}
RISCS = ["brisc", "ncrisc", "trisc0", "trisc1", "trisc2"]
LIMIT = 105472  # kcfg ring at today's ceiling (69 KB + 34 KB), the size the SIB program overflowed

rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["file"].endswith(".xip.elf") or r["kernel"] not in PROGRAMS:
        continue
    rows[(PROGRAMS[r["kernel"]], r["risc"])] = (r["load"] + 15) // 16 * 16
progs = ["pfwc", "sort_ol", "matblend+K2"]
print("| RISC | " + " | ".join(progs) + " | persistent (sum) |")
print("|---|" + "---:|" * (len(progs) + 1))
tot = {p: 0 for p in progs}
for k in RISCS:
    v = [rows.get((p, k), 0) for p in progs]
    for p, x in zip(progs, v):
        tot[p] += x
    print(f"| {k} | " + " | ".join(f"{x:,}" for x in v) + f" | {sum(v):,} |")
s = sum(tot.values())
print("| **program code** | " + " | ".join(f"**{tot[p]:,}**" for p in progs) + f" | **{s:,}** |")
print(f"\npersistent code {s:,} B vs kcfg limit {LIMIT:,} B: over by {s - LIMIT:,} B ({s / LIMIT:.2f}x)")
