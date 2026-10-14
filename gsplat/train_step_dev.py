"""On-device training step chain (#748): prep and per-stage T0 oracle check.

render/train/train_step runs T1 (blend backward) -> T2 (param pass, t697 gather reduce) +
sparse Adam on one device. T1 writes its slots straight into T2's slot-page buffer (4 KB pages,
64 slots of 64 B), so nothing is re-packed on the host between the stages. Forward image and
dL/dimage still come from the T0 dump here (they are the next stages to move on device).

    python3 -m gsplat.train_step_dev prep <t0.npz> <dir>       -> <dir>/t1, <dir>/t2
    python3 -m gsplat.train_step_dev check <t0.npz> <dir> <out>  per-stage report (exit 1 on FAIL)

Stages checked:
  t1     device slots (read back from the shared buffer) vs blend_backward_ref (rel 1e-2),
         plus the count of slots left NaN (the bench poisons the buffer before T1)
  t2     device grads / params / m / v / pfwc vs t2_step fed the ORACLE slots (t2_dev.check)
  t0     device grads vs the dump's autograd grad_* and device params vs adam_bf16_param_*
         (the T0 oracle itself, all SH bands; gated for every sh_degree since #751)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from gsplat import blend_backward_dev as t1
from gsplat import t2_dev
from gsplat import t2_param_backward as t2

f32 = np.float32


def prep(dump: str, out: Path) -> dict:
    d, cfg = t1.load_dump(dump)
    p = t1.prep(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"],
                d["image"], d["dL_dimage"], cfg)
    t1.write(p, str(out / "t1"))
    # T2's slot numbering and reference slots are T1's (3-sigma tile rectangles), not the
    # oracle's frozen-live rectangles: the device T2 reads exactly what the device T1 wrote.
    bw = t1.blend_backward(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"],
                           d["valid"], d["image"], d["dL_dimage"], cfg)
    m2 = t2_dev.prep_dump(dump, out / "t2", bw)
    r = np.load(out / "t2" / "ref_in.npz")
    same = bool(np.array_equal(np.asarray(p["P"], np.int64), np.asarray(r["P"], np.int64))
                and np.array_equal(np.asarray(p["counts"]), np.asarray(r["counts"])))
    if not same or p["n_slots"] != m2["n_slots"]:
        raise SystemExit(f"T1 and T2 slot numbering differ (P/counts equal: {same}, "
                         f"n_slots {p['n_slots']} vs {m2['n_slots']})")
    (out / "t2" / "slots.f32").unlink(missing_ok=True)   # the device fills it from T1
    info = {"dump": str(dump), "n_tiles": len(p["tiles"]), "n_recs": len(p["recs"]),
            "n_slots": p["n_slots"], "n_groups": m2["n_groups"], "n": m2["n"]}
    (out / "chain.json").write_text(json.dumps(info, indent=1))
    return info


def check(dump: str, indir: Path, outdir: Path, slot_bar=1e-2, grad_bar=1e-2,
          adam_bar=1e-3) -> dict:
    d, cfg = t1.load_dump(dump)
    meta = json.loads((indir / "t2" / "meta.json").read_text())
    N, ng, ns = meta["n"], meta["n_groups"], meta["n_slots"]
    dev = np.fromfile(outdir / "slots.f32", f32).reshape(-1, 16)[:ns, :9]
    rep = {"dump": Path(dump).name}
    s = t1.compare_slots(dev, t1.ref_slots(d, cfg))
    s["nan_slots"] = int(np.isnan(dev).any(1).sum())
    s["pass"] = bool(s["rel_err"] <= slot_bar and s["nan_slots"] == 0)
    rep["t1"] = s
    rep["t2"] = t2_dev.check(indir / "t2", outdir, grad_bar, adam_bar)
    # Against the T0 oracle directly (autograd grads, bf16-moment Adam), every SH band (#751:
    # bands 1-15 and the view-direction term from the t2sh passes).
    o = t2_dev.outputs(indir / "t2", outdir)
    t0 = {}
    if "grads" in o:
        t0["grads"] = {k: t2.rel_error(o["grads"][k], d[f"grad_{k}"]) for k in t2.PARAM_NAMES}
    p = o["params"]
    t0["params"] = {k: t2.rel_error(p[k], d[f"adam_bf16_param_{k}"]) for k in t2.PARAM_NAMES}
    pin = {k: np.asarray(d[f"param_{k}"]) for k in t2.PARAM_NAMES}
    t0["delta"] = {k: t2.rel_error(p[k] - pin[k], d[f"adam_bf16_param_{k}"] - pin[k])
                   for k in t2.PARAM_NAMES}
    if p["sh"].shape[1] > 1:  # per band (0, 1-3, 4-8, 9-15) for the report
        bands = {"sh0": (0, 1), "sh1": (1, 4), "sh2": (4, 9), "sh3": (9, 16)}
        t0["sh_bands"] = {f"{w}_{b}": t2.rel_error(a[:, i:j], ref[:, i:j])
                          for b, (i, j) in bands.items() if j <= p["sh"].shape[1]
                          for w, a, ref in (("grad", o.get("grads", {}).get("sh"), d["grad_sh"]),
                                            ("param", p["sh"], d["adam_bf16_param_sh"]))
                          if a is not None}
    t0["worst_grad"] = max(t0.get("grads", {"-": 0.0}).values())
    t0["worst_param"] = max(max(t0["params"].values()), max(t0["delta"].values()))
    t0["pass"] = bool(t0["worst_grad"] <= grad_bar and t0["worst_param"] <= adam_bar)
    rep["t0"] = t0
    rep["pass"] = bool(s["pass"] and rep["t2"]["pass"] and t0["pass"])
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    a1 = sp.add_parser("prep")
    a1.add_argument("dump")
    a1.add_argument("out")
    a2 = sp.add_parser("check")
    a2.add_argument("dump")
    a2.add_argument("indir")
    a2.add_argument("outdir")
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        print(json.dumps(prep(a.dump, Path(a.out))))
        return 0
    rep = check(a.dump, Path(a.indir), Path(a.outdir))
    (Path(a.outdir) / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
