"""#728: hero T0 check with threshold-tie slots skipped (ref_slots_fast ties=True), against the
fp32 oracle (the gate, = `blend_backward_dev check --fast`) and the float64 reference.
Usage: hero_ties.py hero.npz slots.f32 out.json ref64.npy"""
import json
import sys

import numpy as np

from gsplat.blend_backward_dev import compare_slots, load_dump, ref_slots_fast


def main(npz, slots_path, out, p64):
    d, cfg = load_dump(npz)
    dev = np.fromfile(slots_path, np.float32).reshape(-1, 16)[:, :9]
    r32, tie = ref_slots_fast(d, cfg, ties=True)
    rep = {"dev_vs_f32": compare_slots(dev, r32, tie), "dev_vs_f64": compare_slots(dev, np.load(p64), tie),
           "n_slots": int(tie.size), "tie_slots": int(tie.sum())}
    rep["pass"] = bool(rep["dev_vs_f32"]["rel_err"] <= 1e-2)
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
