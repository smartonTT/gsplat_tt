"""#728: device and fp32 oracle slots against a float64 evaluation of the same math on the same
fp32 inputs. Usage: hero_f64.py hero.npz slots.f32 out.json ref32.npy ref64.npy
(each ref .npy is reused if present, else computed and saved; ~25 min each on the hero)."""
import json
import os
import sys

import numpy as np

from gsplat.blend_backward_dev import compare_slots, load_dump, ref_slots_fast


def ref(d, cfg, path, dt):
    if os.path.exists(path):
        return np.load(path)
    r = ref_slots_fast(d, cfg, dt)
    np.save(path, r)
    return r


def main(npz, slots_path, out, p32, p64):
    d, cfg = load_dump(npz)
    dev = np.fromfile(slots_path, np.float32).reshape(-1, 16)[:, :9]
    r32, r64 = ref(d, cfg, p32, np.float32), ref(d, cfg, p64, np.float64)
    rep = {"dev_vs_f32": compare_slots(dev, r32), "dev_vs_f64": compare_slots(dev, r64),
           "f32_vs_f64": compare_slots(r32, r64)}
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    main(*sys.argv[1:])
