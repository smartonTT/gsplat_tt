"""#781: the hero forward gate, computed once on the host: prep_binned + emulate_fast (the numpy
model of the device forward) on the hero dump, compared with its forward_fast image; saves the
model image and final T for drive_loop.sh (fwd_model_image / fwd_model_final_T).
  PYTHONPATH=$PWD python3 docs/train-step/hero_model.py <hero.npz> <out.npz>     (~13 min on the Mac)"""

import json, sys, time
import numpy as np
from gsplat import blend_forward_dev as fwd
from gsplat.blend_backward_dev import cfg_from_dump
d = np.load(sys.argv[1]); cfg = cfg_from_dump(d); H, W = int(d["H"]), int(d["W"])
t = time.time()
p = fwd.prep_binned(d["means2d"], d["conic"], d["opacity"], d["colors"], d["order"], d["valid"], H, W)
t1 = time.time()
print(json.dumps({"prep_s": round(t1 - t, 2), "n_recs": int(p["tiles"][:, 2].sum()), "pages": len(p["recs"]) // 2,
                  "max_tile": int(p["tiles"][:, 2].max())}), flush=True)
img, fT, stop = fwd.to_image(fwd.emulate_fast(p, cfg), p["tiles"], H, W, cfg.background)
t2 = time.time()
rep = fwd.compare(d, img, fT, 1e-4)
rep.update(emulate_s=round(t2 - t1, 1), stopped_px=int(stop.sum()))
print(json.dumps(rep), flush=True)
np.savez(sys.argv[2], image=img.astype(np.float32), final_T=fT.astype(np.float32))
