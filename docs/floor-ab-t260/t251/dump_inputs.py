"""Task #251: dump projected hero-view inputs (cpu_cpp project, fp32) for model.py.

The tests/fixtures/hero/*.npz inputs are a different view from reference_v2 (their
own blend_output.npy is 12.3 dB vs reference_v2), so model.py needs these instead.
Run: PYTHONDONTWRITEBYTECODE=1 python3 dump_inputs.py <gstt2 root (built cpu_cpp)> <out.npz>
"""
import json, math, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
from backends import get_backend
from gsplat.loading_gaussians import load_ply
from gsplat.utils import c2w_to_w2c

cam = json.load(open('benchmarks/cameras_v2.json'))['bicycle']
W = H = 1024
f = 0.5 * W / math.tan(0.5 * math.radians(cam['fov_deg']))
K = torch.tensor([[f, 0, W * .5], [0, f, H * .5], [0, 0, 1]], dtype=torch.float32)
E = c2w_to_w2c(torch.tensor(np.asarray(cam['views']['hero']['c2w'], np.float32)))
g = load_ply('/Users/smarton/dev/gsplat_tt/scenes/bicycle.ply')
be = get_backend('cpu_cpp')
m2, cov, dep, rad, valid = be.project(g.means, g.scales, g.rotations, E, K, H, W,
                                      opacities=g.opacities, sub_timings={})
v = valid.bool()
np.savez(sys.argv[2], means_2d=np.asarray(m2[v] if len(m2) == len(v) else m2, np.float32),
         covs_2d=np.asarray(cov[v] if len(cov) == len(v) else cov, np.float32),
         depths=np.asarray(dep[v] if len(dep) == len(v) else dep, np.float32),
         radii=np.asarray(rad[v] if len(rad) == len(v) else rad, np.float32),
         opacities=g.opacities[v].numpy().astype(np.float32),
         colors=g.colors[v].numpy().astype(np.float32))
print('visible', int(v.sum()), 'm2', tuple(m2.shape), 'rad', tuple(rad.shape))
