# T0 gradient-oracle dumps (task #674, docs/training-plan/PLAN.md T0)

Written by `python3 -m gsplat.train_oracle --out benchmarks/train_oracle` (torch on the
Mac, no device). `tests/unit/test_train_oracle.py::test_dumps_are_current` fails if
they drift from the code. Conventions are in the docstring of `gsplat/train_oracle.py`.

| file | case |
|---|---|
| `t0_sh0_64x64_n50.npz` | 64x64, 50 Gaussians (46 visible), SH degree 0 |
| `t0_sh3_64x64_n50.npz` | same scene layout, SH degree 3 (16 coefficients) |

Self-supervised target: `target` is the render of `scene_*`; `param_*` is a perturbed
copy, and the loss is 0.8 * L1 + 0.2 * (1 - SSIM(render(param), target)).
Forward and backward are fp64; the Adam arrays are fp32.

Keys (N Gaussians, H x W pixels, K = (deg+1)^2):

- camera: `w2c` (4,4) world-to-camera, `K` (3,3), `H`, `W`, `N`, `sh_degree`, `cfg_json`
- params (optimizer space): `scene_<p>` (unperturbed), `param_<p>` (rendered), for
  p in means (N,3), log_scales (N,3), quats (N,4 wxyz, unnormalised), opacity_logits (N,),
  sh (N,K,3)
- forward: `image`, `target` (H,W,3), `final_T` (H,W), `loss`, `means2d` (N,2),
  `conic` (N,3: A,B,C with power = -0.5(A dx^2 + 2B dx dy + C dy^2)), `opacity`,
  `colors` (N,3 after clamp), `depth`, `order` (front to back), `valid` (z > 0.2),
  `visible` (sparse-Adam rows), `live` (N,H,W bool: pair blended into the pixel)
- loss grad (T7): `dL_dimage` (H,W,3)
- blend backward outputs (T1): `dL_dmeans2d` (N,2), `dL_dconic` (N,3), `dL_dopacity` (N,)
  (activated opacity), `dL_dcolors` (N,3)
- param grads (T2/T4): `grad_<p>`
- sparse Adam (T2), one step from step `adam_step_in` = 4 over `visible` rows, lr per
  group in `adam_lr_json` (SH band 0 `sh_dc`, bands 1.. `sh_rest`), `adam_betas`, `adam_eps`:
  - fp32 moments: inputs `adam_m_in_<p>`, `adam_v_in_<p>`; outputs `adam_fp32_param_<p>`,
    `adam_fp32_m_<p>`, `adam_fp32_v_<p>`
  - bf16 moments (stored bf16, math fp32, update from the fp32 m/v, stored back rounded
    to nearest even): inputs `adam_bf16_m_in_<p>`, `adam_bf16_v_in_<p>`; outputs
    `adam_bf16_param_<p>`, `adam_bf16_m_<p>`, `adam_bf16_v_<p>` (bf16 values held in fp32)

Gradient check: autograd vs central differences (eps 1e-6, fp64) with the forward's
discrete decisions held fixed, max relative error per param group (floor 1e-4 of the
group's largest gradient). `tests/unit/test_train_oracle.py` probes a fixed sample of
every group; `python3 -m gsplat.train_oracle --fd-report <out.json>` checks every entry of
the SH0/SH3 cases above plus a dense variant (alpha clamp and early stop active).
