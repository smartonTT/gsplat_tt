# GPU reference bench

The charter's success criterion is "beat GPU performance", so the project needs
a **measured** CUDA 3DGS number on the *same* workload our TT renderer runs:
`scenes/bicycle.ply`, the 30 views of `benchmarks/cameras_v2.json`, 1024×1024,
SH degree 0.

`run_gpu_bench.py` produces exactly that row. It is deliberately a single file
with no repo imports so it can be dropped onto any CUDA host that has
`torch` + `gsplat` (or `diff-gaussian-rasterization`) and run against a copy of
`scenes/`, `benchmarks/`.

## Status: no GPU measured yet

As of 2026-09-30 no NVIDIA GPU is reachable from this environment — see
`opt/cpu-vs-tt-comparison.md` § "GPU reference" for the search that establishes
this. The script is committed ready-to-run; the moment a CUDA host exists,
running it fills in the GPU row of the comparison doc and of `opt/REPORT.html`
automatically (the report reads `opt/cpu-vs-tt/gpu_result.json`).

## Run

```bash
# on the GPU host, from the repo root
python -m venv .venv-gpu && . .venv-gpu/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gsplat plyfile numpy pillow

python bench/gpu_reference/run_gpu_bench.py --backend gsplat --repeats 2
```

Outputs:

| path | what |
|---|---|
| `opt/cpu-vs-tt/gpu_result.json` | GPU name/driver, per-view ms, avg/p50/min/max, hero PSNR |
| `opt/cpu-vs-tt/gpu_hero.png` | the hero render, for a side-by-side |

Then:

```bash
python3 opt/build_report.py      # picks the json up, renders the GPU row
```

and paste the printed avg/p50 into the GPU row of
`opt/cpu-vs-tt-comparison.md`.

## Bench identity (why the numbers are comparable)

These are the things that would silently make the comparison a lie, and what the
script does about each:

* **Intrinsics / extrinsics.** Re-implemented from `gsplat/viewer.py`
  (`f = 0.5·max(W,H)/tan(fov/2)`) and `gsplat/utils.py` (closed-form rigid
  inverse). Verified **bit-identical** (max abs diff 0.0 on K and on all 30
  viewmats) against the repo helpers on 2026-09-30.
* **Activations.** `exp(scale)`, `normalize(quat)`, `sigmoid(opacity)`,
  `clamp(0.5 + C0·f_dc, 0, 1)` — the same four lines as
  `gsplat/loading_gaussians.py::load_ply`. Verified on 2026-09-30 by loading the
  full 6,131,954-Gaussian `bicycle.ply` through both loaders: `means`, `quats`,
  `scales`, `colors` **bit-identical** (max abs diff 0.0); `opacities` differ by
  1 ULP (1.19e-07) from the numpy-vs-tensor `sigmoid` path.
* **Colors, not SH.** Colors go in as *precomputed RGB* with `sh_degree=None`.
  The bicycle ply carries higher SH bands that our pipeline never evaluates; if
  the GPU path evaluated them it would render a different (better) image and a
  slower frame, and the comparison would be meaningless in both directions.
* **Warmup.** The hero view is a warmup and is excluded; stats are over the
  remaining 29 views — the same rule the CPU rows use. `--repeats 2` also
  discards the first full pass so CUDA JIT/autotune is out of the numbers.
* **Timing scope.** CUDA-synchronised wall time around the rasterization call
  only. No ply load, no upload, no PNG encode. This mirrors "wall time per
  `pipeline.render`" on the CPU and TT rows.
* **Quality.** Hero PSNR is computed with the same `-10·log10(MSE)` on `[0,1]`
  against the same `benchmarks/reference_v2/hero.png` the CPU rows use, so a
  GPU render that cheats on quality shows up as a PSNR outlier rather than as a
  free speedup.

The script **refuses to run** without a CUDA device rather than silently falling
back to CPU — the charter forbids estimated numbers, and a CPU-fallback number
mislabelled as GPU would be exactly that.

## Caveats to state alongside any number this produces

* gsplat/INRIA render the full 3DGS forward pass including their own culling and
  tile binning; our pipeline additionally applies a Mahalanobis per-pair and
  per-microblock cull at `contrib_floor = 1/16384`. Compare the hero PSNRs
  before comparing the ms.
* The comparison is **one GPU vs one Blackhole card**, not perf/W or perf/$.
  Record the GPU's TDP and class (datacenter vs consumer) next to the number.
