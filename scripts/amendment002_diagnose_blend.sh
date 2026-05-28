#!/usr/bin/env bash
# Hero blend diagnostics on bh-30: TT vs cpu_cpp_mb vs ref.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

REMOTE="${REMOTE_HOST:-$METAL_REMOTE_HOST}"

ssh -o ConnectTimeout=30 "$REMOTE" bash -s <<'REMOTE'
set -euo pipefail
cd /localdev/smarton/gstt2
source .venv/bin/activate
export TT_METAL_HOME=/localdev/smarton/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
export MESH_DEVICE=P100
export TT_METAL_ARCH_NAME=blackhole

pkill -f 'metal_example_gaussian_splatting --daemon' 2>/dev/null || true
sleep 1

python3 2> /tmp/amend002_diag.log <<'PY'
import json, numpy as np, torch, sys
sys.path.insert(0, ".")
from backends import get_backend
from backends.cpu_cpp import _gsplat_cpu as m

def psnr(a, b):
    d = a.astype(np.float64) - b.astype(np.float64)
    mse = float(np.mean(d * d))
    return float("inf") if mse <= 0 else 10 * np.log10(1.0 / mse)

data = np.load("tests/fixtures/hero/blend_inputs.npz")
ref = np.load("tests/fixtures/hero/blend_output.npy")
H, W = int(data["H"]), int(data["W"])
t = lambda x: torch.from_numpy(x)

cpu_b = get_backend("cpu_cpp_mb")
cpu_img, _ = cpu_b.blend(
    t(data["means_2d"]), t(data["covs_2d"]), t(data["colors"]),
    t(data["opacities"]), t(data["sorted_gaussian_ids"]), t(data["tile_ranges"]), H, W)
cpu = np.asarray(cpu_img, dtype=np.float32)

out = {
    "has_tt_support": bool(m.has_tt_support()),
    "psnr_cpu_cpp_mb_vs_ref_dB": psnr(ref, cpu),
    "psnr_tt_vs_ref_dB": None,
    "psnr_tt_vs_cpu_cpp_mb_dB": None,
    "max_abs_tt_vs_cpu": None,
}

if m.has_tt_support():
    tt_b = get_backend("tt")
    tt_img, sub = tt_b.blend(
        t(data["means_2d"]), t(data["covs_2d"]), t(data["colors"]),
        t(data["opacities"]), t(data["sorted_gaussian_ids"]), t(data["tile_ranges"]), H, W)
    tt = np.asarray(tt_img, dtype=np.float32)
    out["psnr_tt_vs_ref_dB"] = psnr(ref, tt)
    out["psnr_tt_vs_cpu_cpp_mb_dB"] = psnr(cpu, tt)
    out["max_abs_tt_vs_cpu"] = float(np.max(np.abs(cpu - tt)))
    out["device_kernel_ms"] = float(sub.get("device_kernel_ms", 0.0)) if isinstance(sub, dict) else None
    tt_b.close()

with open("/tmp/amend002_diag.json", "w") as f:
    json.dump(out, f, indent=2)
PY
cat /tmp/amend002_diag.json
REMOTE
