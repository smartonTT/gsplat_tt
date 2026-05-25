#!/usr/bin/env bash
# Bisect-step harness. Runs on the box. Mac is responsible for git checkout + devsync.
# Exit codes: 0=PASS (PSNR>=25 vs ref), 1=FAIL (low PSNR or no PNG), 125=SKIP (build broken).
set -uo pipefail
cd /proj_sw/user_dev/smarton/gsplat_tt
export TT_METAL_HOME=/proj_sw/user_dev/smarton/gsplat_tt/backends/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
export TT_MESH_GRAPH_DESC_PATH=/proj_sw/user_dev/smarton/gsplat_tt/backends/tt/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p100_mesh_graph_descriptor.textproto
export TT_METAL_LOGGER_LEVEL=error
export TT_METAL_LOGGER_FILE=/localdev/smarton/ttmetal_daemon.log
export TT_METAL_OPERATION_TIMEOUT_SECONDS=120

OUT=/tmp/luigi_bisect_$(date +%H%M%S).png
REF=/tmp/luigi_hero_basemaster_optdir.png

echo "=== [$(date +%H:%M:%S)] kill leftover daemons ==="
pkill -TERM -f 'metal_example_gaussian_splatting --daemon' 2>/dev/null || true
sleep 2
pgrep -af metal_example_gaussian_splatting || echo "(no daemons running)"

echo
echo "=== [$(date +%H:%M:%S)] touch sources to force rebuild ==="
touch backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend.cpp
touch backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/alpha_blend_host.h
touch backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/dataflow/reader_alpha_blend.cpp
touch backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting/kernels/compute/alpha_blend_compute.cpp

echo
echo "=== [$(date +%H:%M:%S)] ninja build (worst case ~3 min) ==="
cd backends/tt/tt-metal/build
if ! timeout 300 sudo -n ninja metal_example_gaussian_splatting 2>&1 | tail -20 ; then
  echo "BUILD_FAIL"
  exit 125
fi
cd /proj_sw/user_dev/smarton/gsplat_tt

echo
echo "=== [$(date +%H:%M:%S)] clear JIT cache (force kernel recompile against new sources) ==="
rm -rf /home/smarton/.cache/tt-metal-cache/* /localdev/smarton/.cache/tt-metal-cache/* 2>/dev/null
mkdir -p /localdev/smarton/.cache/tt-metal-cache
export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache

echo
echo "=== [$(date +%H:%M:%S)] render luigi hero (timeout 180s) ==="
source venv/bin/activate
if ! timeout 180 python scripts/render_fixed.py luigi hero --backend tt --warmup 1 --frames 1 --out "$OUT" 2>&1 | tail -30 ; then
  echo "RENDER_TIMEOUT_OR_CRASH"
  pkill -TERM -f 'metal_example_gaussian_splatting --daemon' 2>/dev/null || true
  exit 1
fi
pkill -TERM -f 'metal_example_gaussian_splatting --daemon' 2>/dev/null || true

echo
echo "=== [$(date +%H:%M:%S)] verify PNG and PSNR ==="
if [ ! -s "$OUT" ]; then
  echo "NO_PNG_OUTPUT"
  exit 1
fi
ls -la "$OUT"

python3 - "$OUT" "$REF" <<'PYEOF'
import sys, numpy as np
from PIL import Image
out, ref = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(out).convert("RGB"), dtype=np.float32)
b = np.asarray(Image.open(ref).convert("RGB"), dtype=np.float32)
if a.shape != b.shape:
    print(f"SHAPE_MISMATCH out={a.shape} ref={b.shape}")
    sys.exit(1)
mse = np.mean((a - b) ** 2)
psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
mean = a.mean()
print(f"PSNR={psnr:.2f} dB  mean_out={mean:.1f} (ref_mean={b.mean():.1f})")
# Black/garbage image detection.
if mean < 5:
    print("VERDICT=FAIL (image is black)")
    sys.exit(1)
if psnr < 25:
    print(f"VERDICT=FAIL (PSNR {psnr:.2f} < 25 dB)")
    sys.exit(1)
print(f"VERDICT=PASS")
PYEOF
PYRC=$?
echo "[$(date +%H:%M:%S)] bisect step done, exit=$PYRC"
exit $PYRC
