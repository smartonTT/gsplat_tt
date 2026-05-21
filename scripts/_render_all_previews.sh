#!/bin/bash
# Render all 9 (scene × view) preview thumbnails on the active TT device.
# Each render spawns a fresh daemon so device state is clean per scene.
set -uo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export TT_METAL_HOME="$PWD/backends/tt/tt-metal"
export TT_METAL_RUNTIME_ROOT="$PWD/backends/tt/tt-metal"

scenes=(stitch luigi strawberry)
views=(hero side top)

for scene in "${scenes[@]}"; do
    for view in "${views[@]}"; do
        echo ""
        echo "=== $scene $view ==="
        pkill -9 -f metal_example_gaussian 2>/dev/null || true
        sleep 1
        python scripts/render_fixed.py "$scene" "$view" --backend tt --warmup 2 --frames 3 2>&1 | tail -12
    done
done
echo ""
echo "=== done ==="
