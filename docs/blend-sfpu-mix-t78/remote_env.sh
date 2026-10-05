# sourced by the t78 remote scripts; tree = /localdev/smarton/gstt2-t78
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
T=/localdev/smarton/gstt2-t78
cd $T || exit 1
source .venv/bin/activate
# ablation value -> JIT cache dir
cache() { echo /localdev/smarton/.cache/ttmc-gstt2-t78-a$1${2:-}; }
