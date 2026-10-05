# sourced by the t83 remote scripts; tree = /localdev/smarton/gstt2-t83 (override T83_TREE)
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
T=${T83_TREE:-/localdev/smarton/gstt2-t83}
cd $T || exit 1
source .venv/bin/activate
# ablation value (+ variant tag) -> JIT cache dir
cache() { echo /localdev/smarton/.cache/ttmc-$(basename $T)-a$1${2:-}; }
