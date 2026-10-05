# sourced by the t80 remote scripts; tree = /localdev/smarton/gstt2-t80
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
T=${T80_TREE:-/localdev/smarton/gstt2-t80}   # gstt2-t80 = base+ablations, gstt2-t80b = change
cd $T || exit 1
source .venv/bin/activate
# ablation value (+ variant tag) -> JIT cache dir
cache() { echo /localdev/smarton/.cache/ttmc-$(basename $T)-a$1${2:-}; }
