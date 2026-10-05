# sourced by the t86 remote scripts; tree = /localdev/smarton/gstt2-t86 (override T86_TREE)
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_ALLOW_DIRECT=1
T=${T86_TREE:-/localdev/smarton/gstt2-t86}; cd $T || exit 1; source .venv/bin/activate
S=/localdev/smarton/t86_scripts; mkdir -p $S
