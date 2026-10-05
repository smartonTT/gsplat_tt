# sourced by the t68 remote scripts; tree = /localdev/smarton/gstt2-t68
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
T=/localdev/smarton/gstt2-t68
cd $T || exit 1
ln -sfn /localdev/smarton/gstt2/.venv .venv; ln -sfn /localdev/smarton/gstt2/scenes scenes
mkdir -p backends/cpu_cpp tmp; cp -n /localdev/smarton/gstt2/backends/cpu_cpp/_gsplat_cpu*.so backends/cpu_cpp/ 2>/dev/null
source .venv/bin/activate
# knob value -> JIT cache dir (defines differ, keep caches apart anyway)
cache() { echo /localdev/smarton/.cache/ttmc-gstt2-t68-k$1${2:-}; }
