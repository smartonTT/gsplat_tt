#!/usr/bin/env bash
# One-time setup of a fresh IRD box for the live viewer (runs ON the box).
# Builds tt-metal at the commit the ledger box uses, makes the project venv and
# the scenes dir that opt/sync_remote.sh links into each deploy dir. Idempotent:
# finished steps are skipped. Writes $ROOT/viewer/setup.rc (0 = ok) at the end.
#   ssh <box> 'cat > /localdev/$USER/viewer/setup_box.sh' < opt/viewer/setup_box.sh
#   ssh <box> '(setsid nohup bash /localdev/$USER/viewer/setup_box.sh > /localdev/$USER/viewer/setup.log 2>&1 < /dev/null &)'
# (A backgrounded `bash -s` reads /dev/null, not the piped script.)
set -uo pipefail
# The ledger box (yyzo-bh-07) runs tt-metal e77780fe, a local commit that is not
# on GitHub; it pins sfpi 7.49.0. 437bc366 (2026-05-15) is the parent of
# 36c276a8 (#44262), which removed _llk_math_eltwise_unary_sfpu_start_/_done_
# that render/kernels/compute/project_pfwc_compute.cpp calls; it pins sfpi 7.49.0.
TT_SHA=${TT_SHA:-437bc3664390ce8e7d564db2133083ff9035e7cd}
ROOT=${ROOT:-/localdev/$USER}
TT=$ROOT/tt-metal BASE=$ROOT/gstt2 VDIR=$ROOT/viewer
mkdir -p "$ROOT" "$BASE/scenes" "$VDIR"
rm -f "$VDIR/setup.rc"
steps() {
  set -ex
  if [ ! -f "$TT/build/tt_metal/libtt_metal.so" ] || [ "$(git -C "$TT" rev-parse HEAD)" != "$TT_SHA" ]; then
    [ -d "$TT/.git" ] || git clone -q https://github.com/tenstorrent/tt-metal.git "$TT"
    cd "$TT"
    git checkout -q "$TT_SHA"
    git submodule update --init --recursive -q
    # Same options as the ledger box's build/CMakeCache.txt, minus python bindings.
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \
      -DENABLE_TRACY=ON -DWITH_PYTHON_BINDINGS=OFF -DTT_METAL_BUILD_TESTS=OFF \
      -DBUILD_PROGRAMMING_EXAMPLES=OFF
    cmake --build build -j "$(nproc)"
  fi
  if [ ! -x "$BASE/.venv/bin/python" ]; then
    python3 -m venv "$BASE/.venv"
    "$BASE/.venv/bin/pip" install -q --upgrade pip
    "$BASE/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
    "$BASE/.venv/bin/pip" install -q numpy==2.2.6 viser==1.0.27 nerfview==0.1.3 \
      pybind11==3.0.4 plyfile==1.1.3 pillow scipy matplotlib jaxtyping imageio rich splines==0.3.3 websockets==15.0.1
  fi
  # viser encodes frames with cv2 when it is importable: one GIL-free call, so the JPEG encode
  # on the viewer's sender thread no longer slows the render (PIL's chunked encode cost the
  # render +1.6 ms/frame; task #257).
  "$BASE/.venv/bin/python" -c "import cv2" 2>/dev/null || \
    "$BASE/.venv/bin/pip" install -q --no-deps opencv-python-headless==5.0.0.93
  "$BASE/.venv/bin/python" -c "import torch, viser, nerfview, pybind11, cv2; print('venv ok')"
}
( steps ); rc=$?
echo "$rc" > "$VDIR/setup.rc"
echo "[setup_box] rc=$rc"
