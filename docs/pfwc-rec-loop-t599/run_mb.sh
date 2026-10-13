#!/usr/bin/env bash
set -eo pipefail
SSH="ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes"
D=/localdev/smarton/gstt2-t599mb
git archive --format=tar HEAD render/bench render/kernels cmake | $SSH bh-33 "mkdir -p $D && tar -xmf - -C $D && df -h $D /home/smarton | tail -2"
$SSH bh-33 "cd $D && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole && \
  (cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build render/bench/build 2>&1 | tail -5) && \
  for nb in 8 7; do MB_NB=\$nb MB_SUITE=pfwcrec TT_METAL_CACHE=$D/mbcache TT_METAL_RUNTIME_ROOT=\$TT_METAL_HOME timeout 600 render/bench/build/risc_microbench 2>&1 | grep -E '^\[MB\]|error|Error' | grep -v '^\s*$' | tail -12; done"
