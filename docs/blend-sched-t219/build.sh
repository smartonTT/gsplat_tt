#!/bin/bash
# t219: compile-only build of the blend TRISC1 ELF for BLEND_SCHED 0, 1, 2 on
# the reserved host, in a private scratch dir (no device, no shared build dir).
# Inputs: the t188 JIT cache entry of alpha_blend_compute_mb (generated headers,
# default blend defines, firmware symbols) and the JIT compile/link commands
# that tt-metal logged for this kernel in t60 (TT_METAL_LOG_KERNELS_COMPILE_COMMANDS).
# Usage (on the host): [LEVELS="0 1 2"] build.sh [src tree holding render/kernels] [scratch dir]
set -u
T=/localdev/smarton/tt-metal
W=${2:-/localdev/smarton/gstt2-t219-scratch}
SRC=${1:-$W/src}
mkdir -p $W
C=/localdev/smarton/.cache/ttmc-gstt2-t188/tt-metal-cache431005707817566179
K=$C/kernels/alpha_blend_compute_mb/15508740407247128523
BIN=$T/runtime/sfpi/compiler/bin
CXX=$BIN/riscv-tt-elf-g++
FL="-O3 -std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -flto=auto -ffast-math -fno-exceptions -g"
CFL="$FL -MMD -fno-use-cxa-atexit -Wall -Werror -Wno-error=deprecated-declarations -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function -mcpu=tt-bh-tensix"
INC="-I. -I.. -I$T/ -I$T/ttnn -I$T/ttnn/cpp -I$T/tt_metal -I$T/tt_metal/hw/inc -I$T/tt_metal/tt-llk/common -I$T/tt_metal/hostdevcommon/api -I$T/tt_metal/api/ -I$T/runtime/sfpi/include -I$T/tt_metal/hw/ckernels/blackhole/metal/common -I$T/tt_metal/hw/ckernels/blackhole/metal/llk_io -I$T/tt_metal/hw/inc/internal/tt-1xx -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc -I$T/tt_metal/tt-llk/tt_llk_blackhole/common/inc -I$T/tt_metal/tt-llk/tt_llk_blackhole/llk_lib -I$T/tt_metal/hw/ckernels/blackhole/metal/llk_api -I$T/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu -I$T/tt_metal/hw/firmware/src/tt-1xx -I$SRC/render/kernels/compute"
DEF="-DIS_NOT_POW2_NUM_DRAM_BANKS=1 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DNUM_DRAM_BANKS=7 -DNUM_L1_BANKS=110 -DPCIE_NOC_X=19 -DPCIE_NOC_Y=24 -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DPROCESSOR_INDEX=3 -DUCK_CHLKC_MATH -DCOMPILE_FOR_TRISC=1 -DARCH_BLACKHOLE -DDISPATCH_MESSAGE_ADDR=4290184248 -DKERNEL_BUILD"
LFL="-O3 -Wl,--just-symbols=$C/firmware/trisc1/trisc1_weakened.elf $FL -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -mcpu=tt-bh-tensix -T$T/runtime/hw/toolchain/blackhole/kernel_trisc1.ld -Wl,--emit-relocs $T/runtime/hw/lib/blackhole/substitutes.o"
build() {
    L=$1 D=$W/k$1
    rm -rf $D && mkdir -p $D/trisc1 && cp $K/chlkc_*.h $D/
    { cat $K/defines_generated.h; echo "#define BLEND_SCHED $L"; } > $D/defines_generated.h
    printf '#define TRISC_MATH\n#include "defines_generated.h"\n#include "%s"\n' \
        "$SRC/render/kernels/compute/alpha_blend_compute_mb.cpp" > $D/chlkc_math.cpp
    cd $D/trisc1 || return 1
    $CXX $CFL $INC -c -o trisck.o $T/tt_metal/hw/firmware/src/tt-1xx/trisck.cc $DEF > build.log 2>&1 &&
        $CXX $LFL trisck.o -o trisc1.elf >> build.log 2>&1 &&
        $BIN/riscv-tt-elf-objdump -d -C trisc1.elf > $W/trisc1-sched$L.dis
    echo "level $L rc=$? warnings/errors: $(grep -c -E 'warning|error' build.log), text bytes: $($BIN/riscv-tt-elf-size trisc1.elf 2>/dev/null | awk 'NR==2{print $1}')"
}
for L in ${LEVELS:-0 1 2}; do build $L > $W/build$L.txt 2>&1 & done
wait
for L in ${LEVELS:-0 1 2}; do cat $W/build$L.txt; done
