#!/bin/bash
# t231: compile-only build of the blend TRISC0/1/2 ELFs for BLEND_DECODE_AHEAD
# 0, 1, 2 on the reserved host, in a private scratch dir (no device, no shared
# build dir). Copy of docs/blend-sched-t219/build.sh: the t229 JIT cache entry
# of alpha_blend_compute_mb (BLEND_SCHED=2 defaults) gives the generated headers,
# default blend defines and firmware symbols.
# Usage (on the host): [LEVELS="0 1 2"] build.sh [src tree holding render/kernels] [scratch dir]
set -u
T=/localdev/smarton/tt-metal
W=${2:-/localdev/smarton/gstt2-t231-scratch}
SRC=${1:-$W/src}
mkdir -p $W
C=/localdev/smarton/.cache/ttmc-gstt2-t229/tt-metal-cache5481144561508939120
K=$C/kernels/alpha_blend_compute_mb/629917748786089677
BIN=$T/runtime/sfpi/compiler/bin
CXX=$BIN/riscv-tt-elf-g++
FL="-O3 -std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -flto=auto -ffast-math -fno-exceptions -g"
CFL="$FL -MMD -fno-use-cxa-atexit -Wall -Werror -Wno-error=deprecated-declarations -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable -Wno-unused-variable -Wno-unused-function -mcpu=tt-bh-tensix"
INC="-I. -I.. -I$T/ -I$T/ttnn -I$T/ttnn/cpp -I$T/tt_metal -I$T/tt_metal/hw/inc -I$T/tt_metal/tt-llk/common -I$T/tt_metal/hostdevcommon/api -I$T/tt_metal/api/ -I$T/runtime/sfpi/include -I$T/tt_metal/hw/ckernels/blackhole/metal/common -I$T/tt_metal/hw/ckernels/blackhole/metal/llk_io -I$T/tt_metal/hw/inc/internal/tt-1xx -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines -I$T/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc -I$T/tt_metal/tt-llk/tt_llk_blackhole/common/inc -I$T/tt_metal/tt-llk/tt_llk_blackhole/llk_lib -I$T/tt_metal/hw/ckernels/blackhole/metal/llk_api -I$T/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu -I$T/tt_metal/hw/firmware/src/tt-1xx -I$SRC/render/kernels/compute"
DEFC="-DIS_NOT_POW2_NUM_DRAM_BANKS=1 -DIS_NOT_POW2_NUM_L1_BANKS=1 -DNUM_DRAM_BANKS=7 -DNUM_L1_BANKS=110 -DPCIE_NOC_X=19 -DPCIE_NOC_Y=24 -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DCOMPILE_FOR_TRISC=1 -DARCH_BLACKHOLE -DDISPATCH_MESSAGE_ADDR=4290184248 -DKERNEL_BUILD"
build() {  # level trisc(0|1|2)
    L=$1 N=$2 D=$W/k$1
    case $N in 0) TH=UNPACK F=unpack P=2 ;; 1) TH=MATH F=math P=3 ;; 2) TH=PACK F=pack P=4 ;; esac
    mkdir -p $D/trisc$N
    printf '#define TRISC_%s\n#include "defines_generated.h"\n#include "%s"\n' \
        $TH "$SRC/render/kernels/compute/alpha_blend_compute_mb.cpp" > $D/chlkc_$F.cpp
    LFL="-O3 -Wl,--just-symbols=$C/firmware/trisc$N/trisc${N}_weakened.elf $FL -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -mcpu=tt-bh-tensix -T$T/runtime/hw/toolchain/blackhole/kernel_trisc$N.ld -Wl,--emit-relocs $T/runtime/hw/lib/blackhole/substitutes.o"
    cd $D/trisc$N || return 1
    $CXX $CFL $INC -c -o trisck.o $T/tt_metal/hw/firmware/src/tt-1xx/trisck.cc $DEFC -DPROCESSOR_INDEX=$P -DUCK_CHLKC_$TH > build.log 2>&1 &&
        $CXX $LFL trisck.o -o trisc$N.elf >> build.log 2>&1 &&
        $BIN/riscv-tt-elf-objdump -d -C trisc$N.elf > $W/trisc$N-da$L.dis
    echo "level $L trisc$N rc=$? warnings/errors: $(grep -c -E 'warning|error' build.log), text bytes: $($BIN/riscv-tt-elf-size trisc$N.elf 2>/dev/null | awk 'NR==2{print $1}')"
    [ -s build.log ] && grep -m 8 -E 'error|warning' build.log
}
for L in ${LEVELS:-0 1 2}; do
    D=$W/k$L; rm -rf $D && mkdir -p $D && cp $K/chlkc_*.h $D/
    { cat $K/defines_generated.h; echo "#define BLEND_DECODE_AHEAD $L"; } > $D/defines_generated.h
    for N in 0 1 2; do build $L $N > $W/build$L-$N.txt 2>&1 & done
done
wait
for L in ${LEVELS:-0 1 2}; do for N in 0 1 2; do cat $W/build$L-$N.txt; done; done
