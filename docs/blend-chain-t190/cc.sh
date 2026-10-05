#!/bin/bash
# Task #190: build the blend TRISC1 ELF the way the tt-metal JIT does, with no
# device: flags from tt_metal/jit_build/build.cpp and bh_hal.cpp (codegen flags
# match the .gnu.lto_.opts of the t188 trisck.o), generated files and weakened
# firmware from the t188 JIT cache, one ELF per BLEND_CHAIN_WALK value.
#   cc.sh <src root> <out dir> <knob value>...
# Writes <out>/cw<v>/trisc1/{trisc1.elf,trisc1.dis,trisc1.sym,build.log}.
set -u
SRC=$1; OUT=$2; shift 2
R=/localdev/smarton/tt-metal
G=$R/runtime/sfpi/compiler/bin
C188=/localdev/smarton/.cache/ttmc-gstt2-t188/tt-metal-cache431005707817566179
K188=$C188/kernels/alpha_blend_compute_mb/15508740407247128523
FLAGS="-std=c++17 -ftt-nttp -ftt-constinit -ftt-consteval -flto=auto -ffast-math -fno-exceptions -g"
CFLAGS="$FLAGS -MMD -fno-use-cxa-atexit -Wall -Werror -Wno-error=deprecated-declarations
 -Wno-error=multistatement-macros -Wno-error=parentheses -Wno-error=unused-but-set-variable
 -Wno-unused-variable -Wno-unused-function -mcpu=tt-bh-tensix"
INC="-I. -I.. -I$R -I$R/ttnn -I$R/ttnn/cpp -I$R/tt_metal -I$R/tt_metal/hw/inc -I$R/tt_metal/tt-llk/common
 -I$R/tt_metal/hostdevcommon/api -I$R/tt_metal/api/ -I$R/runtime/sfpi/include
 -I$R/tt_metal/hw/ckernels/blackhole/metal/common -I$R/tt_metal/hw/ckernels/blackhole/metal/llk_io
 -I$R/tt_metal/hw/inc/internal/tt-1xx -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole
 -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole/blackhole_defines
 -I$R/tt_metal/hw/inc/internal/tt-1xx/blackhole/noc -I$R/tt_metal/tt-llk/tt_llk_blackhole/common/inc
 -I$R/tt_metal/tt-llk/tt_llk_blackhole/llk_lib -I$R/tt_metal/hw/ckernels/blackhole/metal/llk_api
 -I$R/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu -I$R/tt_metal/hw/firmware/src/tt-1xx
 -I$SRC/render/kernels/compute"
DEFS="-DNUM_DRAM_BANKS=7 -DNUM_L1_BANKS=110 -DIS_NOT_POW2_NUM_DRAM_BANKS=1 -DIS_NOT_POW2_NUM_L1_BANKS=1
 -DPCIE_NOC_X=19 -DPCIE_NOC_Y=24 -DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 -DPROCESSOR_INDEX=3
 -DUCK_CHLKC_MATH -DCOMPILE_FOR_TRISC=1 -DARCH_BLACKHOLE -DDISPATCH_MESSAGE_ADDR=0 -DKERNEL_BUILD"
LFLAGS="$FLAGS -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -mcpu=tt-bh-tensix
 -T$R/runtime/hw/toolchain/blackhole/kernel_trisc1.ld -Wl,--emit-relocs"
rc=0
for v in "$@"; do
  d=$OUT/cw$v; rm -rf "$d"; mkdir -p "$d/trisc1"
  cp "$K188"/chlkc_descriptors.h "$d/"
  for f in "$K188"/chlkc_*.cpp; do
    sed "s|/localdev/smarton/gstt2-t188/|$SRC/|" "$f" > "$d/$(basename "$f")"
  done
  { cat "$K188/defines_generated.h"; echo "#define BLEND_CHAIN_WALK $v"; } > "$d/defines_generated.h"
  (
    cd "$d/trisc1" &&
    $G/riscv-tt-elf-g++ -O3 $CFLAGS $INC -c -o trisck.o $R/tt_metal/hw/firmware/src/tt-1xx/trisck.cc \
      -MF trisck.d $DEFS &&
    $G/riscv-tt-elf-g++ -O3 -Wl,--just-symbols=$C188/firmware/trisc1/trisc1_weakened.elf $LFLAGS \
      $R/runtime/hw/lib/blackhole/substitutes.o trisck.o -o trisc1.elf &&
    $G/riscv-tt-elf-objdump -d trisc1.elf > trisc1.dis &&
    $G/riscv-tt-elf-nm -S --size-sort trisc1.elf > trisc1.sym &&
    $G/riscv-tt-elf-size -A trisc1.elf > trisc1.size
  ) > "$d/trisc1/build.log" 2>&1
  r=$?; echo "cw$v rc=$r"; [ $r -eq 0 ] || { tail -30 "$d/trisc1/build.log"; rc=1; }
done
# Reference: the t188 default ELF, same objdump.
$G/riscv-tt-elf-objdump -d "$K188/trisc1/trisc1.elf" > "$OUT/t188-trisc1.dis"
$G/riscv-tt-elf-size -A "$K188/trisc1/trisc1.elf" > "$OUT/t188-trisc1.size"
exit $rc
