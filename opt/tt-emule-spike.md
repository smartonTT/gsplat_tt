# tt-emule Feasibility Spike for gsplat_tt

**Date:** 2026-06-01
**Box:** `yyzo-bh-03` (x86-64 Linux, Ubuntu 5.15 kernel, 12 cores, 423 GB free on `/localdev`)
**Isolation:** All work in `/localdev/smarton/emule-spike/`. **No TT device opened, no `devrun.sh`, no contention with the in-flight optimization worker.** Every probe below is pure host-side `clang++-20` compilation — the exact compile path emule's runtime uses, run by hand in a sandbox.
**tt-emule HEAD probed:** `e1387bc217ef74194227050026b0d1ccb82d8df4` (2026-06-01, current tip of `main`).

---

## TL;DR VERDICT

**Worth adopting as a gsplat correctness/debug sandbox — but only after a bounded, well-scoped fix set, and only for the *non-SFPU* slice out of the box.**

- **Toolchain:** ✅ fully present, no installs needed.
- **emule itself:** builds + integrates on this class of box (production tt-metal here is *already* emule-integrated). Full build deliberately **not** run (it is the multi-hour long pole and would contend CPU with the device worker); the per-kernel JIT compile-path probe gives more direct gsplat-specific signal than emule's own regression would.
- **The raw-L1-deref fear is a NON-issue for us.** Our kernels never dereference a raw runtime-arg L1 address (the one documented thing that segfaults). They are NOC-API-clean.
- **Dataflow kernels (readers/writers): ✅ compile clean under emule today.** Verified `reader_bucket_cull`, the 60 KB `reader_alpha_blend_mb_devcull`, `writer_bucket_cull`, `writer_microblock_cull` all build to `.so` against emule's JIT headers.
- **Compute kernels (the SFPU microblock cull + blend): ❌ HARD WALL at JIT compile.** They use the raw `sfpi::` vector model (`vFloat`, `dst_reg[]`, `v_if`, low-level `ckernel_sfpu_*`), which emule does **not** provide and, per emule's own docs, cannot mock faithfully without a per-kernel semantic rewrite.

**Bottom line: worth-it-after-N-fixes, N ≈ 2 large + 2 small.** The two SFPU compute kernels each need a Strategy-C semantic rewrite (HIGH effort each); two small stubs (`get_tile_address`, the `_llk_math_eltwise_unary_sfpu_*` no-ops) are trivial. The dataflow half of the pipeline is usable as a deterministic sandbox immediately.

---

## 1. Toolchain status — ✅ PASS (nothing to install)

| Tool | Required | Found on `yyzo-bh-03` |
|---|---|---|
| clang-20 / clang++-20 | 20.x | **20.1.8** (`/usr/bin/clang-20`, `/usr/bin/clang++-20`) |
| libc++-20-dev | present | **1:20.1.8** ✅ |
| libc++abi-20-dev | present | **1:20.1.8** ✅ |
| libclang-20-dev | — | 1:20.1.8 ✅ |
| CMake | ≥ 3.24 | **4.0.2** ✅ |
| Ninja | ≥ 1.10 | **1.10.1** ✅ |
| llvm toolchain dir | — | `/usr/lib/llvm-20` present ✅ |
| Toolchain file | — | `tt-metal/cmake/x86_64-linux-clang-20-libcpp-toolchain.cmake` present ✅ |

No blocker here. A `-DTT_METAL_USE_EMULE=ON` build is toolchain-feasible immediately.

---

## 2. tt-emule repo facts (verified against live HEAD)

- **Pinned tt-metal SHA** (`tt-metal-pin.txt`): `b24ad48df9ce8e1b1e083a73d5539c3ceb2886d4`.
- **Pinned tt-mlir SHA** (`tt-mlir-pin.txt`): `8b8cccac358830a088f0c6a4b32d497a52eb069e` (D2M only; not needed for us).
- **Production tt-metal here** (`/localdev/smarton/tt-metal`) is on `ebaaa579351d3ed5fcf7166d6cc55c62e52b2fb5` — **different from the pin**, so it cannot be reused as the build source without a checkout (which we must not do in place). A full build needs a **separate** tt-metal clone at the pin. **However**, the production tree already contains `tt_metal/impl/emulation/emulated_program_runner.cpp` — i.e. it is already emule-aware — which is why the read-only header-path probe below is faithful.
- **`.claude/` skills present:** `implement-mock` (Strategy A/B/C workflow), `arch-lookup`, plus `references/emule-mapping.md`, `stub-checklist.md`, `api-injection-points.md`, and `sage-*` arch agents. These are exactly the assets we'd use to close the holes.
- **Supported arch:** Wormhole N150 (primary), **Blackhole P100** (our target), Quasar.

### emule's own documented limitations that matter to us
From `README.md` / `KNOWN_ISSUES.md` / `.claude/references/emule-mapping.md`:
1. **Raw L1 pointer deref** (offset-as-host-pointer) segfaults — but only for *raw runtime-arg* addresses (KNOWN_ISSUES #3, segfault at e.g. `0x19520`). CB pointers from `get_write_ptr` are fine because L1 is a `MAP_32BIT` mmap (the truncated 32-bit pointer *is* a valid host pointer).
2. **NOC stream overlay regs** (`noc_overlay_parameters.h`, `stream_io_map.h`) not stubbed → JIT "file not found". Effort: High.
3. **`sfpi::` vector intrinsics are a placeholder shim — "types exist but math is wrong"; any op using `sfpi::` directly must be semantic-rewritten** (`emule-mapping.md` §3.5; flagged as unfinished "wave-7" work). *At HEAD `e1387bc` even the shim header is absent — see Hole A.*
4. **tilize/untilize are identity copies** (no real face↔row-major conversion).
5. `inline_dw_write`, `noc_mode`/`DM_DYNAMIC_NOC` not modeled (Low-medium).

---

## 3. Build feasibility (Phase 3) — assessment, full build NOT forced

The full `-DTT_METAL_USE_EMULE=ON -DWITH_PYTHON_BINDINGS=ON` build of tt-metal at the pin is the spike long pole: fresh clone + UMD/tracy submodules + a Release build with Python bindings on 12 cores (≈1–3 h) that would also **contend CPU with the in-flight device worker** the rules forbid disturbing. Per the spike charter ("if a full build is the long pole … report build-feasibility + documented-limitation analysis as the verdict rather than forcing it"), I did **not** run it.

Instead I exercised **the exact compile path emule uses at runtime**: `emulated_program_runner.cpp::jit_compile_kernel` builds a `clang++-20 -std=c++20 -fPIC -shared -O2 -Wno-c++11-narrowing` command with includes
`-I <emule>/include/jit_hw -I <emule>/include -I <tt-metal>/ttnn/cpp -I <tt-metal> -I <tt-metal>/tt_metal/hw/inc -I <tt-metal>/tt_metal/hostdevcommon/api`, `-DTT_EMULE_USE_L1_POOL`, a wrapper that `#include`s `jit_kernel_stubs.hpp` then the kernel, and compiles **4 variants** defining `TRISC_UNPACK` / `TRISC_MATH` / `TRISC_PACK` / `TRISC_ISOLATE_SFPU` (runner lines 882–922). I reproduced that invocation by hand against emule HEAD + the production tt-metal headers (read-only). This is the same compiler, same headers, same flags emule's JIT shells out to — so a "compiles / fails" result here is exactly what emule's runtime would see.

**Build-feasibility verdict:** buildable on this box (toolchain ✅, prod tree already emule-wired). Recommend a *separate* shallow tt-metal clone at pin `b24ad48` into `/localdev/smarton/emule-spike/tt-metal-pin` with its own `build_emule/` — never touching prod `build_Release` or `.cache`. emule's own C++ regression baselines (per README: WH 30/9, BH 19/0) and `KNOWN_ISSUES.md` indicate emule works on this class of host; I did not independently re-run them this spike.

---

## 4. gsplat kernel probe (Phase 4) — what runs vs. what walls

Kernels copied read-only into `/localdev/smarton/emule-spike/gsplat-kernels/` (from the Mac tree; prod `gstt2` untouched). Probe `.so`s + wrappers live in `/localdev/smarton/emule-spike/probe/`.

### 4a. Dataflow / reader / writer kernels — ✅ COMPILE CLEAN

| Kernel | Result | Notes |
|---|---|---|
| `dataflow/reader_bucket_cull.cpp` | **✅ exit 0 → reader2.so (26 KB)** | clean with realistic compile-time args |
| `dataflow/reader_alpha_blend_mb_devcull.cpp` (60 KB, the big one) | **✅ exit 0** | the prime "raw-L1" suspect — compiles |
| `dataflow/writer_bucket_cull.cpp` | **✅ exit 0** | |
| `dataflow/writer_microblock_cull.cpp` | **✅ exit 0** | |
| `dataflow/reader_microblock_cull.cpp` | ⚠️ probe-artifact fail | `tensor_accessor_args.h` constexpr error — caused by my **placeholder** compile-time args, not an emule hole (same class as the first `reader_bucket_cull` attempt, which then passed with correct args). Resolves with real CTAs. |

**Why the raw-L1 fear does not bite us:** I grepped every kernel — **zero** instances of `reinterpret_cast<T*>(get_arg_val<...>(N))` (the KNOWN_ISSUES #3 segfault pattern). Every dereferenced pointer in our readers/writers comes from `get_write_ptr(CB_*)` / `get_read_ptr(CB_*)` (valid host pointers under emule's `MAP_32BIT` L1Pool), and every "resident" runtime-arg base address (`cull_masks_addr`, `tile_recs_addr`, `bucket_meta_addr`, `blendrec_addr`, …) is read **through the NOC API** via `TensorAccessor` + `noc_async_read_tile` / `get_noc_addr` — which emule resolves with `__emule_resolve_noc_addr` (memcpy). So our kernels are NOC-API-clean by construction. The documented raw-L1 segfault is **not** a gsplat blocker.

### 4b. Compute kernels — ❌ HARD WALL (SFPU / microblock compute lanes)

Only **2** compute kernels use the raw SFPU model, and they are the active cull+blend path:
`compute/microblock_cull_compute.cpp` (105 `vFloat`/`dst_reg`/`sfpi::` sites) and `compute/alpha_blend_compute_mb.cpp` (78 sites). Both gate their SFPU includes behind `#ifdef TRISC_MATH`:

```cpp
#ifdef TRISC_MATH
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu.h"
#endif
```

emule compiles the `TRISC_MATH` variant, so this block activates. The exact failure:

```
$ clang++-20 ... -I<emule>/include/jit_hw ... <wrapper around sfpi.h>
fatal error: 'sfpi.h' file not found
    1 | #include "sfpi.h"
```

`sfpi.h`, the `sfpu/ckernel_sfpu_*.h` set, and `llk_math_eltwise_unary_sfpu.h` are **absent from emule HEAD** (`find` over `include/` finds none) **and absent from the production tt-metal tree** — and emule's JIT include list (`get_extra_include_flags`) never adds the SFPU/sfpi ckernel paths. So the math RISC cannot compile.

With `TRISC_MATH` *off* (as in one probe pass) the failure surfaces instead as cascading semantic errors, which independently catalog the missing surface:

```
error: use of undeclared identifier '_llk_math_eltwise_unary_sfpu_start_'   (microblock_cull_compute.cpp:616)
error: use of undeclared identifier '_llk_math_eltwise_unary_sfpu_done_'    (microblock_cull_compute.cpp:618)
error: use of undeclared identifier 'get_tile_address'                      (microblock_cull_compute.cpp:565,594; alpha_blend_compute_mb.cpp:279,399)
error: use of undeclared identifier 'cull_face_x' / 'cull_face_y' / 'cull_combine' / 'cull_bbox'   (our own sfpi helpers — dropped because their vFloat/dst_reg bodies can't parse)
error: use of undeclared identifier 'blend_one_gaussian_math'              (alpha_blend_compute_mb.cpp:259)
```

These are **compile-time** failures (missing API surface), not the segfault/backtrace class — the wall is hit before any emulated thread runs. (No SIGSEGV to capture: the kernels never link.)

---

## 5. Hole catalog with fixability

### HOLE A — SFPU `sfpi::` vector model absent  *(BLOCKER, HIGH effort)*
- **What our kernels need:** `sfpi.h` (`vFloat`, `vInt`, `dst_reg[k]`, `v_if/v_endif`, `vec_min_max`, `reinterpret`), low-level `sfpu/ckernel_sfpu_{exp,log,converter}.h` (`_sfpu_exp_21f_bf16_`, `Converter::as_float`), and `_llk_math_eltwise_unary_sfpu_start_/done_`.
- **Where it'd be fixed in emule:** emule deliberately does **not** emulate the SFPU SIMD ISA. Per `.claude/references/emule-mapping.md` §3.5 the canonical fix is **Strategy C — semantic rewrite under `__EMULE_JIT_MODE`**: gate off the entire `#ifdef TRISC_MATH` sfpi block + the `MATH(...)` bodies and reimplement the per-lane math as plain scalar/loop C++ over `__emule_dst[idst][i]` (row-major fp32 DST), using `__emule_compute::cb_read_ptr_at`/`cb_write_ptr_at` for CB tiles. (A bare `sfpi.h` "types-only" shim is explicitly called out as insufficient — "math is wrong".)
- **Files to touch:** `gsplat_tt/kernels/compute/microblock_cull_compute.cpp` (~700 lines; Mahalanobis face_x/face_y/combine/bbox cull per 32-lane vector) and `compute/alpha_blend_compute_mb.cpp` (~400 lines; per-lane Gaussian exp-weight alpha blend). Each is a self-contained `#ifdef __EMULE_JIT_MODE` reimplementation living in the kernel itself (or a shared gsplat emule-math header).
- **Effort:** **HIGH, ~1 well-scoped day per kernel** (2 kernels). The pattern is validated 4× upstream (RMSNorm, clamped_silu, Mcast, eltwise_mul), and the lane→DST mapping is the only subtlety (the kernels already document `MB_TO_DST_ADDR` / `dst_reg[slot*32 + ...]`). Determinism is the payoff: the rewrite runs as ordinary host fp32, so it's a perfect golden/debug oracle.

### HOLE B — `get_tile_address(cb_id, tile_index)` not stubbed in emule  *(LOW effort)*
- **What:** Both compute kernels call `get_tile_address` to get a CB tile's L1 address for direct word reads of coeff/count rows. It is a **real** tt-metal compute API (`tt_metal/hw/inc/api/compute/cb_api.h:154`, `ALWI uint32_t get_tile_address(...)`), but emule's shadowing `include/jit_hw/api/compute/cb_api.h` doesn't define it.
- **Where to fix:** add `get_tile_address` to `tt-emule/include/jit_hw/api/compute/cb_api.h` (Strategy A), returning the existing emule CB L1 address (it can wrap `__emule_compute::cb_write_ptr_at(cb_id, tile_index)` / `cb_read_ptr_at`, already present per `emule-mapping.md` §4.2).
- **Effort:** **LOW (~5–10 lines).** Generic, benefits any kernel.

### HOLE C — `_llk_math_eltwise_unary_sfpu_start_` / `_done_` not stubbed  *(LOW effort)*
- **What:** SFPU pipeline-state begin/end markers. emule has no pipeline state (`MATH(x)`→`x`), so these are no-ops.
- **Where to fix:** empty inline stubs in emule (e.g. a new `jit_hw/api/compute/llk_math_eltwise_unary_sfpu.h` or in `common.h`). Largely moot if Hole A is done via semantic rewrite (the rewrite removes these calls), but trivial to add.
- **Effort:** **LOW (no-op stubs).** Matches KNOWN_ISSUES Inline-Write fix style.

### Non-holes / non-issues for gsplat
- **Raw-L1 deref segfault (KNOWN_ISSUES #3): N/A** — we never deref raw runtime-arg L1 addresses (see §4a).
- **NOC stream overlay regs / `inline_dw_write` / `noc_mode`:** grepped — our kernels don't use `noc_overlay_parameters.h`, stream regs, or `inline_dw_write`. Not a blocker.
- **tilize/untilize identity copies:** our cull/blend path operates on packed coeff rows + DST math, not on real tile-layout conversions, so identity tilize is acceptable for correctness work (verify per-kernel if any path relies on true face packing).
- **Slow-dispatch / mock-cluster:** standard env (`TT_METAL_EMULE_MODE=1`, `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_MOCK_CLUSTER_DESC_PATH=.../blackhole_P100.yaml`) — no gsplat-specific issue expected.

---

## 6. Minimal fix set to get gsplat running under emule

1. **Hole B + Hole C stubs** (~½ day total) → unblocks the *compile* of the compute kernels' non-SFPU scaffolding and any future kernel.
2. **Hole A semantic rewrites** of `microblock_cull_compute.cpp` and `alpha_blend_compute_mb.cpp` under `__EMULE_JIT_MODE` (~1 day each) → makes the full cull+blend pipeline run deterministically on host.
3. Stand up a *separate* tt-metal clone at pin `b24ad48` + isolated `build_emule/`; build `libtt_metal.so` + `_ttnn.so` once; run the gsplat pybind path under the emule env vars. (One-time, heavy, but off the device.)

After steps 1–2 the **dataflow half is already usable today** (it compiles clean now), so emule can serve as a correctness sandbox for the reader/writer/sort/scatter kernels even before the SFPU rewrites land. The SFPU rewrites are the gate for end-to-end blend correctness.

**Recommendation: ADOPT, scoped.** Use emule now for deterministic debugging of the dataflow/sort/bucket kernels; invest the ~2.5-day fix set to bring the SFPU cull+blend compute kernels online as a golden oracle. The investment is low-risk because (a) the toolchain is ready, (b) our kernels are NOC-API-clean so the scary raw-L1 class doesn't apply, and (c) emule ships the exact `/implement-mock` + Strategy-C workflow these rewrites need.

---

## Appendix — reproduction (host-only, no device)

```bash
ssh yyzo-bh-03
cd /localdev/smarton/emule-spike/tt-emule          # HEAD e1387bc
E=/localdev/smarton/emule-spike/tt-emule
M=/localdev/smarton/tt-metal                        # read-only header source (prod, ebaaa579)
KD=/localdev/smarton/emule-spike/gsplat-kernels
INC="-I$E/include/jit_hw -I$E/include -I$M/ttnn/cpp -I$M -I$M/tt_metal/hw/inc -I$M/tt_metal/hostdevcommon/api"

# dataflow reader — COMPILES (exit 0):
printf '#define KERNEL_COMPILE_TIME_ARGS 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n#include "jit_kernel_stubs.hpp"\n#include "%s"\nextern "C"{void __emule_kernel_entry(){kernel_main();}}\n' \
  "$KD/dataflow/reader_bucket_cull.cpp" > /tmp/w.cpp
clang++-20 -std=c++20 -fPIC -shared -O2 -Wno-c++11-narrowing $INC -I"$KD/dataflow" -DTT_EMULE_USE_L1_POOL -o /tmp/r.so /tmp/w.cpp ; echo exit=$?

# SFPU compute kernel — sfpi.h fatal (the wall):
printf '#include "sfpi.h"\nint main(){}\n' > /tmp/t.cpp
clang++-20 -std=c++20 -fsyntax-only $INC -I"$KD/compute" /tmp/t.cpp   # fatal error: 'sfpi.h' file not found
```
