# SFPU microblock-cull — next steps

Updated: 2026-05-31 (bh-30 **RECOVERED** via device reset — see recovery procedure below)

## bh-30 device recovery (known-good)

Symptom of a wedged chip: warmup + `[Real-time profiler] Device 0 sync complete` succeed,
but the **first real compute dispatch hangs forever** (host pins one core at ~100% in a
`Finish()` busy-wait, `futex_wait_queue`). This is a firmware/PCIe wedge, NOT a code bug —
clean HEAD hangs identically.

**Fix:** `tt-smi` 5.2.0 is in the venv (not on PATH). Reset the PCI device:

```
ssh bh-30 'pkill -9 -f a003_verify; sleep 2; rm -f /dev/shm/tt_*; \
  /localdev/smarton/gstt2/.venv/bin/tt-smi -r'
```

Reset takes ~60 s ("Resetting all PCI devices: [3] ... Re-initializing boards"). Confirm
health with `tt-smi -s` (`DDR_STATUS: 0x5555` = banks trained). Then re-run verify.

## Current local fix (uncommitted)

**File:** `src/gsplat_tt/kernels/compute/microblock_cull_compute.cpp`

**Change:** Match `alpha_blend_compute_mb.cpp` DEST lifecycle — **one `tile_regs_acquire()` per screen tile**, not per 32-gaussian batch.

| Before (buggy) | After (this fix) |
|----------------|------------------|
| `tile_regs_acquire()` at start of each batch | `tile_regs_acquire()` once after `CB_CULL_COUNTS` read |
| `tile_regs_release()` after each `pack_tile` | `tile_regs_commit()` / `tile_regs_wait()` per batch only |
| Box ramps copied only inside batch loop | Initial box copy after tile acquire; **reload** ox/oy before each SFPU batch (pack clobbers DEST) |
| Only `fill_tile(DR_KEEP)` | Also zero `DR_QV` / `DR_QH` scratch before each batch |

**Hypothesis:** Per-batch acquire/release broke math↔packer double-buffer handshakes when multiple `pack_tile` calls happen inside one tile. Symptoms matched s19: batch-granular **+32** association (`STORED[k] == ref[k+32]`), run-to-run nondeterministic 18–42 dB.

**Gate when device is back:** `GSPLAT_TT_SFPU_CULL=1` + full resident chain, `a003_verify.py` **×2** @ hero_vs_ref **63.85 dB** (30-view median ms/view should drop ~310 ms vs scalar cull).

---

## ITERATION 3 FINDING (2026-05-31, supervisor-driven) — root cause narrowed

Ran `GSPLAT_TT_SFPU_CULL=1 + SFPU_CULL_DUMPALL=1 + CULL_VALS=1`, DPRINT core (0,0),
dumped `(k, ref, sfpu)` for 1600 candidates. **23 mismatches**, and they cluster
at **batch boundaries** (SFPU batch = 32): mismatched `pos` values are
0, 32, 33, 34, 96, 97, 128–133 (i.e. the first 1–6 candidates after a multiple
of 32), varying per tile. Everywhere else `sfpu == ref` bit-exact.

**Conclusion:** NOT a uniform +32 shift, NOT a DRAM-placement bug (iter-1 KVAL =
0 mismatches), NOT a nondeterministic race (iter-2 = deterministic 29.7 dB ×2).
It is **deterministic mask corruption of the first few gaussians of (some) 32-wide
SFPU batches** — a batch-boundary data hazard in the cull compute/reader handshake
(follow-ups B/C below: coeff CB page overlap when reader runs ahead, and/or stale
box-ramp / DEST scratch for the first lanes of a new batch). The reverted per-tile
`tile_regs_acquire` change addressed DEST lifecycle but not this boundary hazard.
Fix target: ensure the first lanes of every batch get freshly-written coeff +
box-ramp (no read-before-write, no stale DST) — see B (tie reader chunk to BATCH=32
or add back-pressure) and C (zero scratch for partial/leading lanes).

## ITERATION 4 FINDING (2026-05-31) — NOT compute-order; suspect stale coeffs

Reversed the SFPU dispatch order (V=31->0, `GSPLAT_TT_CULL_REVERSE=1` ->
`CULL_REVERSE_DISPATCH`, packing position DR_KEEP[V] unchanged). Mismatches
**stayed at low V** (V=0,1,2; pos 32/64/96->V0, 33/65/97/129->V1, 98->V2) — they
did NOT follow compute order to high V. PSNR barely moved (42.19 -> 42.95 dB).

**Therefore the corruption is tied to batch-LOCAL position V (the first 1-3
gaussians of a batch), independent of when they compute.** Ruled out: compute
order, copy_tile->SFPU settle-race (reverse gives low-V MORE settle time, still
wrong), DRAM placement (iter-1 KVAL), box geometry (DUMPBOX). What remains and
reads the same `a[V]/b[V]/.../op[V]` regardless of order: the **per-gaussian coeff
arrays loaded from `CB_CULL_COEFF`** at the top of each batch (compute lines
291-301). Leading hypothesis: a cross-RISC producer(reader)->consumer(compute)
visibility/ordering issue on the FIRST coeff rows of (some) batches — compute's
`cb_wait_front(CB_CULL_COEFF,1)` returns before the reader's L1 row store for that
slot is visible, so `a[0..2]` are stale.

**Next experiments:**
1. In compute, dump `a[V],b[V],c[V],mx[V],my[V],op[V]` for the wrong low-V slots
   (extend CULLVAL beyond first-3-batches) and compare to the reader's CULLIN g +
   the resident SoA values for that g. If coeffs are stale -> confirmed.
2. Fix candidates: (a) add `noc_async_read_barrier()` / explicit fence is N/A
   (direct L1 store) — instead verify CB_CULL_COEFF depth + that the reader does
   not advance the write ptr before the store retires; (b) bump CB_CULL_COEFF
   depth; (c) have compute re-read the row twice / add a CB credit handshake.
3. Cheap isolation: `GSPLAT_TT_CULL_KEEPALL=1` (keepv=1) — if masks become all-1
   correctly, math path is fine and bug is purely coeff/keep transport.

## ITERATION 5 FINDING (2026-05-31) — ROOT CAUSE: data-dependent SFPU math, NOT transport

Dumped compute-received coeffs (`CULLVAL`, EMIT_M2) vs reader-written coeffs
(`CULLCOEF`, SELFCHECK) at matching local positions. **They are byte-identical**
(local=0: a=21365.5 b=-3966.93 c=4066.68 mx=54.6123 my=799.237 op=0.0293077 on
BOTH sides; same for local 1,2,32,33,34). **CB_CULL_COEFF transport is correct.**

=> The wrong masks come from the **SFPU cull math**, and it is **data-dependent**:
the affected gaussians have extreme / near-degenerate covariance. Example wrong
gaussian: a=32.77, c=141.62, b=-67.74 -> a*c=4641.7, b*b=4588.7, det≈53 (det/ac
≈ 0.011, catastrophic cancellation). The combine `approx_recip(det)` (2 Newton
iters off the bare HW seed) and the per-face `approx_recip(cov_a/cov_c)` lose
precision for these, flipping `keep = (m2 <= thresh)` near the boundary.

**The earlier "low-V / batch-leading" clustering was a SAMPLING ARTIFACT** (DUMPALL
caps + depth-sort order), not a structural batch hazard. Iters 3-4 already ruled
out race / +32 shift / placement / compute-order; iter 5 rules out coeff transport.

**CONFIRMED-ELIMINATED:** nondeterministic race, +32 DRAM shift, compute-order/
copy_tile->SFPU hazard, box geometry, coeff transport.

**Next:** confirm + fix the numeric path.
1. Compare SFPU `m2` vs scalar `m2` for the degenerate gaussians (EMIT_M2 writer
   read-back vs host `compute_microblock_mask` m2) to prove it's the reciprocal.
2. Fix candidates (cheapest first): (a) add a 3rd Newton iteration to
   `approx_recip(det)` and the face recips; (b) compute `det` more stably; (c)
   guard the near-degenerate branch. Re-gate ≥63.6 dB ×2, blend ~110ms.

## ITERATION 6 (2026-05-31) — recip fix NO-OP; perf CONFIRMED

Added a 3rd Newton iteration to all four `approx_recip` (det, cov_a, cov_c, floor)
in `microblock_cull_compute.cpp`. **No change: 29.64 dB** (== iter-2's 29.7). So
**reciprocal precision is NOT the bug** (2 Newton iters from the HW seed already
give full fp32, as expected). The extra iteration is harmless; can revert later.

**PERF CONFIRMED on the same run:** `ms/view=349.5 (proj=61.2 ta=119.7 sort=57.5
blend=110.0)`. SFPU cull cuts blend **422 -> 110 ms** and the frame **~660 -> ~350
ms (1.9x)**. The entire ~310 ms/view win is gated solely on fixing the cull-math
correctness (masks deterministically wrong -> 29.6 dB).

Remaining suspects (recip eliminated): (a) constrained-min Mahalanobis geometry
(two-face edge projection in cull_face_x/y) systematically off; (b) `log` threshold
`_calculate_log_body_no_init_`. 29.6 dB is a LARGE error (many masks wrong), which
argues for a SYSTEMATIC geometry error over near-boundary log flips. NOTE: the old
"SELFCHECK m2 diff < 0.55 on tile 584" was a non-representative 100-gaussian subset.

**Next:** run CULL_SELFCHECK (reader scalar m2 vs SFPU m2) across many tiles; if m2
diverges broadly -> geometry bug in cull_face_x/y; if m2 matches but keep flips ->
threshold/log. Then fix and re-gate >=63.6 dB x2 (perf already proven).

## ITERATION 7 (2026-05-31) — PIVOTAL: formula CORRECT, device EXECUTION wrong

Ran `opt/cull_check.py` (offline float64 harness: scalar `ref_mask` with
x_inside/y_inside + AABB; `sfpu_mask` = the kernel's unconditional min(Qv,Qh) over
all 32). Result for every test gaussian:
- `sfpu_py == ref_py` -> **TRUE**: the SFPU FORMULA reproduces the scalar reference
  exactly in float64. The geometry / min(Qv,Qh) / threshold math is CORRECT.
- `sfpu_dev != sfpu_py`: the on-DEVICE mask differs grossly (e.g. g1196991
  sfpu_py=0x00888888 pop6 vs sfpu_dev=0x00000008 pop1; g1544670 0x000fffff pop20 vs
  device 0xffffffff pop32).

Per-microblock m2 (offline vs device): MATCHES for a=18558 (err 8e-5) and a=0.666
(err 0); GROSSLY WRONG for a in ~[21,45] (err ~4600). The wrong device m2 for an
a=32.77 gaussian numerically equals the OFFLINE m2 of the a=0.666 gaussian — i.e.
the device produced another gaussian's / wrong-magnitude result.

**Conclusion: this is an SFPU CODEGEN / EXECUTION bug, not math, not inputs, not
placement.** The correct float64 formula is mis-executed on hardware for mid-range
covariance. Prime suspects (kernel header lines 92-94 footgun): a `Converter::
as_float(...)` emitting a NON-uniform per-lane load instead of a broadcast for some
value range; DR_QV/DR_QH DEST-scratch cross-vector aliasing in the split noinline
funcs; or `vec_min_max` semantics vs the assumed clamp for specific magnitudes.

**Next experiments (device):**
1. Bisect: `sfpu_oneface` exists in cull_check — on device, emit Qv-only vs Qh-only
   vs min to see which face diverges (g1196991: Qv=0x888800, Qh=0x88888, min=
   0x888888 offline; compare device).
2. EMIT raw DR_QV / DR_QH per (g,m) for an a~30 gaussian; compare to offline Qv/Qh
   -> pinpoint which of face_x / face_y / combine mis-executes.
3. Check each `cs::Converter::as_float` materialization + the `vFloat(dst_reg[...])`
   reads compile to uniform broadcasts (inspect generated SFPU asm if needed).

## Ruled out (s19 diagnosis — do not re-litigate)

1. **Cull → blend cross-pass race** — `Finish()` at end of cull pass fences before blend (`blend_device.cpp`).
2. **Lane ↔ microblock geometry** — `GSPLAT_TT_CULL_DUMPBOX`: 0/512 BAD; `perm(g,m)` round-trip bit-perfect.
3. **Per-lane m2 numeric** — `GSPLAT_TT_CULL_SELFCHECK` (100 gaussians, tile 584): masks match scalar, m2 diff &lt; 0.55.
4. **Allocator overlap** — cull DRAM now allocated from **blend** `DeviceContext` (not separate cull ctx).

---

## If tile-acquire fix is insufficient — ordered follow-ups

### A. CB_KEEP producer/consumer (highest prior s19 suspect)

- **Symptom:** `CULLW` (writer store) vs `CULLMM` (blend read) off by **+32** on first gaussians of some batches; which side is wrong varies per run.
- **Checks:**
  - Confirm compute `cb_push_back(CB_KEEP, 1)` count == writer `cb_wait_front`/`cb_pop_front` per tile (no extra/missing push on partial final batch).
  - Bump `CB_KEEP` depth in `cull::build_program_and_workload` (currently **4**) only if profiling shows writer falling behind compute (unlikely at depth 4 for serial batch loop).
  - Add compile-time assert path: writer records `processed` + `nb` in DPRINT; blend `CULLMM` must match `id_start + processed + g`.
- **Experiment:** `GSPLAT_TT_CULL_KVAL=1` — writer stores global index `k` instead of mask; blend reader must read exact `k` (isolates DRAM placement from SFPU math).

### B. Reader ↔ compute coeff ordering

- Reader emits coeff rows in **chunks of ≤16** (`CHUNK_MAX`); compute consumes **batches of 32**.
- `CB_CULL_COEFF` depth **32** — verify no overlap when reader runs ahead on fast tiles (same class of bug as s5 big-page reader writing into `CB_MB_COUNTS`).
- **Mitigation:** Tie reader chunk size to `BATCH` (32) or add explicit back-pressure (compute-driven credits).

### C. Partial final batch / mask tail

- Writer pads mask write to whole pages (`nb_pad`); compute zero-fills unused SFPU vectors — confirm writer does not read stale `keep[perm(g,m)]` for `g >= nb` from previous batch’s CB page (should be zero from compute `fill_tile(DR_KEEP)`).
- **Experiment:** `GSPLAT_TT_CULL_KEEPALL=1` — if PSNR still noisy, race is not in Mahalanobis math.

### D. Writer DRAM page alignment (reverted once — re-validate only if +32 shifts by 16)

- Masks are page-aligned via `cull_mask_base[tile]`; writer uses whole-page `noc_async_write`.
- Re-check only if diagnostics show **16-element** shifts (not 32).

### E. Fuse cull into blend reader (structural, post-correctness)

- Eliminate separate 3-kernel pass + `cull_masks` DRAM round-trip.
- Reader loads precomputed mask page with gaussian gather (one NoC per gaussian).
- Depends on stable mask path above.

### F. Numeric edge (low priority after s19 selfcheck)

- Threshold uses `_calculate_log_body_no_init_(opacity/floor)` per gaussian (scalar in combine template) — already avoids per-lane bf16 `exp`.
- If borderline masks still differ: compare against scalar `log` threshold in host offline on `M2DMP` dumps.

---

## Diagnostic env (retained in tree)

| Env | Purpose |
|-----|---------|
| `GSPLAT_TT_SFPU_CULL=1` | Enable pass |
| `GSPLAT_TT_CULL_WDUMP` / blend `CULLMM` | Writer vs reader placement |
| `GSPLAT_TT_CULL_KVAL` | Store `k` not mask |
| `GSPLAT_TT_CULL_DUMPBOX` | Box-origin geometry |
| `GSPLAT_TT_CULL_SELFCHECK` | Scalar mask compare |
| `GSPLAT_TT_CULL_EMIT_M2` | Per-lane m2 dump |
| `GSPLAT_TT_SFPU_CULL_{DEBUG,DUMPALL,ALLONE,BLOCKING,USEREF}` | Blend-side isolation |

---

## Device ops (when bh-30 recovers)

1. Kill stray verify: `ps aux | grep '[a]003_verify' | awk '{print $2}' | xargs kill -9` — **never** `pkill -f a003_verify`.
2. One job at a time: `scripts/a003_verify.py` or `scripts/bh30_recover_and_verify.sh` (not parallel).
3. Record in `opt/metal-iters.jsonl` only after **×2 deterministic** 63.85 dB gate.

---

## Perf expectation (already measured s19, pre-correctness fix)

- Scalar devcull blend ~422 ms → SFPU cull ~37 ms + blend ~73 ms (~110 ms blend stage).
- ~310 ms/view win at 30-view scale once correctness holds.
