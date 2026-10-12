# t511: FPU quadratic-form blend alpha, device timing (run 2026-10-12: shelve)

Task #511, 2026-10-12 (PDT), base best-iter-223 (5ca019ae) via cd33efcd.

#510 (ttp/t510-lever-fpu-quadratic-form-blend-alpha-cpu, e6c4d158) pushed only the CPU
precision model and a cost model, no device code. Its cost model puts the FPU form at break-even
or worse: per live record it saves ~41-63 math-thread cycles (x/y loads + 13-op front per
microblock, ~3 microblocks per record) and costs ~59-118 (tile matmul at HiFi4, coefficient
operand build, SFPU<->FPU hand-offs). A working FPU path also needs TRISC0 (now busy with
decode-ahead) to feed srcA/srcB or a DEST->src move per record. So this task first measures, on
device, the two numbers that decide it, before anyone builds the real path:

| env (default 0) | what it does | measures |
|---|---|---|
| `GSPLAT_TT_BLEND_QF_PROBE=1` | SCHED=2 single body runs x/y + front twice (same LREGs: bit-identical) | S = what an FPU power removes from the SFPU (ceiling of the saving) |
| `GSPLAT_TT_BLEND_QF_PROBE=2` | front replaced by one SFPLOAD of the x ramp (output wrong); paired with `BLEND_T_PERIOD=0u` | cross-check of S, as #111 did on the old path |
| `GSPLAT_TT_BLEND_QF_PAD=32` | 32 SFPNOPs per live record on the math thread | ms/view per cycle of per-record work, to price the FPU side |

Break-even: the FPU form wins only if its per-record math-thread cost in cycles is below
S / (pad32 delta / 32). With #510's 59-118 cycles that decides the lever without a full build.

## Run

```
[HOST=<measurement box>] ttp detach t511 -- ttp lock p100 -- docs/blend-qf-t511/drive511.sh <rev>
```

Arms off / f2 / pad32 / p0 / q2, one warm-up, 3 rotated b2b rounds (`render/run.py
--back-to-back`, headline = median), one latency run per arm (stage split, secondary only), and
`--dump-views` runs for off/f2/pad32 (md5; f2 must match off) with the device hero copied out.

## Result (2026-10-12 08:49-08:54 PDT, bh-33 p100a aiclk 800, IRD 138447, sha 96a5ab7e)

Headline: `render/run.py --back-to-back`, 30 bicycle views, 20 passes per run, median ms/view.

| arm | r1 | r2 | r3 | median | delta vs off |
|---|---|---|---|---|---|
| off | 9.363 | 9.362 | 9.372 | 9.363 | - |
| f2 (front twice) | 9.819 | 9.827 | 9.818 | 9.819 | +0.456 |
| pad32 (32 SFPNOP/record) | 9.827 | 9.812 | 9.809 | 9.812 | +0.449 |
| p0, q2 | build failed | | | | |

Latency stage split (secondary, one run each, blend ms/view): off 5.784, f2 6.099, pad32 6.158.
Only blend moves; project/sort differences are run-to-run noise.

md5: off, f2 and pad32 dump the same 30 views (list 906e0435 = golden, 30/30). Device hero
(`out/hero-off.png`, byte-identical in all three arms) PSNR 42.51 dB vs benchmarks/reference_v2/hero.png;
diff x10 in `out/diff-off-x10.png`: edge/spoke residue only, no tile seams.

p0/q2 did not build: with `BLEND_T_PERIOD=0u` GCC flags `g_seen % kBlendTPeriod` in the dead branch at
alpha_blend_compute_mb.cpp:1512 as division by zero under -Werror. The cross-check is not needed: f2 gives S.

### Verdict: shelve

- One cycle of per-record math-thread work costs 0.449 / 32 = 0.014 ms/view.
- S (all the SFPU work an FPU quadratic form could remove: x/y loads + front) = 0.456 ms/view, i.e. about 32
  cycles per record.
- So the FPU form wins only if its whole per-record math-thread cost (matmul, coefficient operand build,
  SFPU<->FPU hand-off through DEST/src) stays under about 32 cycles. #510 prices it at 59-118 cycles: a net
  loss of about 0.38-1.2 ms/view. Even a free matmul would need a DEST->srcA move per record (~16+ cycles)
  plus the coefficient tile build, which leaves no room.
- A cheaper form would have to remove the front without adding per-record hand-offs, e.g. by computing
  the exponent for a whole batch of records in one matmul ahead of blend on another core. That moves
  the cost into data movement (one 32x32 fp16 tile per record per pixel block) and is not promising here.
