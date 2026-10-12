# t511: FPU quadratic-form blend alpha, device timing (not yet run)

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

## Status

Not run: at 02:10 PDT the bh-33 measurement reservation (IRD 138447) was gone (ssh preflight
first unreachable, then `auth_failed`; `opt/ird/renew.sh measure --dry-run` plans a new
reservation on bh-50). This task may not reserve, so it stopped here.
