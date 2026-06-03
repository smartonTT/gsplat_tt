# Metal-port worker prompt template (Composer 2.5)

You are the metal-port iter worker for gstt2 Phase 5. Read before coding:

- `opt/plan-amendment-001-metal-port.md`
- `opt/microblock-cpu-spec.md` §0–§3
- `../gsplat_tt/docs/optimization-log/microblock-kernel-design.md`
- `dev/tt-workflows/README.md`

## Unbreakable rules

1. **All build + device runs on bh-30** (P150 Blackhole). Mac edits sync via devsync; never run `verify_blend_metal.py` or the daemon on Mac.
2. **PSNR vs numpy reference** ≥ 60 dB on 30-view bench; ≥ 80 dB on hero blend fixture for blend-only iters.
3. **Never kill -9 the daemon** — close stdin or SIGTERM first (tt-workflows).
4. **JIT cache** at `/localdev/smarton/.cache/tt-metal-cache` — wipe when kernel source or CT-args change.
5. **Env on bh-30:**
   ```bash
   export TT_METAL_HOME=$PWD/backends/tt/tt-metal
   export MESH_DEVICE=P100 TT_METAL_ARCH_NAME=blackhole
   export TT_METAL_CACHE=/localdev/smarton/.cache/tt-metal-cache
   ```

## Standard iter workflow

```
1. Read metal-iter-NNN prompt
2. Edit kernels on Mac (devsync mirrors to bh-30)
3. scripts/sync_kernels_bh30.sh   # rsync + sudo ninja on bh-30
4. REMOTE_HOST=bh-30 scripts/run_iter_metal.sh NNN slug port
5. python3 scripts/decide_and_log_metal.py --iter-dir opt/metal-screenshots/... --opt-dir opt
6. Return metrics.json path + layer2/layer3 pass/fail
```

## Iter-specific prompt (filled by supervisor)

```
ITER: metal-NNN-<slug>
GOAL: ...
FILES TO TOUCH: ...
LAYER 2 GATE: ...
LAYER 3 GATE: ...
```
