# t562: B4a fold hang under p150 ETH dispatch

## Finding
- The iter-226 bench (docs/b4a-t554) ran on bh-33, which is a **p100a**, with
  **worker dispatch, 11x10 grid** (all 12 logs: `[eth] card p100a: worker dispatch`).
  The viewer on bh-30 is a **p150, ETH dispatch, 12x10**. So the iter-226 headline
  config does **not** match the viewer's, and the fold was never run under ETH
  dispatch before landing.
- The hang (docs/viewer-t556 on ttp/t556-viewer-deploy-226 @0c06a012) could not be
  reproduced here: the only reachable p150 is the viewer box bh-30, which a
  measurement task may not use. Root cause (ETH dispatch vs 120-core grid) is open.

## Fix (1e0daf5b)
- `b4a_fold_on()` in render/host/dispatch_select.h: fold defaults **off under ETH
  dispatch**, on under worker dispatch. `GSPLAT_TT_B4A_FOLD=1` / `=0` still force it.
- device_state records `eth_dispatch`; tile_assign_device.cpp uses the gate.
- tests/unit/test_dispatch_select.cpp covers env on/off/unset x ETH/worker.

## p100 check that the gate is a no-op there (bh-33, bc2b1a9a, 2026-10-12 PDT)
| check | result |
|---|---|
| fold verify (latency, 3 views) | 0/3 views bad |
| fold verify (b2b) | OK |
| device hero sweep md5 | 906e0435, 30/30 views identical |
| hero PSNR vs benchmarks/reference_v2/hero.png | 42.51 dB, golden match, max 0 LSB |
| eye check hero.png + hero_diff10.png | no tile seams/artifacts |
| b2b fold off (worker 11x10) | 8.993 ms/view (111.2 FPS) |
| b2b fold on (worker 11x10, default) | 8.591 ms/view (116.4 FPS) |

## Open
- p150 ETH A/B (fold forced on, with GSPLAT_TT_B4A_VERIFY) needs a p150; on bh-30 it
  needs a task holding resource 'viewer' (user exception 2026-10-07).
