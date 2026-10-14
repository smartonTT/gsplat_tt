# Training-view bench (task #667) — status: blocked on camera poses

Goal: measure best-iter-229 b2b ms/view on bicycle's own dataset cameras (training split in
seeded random order, and the every-8th test split) at 1237x822, next to the orbit control.

## Status (2026-10-13 19:50 PDT)
No bicycle camera poses are reachable from the allowed places:
- repo (`git ls-files`) and the project root: only `benchmarks/cameras_v2.json` (orbit) and the old
  `benchmarks/cameras.json`; the hero there is one dataset pose (`_DSC8679`), no others.
- bh-33 dirs named by repo scripts (`/localdev/smarton/gstt2-t513`, `gstt2`, `p150bench`):
  only `scenes/bicycle.ply` and `cameras_v2.json`. `bicycle.ply` was rsync'd from the Mac
  (docs/meas-host-t320) without its training output folder.
Missing: the Mip-NeRF360 bicycle COLMAP model (`360_v2/bicycle/sparse/0/{cameras,images}.bin`)
or the 3DGS training output `cameras.json` that came with `bicycle.ply`; for PSNR against the
dataset also `360_v2/bicycle/images_4/` (1237x822 JPEGs).

## Ready tooling
- `scripts/training_cameras.py --colmap <sparse/0> | --gs-cameras <cameras.json> [--seed 0]` writes
  `benchmarks/cameras_train_shuffled.json`, `cameras_test.json` and `cameras_orbit_1237.json`
  (orbit at 1237x822, control). Intrinsics: the dataset's fx/fy/cx/cy scaled to 1237x822, stored
  as `"intrinsics"`; split = sorted names, index % 8 == 0 is test. It checks that `_DSC8679`
  matches the cameras_v2 hero c2w (synthetic round-trip test: 6e-8).
- `render/run.py`: a camera file's optional `"intrinsics"` {fx, fy, cx, cy} overrides the
  fov-derived K (main sweep and the CPU hero reference). Select the bench with `--cameras <file>`.

Note: the headline orbit bench (cameras_v2.json) renders at 1024x1024 (fov 50 on the long side),
not 1237x822; `cameras_orbit_1237.json` gives the orbit at the GPU-comparison size.
