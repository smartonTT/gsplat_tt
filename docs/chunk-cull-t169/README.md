# Task #169: chunk frustum cull before pfwc

Goal: shorten pfwc (2.98 ms/view at the 18.59 ms tip, yyzo-bh-07 p100a) by not
reading or projecting 1024-gaussian tiles that cannot hold a visible gaussian.

## Step 1: skip share on the Mac (bicycle, 30 views, no device)

`skipshare.py` (box = means only, an upper bound) and `skipshare_safe.py` (the
conservative test the device path uses). Share of N skipped, mean of 30 views:

| order / chunk | means-box (UB) | safe K=1.25 | safe K=1.5 | safe K=2.0 |
|---|---|---|---|---|
| ply, any size | 0.000 | - | - | - |
| Morton 256  | 0.638 | 0.505 | 0.501 | 0.469 |
| Morton 1024 | 0.596 | 0.400 | 0.394 | 0.356 |
| Morton 4096 | 0.524 | 0.266 | 0.258 | 0.218 |

x 2.98 ms: Morton 1024, K=1.5 = 1.17 ms upper bound. 0 visible gaussians lost
on every view and setting. 27.7% of N is visible on average. The ply order has
no spatial locality, so the cull needs the Morton reorder. 1024 matches the
pfwc tile, so the cull needs no change to the work unit. K=1.5 is the default
(`GSPLAT_TT_CHUNK_K`).

## Conservative test

Per gaussian, rho = 3 sqrt(trace cov3d) >= 3 sigma_max. Per tile: box = union of
mean +- K rho, and rho_max. Per view, with camera-space box corners:

- skip if every corner has z <= k_near (0.2), or
- skip if every corner lies beyond one side plane x = U0 z (U0 = (W - cx + 5)/fx;
  the same for -x, +y, -y), and rho_max / max(zmin, k_near) <= s_K, with
  u_K = sqrt(K^2 (1 + U0^2) - 1) and s_K = (u_K - U0) / sqrt(1 + u_K^2).

Why it holds: a gaussian inside the box but beyond the plane has its centre at
u = x/z > U0. Its 3-sigma sphere has radius rho <= rho_max, at depth >= zmin.
The device Jacobian is not clamped, so the linearised 2D extent is at most
(f/z) rho sqrt(1 + u^2). That extent reaches back over the image edge only if
rho/z > s_K. The 5 px pad covers the 0.3 dilation (3 sqrt(0.3) = 1.6 px), the
ceil of the radius and the rounding. Non-finite members mark a tile "never skip".

## Device path (GSPLAT_TT_CHUNK_CULL=1, 0 = kill switch)

- `render/run.py`: Morton-orders the scene once (`morton_reorder`).
- `pfwc_device.cpp`:
  - `chunk_table` builds the per-scene box table, cached by means pointer.
  - `chunk_survivors` runs per view.
  - The survivors are dealt strided over the pfwc cores (core c: list[c],
    list[c+C], ...). So each core's count stays <= the full-scene SeqMap count
    and its segment base, which K2 recomputes, still holds.
- The tile ids go as runtime args, padded to ceil(num_tiles/C), because the
  argument count cannot change between launches:
  - `reader_pfwc.cpp` args 13.. under `PFWC_TILE_LIST`;
  - `writer_pfwc_fuse.cpp` args 25.. under the same define.
- `GSPLAT_TT_CHUNK_SKIP=0` keeps all tiles: reorder only, a diagnostic.
- `GSPLAT_TT_CHUNK_LOG=1` prints the kept count.
- Host cost is the `project_pfwc_chunkcull` stage timer.

## Device A/B

`drive.sh` + `remote_job.sh` (see results below).
