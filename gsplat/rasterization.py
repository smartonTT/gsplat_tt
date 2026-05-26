import time
from contextlib import contextmanager

import numpy as np
import torch

from gsplat.utils import build_covariance_3d


# Sub-stage timer used by sort_and_bin / prepare_kernel_inputs. Both
# accept an optional dict; when one is passed in, each sub-step records
# its wall-clock millis into the dict. This is a permanent profiling
# hook — Pipeline.render() always passes a dict so every frame's
# breakdown is captured. Overhead is one perf_counter call per zone
# (~50 ns), negligible against the millisecond-scale stages.
@contextmanager
def _sub_timer(d: dict[str, float] | None, name: str):
    if d is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        d[name] = d.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0


def project_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    opacities: torch.Tensor | None = None,
    min_opacity: float = 1.0 / 255.0,
    sub_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D Gaussians to 2D screen-space ellipses.

    This implements the EWA (Elliptical Weighted Average) splatting approach:
    1. Build 3D covariance from scale + rotation
    2. Transform Gaussian centers to camera space using the view matrix
    3. Cull Gaussians behind the camera or outside the frustum
    4. Approximate the projective transform with a first-order Taylor expansion (Jacobian)
    5. Project 3D covariance to 2D using: Σ_2D = J @ W @ Σ_3D @ W^T @ J^T
    6. Compute screen-space position using the intrinsic matrix

    Args:
        means: (N, 3) Gaussian centers in world space.
        scales: (N, 3) Gaussian scales.
        rotations: (N, 4) unit quaternions (w, x, y, z).
        extrinsics: (4, 4) world-to-camera transformation matrix.
        intrinsics: (3, 3) camera intrinsic matrix.
        image_height: output image height in pixels.
        image_width: output image width in pixels.
        opacities: optional (N,) per-Gaussian opacity. If provided, Gaussians
            with opacity < min_opacity are culled — their peak pixel
            contribution would be below 8-bit quantization anyway.
        min_opacity: opacity threshold (default 1/255 = 0.0039). Only used
            when `opacities` is provided.

    Returns:
        means_2d: (M, 2) screen-space positions of visible Gaussians.
        covs_2d: (M, 2, 2) 2D covariance matrices.
        depths: (M,) camera-space depth values.
        radii: (M, 2) axis-aligned 3σ bounding-box half-extents (rx, ry) in
            pixels. Strictly tighter than a circular radius for elongated
            Gaussians (off-diagonal cov), identical for circular ones.
        valid_mask: (N,) boolean mask indicating which input Gaussians are visible.
    """
    N = means.shape[0]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # --- Step 1: Build 3D covariance matrices ---
    with _sub_timer(sub_timings, "project.cov3d"):
        cov3d = build_covariance_3d(scales, rotations)  # (N, 3, 3)

    # --- Step 2: Transform Gaussian centers to camera space ---
    with _sub_timer(sub_timings, "project.cam_xform"):
        R = extrinsics[:3, :3]  # (3, 3)
        t = extrinsics[:3, 3]   # (3,)
        means_cam = means @ R.T + t  # (N, 3) @ (3, 3) + (3,) -> (N, 3)

        # --- Step 3: Frustum culling ---
        near = 0.2
        valid_mask = means_cam[:, 2] > near  # z > near plane

    # --- Step 4: Compute screen-space positions (exact pinhole projection) ---
    with _sub_timer(sub_timings, "project.screen_xy"):
        tx, ty, tz = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
        means_2d = torch.stack([fx * tx / tz + cx, fy * ty / tz + cy], dim=-1)  # (N, 2)

    # --- Step 5+6: Compute the Jacobian + project 3D covariance to 2D ---
    with _sub_timer(sub_timings, "project.cov2d"):
        tz2 = tz * tz

        J = torch.zeros(N, 2, 3, dtype=means.dtype)
        J[:, 0, 0] = fx / tz
        J[:, 0, 2] = -fx * tx / tz2
        J[:, 1, 1] = fy / tz
        J[:, 1, 2] = -fy * ty / tz2

        # Σ_2D = J @ R @ Σ_3D @ R^T @ J^T
        cov_cam = R @ cov3d @ R.T
        covs_2d = J @ cov_cam @ J.transpose(1, 2)

        # Add low-pass filter for anti-aliasing.
        covs_2d[:, 0, 0] += 0.3
        covs_2d[:, 1, 1] += 0.3

    # --- Step 7: Compute per-axis AABB radii from the 2D covariance ---
    # For Σ_2D = [[a, b], [b, c]], the diagonals a and c are the variances
    # along screen-x and screen-y. The axis-aligned σ-bbox is therefore
    # rx = k*sqrt(a), ry = k*sqrt(c) regardless of the rotation encoded in b.
    #
    # k is chosen *per Gaussian* from its opacity ω. The peak contribution at
    # Mahalanobis distance d² (in σ units) is ω * exp(-d²/2). Setting this
    # equal to the 8-bit perceptual floor 1/255 yields:
    #     d² = 2 * ln(ω * 255)   →   k(ω) = sqrt(2 * ln(ω * 255))
    # For ω≥1/255 this is well-defined; the min_opacity cull already drops
    # ω<1/255. We additionally clamp k ≤ 3 so high-opacity Gaussians never
    # grow beyond the iter-022 3σ baseline (k(1.0)=3.33 without the cap).
    # On stitch_doll (most Gaussians have ω∈[0.01, 0.5]) this cuts another
    # ~11% (gaussian, tile) pairs on top of iter-022 — measurement:
    # scripts/measure_splat_count.py.
    with _sub_timer(sub_timings, "project.radii"):
        a = covs_2d[:, 0, 0]
        c = covs_2d[:, 1, 1]
        if opacities is not None:
            arg = torch.clamp(opacities * 255.0, min=1.0)
            k = torch.clamp(torch.sqrt(2.0 * torch.log(arg)), max=3.0)
        else:
            k = torch.full_like(a, 3.0)
        rx = torch.ceil(k * torch.sqrt(torch.clamp(a, min=0.0)))
        ry = torch.ceil(k * torch.sqrt(torch.clamp(c, min=0.0)))
        radii = torch.stack([rx, ry], dim=-1)  # (N, 2)

        # Also cull Gaussians that project entirely outside the screen.
        # Use per-axis radii so an elongated Gaussian whose minor axis crosses
        # the edge is still kept based on the relevant axis.
        valid_mask = valid_mask & (means_2d[:, 0] + rx > 0)
        valid_mask = valid_mask & (means_2d[:, 0] - rx < image_width)
        valid_mask = valid_mask & (means_2d[:, 1] + ry > 0)
        valid_mask = valid_mask & (means_2d[:, 1] - ry < image_height)
        valid_mask = valid_mask & (rx > 0) & (ry > 0)

    # Cap the bounding radius. The Jacobian linearization of the perspective
    # transform (Step 5 above) breaks down when a Gaussian's 3D extent is
    # comparable to its distance from the camera — producing wildly wrong 2D
    # covariances and massive bounding boxes. Visually these show up as
    # giant fuzzy blobs right in front of the camera when zooming in.
    # Drop any Gaussian where either AABB half-extent exceeds half the smaller
    # image dim, since a single splat covering more than half the viewport is
    # almost always an artifact, not real geometry.
    max_radius = min(image_height, image_width) // 2
    valid_mask = valid_mask & (rx <= max_radius) & (ry <= max_radius)

    # Optional opacity cull: a Gaussian's peak per-pixel contribution is
    # `opacity * exp(0) = opacity` (at its center). If that's below the 8-bit
    # quantization step (1/255), the Gaussian is invisible everywhere and can
    # be dropped — significant kernel speedup on translucent-heavy scenes
    # like Mip-NeRF 360 captures (median opacity ~0.16). Synthetic / luigi
    # scenes are typically opaque, so this filter is a no-op for them.
    if opacities is not None:
        valid_mask = valid_mask & (opacities >= min_opacity)

    depths = means_cam[valid_mask, 2]

    return means_2d[valid_mask], covs_2d[valid_mask], depths, radii[valid_mask], valid_mask


def get_tile_assignments(
    means_2d: torch.Tensor,
    radii: torch.Tensor,
    image_height: int,
    image_width: int,
    tile_size: int = 32,
    sub_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each visible Gaussian to the screen tiles it overlaps.

    The screen is divided into a grid of tile_size x tile_size pixel tiles.
    Each Gaussian's bounding circle (center + radius) is tested against the tile grid
    to find all tiles it touches. This produces a list of (gaussian_idx, tile_id) pairs
    that tells us which Gaussians contribute to which tiles.

    Args:
        means_2d: (M, 2) screen-space positions of visible Gaussians.
        radii: (M, 2) per-axis 3σ AABB half-extents (rx, ry) in pixels.
        image_height: output image height in pixels.
        image_width: output image width in pixels.
        tile_size: tile dimension in pixels (default 32x32, matches the kernel).

    Returns:
        gaussian_ids: (P,) index into means_2d for each Gaussian-tile pair.
        tile_ids: (P,) flat tile index for each Gaussian-tile pair.
        tiles_per_gaussian: (M,) how many tiles each Gaussian overlaps (useful for debugging).
    """
    tiles_x = (image_width + tile_size - 1) // tile_size
    tiles_y = (image_height + tile_size - 1) // tile_size

    # `radii` is (M, 2): per-axis 3σ AABB half-extents from the 2D covariance
    # diagonal (see project_gaussians step 7). This is strictly tighter than a
    # lambda_max-based circle for elongated/tilted Gaussians — equivalent to
    # the original CUDA implementation (diff-gaussian-rasterization).
    rx = radii[:, 0]
    ry = radii[:, 1]

    # Compute the tile range each Gaussian's AABB covers.
    with _sub_timer(sub_timings, "tile_assign.bbox"):
        tile_min_x = torch.clamp((means_2d[:, 0] - rx) / tile_size, min=0, max=tiles_x - 1).int()
        tile_max_x = torch.clamp((means_2d[:, 0] + rx) / tile_size, min=0, max=tiles_x - 1).int()
        tile_min_y = torch.clamp((means_2d[:, 1] - ry) / tile_size, min=0, max=tiles_y - 1).int()
        tile_max_y = torch.clamp((means_2d[:, 1] + ry) / tile_size, min=0, max=tiles_y - 1).int()

        widths = tile_max_x - tile_min_x + 1
        heights = tile_max_y - tile_min_y + 1
        tiles_per_gaussian = widths * heights

    P = tiles_per_gaussian.sum().item()

    # Repeat-interleave: expand each Gaussian's index by its tile count.
    with _sub_timer(sub_timings, "tile_assign.repeat"):
        gaussian_ids = torch.repeat_interleave(torch.arange(means_2d.shape[0]), tiles_per_gaussian)
        min_x_rep = torch.repeat_interleave(tile_min_x, tiles_per_gaussian)
        min_y_rep = torch.repeat_interleave(tile_min_y, tiles_per_gaussian)
        widths_rep = torch.repeat_interleave(widths, tiles_per_gaussian)

    # Compute flat tile index per (Gaussian, tile-slot) pair.
    with _sub_timer(sub_timings, "tile_assign.flat_ids"):
        offsets = torch.arange(P) - torch.repeat_interleave(
            torch.cat([torch.zeros(1, dtype=torch.int32), tiles_per_gaussian.cumsum(0)[:-1]]),
            tiles_per_gaussian,
        )

        dy = offsets // widths_rep
        dx = offsets % widths_rep

        tile_ids = (min_y_rep + dy) * tiles_x + (min_x_rep + dx)

    return gaussian_ids, tile_ids, tiles_per_gaussian


def sort_and_bin(
    gaussian_ids: torch.Tensor,
    tile_ids: torch.Tensor,
    depths: torch.Tensor,
    tiles_x: int,
    tiles_y: int,
    sub_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort Gaussians by (tile_id, depth) and compute per-tile start/end ranges.

    After tile assignment, we have a flat list of (gaussian_idx, tile_id) pairs.
    This function sorts them so that all Gaussians for the same tile are contiguous,
    and within each tile they are ordered front-to-back by depth. It then computes
    a range table so we can quickly look up which slice of the sorted array belongs
    to each tile.

    This stays on CPU — sorting is hard to parallelize on tt-metal and even the
    original CUDA implementation uses a GPU radix sort (a highly specialized algorithm).

    Args:
        gaussian_ids: (P,) Gaussian index for each pair.
        tile_ids: (P,) tile index for each pair.
        depths: (M,) depth of each visible Gaussian (indexed by gaussian_ids).
        tiles_x: number of tiles horizontally.
        tiles_y: number of tiles vertically.

    Returns:
        sorted_gaussian_ids: (P,) Gaussian indices sorted by (tile_id, depth).
        tile_ranges: (num_tiles, 2) start and end index in sorted array for each tile.
    """
    num_tiles = tiles_x * tiles_y

    # Composite int64 sort key: (tile_id << 32) | depth_bits.
    #
    # Depth bits are the float32 bit pattern reinterpreted as int32. For
    # positive floats (always true here — z > near plane > 0 after the
    # visibility filter), the int32 view is monotonic in the float value,
    # so a single int64 argsort produces exact lexicographic (tile_id,
    # depth) ordering with no precision loss.
    #
    # The previous implementation built `tile_id * max_depth + depth` as
    # float32 — at ~30000 magnitude that gives ~0.001 precision, collapsing
    # adjacent depths within a tile and producing approximate ordering.
    # torch.argsort on int64 also uses radix internally, which is ~7×
    # faster than mergesort on float32 keys at this size (iter-020: 61→7 ms).
    with _sub_timer(sub_timings, "sort.keys"):
        depth_bits = depths[gaussian_ids].view(torch.int32).to(torch.int64)
        sort_keys = (tile_ids.to(torch.int64) << 32) | depth_bits

    with _sub_timer(sub_timings, "sort.argsort"):
        sorted_indices = torch.argsort(sort_keys)
    with _sub_timer(sub_timings, "sort.gather"):
        sorted_gaussian_ids = gaussian_ids[sorted_indices]
        sorted_tile_ids = tile_ids[sorted_indices]

    # Build tile ranges: for each tile, find where its Gaussians start and end
    # in the sorted array. Tiles with no Gaussians get range (0, 0).
    with _sub_timer(sub_timings, "sort.ranges"):
        tile_ranges = torch.zeros(num_tiles, 2, dtype=torch.int64)

        if sorted_tile_ids.numel() > 0:
            # Detect where tile_id changes in the sorted array
            changes = sorted_tile_ids[1:] != sorted_tile_ids[:-1]
            change_indices = torch.where(changes)[0] + 1

            # Start indices: position 0 + every change point
            starts = torch.cat([torch.zeros(1, dtype=torch.int64), change_indices])
            # End indices: every change point + final position
            ends = torch.cat([change_indices, torch.tensor([len(sorted_tile_ids)])])

            # The tile at each segment
            segment_tiles = sorted_tile_ids[starts]

            tile_ranges[segment_tiles, 0] = starts
            tile_ranges[segment_tiles, 1] = ends

    return sorted_gaussian_ids, tile_ranges


def alpha_blend(
    means_2d: torch.Tensor,
    covs_2d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    sorted_gaussian_ids: torch.Tensor,
    tile_ranges: torch.Tensor,
    image_height: int,
    image_width: int,
    tile_size: int = 32,
) -> torch.Tensor:
    """Render the final image by compositing Gaussians front-to-back per pixel.

    For each tile, iterates through its sorted Gaussians. For each pixel in the tile,
    evaluates the 2D Gaussian weight and accumulates color using alpha compositing:

        C_pixel += alpha_i * T * color_i
        T *= (1 - alpha_i)

    where T is the accumulated transmittance (starts at 1, decreases as Gaussians are
    composited). Early termination when T < 0.0001 (pixel is effectively saturated).

    The alpha for each Gaussian at a pixel is:
        alpha = opacity * exp(-0.5 * d^T @ Σ_2D⁻¹ @ d)
    where d = pixel_position - gaussian_center.

    Args:
        means_2d: (M, 2) screen-space Gaussian centers.
        covs_2d: (M, 2, 2) 2D covariance matrices.
        colors: (M, 3) RGB colors per Gaussian.
        opacities: (M,) opacity values in [0, 1].
        sorted_gaussian_ids: (P,) Gaussian indices sorted by (tile_id, depth).
        tile_ranges: (num_tiles, 2) start/end indices per tile in sorted array.
        image_height: output image height in pixels.
        image_width: output image width in pixels.
        tile_size: tile dimension in pixels (default 32x32, matches the kernel).

    Returns:
        image: (image_height, image_width, 3) rendered RGB image.
    """
    tiles_x = (image_width + tile_size - 1) // tile_size
    tiles_y = (image_height + tile_size - 1) // tile_size

    # Convert to numpy: lower per-op overhead (~2µs vs ~10µs for PyTorch on small arrays),
    # giving ~3-5x speedup on the inner loop which dominates rendering time.
    means_np = means_2d.numpy()
    colors_np = colors.numpy()
    opacities_np = opacities.numpy()
    gids_np = sorted_gaussian_ids.numpy()
    ranges_np = tile_ranges.numpy()

    # Precompute inverse covariances in numpy
    covs_np = covs_2d.numpy()
    a, b, c = covs_np[:, 0, 0], covs_np[:, 0, 1], covs_np[:, 1, 1]
    det = np.maximum(a * c - b * b, 1e-6)
    cov_inv_np = np.zeros_like(covs_np)
    cov_inv_np[:, 0, 0] = c / det
    cov_inv_np[:, 0, 1] = -b / det
    cov_inv_np[:, 1, 0] = -b / det
    cov_inv_np[:, 1, 1] = a / det

    image = np.zeros((image_height, image_width, 3), dtype=np.float32)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_id = ty * tiles_x + tx
            start, end = ranges_np[tile_id, 0], ranges_np[tile_id, 1]

            if start == end:
                continue

            # Pixel coordinates for this tile
            py_start = ty * tile_size
            px_start = tx * tile_size
            py_end = min(py_start + tile_size, image_height)
            px_end = min(px_start + tile_size, image_width)

            tile_h = py_end - py_start
            tile_w = px_end - px_start

            # Grid of pixel centers (offset by 0.5 to sample at pixel center)
            py = np.arange(py_start, py_end, dtype=np.float32) + 0.5
            px = np.arange(px_start, px_end, dtype=np.float32) + 0.5
            grid_y, grid_x = np.meshgrid(py, px, indexing="ij")  # (tile_h, tile_w)

            # Accumulated color and transmittance per pixel
            accumulated_color = np.zeros((tile_h, tile_w, 3), dtype=np.float32)
            transmittance = np.ones((tile_h, tile_w), dtype=np.float32)

            # Iterate through Gaussians for this tile (front-to-back)
            for idx in range(start, end):
                g = gids_np[idx]

                dx = grid_x - means_np[g, 0]  # (tile_h, tile_w)
                dy = grid_y - means_np[g, 1]

                ci = cov_inv_np[g]
                power = -0.5 * (ci[0, 0] * dx * dx + 2.0 * ci[0, 1] * dx * dy + ci[1, 1] * dy * dy)
                gauss_weight = np.exp(np.minimum(power, 0.0))

                alpha = np.clip(opacities_np[g] * gauss_weight, None, 0.99)

                # NOTE: the original CUDA implementation skips pixels where alpha < 1/255
                # (invisible in 8-bit output). We skip that here because masking individual
                # pixels in a vectorized tile operation adds more overhead than computing
                # the near-zero contribution. Worth doing in the tt-metal kernel.

                # Alpha compositing: color += alpha * T * gaussian_color
                accumulated_color += (alpha * transmittance)[:, :, np.newaxis] * colors_np[g]

                # Update transmittance
                transmittance *= (1.0 - alpha)

                # Early termination: if all pixels in tile are saturated, stop
                if transmittance.max() < 0.0001:
                    break

            image[py_start:py_end, px_start:px_end] = accumulated_color

    return torch.from_numpy(image)


def prepare_kernel_inputs(
    means_2d: torch.Tensor,
    covs_2d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    sorted_gaussian_ids: torch.Tensor,
    tile_ranges: torch.Tensor,
    image_height: int,
    image_width: int,
    sub_timings: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack per-tile Gaussian attributes for the tt-metal kernel.

    Produces:
      attribute_packs: (N_entries, 9) fp32, per row:
          [mean_x, mean_y, cov_inv_a, 2*cov_inv_b, cov_inv_c, R, G, B, opacity]
      tile_offsets: (num_tiles + 1,) uint32, cumulative prefix sum.
      px_tiles, py_tiles: (num_tiles, 32, 32) fp32, global screen coords.
    """
    tiles_x = (image_width + 31) // 32
    tiles_y = (image_height + 31) // 32
    num_tiles = tiles_x * tiles_y

    # Invert covariances (same math as alpha_blend)
    with _sub_timer(sub_timings, "prep.cov_inv"):
        a = covs_2d[:, 0, 0]
        b = covs_2d[:, 0, 1]
        c = covs_2d[:, 1, 1]
        det = torch.clamp(a * c - b * b, min=1e-6)
        cov_inv_a = (c / det).numpy()
        cov_inv_b = (-b / det).numpy()
        cov_inv_c = (a / det).numpy()

        means_np = means_2d.numpy()
        colors_np = colors.numpy()
        opacities_np = opacities.numpy()
        gids_np = sorted_gaussian_ids.numpy()
        ranges_np = tile_ranges.numpy()

    # Build attribute_packs by a single gather over sorted_gaussian_ids.
    # The previous implementation iterated tile-by-tile in Python and built one
    # 9-element list per entry — at 45K entries that's ~250 ms of pure
    # interpreter overhead. The flat array is already in the correct order
    # (sort_and_bin sorts by tile_id then depth), so a per-column gather is
    # equivalent and ~100x faster.
    total_entries = gids_np.shape[0]

    # tile_offsets: cumulative count up to each tile, plus a final total.
    with _sub_timer(sub_timings, "prep.offsets"):
        counts = (ranges_np[:, 1] - ranges_np[:, 0]).astype(np.uint32)
        tile_offsets = np.zeros(num_tiles + 1, dtype=np.uint32)
        tile_offsets[1:] = np.cumsum(counts)

    # Per-entry tile origin: subtract from each Gaussian's mean so the pack
    # stores tile-LOCAL means matching the tile-local px/py grid.
    with _sub_timer(sub_timings, "prep.tile_origin"):
        tile_id_per_entry = np.repeat(np.arange(num_tiles, dtype=np.int32), counts)
        tile_origin_x = (tile_id_per_entry % tiles_x).astype(np.float32) * 32.0
        tile_origin_y = (tile_id_per_entry // tiles_x).astype(np.float32) * 32.0

    with _sub_timer(sub_timings, "prep.gather"):
        attribute_packs = np.empty((total_entries, 9), dtype=np.float32)
        attribute_packs[:, 0] = means_np[gids_np, 0] - tile_origin_x
        attribute_packs[:, 1] = means_np[gids_np, 1] - tile_origin_y
        attribute_packs[:, 2] = cov_inv_a[gids_np]
        attribute_packs[:, 3] = 2.0 * cov_inv_b[gids_np]
        attribute_packs[:, 4] = cov_inv_c[gids_np]
        attribute_packs[:, 5] = colors_np[gids_np, 0]
        attribute_packs[:, 6] = colors_np[gids_np, 1]
        attribute_packs[:, 7] = colors_np[gids_np, 2]
        attribute_packs[:, 8] = opacities_np[gids_np]

    # px/py grids depend only on (H, W) — cache them per resolution. During
    # interactive viewing the same resolution is reused thousands of times;
    # in benchmark runs each scene/resolution combo computes once.
    with _sub_timer(sub_timings, "prep.px_py_grid"):
        px_tiles, py_tiles = _get_px_py_grids(image_height, image_width)

    return attribute_packs, tile_offsets, px_tiles, py_tiles


# Cache of (px_tiles, py_tiles) keyed by (image_height, image_width).
# Each entry is ~num_tiles * 32 * 32 * 4 * 2 bytes — at 640x640 that's ~3 MB.
# Bounded by the small set of distinct resolutions an interactive session
# produces, so a plain dict is fine (no LRU needed in practice).
_px_py_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_px_py_grids(image_height: int, image_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (px_tiles, py_tiles) of shape (num_tiles, 32, 32) — cached.

    Each tile stores **tile-local** pixel coordinates:
        px[i, j] = j + 0.5
        py[i, j] = i + 0.5
    Same grid for every tile. Per-Gaussian `mean_x`, `mean_y` in the packs
    are pre-shifted by the tile's origin, so `dx = px - mean_x` produces
    the same value as global coords would.

    Why tile-local: the kernel stores px/py as bf16 in CB_PX/CB_PY. bf16
    has 7 mantissa bits → for values in [1024, 2048) the representable
    step is 8, so adjacent pixels in right-side tiles of a 1920-wide
    render would round to the same px and produce identical output —
    visible as ~8-pixel-wide blocky stripes on the right side. Tile-local
    coords stay in [0, 32) where bf16 has sub-0.25 precision.
    """
    key = (image_height, image_width)
    cached = _px_py_cache.get(key)
    if cached is not None:
        return cached

    tiles_x = (image_width + 31) // 32
    tiles_y = (image_height + 31) // 32
    num_tiles = tiles_x * tiles_y
    i_grid = np.arange(32, dtype=np.float32)
    j_grid = np.arange(32, dtype=np.float32)
    # Single tile-local grid, broadcast to all num_tiles slots.
    px_tiles = np.broadcast_to(
        (j_grid + 0.5)[None, None, :], (num_tiles, 32, 32)
    ).copy()
    py_tiles = np.broadcast_to(
        (i_grid + 0.5)[None, :, None], (num_tiles, 32, 32)
    ).copy()
    _px_py_cache[key] = (px_tiles, py_tiles)
    return px_tiles, py_tiles
