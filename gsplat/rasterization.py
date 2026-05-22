import numpy as np
import torch

from gsplat.utils import build_covariance_3d


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
        radii: (M, 2) screen-axis bbox half-widths in pixels (at 3σ).
            radii[:, 0] = X half-width = 3·√a, radii[:, 1] = Y = 3·√c.
            Tighter than a single bounding circle for anisotropic Gaussians.
        valid_mask: (N,) boolean mask indicating which input Gaussians are visible.
    """
    N = means.shape[0]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # --- Step 1: Build 3D covariance matrices ---
    cov3d = build_covariance_3d(scales, rotations)  # (N, 3, 3)

    # --- Step 2: Transform Gaussian centers to camera space ---
    # p_cam = R @ p_world + t, using extrinsics directly
    R = extrinsics[:3, :3]  # (3, 3)
    t = extrinsics[:3, 3]   # (3,)
    means_cam = means @ R.T + t  # (N, 3) @ (3, 3) + (3,) -> (N, 3)

    # --- Step 3: Frustum culling ---
    # Keep only Gaussians in front of the near plane
    near = 0.2
    valid_mask = means_cam[:, 2] > near  # z > near plane

    # --- Step 4: Compute screen-space positions (exact pinhole projection) ---
    tx, ty, tz = means_cam[:, 0], means_cam[:, 1], means_cam[:, 2]
    means_2d = torch.stack([fx * tx / tz + cx, fy * ty / tz + cy], dim=-1)  # (N, 2)

    # --- Step 5: Compute the Jacobian of the perspective projection ---
    # The perspective projection maps (x, y, z) -> (fx*x/z + cx, fy*y/z + cy).
    # Its Jacobian at point (x, y, z) is:
    #   J = [[fx/z,  0,   -fx*x/z²],
    #        [0,     fy/z, -fy*y/z²]]
    # This linearizes the nonlinear projection around each Gaussian's center, 
    # which is a good approximation if the size of the Gaussian is small 
    # compared to the distance to the camera.
    tz2 = tz * tz

    J = torch.zeros(N, 2, 3, dtype=means.dtype)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / tz2
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / tz2

    # --- Step 6: Project 3D covariance to 2D ---
    # Σ_2D = J @ R @ Σ_3D @ R^T @ J^T
    # First transform to camera space: Σ_cam = R @ Σ_3D @ R^T
    cov_cam = R @ cov3d @ R.T  # (3, 3) @ (N, 3, 3) @ (3, 3) -> (N, 3, 3)

    # Then apply Jacobian: Σ_2D = J @ Σ_cam @ J^T
    covs_2d = J @ cov_cam @ J.transpose(1, 2)  # (N, 2, 2)

    # Add low-pass filter for anti-aliasing (0.3px variance, per original implementation)
    covs_2d[:, 0, 0] += 0.3
    covs_2d[:, 1, 1] += 0.3

    # --- Step 7: Compute screen-axis bounding box from 2D covariance ---
    # For a 2D Gaussian with covariance Σ = [[a, b], [b, c]], the screen-axis
    # half-widths of the 3σ contour are exactly 3√a (X) and 3√c (Y),
    # independent of the off-diagonal b. Proof: max |x| s.t.
    # (x,y)ᵀ Σ⁻¹ (x,y) = 9 gives x² = 9·Σ⁻¹_yy/det(Σ⁻¹) = 9·a → |x| = 3√a.
    # The screen-aligned bbox is always tighter than the bounding box of the
    # circumscribed circle (radius 3√λ_max), with the gap growing for
    # anisotropic Gaussians (the common case for surface-aligned splats).
    # `radii` is now (M, 2): radii[:, 0] = X half-width, radii[:, 1] = Y.
    a = covs_2d[:, 0, 0]
    c = covs_2d[:, 1, 1]
    radii_x = torch.ceil(3.0 * torch.sqrt(torch.clamp(a, min=0.0)))
    radii_y = torch.ceil(3.0 * torch.sqrt(torch.clamp(c, min=0.0)))
    radii = torch.stack([radii_x, radii_y], dim=-1)  # (M, 2)

    valid_mask = valid_mask & (means_2d[:, 0] + radii_x > 0)
    valid_mask = valid_mask & (means_2d[:, 0] - radii_x < image_width)
    valid_mask = valid_mask & (means_2d[:, 1] + radii_y > 0)
    valid_mask = valid_mask & (means_2d[:, 1] - radii_y < image_height)
    valid_mask = valid_mask & ((radii_x > 0) & (radii_y > 0))

    # Cap the bounding box. The Jacobian linearization of the perspective
    # transform (Step 5 above) breaks down when a Gaussian's 3D extent is
    # comparable to its distance from the camera — producing wildly wrong 2D
    # covariances and massive bounding boxes. Cap each axis to half the
    # smaller image dim; the projection-breakdown cases were already culled
    # by the radius check; the genuine close-ups are still kept.
    max_radius = min(image_height, image_width) // 2
    valid_mask = valid_mask & (radii_x <= max_radius) & (radii_y <= max_radius)

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each visible Gaussian to the screen tiles it overlaps.

    The screen is divided into a grid of tile_size x tile_size pixel tiles.
    Each Gaussian's screen-axis 3σ bounding box is tested against the tile
    grid to find all tiles it touches. Produces a list of (gaussian_idx,
    tile_id) pairs telling us which Gaussians contribute to which tiles.

    Args:
        means_2d: (M, 2) screen-space positions of visible Gaussians.
        radii: per-axis screen-aligned 3σ bbox half-widths in pixels. Accepts
            either (M, 2) [preferred — radii[:, 0] = X, radii[:, 1] = Y]
            or (M,) [legacy bounding-circle radius, used for both axes].
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

    if radii.dim() == 2:
        # Tight per-axis bbox (the new path, post-iter-030). Saves ~25-35% of
        # tile-pair count vs the old single-radius circle for anisotropic
        # Gaussians, with no visual change (the circle was a strict superset
        # of the ellipse bbox).
        radii_x = radii[:, 0]
        radii_y = radii[:, 1]
    else:
        radii_x = radii
        radii_y = radii

    tile_min_x = torch.clamp((means_2d[:, 0] - radii_x) / tile_size, min=0, max=tiles_x - 1).int()
    tile_max_x = torch.clamp((means_2d[:, 0] + radii_x) / tile_size, min=0, max=tiles_x - 1).int()
    tile_min_y = torch.clamp((means_2d[:, 1] - radii_y) / tile_size, min=0, max=tiles_y - 1).int()
    tile_max_y = torch.clamp((means_2d[:, 1] + radii_y) / tile_size, min=0, max=tiles_y - 1).int()

    # Width and height of each Gaussian's tile bounding box
    widths = tile_max_x - tile_min_x + 1
    heights = tile_max_y - tile_min_y + 1
    tiles_per_gaussian = widths * heights

    # Total number of (gaussian, tile) pairs
    P = tiles_per_gaussian.sum().item()

    # Repeat each Gaussian's index and tile-box params by its tile count
    gaussian_ids = torch.repeat_interleave(torch.arange(means_2d.shape[0]), tiles_per_gaussian)
    min_x_rep = torch.repeat_interleave(tile_min_x, tiles_per_gaussian)
    min_y_rep = torch.repeat_interleave(tile_min_y, tiles_per_gaussian)
    widths_rep = torch.repeat_interleave(widths, tiles_per_gaussian)

    # For each pair, compute the local offset within that Gaussian's tile grid
    # local_idx goes 0, 1, 2, ... tiles_per_gaussian[i]-1 for each Gaussian
    offsets = torch.arange(P) - torch.repeat_interleave(
        torch.cat([torch.zeros(1, dtype=torch.int32), tiles_per_gaussian.cumsum(0)[:-1]]),
        tiles_per_gaussian,
    )

    # Convert local offset to (dx, dy) within the Gaussian's tile bounding box
    dy = offsets // widths_rep
    dx = offsets % widths_rep

    # Compute flat tile index: (min_y + dy) * tiles_x + (min_x + dx)
    tile_ids = (min_y_rep + dy) * tiles_x + (min_x_rep + dx)

    return gaussian_ids, tile_ids, tiles_per_gaussian


def sort_and_bin(
    gaussian_ids: torch.Tensor,
    tile_ids: torch.Tensor,
    depths: torch.Tensor,
    tiles_x: int,
    tiles_y: int,
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

    # Build an int64 composite key: upper 32 bits = tile_id, lower 32 bits = depth
    # reinterpreted as uint32.  IEEE 754 positive floats preserve their ordering when
    # viewed as uint32 bit patterns, so this single int64 key sorts lexicographically
    # by (tile_id, depth) — identical semantics to the old float composite key but
    # without floating-point precision loss and faster via numpy's C-level sort.
    import numpy as np  # already a dep; local import avoids module-level side effects
    depth_per_pair = depths[gaussian_ids]  # (P,) float32, non-negative camera-space Z
    tile_ids_np = tile_ids.numpy().astype(np.int64)  # no copy on CPU contiguous tensor
    # view float32 bits as uint32, then widen to int64 for the bit-pack
    depth_bits = depth_per_pair.numpy().view(np.uint32).astype(np.int64)
    composite_keys = (tile_ids_np << np.int64(32)) | depth_bits  # (P,) int64

    # np.argsort with kind='stable' (mergesort) preserves relative order of equal keys,
    # matching torch.argsort's stable guarantee.
    sorted_indices_np = np.argsort(composite_keys, kind="stable")
    sorted_indices = torch.from_numpy(sorted_indices_np)
    sorted_gaussian_ids = gaussian_ids[sorted_indices]
    sorted_tile_ids = tile_ids[sorted_indices]

    # Build tile ranges: for each tile, find where its Gaussians start and end
    # in the sorted array. Tiles with no Gaussians get range (0, 0).
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
    *,
    static_handled_externally: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    """Pack per-tile Gaussian attributes for the tt-metal kernel.

    When *static_handled_externally* is False (CPU / legacy path), produces:
      attribute_packs: (N_entries, 9) fp32, per row:
          [mean_x, mean_y, cov_inv_a, 2*cov_inv_b, cov_inv_c, R, G, B, opacity]
      tile_offsets, px_tiles, py_tiles

    When True (TT daemon with SCN1 static DRAM), color/opacity are omitted;
    returns a 6-col dyn_pack [A,B,C,D,E,F], the same offsets/px/py, and sorted_gids (uint32).
    """
    tiles_x = (image_width + 31) // 32
    tiles_y = (image_height + 31) // 32
    num_tiles = tiles_x * tiles_y

    # Invert covariances (same math as alpha_blend)
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
    n_cols = 6 if static_handled_externally else 9
    attribute_packs = np.empty((total_entries, n_cols), dtype=np.float32)

    # tile_offsets: cumulative count up to each tile, plus a final total.
    # Equivalent to walking tile_ranges in order, since sort_and_bin produces
    # contiguous ranges for non-empty tiles and (0, 0) for empties.
    counts = (ranges_np[:, 1] - ranges_np[:, 0]).astype(np.uint32)
    tile_offsets = np.zeros(num_tiles + 1, dtype=np.uint32)
    tile_offsets[1:] = np.cumsum(counts)

    # Per-entry tile origin: subtract from each Gaussian's mean so the pack
    # stores tile-LOCAL means matching the tile-local px/py grid. This keeps
    # all coordinates in a small range where bf16 has sub-0.25 precision —
    # without this, right-side tiles at high resolution stored mean_x/px in
    # bf16's coarse range (step = 8 in [1024, 2048)) and produced visible
    # ~8-pixel blocky stripes.
    tile_id_per_entry = np.repeat(np.arange(num_tiles, dtype=np.int32), counts)
    tile_origin_x = (tile_id_per_entry % tiles_x).astype(np.float32) * 32.0
    tile_origin_y = (tile_id_per_entry // tiles_x).astype(np.float32) * 32.0

    mx = means_np[gids_np, 0] - tile_origin_x
    my = means_np[gids_np, 1] - tile_origin_y
    A = cov_inv_a[gids_np]
    B = 2.0 * cov_inv_b[gids_np]
    C = cov_inv_c[gids_np]
    attribute_packs[:, 0] = A
    attribute_packs[:, 1] = B
    attribute_packs[:, 2] = C
    attribute_packs[:, 3] = -2.0 * A * mx - B * my
    attribute_packs[:, 4] = -2.0 * C * my - B * mx
    attribute_packs[:, 5] = A * mx * mx + B * mx * my + C * my * my
    if not static_handled_externally:
        attribute_packs[:, 5] = colors_np[gids_np, 0]
        attribute_packs[:, 6] = colors_np[gids_np, 1]
        attribute_packs[:, 7] = colors_np[gids_np, 2]
        attribute_packs[:, 8] = opacities_np[gids_np]

    # px/py grids depend only on (H, W) — cache them per resolution. During
    # interactive viewing the same resolution is reused thousands of times;
    # in benchmark runs each scene/resolution combo computes once.
    px_tiles, py_tiles = _get_px_py_grids(image_height, image_width)

    if static_handled_externally:
        return (
            attribute_packs,
            tile_offsets,
            px_tiles,
            py_tiles,
            gids_np.astype(np.uint32, copy=False),
        )
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
