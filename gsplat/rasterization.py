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
    max_radius: int = 0,
    contrib_floor: float = 1.0 / 16384.0,
    k_cap: float = 3.0,
    use_isoellipse: bool = False,
    ground_truth: bool = False,
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
    # ω<1/255. We additionally clamp k ≤ k_cap (default 3) so high-opacity
    # Gaussians never grow beyond the iter-022 3σ baseline (k(1.0)=3.33
    # without the cap). On stitch_doll (most ω∈[0.01, 0.5]) this cuts ~11 %
    # (Gaussian, tile) pairs on top of iter-022.
    #
    # NB: there used to be a `torch.clamp(k, min=3.0)` floor here that the
    # cpu_cpp staged path never had; combined with `k_cap=3` it forced k=3
    # everywhere and silently inflated low-opacity AABBs (typical bicycle
    # grass blades with ω≈1/255 want k≈2.9). Removed for consistency with
    # the cpu_cpp `apply_k_cap` formula and the staged path. Regression
    # locked by tests/spec/test_thin_splat_stipple.py.
    with _sub_timer(sub_timings, "project.radii"):
        a = covs_2d[:, 0, 0]
        b = covs_2d[:, 0, 1]
        c = covs_2d[:, 1, 1]
        if ground_truth:
            k = torch.full_like(a, 3.0)
        elif opacities is not None:
            floor = max(float(contrib_floor), 1e-12)
            arg = torch.clamp(opacities / floor, min=1.0)
            k = torch.sqrt(2.0 * torch.log(arg))
            if k_cap > 0.0:
                k = torch.clamp(k, max=float(k_cap))
        else:
            k = torch.full_like(a, 3.0)

        if use_isoellipse:
            trace = a + c
            disc = torch.sqrt(torch.clamp(trace * trace - 4.0 * (a * c - b * b), min=0.0))
            l1 = 0.5 * (trace + disc)
            l2 = 0.5 * (trace - disc)
            theta = 0.5 * torch.atan2(2.0 * b, a - c)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            cos2 = cos_t * cos_t
            sin2 = sin_t * sin_t
            rx = torch.ceil(k * torch.sqrt(torch.clamp(l1 * cos2 + l2 * sin2, min=0.0)))
            ry = torch.ceil(k * torch.sqrt(torch.clamp(l1 * sin2 + l2 * cos2, min=0.0)))
        else:
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

    # Cap the bounding radius. max_radius: 0 = min(H,W)/2; >0 = cap; <0 = off.
    if max_radius < 0:
        pass
    else:
        cap = min(image_height, image_width) // 2 if max_radius == 0 else int(max_radius)
        valid_mask = valid_mask & (rx <= cap) & (ry <= cap)

    if opacities is not None and not ground_truth:
        valid_mask = valid_mask & (opacities >= min_opacity)

    depths = means_cam[valid_mask, 2]

    return means_2d[valid_mask], covs_2d[valid_mask], depths, radii[valid_mask], valid_mask


def get_tile_assignments(
    means_2d: torch.Tensor,
    radii: torch.Tensor,
    image_height: int,
    image_width: int,
    tile_size: int = 32,
    covs_2d: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
    contrib_floor: float = 1.0 / 16384.0,
    sub_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign each visible Gaussian to the screen tiles it overlaps.

    The screen is divided into a grid of tile_size x tile_size pixel tiles.
    Each Gaussian's bounding circle (center + radius) is tested against the tile grid
    to find all tiles it touches. This produces a list of (gaussian_idx, tile_id) pairs
    that tells us which Gaussians contribute to which tiles.

    When both `covs_2d` and `opacities` are provided, an additional per-pair
    Mahalanobis cull runs after the bbox assignment: for each candidate
    (Gaussian, tile) pair we compute the closest point in the tile to the
    Gaussian center and drop the pair if the Gaussian's peak contribution
    inside the tile (`ω·exp(-½·m²)`) is below `contrib_floor`. This catches
    pairs the AABB still kept whose actual in-tile contribution is invisible.

    Args:
        means_2d: (M, 2) screen-space positions of visible Gaussians.
        radii: (M, 2) per-axis 3σ AABB half-extents (rx, ry) in pixels.
        image_height: output image height in pixels.
        image_width: output image width in pixels.
        tile_size: tile dimension in pixels (default 32x32, matches the kernel).
        covs_2d: (M, 2, 2) 2D covariance (optional; required for the cull).
        opacities: (M,) opacities (optional; required for the cull).
        contrib_floor: pairs whose peak in-tile contribution falls below this
            value are dropped (default 1/255 — the 8-bit perceptual floor).

    Returns:
        gaussian_ids: (P,) index into means_2d for each Gaussian-tile pair.
        tile_ids: (P,) flat tile index for each Gaussian-tile pair.
        tiles_per_gaussian: (M,) how many tiles each Gaussian overlaps after
            the cull (sums to P).
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

    # Per-pair Mahalanobis cull: drop (G, tile) pairs whose actual peak
    # contribution inside the tile is below the perceptual floor.
    #
    # CORRECTNESS NOTE: the previous implementation used an L∞-clamp closest
    # point (per-axis clamp of the Gaussian center onto the tile rectangle)
    # to evaluate m². That is WRONG for tilted (off-diagonal cov) Gaussians:
    # the L∞-clamp point can have m² much larger than the true min m² over
    # the tile, so the cull drops Gaussians whose elongated axis crosses the
    # tile diagonally — producing visible 32×32 tile artifacts at silhouettes
    # (chunks missing, sharp tile-boundary cuts in fur, semi-transparency).
    #
    # The proper-min formulation: for a positive-definite quadratic form
    # f(u, v) = c·u² - 2b·u·v + a·v² (with Σ = [[a,b],[b,c]], det > 0), the
    # 1D min on a vertical edge u = u_fixed is at v* = b·u_fixed/a, clamped
    # to the edge range. The constrained min over an axis-aligned rectangle
    # is either 0 (center inside), or the smaller of the two facing-edge
    # 1D-mins. This is exact and only ~2× the cost of the L∞-clamp.
    if covs_2d is not None and opacities is not None and P > 0:
        with _sub_timer(sub_timings, "tile_assign.contrib_cull"):
            a = covs_2d[gaussian_ids, 0, 0]
            b = covs_2d[gaussian_ids, 0, 1]
            c = covs_2d[gaussian_ids, 1, 1]
            det = torch.clamp(a * c - b * b, min=1e-6)

            px = means_2d[gaussian_ids, 0]
            py = means_2d[gaussian_ids, 1]
            tx_tile = (tile_ids % tiles_x).float() * tile_size
            ty_tile = (tile_ids // tiles_x).float() * tile_size

            # Displacement bounds (Gaussian-centered): rect is
            # u in [u_lo, u_hi], v in [v_lo, v_hi].
            u_lo = tx_tile - px
            u_hi = u_lo + tile_size
            v_lo = ty_tile - py
            v_hi = v_lo + tile_size

            x_inside = (u_lo <= 0.0) & (0.0 <= u_hi)
            y_inside = (v_lo <= 0.0) & (0.0 <= v_hi)
            inside = x_inside & y_inside

            INF = torch.full_like(u_lo, float("inf"))

            # Vertical-edge candidate: choose facing edge.
            u_fix = torch.where(u_lo > 0.0, u_lo, u_hi)
            a_safe = torch.clamp(a, min=1e-12)
            v_star = (b * u_fix) / a_safe
            v_star = torch.clamp(v_star, min=v_lo, max=v_hi)
            m2_v_edge = (
                c * u_fix * u_fix - 2.0 * b * u_fix * v_star + a * v_star * v_star
            )
            m2_v_edge = torch.where(x_inside, INF, m2_v_edge)

            # Horizontal-edge candidate: choose facing edge.
            v_fix = torch.where(v_lo > 0.0, v_lo, v_hi)
            c_safe = torch.clamp(c, min=1e-12)
            u_star = (b * v_fix) / c_safe
            u_star = torch.clamp(u_star, min=u_lo, max=u_hi)
            m2_h_edge = (
                c * u_star * u_star - 2.0 * b * u_star * v_fix + a * v_fix * v_fix
            )
            m2_h_edge = torch.where(y_inside, INF, m2_h_edge)

            m2_min = torch.minimum(m2_v_edge, m2_h_edge)
            m2_min = torch.where(inside, torch.zeros_like(m2_min), m2_min)
            m2 = m2_min / det

            keep = opacities[gaussian_ids] * torch.exp(-0.5 * m2) >= contrib_floor

            gaussian_ids = gaussian_ids[keep]
            tile_ids = tile_ids[keep]
            # Recompute tiles_per_gaussian from the surviving pairs so any
            # downstream consumer keeps a consistent view.
            tiles_per_gaussian = torch.bincount(gaussian_ids, minlength=means_2d.shape[0])

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
    transmittance_threshold: float = 1.0 / 255.0,
) -> torch.Tensor:
    """Render the final image by compositing Gaussians front-to-back per pixel.

    For each tile, iterates through its sorted Gaussians. For each pixel in the tile,
    evaluates the 2D Gaussian weight and accumulates color using alpha compositing:

        C_pixel += alpha_i * T * color_i
        T *= (1 - alpha_i)

    where T is the accumulated transmittance (starts at 1, decreases as Gaussians are
    composited). Early termination when T < transmittance_threshold (default
    1/255 — the 8-bit perceptual floor; further Gaussians cannot change output).

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
                if transmittance.max() < transmittance_threshold:
                    break

            image[py_start:py_end, px_start:px_end] = accumulated_color

    return torch.from_numpy(image)


_NUM_MICROBLOCKS = 32
_MB_ORIGIN_X = (np.arange(_NUM_MICROBLOCKS, dtype=np.int32) & 3) * 8
_MB_ORIGIN_Y = (np.arange(_NUM_MICROBLOCKS, dtype=np.int32) >> 2) * 4


def microblock_cull(
    means_2d: torch.Tensor,
    covs_2d: torch.Tensor,
    opacities: torch.Tensor,
    sorted_gaussian_ids: torch.Tensor,
    tile_ranges: torch.Tensor,
    tiles_x: int,
    tiles_y: int,
    tile_size: int = 32,
    mb_contrib_floor: float = 1.0 / 16384.0,
    contrib_floor: float = 15.0 / 255.0,
    sub_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Per-microblock Mahalanobis cull; emit mb_header and mb_stream.

    `mb_contrib_floor` is the tight per-microblock keep threshold (default
    1/16384). Hero hits ~63 dB at 1/4096 but a few off-axis benchmark views
    (chal_bottom, orbit_*) need a tighter floor to stay ≥ 60 dB on the full
    30-frame sweep. `contrib_floor`
    is retained for backwards compatibility with callers that want the looser
    per-tile floor.

    mb_stream stores GLOBAL gaussian ids (same M-space as sorted_gaussian_ids),
    filtered per (tile, microblock) by the §2 closest-point-in-microblock test.
    """
    del contrib_floor  # explicit: per-microblock cull uses mb_contrib_floor.
    del sub_timings    # reserved for iter-007 profiling hooks

    means_np = means_2d.numpy()
    covs_np = covs_2d.numpy()
    opacities_np = opacities.numpy()
    gids_np = sorted_gaussian_ids.numpy()
    ranges_np = tile_ranges.numpy()

    num_tiles = tiles_x * tiles_y
    mb_header = np.zeros((num_tiles, _NUM_MICROBLOCKS, 2), dtype=np.int64)
    stream_chunks: list[np.ndarray] = []

    pairs_in = int(gids_np.shape[0])
    tile_pairs_dropped = 0
    stream_offset = 0
    # Use float32 to match the C++ fixture-bit-identity contract; the proper-min
    # cull is well-conditioned so f32 is sufficient for stability.
    mb_ox_local = _MB_ORIGIN_X.astype(np.float32)
    mb_oy_local = _MB_ORIGIN_Y.astype(np.float32)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_id = ty * tiles_x + tx
            start, end = ranges_np[tile_id, 0], ranges_np[tile_id, 1]
            if start == end:
                continue

            tile_g_ids = gids_np[start:end]
            tx_tile = float(tx * tile_size)
            ty_tile = float(ty * tile_size)

            mean_x = means_np[tile_g_ids, 0]
            mean_y = means_np[tile_g_ids, 1]
            a = covs_np[tile_g_ids, 0, 0]
            b = covs_np[tile_g_ids, 0, 1]
            c = covs_np[tile_g_ids, 1, 1]
            det = np.maximum(a * c - b * b, 1e-6)
            ci_a = c / det
            ci_b = -b / det
            ci_c = a / det
            g_op = opacities_np[tile_g_ids]

            # Per-(g, m) keep mask using the TRUE min m² over each microblock
            # rectangle. The previous version used L∞-clamp (per-axis clamp of
            # the Gaussian center onto the microblock), which is wrong for
            # tilted Gaussians — see get_tile_assignments for the full bug
            # write-up. This vectorised constrained-min form is exact.
            mb_ox = tx_tile + mb_ox_local        # (M,) microblock x origins
            mb_oy = ty_tile + mb_oy_local        # (M,) microblock y origins
            u_lo = mb_ox[np.newaxis, :] - mean_x[:, np.newaxis]   # (G, M)
            u_hi = u_lo + 8.0
            v_lo = mb_oy[np.newaxis, :] - mean_y[:, np.newaxis]
            v_hi = v_lo + 4.0

            x_inside = (u_lo <= 0.0) & (0.0 <= u_hi)
            y_inside = (v_lo <= 0.0) & (0.0 <= v_hi)

            ci_a_safe = np.maximum(ci_a, 1e-12)[:, np.newaxis]
            ci_c_safe = np.maximum(ci_c, 1e-12)[:, np.newaxis]
            ci_a_b = ci_a[:, np.newaxis]
            ci_b_b = ci_b[:, np.newaxis]
            ci_c_b = ci_c[:, np.newaxis]

            INF = np.full_like(u_lo, np.inf)

            # Vertical-edge candidate.
            u_fix = np.where(u_lo > 0.0, u_lo, u_hi)
            v_star = -ci_b_b * u_fix / ci_c_safe
            v_star = np.clip(v_star, v_lo, v_hi)
            m2_v = (
                ci_a_b * u_fix * u_fix
                + 2.0 * ci_b_b * u_fix * v_star
                + ci_c_b * v_star * v_star
            )
            m2_v = np.where(x_inside, INF, m2_v)

            # Horizontal-edge candidate.
            v_fix = np.where(v_lo > 0.0, v_lo, v_hi)
            u_star = -ci_b_b * v_fix / ci_a_safe
            u_star = np.clip(u_star, u_lo, u_hi)
            m2_h = (
                ci_a_b * u_star * u_star
                + 2.0 * ci_b_b * u_star * v_fix
                + ci_c_b * v_fix * v_fix
            )
            m2_h = np.where(y_inside, INF, m2_h)

            m2_min = np.minimum(m2_v, m2_h)
            m2_min = np.where(x_inside & y_inside, np.float32(0.0), m2_min)

            # Equivalent exp-free form (matches C++ cull bit-for-bit). Test
            # `g_op * exp(-0.5*m²) >= floor` ⇔ `m² <= -2*log(floor/g_op)`.
            # For g_op <= floor the LHS can never reach floor (since exp ≤ 1),
            # so set thresh = -inf to force drop.
            g_op_f = g_op.astype(np.float32)
            with np.errstate(divide="ignore"):
                log_thresh_g = np.log(np.float32(mb_contrib_floor) / g_op_f)
            thresh_m2_g = np.where(
                g_op_f > np.float32(mb_contrib_floor),
                (-2.0 * log_thresh_g).astype(np.float32),
                np.float32(-np.inf),
            )
            keep_mask = m2_min <= thresh_m2_g[:, np.newaxis]
            tile_pairs_dropped += int(np.sum(~keep_mask.any(axis=1)))

            for m in range(_NUM_MICROBLOCKS):
                kept = tile_g_ids[keep_mask[:, m]]
                count = int(kept.shape[0])
                mb_header[tile_id, m, 0] = stream_offset
                mb_header[tile_id, m, 1] = count
                if count > 0:
                    stream_chunks.append(kept)
                    stream_offset += count

    if stream_chunks:
        mb_stream_np = np.concatenate(stream_chunks).astype(np.int64)
    else:
        mb_stream_np = np.empty(0, dtype=np.int64)

    pairs_out = int(mb_stream_np.shape[0])
    drop_pct = 0.0 if pairs_in == 0 else 100.0 * tile_pairs_dropped / pairs_in
    full_replay = pairs_in * _NUM_MICROBLOCKS
    work_reduction_pct = (
        0.0 if full_replay == 0 else 100.0 * (1.0 - pairs_out / full_replay)
    )

    stats = {
        "pairs_in": pairs_in,
        "pairs_out": pairs_out,
        "drop_pct": drop_pct,
        "work_reduction_pct": work_reduction_pct,
    }
    return (
        torch.from_numpy(mb_header),
        torch.from_numpy(mb_stream_np),
        stats,
    )


def alpha_blend_microblock(
    means_2d: torch.Tensor,
    covs_2d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    mb_header: torch.Tensor,
    mb_stream: torch.Tensor,
    image_height: int,
    image_width: int,
    tile_size: int = 32,
) -> torch.Tensor:
    """Microblock-major alpha compositing (same per-pixel math as alpha_blend)."""
    means_np = means_2d.numpy()
    colors_np = colors.numpy()
    opacities_np = opacities.numpy()
    header_np = mb_header.numpy()
    stream_np = mb_stream.numpy()

    covs_np = covs_2d.numpy()
    a, b, c = covs_np[:, 0, 0], covs_np[:, 0, 1], covs_np[:, 1, 1]
    det = np.maximum(a * c - b * b, 1e-6)
    cov_inv_np = np.zeros_like(covs_np)
    cov_inv_np[:, 0, 0] = c / det
    cov_inv_np[:, 0, 1] = -b / det
    cov_inv_np[:, 1, 0] = -b / det
    cov_inv_np[:, 1, 1] = a / det

    tiles_x = (image_width + tile_size - 1) // tile_size
    tiles_y = (image_height + tile_size - 1) // tile_size
    image = np.zeros((image_height, image_width, 3), dtype=np.float32)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_id = ty * tiles_x + tx
            py_tile = ty * tile_size
            px_tile = tx * tile_size
            py_end_tile = min(py_tile + tile_size, image_height)
            px_end_tile = min(px_tile + tile_size, image_width)

            for m in range(_NUM_MICROBLOCKS):
                off, cnt = header_np[tile_id, m, 0], header_np[tile_id, m, 1]
                if cnt == 0:
                    continue

                mb_ox = (m & 3) * 8
                mb_oy = (m >> 2) * 4
                py_start = py_tile + mb_oy
                px_start = px_tile + mb_ox
                py_end = min(py_start + 4, py_end_tile)
                px_end = min(px_start + 8, px_end_tile)
                if py_start >= py_end or px_start >= px_end:
                    continue

                mb_h = py_end - py_start
                mb_w = px_end - px_start

                py = np.arange(py_start, py_end, dtype=np.float32) + 0.5
                px = np.arange(px_start, px_end, dtype=np.float32) + 0.5
                grid_y, grid_x = np.meshgrid(py, px, indexing="ij")

                accumulated_color = np.zeros((mb_h, mb_w, 3), dtype=np.float32)
                transmittance = np.ones((mb_h, mb_w), dtype=np.float32)

                for idx in range(int(off), int(off + cnt)):
                    g = int(stream_np[idx])

                    dx = grid_x - means_np[g, 0]
                    dy = grid_y - means_np[g, 1]

                    ci = cov_inv_np[g]
                    power = -0.5 * (
                        ci[0, 0] * dx * dx + 2.0 * ci[0, 1] * dx * dy + ci[1, 1] * dy * dy
                    )
                    gauss_weight = np.exp(np.minimum(power, 0.0))
                    alpha = np.clip(opacities_np[g] * gauss_weight, None, 0.99)

                    accumulated_color += (alpha * transmittance)[:, :, np.newaxis] * colors_np[g]
                    transmittance *= 1.0 - alpha

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


def _packs_to_basis_coeff_rows(
    mean_x: np.ndarray,
    mean_y: np.ndarray,
    cov_a: np.ndarray,
    two_cov_b: np.ndarray,
    cov_c: np.ndarray,
    opacity: np.ndarray,
    color_r: np.ndarray,
    color_g: np.ndarray,
    color_b: np.ndarray,
) -> np.ndarray:
    """Convert per-gaussian tile-local packs to basis-form coeff rows for metal.

    Q = A·x² + B·xy + C·y² + D·x + E·y + F  with x,y tile-local px/py.
    Host folds -0.5 into A..F so the device evaluates exp(Q) directly.
    Returns (N, 10) fp32: A, B, C, D, E, F, opacity, R, G, B.
    """
    neg_half = -0.5
    a = cov_a
    b = two_cov_b  # already 2·cov_inv_b in packs
    c = cov_c
    mx = mean_x
    my = mean_y
    a_q = neg_half * a
    b_q = neg_half * b
    c_q = neg_half * c
    d_q = neg_half * (-2.0 * a * mx - b * my)
    e_q = neg_half * (-2.0 * c * my - b * mx)
    f_q = neg_half * (a * mx * mx + b * mx * my + c * my * my)
    return np.stack(
        [a_q, b_q, c_q, d_q, e_q, f_q, opacity, color_r, color_g, color_b],
        axis=1,
    ).astype(np.float32)


def prepare_microblock_payload(
    means_2d: torch.Tensor,
    covs_2d: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    sorted_gaussian_ids: torch.Tensor,
    tile_ranges: torch.Tensor,
    image_height: int,
    image_width: int,
    mb_contrib_floor: float = 1.0 / 16384.0,
    sub_timings: dict[str, float] | None = None,
) -> dict[str, np.ndarray | dict]:
    """Pack legacy blend inputs plus microblock-major payloads for metal iter-001.

    Returns a dict with:
      attribute_packs, tile_offsets, px_tiles, py_tiles  — legacy daemon path
      coeff_table  — (total_entries, 10) fp32 basis coeffs (same row order as packs)
      mb_header    — (num_tiles, 32, 2) uint32 (offset, count) into mb_stream
      mb_stream    — (L_prime,) uint32 LOCAL gaussian indices per tile list
      mb_stats     — drop/work reduction stats from microblock_cull
    """
    packs, tile_offsets, px_tiles, py_tiles = prepare_kernel_inputs(
        means_2d,
        covs_2d,
        colors,
        opacities,
        sorted_gaussian_ids,
        tile_ranges,
        image_height,
        image_width,
        sub_timings=sub_timings,
    )
    tiles_x = (image_width + 31) // 32
    tiles_y = (image_height + 31) // 32

    mb_header_t, mb_stream_global, mb_stats = microblock_cull(
        means_2d,
        covs_2d,
        opacities,
        sorted_gaussian_ids,
        tile_ranges,
        tiles_x,
        tiles_y,
        32,
        mb_contrib_floor=mb_contrib_floor,
    )
    mb_header = mb_header_t.numpy().astype(np.uint32)
    stream_global = mb_stream_global.numpy()

    # coeff_table: same row order as attribute_packs (depth-sorted per tile).
    coeff_table = _packs_to_basis_coeff_rows(
        packs[:, 0],
        packs[:, 1],
        packs[:, 2],
        packs[:, 3],
        packs[:, 4],
        packs[:, 5],
        packs[:, 6],
        packs[:, 7],
        packs[:, 8],
    )

    # Metal kernel indexes coeff_table by LOCAL gaussian index; legacy CPU
    # alpha_blend_microblock still consumes GLOBAL g-ids (see mb_stream).
    gids_np = sorted_gaussian_ids.numpy()
    ranges_np = tile_ranges.numpy()
    num_tiles = tiles_x * tiles_y
    if stream_global.size == 0:
        mb_stream_local = np.empty(0, dtype=np.uint32)
    else:
        local_stream: list[int] = []
        for tile_id in range(num_tiles):
            start, end = ranges_np[tile_id, 0], ranges_np[tile_id, 1]
            if start == end:
                continue
            tile_gids = gids_np[start:end]
            g_to_local = {int(g): i for i, g in enumerate(tile_gids)}
            pack_base = int(tile_offsets[tile_id])
            for m in range(_NUM_MICROBLOCKS):
                off = int(mb_header[tile_id, m, 0])
                cnt = int(mb_header[tile_id, m, 1])
                for g in stream_global[off : off + cnt]:
                    local_stream.append(pack_base + g_to_local[int(g)])
        mb_stream_local = np.asarray(local_stream, dtype=np.uint32)

    return {
        "attribute_packs": packs,
        "tile_offsets": tile_offsets,
        "px_tiles": px_tiles,
        "py_tiles": py_tiles,
        "coeff_table": coeff_table,
        "mb_header": mb_header,
        "mb_stream": stream_global.astype(np.uint32),
        "mb_stream_local": mb_stream_local,
        "mb_stats": mb_stats,
    }


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
