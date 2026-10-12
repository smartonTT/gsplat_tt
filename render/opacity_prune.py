"""Task #536 (lever 4 of docs/structural-plan, GSPLAT_TT_OPACITY_PRUNE, default on): static
opacity prune with the downstream order kept, so the image stays bit-exact.

A gaussian with opacity < min_opacity is never visible (pfwc's predicate) and
never contributes (alpha <= opacity < 1/255 = the contribution floor), so
dropping it once at load only cuts the bytes pfwc streams per view (bicycle:
572k of 6.13M gaussians).

The order sort and blend see is not the ply order: pfwc deals 1024-gaussian
tiles strided over its C cores (core c owns tiles c, c + C, ...) and core c's
visible gaussians get the storage ids [seg_base(c), ...) in its chunk order
(render/kernels/dataflow/pfwc_fuse.h, vis_tile::SeqMap). That storage id is the
depth-tie key and the record address downstream. Plain compaction would move
gaussians across tiles and cores and change the tie order (t479: 254 px on the
hero). Instead the survivors are placed so that the pruned scene's sequence
order (tile sequence of the strided deal over the new tile count) lists them in
exactly the old sequence order: the storage ids map monotonically, so pairs,
ties and records are the old ones.
"""
import numpy as np

TILE = 1024


def seq_positions(n, num_cores):
    """Gaussian indices of an n-gaussian scene in pfwc sequence order: the
    tiles of core 0 (0, C, 2C, ...), then core 1, ..., each tile's gaussians in
    index order; the last, partial tile only up to n."""
    t = (n + TILE - 1) // TILE
    tiles = np.concatenate([np.arange(c, t, num_cores, dtype=np.int64)
                            for c in range(min(num_cores, t))])
    pos = (tiles[:, None] * TILE + np.arange(TILE, dtype=np.int64)).ravel()
    return pos[pos < n]


def prune_layout(opacities, min_opacity, num_cores):
    """Source index for every slot of the pruned scene: new[i] = old[src[i]].
    Keeps exactly the gaussians with opacity >= min_opacity (fp32 compare, the
    value pfwc gets) and lists them in the old sequence order."""
    op = np.asarray(opacities, dtype=np.float32).reshape(-1)
    n = op.shape[0]
    old_seq = seq_positions(n, num_cores)
    kept_seq = old_seq[op[old_seq] >= np.float32(min_opacity)]
    m = kept_seq.shape[0]
    src = np.empty(m, dtype=np.int64)
    src[seq_positions(m, num_cores)] = kept_seq
    return src
