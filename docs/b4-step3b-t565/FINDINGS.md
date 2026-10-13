# t565 findings (2026-10-12 18:10 PDT)

- Sort-in-blend code (29022227) fits the kernel-config buffer at KCFG 48 KB (-Os program 117360 B).
- kcfg48 arm (47da5355, out-kcfg48/): standalone sort alone at GSPLAT_TT_KCFG_EXTRA_KB=48, sort in blend OFF,
  fails at once: `TT_THROW: Statically allocated circular buffers on core range [0-0 - 10-9] grow to
  1587072 B which is beyond max L1 size of 1572864 B` (14,208 B over). At KCFG 34 it fits with ~128 B spare.
- So the earlier fuse check hang in view 0 (out-check-fuse/) is the standalone onelaunch sort not fitting
  L1 at KCFG 48 (there it hung rather than threw; cause of hang-vs-throw not checked).
- KCFG is per device, so the frame-0 standalone sort and the fused blend must both fit at the same KCFG.

## Next step
Either (a) shrink the standalone onelaunch sort's static CBs by >=14.2 KB when SORT_IN_BLEND=1
(e.g. OL_RING_DEPTH 8->4 or smaller OL_PB, only for the bootstrap frame), or (b) cut the
sort-in-blend program to <= KCFG 34 KB (needs ~12 KB less code). Then rerun the fuse check and the
>=3-round b2b A/B vs 8.615 ms/view (best-iter-226).
