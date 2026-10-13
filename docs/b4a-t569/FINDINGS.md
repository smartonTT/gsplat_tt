# t569: B4a fold hang under p150 ETH dispatch — launch-msg evidence and first fix

Probe 4 (docs/b4a-t569/probe4.sh via drive.sh, bh-30, viewer reservation, tree ec53d065;
raw dumps in out/p4/kcfg-K{1,2,3}.txt, mailbox read with tt-exalens, kcfg.py).

- K1 eth, fold=1, dumped while hung (view 2, first folded view; view 1 nseg=0 completed):
  all 120 cores rd_ptr=0; current entry = matblend: base=0x9e00 lcb=656 rcb=1664 minrcb=64
  mask=0x400135ff07ff07ff (bit 62 = kB4aCb), txt BRISC 0x680 .. TRISC2 0x17230, en=0x1f.
  117 cores go_signal=GO (stuck), 3 DONE: (9,0), (6,1), (2,9). (6,1) is the core whose NCRISC
  firmware t567 found corrupted.
- K2 worker, fold=1: passes (rd_ptr=6, all DONE). K3 eth, fold=0: passes.
- Launch messages are consistent: min_remote_cb_start_index=64, so setup_remote_cb_interfaces
  should do nothing. A BRISC stuck there (t567) means corrupted firmware/data, not a bad
  launch message. Only difference of the fold image in the CB config: 63 local CB entries
  (index 62) instead of 27; both hanging cases (fold, shadow) use CB 62.
- Kernel ids differ per run, so the K1/K2 matblend entries still need a field-by-field match
  (next run), plus the ring extent vs worker_l1_unreserved_start (memmap.txt lacks the Tensix
  KERNEL_CONFIG lines; read them from the dev_mem_map.h in the overlay).

Fix tried (0f1231fe): K2's scratch uses CB 21 (CB_BSORT, same address; K2 only takes
get_write_ptr) instead of a second index 62 on the same CB. ETH default stays off in
dispatch_select.h b4a_fold_on until docs/b4a-t569/bench.sh (verify, md5, hero, b2b A/B) passes
with GSPLAT_TT_B4A_FOLD=1 under ETH.

## t570 (2026-10-12)

- Fix 0f1231fe does not fix it: s3 (45ff1ff5, bh-30, ETH, GSPLAT_TT_B4A_FOLD=1 VERIFY=3) hangs again at
  the first folded view (view 2, cores_with_k2_args=120/120), rc=124, reset OK. CB 62 is ruled out.
- Step 3, K1 (ETH, hang) vs K2 (worker, pass) matblend entries, field by field: base 0x9e00, sem 640,
  lcb 656, rcb 1664, minrcb 64, mask 0x400135ff07ff07ff, text offsets 0x680/0x75a0/0xe350/0x102b0/0x17230,
  rta (0,0)/(360,0)/(608,0)x3: identical. Only the watcher kernel ids differ (47/45/46 vs 53/51/52).
- Ring extent: dev_mem_map.h is the same file in the overlay and the host tt-metal (md5 a87cffab).
  KERNEL_CONFIG base = MEM_MAP_END = 0x9e00; kcfg = (69 + 32) KB = 0x19400 (split pfwc + BRISC reader
  defaults), so l1_unreserved_base = 0x23200. Fold matblend ends at 0x9e00 + 0x17230 + TRISC2 .text
  0xa90 (2704 B, JIT cache) = 0x21ac0: 5952 B under the unreserved start. tt-metal also TT_FATALs a
  program larger than the ring (program.cpp get_ringbuffer_size), so no ring overflow.
- So the launch message and the config ring are not the cause. What differs is ETH itself and the
  core count (120 vs 110; ETH adds logical column 11). Probe 5 (probe5.sh) bisects under ETH:
  A grid 11x10, B grid 12x9, C one command queue (GSPLAT_TT_MAT_CQ1=0).
