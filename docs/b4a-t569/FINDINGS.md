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
