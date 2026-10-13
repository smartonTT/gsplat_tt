# t567: B4a K2 fold/shadow hang under p150 ETH dispatch (bh-30) — findings so far

Rounds (bh-30, viewer stopped only for each probe, restarted after each; localhost:8091 -> 200 every time):
- R1 repro (out/repro): ETH 12x10 hangs (E1/E2/E6), ETH stage grid capped 11x10 (E3) and 12x9 (E7) still hang,
  worker dispatch 11x10 (E4) passes verify (gids 2090575 words equal). Core count is not the cause.
- R2 (out/r2): ETH + shadow K2 with fold OFF hangs (H1); ETH overlay with worker dispatch + fold passes (H2);
  ETH fold without CQ1 hangs (H3); ETH fold hangs (H4). Fold view 1 (nseg=0, same binary, K2 skipped) completes.
  Host is stuck in Finish/EventSynchronize on the mat+blend program.
- R3 (out/r3, tt-triage via tt-exalens 0.3.19 at the hang):
  * All matblend BRISCs (kernel 47) are in FIRMWARE, never entered the kernel: PCs 0x45d0/0x45d2/0x46ae resolve
    (riscv-tt-elf-addr2line on firmware/brisc/brisc.elf) to experimental::setup_remote_cb_interfaces
    (circular_buffer_init.h:189/201) and resize_remote_sender_cb_interface -> update_pages_sent
    (remote_circular_buffer.h:45/140), i.e. the per-launch remote-CB setup in brisc.cc:508-512.
    Our programs use no remote CBs, so this loop should run zero times (min_remote_cb_start_index == 64).
  * NCRISC/TRISCs on those cores are DONE-waiting (ncrisc wait_for_brisc_notification).
  * Core (2,4) BRISC NOC0: HW NIU_MST_NONPOSTED_WR_REQ_SENT 224930 vs SW counter 8338: BRISC issued ~216k
    untracked NoC writes (update_pages_sent sends sem incs to "receivers" read from garbage config).
  * check_binary_integrity: core (4,0) still on pfwc kernel 38 with .text mismatches in its pfwc BRISC/NCRISC/TRISC
    binaries; core (6,1) NCRISC firmware .text mismatch at 0x5ae0. Stray NoC writes are corrupting L1 on other cores.
  * R2-triage: "All NOCs hung", ARC heartbeat 0 (the chip is wedged by the stray writes).

Cause (as far as the data shows): under ETH dispatch, the BRISC launch path of the K2-carrying mat+blend program
reads a bad kernel_config (min_remote_cb_start_index / remote_cb_offset relative to kernel_config_base), so
setup_remote_cb_interfaces walks garbage "remote CB" configs and fires NoC semaphore writes at garbage
addresses, which corrupt other cores' kernel text/firmware and wedge the NoC. It is not K2's own code
(BRISC never reaches the kernel), not CQ1, not the pfwc reorder, not grid size. Leading explanation: the K2-enlarged
program's kernel-config image (CB configs up to CB 62 + RTAs + BRISC text) overruns / races the kernel-config
ring under the ETH-dispatch overlay (ttm-eth12-t427), so the launch message points into a region being rewritten
by the next program's preload (consistent with (4,0)'s pfwc .text being overwritten while it still runs).

Next probe (not yet run): at the hang, read the launch msg kernel_config on a stuck core with tt-exalens
(kernel_config_base, local_cb_offset, remote_cb_offset, min_remote_cb_start_index, local_cb_mask, kernel_text_offset[])
and compare with worker dispatch; print the program's kernel-config size with and without K2 vs the tensix
kernel-config ring size in the ETH overlay. Candidate fixes to A/B: move kB4aCb (62) below 32 / drop the extra CB,
shrink the program's config image, or bound the ring in the ETH overlay.
