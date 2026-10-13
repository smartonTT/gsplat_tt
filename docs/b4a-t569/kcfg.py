#!/usr/bin/env python3
# t569: dump every tensix core's launch-message ring (kernel_config fields) and go state with tt-exalens.
#   triage-venv/bin/python kcfg.py <brisc firmware elf> [x,y ...]   (logical coords; default: all workers)
# Read-only (L1 reads). One line per (core, launch entry); '*' marks the entry at launch_msg_rd_ptr.
import sys
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import parse_elf
from ttexalens.memory_access import MemoryAccess
from ttexalens.coordinate import OnChipCoordinate

ctx = init_ttexalens()
dev = ctx.devices[0]
elf = parse_elf(sys.argv[1], ctx)
want = sys.argv[2:]
locs = ([OnChipCoordinate.create(w, dev, "logical") for w in want] if want
        else dev.get_block_locations("functional_workers"))
F = ("local_cb_offset", "remote_cb_offset", "min_remote_cb_start_index", "mode", "enables",
     "host_assigned_id", "brisc_noc_id", "brisc_noc_mode", "preload")
for loc in locs:
    name = loc.to_str("logical")
    try:
        m = elf.read_global("mailboxes", MemoryAccess.create_l1(loc))
        rd = int(m.launch_msg_rd_ptr)
        gi = int(m.go_message_index)
        print(f"core {name} rd_ptr={rd} go_idx={gi} go_signal=0x{int(m.go_messages[gi].signal):x}")
        for i in range(8):
            k = m.launch[i].kernel_config
            v = {f: int(getattr(k, f)) for f in F}
            base = int(k.kernel_config_base[0])
            txt = [int(k.kernel_text_offset[j]) for j in range(5)]
            rta = [(int(k.rta_offset[j].rta_offset), int(k.rta_offset[j].crta_offset)) for j in range(5)]
            ids = [int(k.watcher_kernel_ids[j]) for j in range(5)]
            print(f"  {'*' if i == rd else ' '}[{i}] base=0x{base:x} sem={int(k.sem_offset[0])} "
                  f"lcb={v['local_cb_offset']} rcb={v['remote_cb_offset']} minrcb={v['min_remote_cb_start_index']} "
                  f"mask=0x{int(k.local_cb_mask):x} txt={[hex(t) for t in txt]} rta={rta} ids={ids} "
                  f"en=0x{v['enables']:x} hid={v['host_assigned_id']} mode={v['mode']} noc={v['brisc_noc_id']}/"
                  f"{v['brisc_noc_mode']} ncsz16={int(k.ncrisc_kernel_size16)} pre={v['preload']}")
    except Exception as e:  # keep going: a corrupted core must not hide the others
        print(f"core {name} ERROR {type(e).__name__}: {e}")
