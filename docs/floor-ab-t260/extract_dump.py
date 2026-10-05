"""t260 (remote): compact the GSPLAT_TT_DUMP_PROJ dump to the valid blend records.
Keeps storage rows whose record looks written (words 9, 13-15 zero, 0 < opacity <= 1,
depth word == proj_m_depth). Output proj_dev.npz: sid (storage index), rec words 0-8,12
as float32, aabb u32, counts, M."""
import sys, numpy as np
d = sys.argv[1]
rec = np.fromfile(f"{d}/proj_m_blendrec.u32", np.uint32).reshape(-1, 16)
dep = np.fromfile(f"{d}/proj_m_depth.u32", np.uint32)
aabb = np.fromfile(f"{d}/proj_m_aabb.u32", np.uint32)
cnt = np.fromfile(f"{d}/pfwc_fuse_counts.u32", np.uint32).reshape(-1, 16)
M = int(np.fromfile(f"{d}/proj_M.u32", np.uint32)[0])
n = min(len(rec), len(dep))
op = rec[:n, 5].view(np.float32)
ok = (rec[:n, 9] == 0) & (rec[:n, 13:16] == 0).all(1) & (op > 0) & (op <= 1) & (rec[:n, 12] == dep[:n])
sid = np.nonzero(ok)[0]
f = rec[sid][:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 12]].view(np.float32)
np.savez_compressed(sys.argv[2], sid=sid, f=f, w1011=rec[sid][:, 10:12], aabb=aabb[sid], counts=cnt, M=M)
print("M", M, "valid_rows", len(sid), "counts_sum", int(cnt[:, 0].sum()))
