"""Task #637: fit the flush latency L to t479's p100a Tracy (views 0:10, max-core ms/view):
sort_ol_emit off 1.42, ro (Morton, R8 D8) 2.07, ro48 (Morton, R4 D8) 1.88; run wait wfl per TRISC
launch 0.06 / 0.46 / 0.25 ms. Model: ms per record-time = 1.42 / mean over views of the off arm's
max end time (R8 D8); predicted emit = that x max end time of each arm. Two simulators:
end_max (one packer, ring wait only) and e3_max (3 TRISC owners t%3, 4 batch slots, ring wait).
Usage: python3 docs/emit-burst-model/calib.py docs/emit-burst-model/out-C110-v0-10.json"""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))["res"]
views = list(d)
Ls = sorted({int(k.split("/")[2]) for k in d[views[0]]["today"]["sim"]})
meas = {("morton", "8/8"): 2.07, ("morton", "4/8"): 1.88, ("today", "4/8"): 1.42}
print("L   off88 rec  | pred ms: off 8/8  off 4/8  ro 8/8  ro 4/8 | meas: 1.42  (off48 ~1.40)  2.07  1.88")
for key in ("end_max", "e3_max"):
  print(key)
  for L in Ls:
    em = lambda o, rd: np.mean([d[v][o]["sim"][f"{rd}/{L}"][key] for v in views])  # noqa: E731
    u = 1.42 / em("today", "8/8")
    print(f"{L:3d} {em('today', '8/8'):8.0f} | {1.42:.2f}  {u * em('today', '4/8'):.2f}  "
          f"{u * em('morton', '8/8'):.2f}  {u * em('morton', '4/8'):.2f}")
