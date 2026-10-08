"""python3 opt/test_md5_golden.py — task #409: md5 golden picked from the run's stage grid."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import md5_golden as g

ROOT = Path(__file__).resolve().parent.parent
W11 = (ROOT / "docs/matblend-ready-t273/t289/md5-golden-906e0435.txt").read_bytes()
E12 = (ROOT / "docs/eth-dispatch-t397/out/md5-r1-E.txt").read_bytes()
W10 = (ROOT / "docs/eth-dispatch-t397/out397b/md5-W10.txt").read_bytes()
LOG_W = "x\n[DEV] dispatch worker (worker), command queues 2, compute grid 11x10\ny\n"
LOG_E = "[DEV] dispatch eth (auto: card has ETH cores, overlay active, card p150b), command queues 2, compute grid 12x10\n"
LOG_E11 = "[DEV] dispatch eth (eth (forced)), command queues 2, compute grid 12x10, stage grid capped to 11x10\n"
LOG_OLD = "[DEV] command queues 1, compute grid 11x10\n"

assert g.grid_from_log(LOG_W) == "11x10"
assert g.grid_from_log(LOG_E) == "12x10"
assert g.grid_from_log(LOG_E11) == "11x10", "a capped stage grid wins"
assert g.grid_from_log(LOG_OLD) == "11x10", "pre-#383 log line"
assert g.grid_from_log("no device line") is None

assert g.list_md5(W11) == "906e0435" and g.list_md5(E12) == "39d84b28"
# Each grid takes its own golden.
assert g.check(W11, LOG_W)[0] == 0
assert g.check(E12, LOG_E)[0] == 0
assert g.check(W11, LOG_E11)[0] == 0, "eth at 11x10 = worker golden"
# A golden from the other grid is a mismatch, not a pass.
assert g.check(E12, LOG_W)[0] == 1
assert g.check(W11, LOG_E)[0] == 1
# Grid with no golden, or no grid in the log: fail (never skip).
assert g.check(W10, "[DEV] dispatch worker (worker), command queues 2, compute grid 10x10\n")[0] == 2
assert g.check(W11, "")[0] == 2
rc, line = g.check(W11[:-10], LOG_W)
assert rc == 1 and "MD5_GOLDEN_FAIL" in line, line
print("test_md5_golden: OK")
