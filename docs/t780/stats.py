"""t772 (copied for t780): per-pair cycles of the blend-backward stage counters (bench stats.json).
   python3 docs/t772/stats.py <stats.json>...   busiest core (max tot) and the all-core mean."""
import json
import sys

PER_PAIR = ["rec", "setup", "s_load", "s_stage9", "s_stage3", "s_mbox", "s_zero", "body", "fin", "pack"]


def summary(path):
    js = json.load(open(path))
    cols, cores = js["cols"], js["cores"]
    rows = [dict(zip(cols, c)) for c in cores]
    busy = max(rows, key=lambda r: r["tot"])
    out = {"file": path, "stats_mode": js.get("stats_mode"), "core": busy["core"], "tot": busy["tot"],
           "np": busy["np"], "nv_per_pair": round(busy["nv"] / max(busy["np"], 1), 2)}
    for k in PER_PAIR:
        if k in busy:
            out[k] = round(busy[k] / max(busy["np"], 1), 1)
            out["mean_" + k] = round(sum(r[k] for r in rows) / max(sum(r["np"] for r in rows), 1), 1)
    sub = sum(busy.get(k, 0) for k in PER_PAIR[2:7])
    out["setup_minus_subs"] = round((busy["setup"] - sub) / max(busy["np"], 1), 1)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            print(json.dumps(summary(p)))
        except (OSError, ValueError, KeyError) as e:
            print(json.dumps({"file": p, "error": repr(e)}))
