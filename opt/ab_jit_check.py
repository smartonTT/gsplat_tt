#!/usr/bin/env python3
"""A/B gate JIT-hash check (#656).

After an A/B run, list each arm's JIT cache (kernels/<name>/<hash>) and fail
when the kernel(s) the gate edits have the same hash set in both arms: that run
was an A/A comparison and its delta is noise (#647/#654).

Each arm is a JIT cache directory, or a text file with one path per line
(e.g. `find <cache> -path '*/kernels/*' -maxdepth 6 -type d` saved on the box).

  python3 opt/ab_jit_check.py --arm base=<cache|listing> --arm cand=<cache|listing> \
      --kernel <name> [--kernel <name> ...]

Exit 0: every --kernel differs between the arms. 1: some --kernel is identical
or missing (the gate must fail or be flagged A/A). 2: usage error.

Runtime-arg-only A/Bs (the knob is a runtime arg/env, e.g. #684
GSPLAT_TT_TEST_PREGROW_TILE_CAP): identical hashes are then correct, so the
check flips. It requires the same hashes for every --kernel AND each arm's
--marker string in that arm's --log file(s), proving the knob took effect:

  python3 opt/ab_jit_check.py --runtime-arg-only --arm base=<c> --arm cand=<c> \
      --kernel <name> --marker base='<str>' --marker cand='<str>' \
      --log base=<run.log> [--log base=...] --log cand=<run.log> [...]
"""
import argparse
import os
import re
import sys

_KERNEL_RE = re.compile(r"(?:^|/)kernels/([^/]+)/([^/\s]+)")


def _paths(src):
    if os.path.isdir(src):
        for root, dirs, _ in os.walk(src):
            for d in dirs:
                yield os.path.join(root, d)
    else:
        with open(src) as f:
            for line in f:
                yield line.strip()


def kernel_hashes(src):
    """Map kernel name -> set of hashes found under src."""
    out = {}
    for p in _paths(src):
        m = _KERNEL_RE.search(p)
        if m:
            out.setdefault(m.group(1), set()).add(m.group(2))
    return out


def check(arms, kernels):
    """Return (ok, report lines). arms: list of (label, {name: hashes})."""
    lines, ok = [], True
    (la, a), (lb, b) = arms
    for k in kernels:
        ha, hb = a.get(k, set()), b.get(k, set())
        if not ha or not hb:
            ok = False
            lines.append(f"MISSING {k}: {la}={sorted(ha)} {lb}={sorted(hb)}")
        elif ha == hb:
            ok = False
            lines.append(f"A/A {k}: same hashes {sorted(ha)} in {la} and {lb}")
        else:
            lines.append(f"OK {k}: {la}={sorted(ha)} {lb}={sorted(hb)}")
    same = sorted(n for n in set(a) & set(b) if a[n] == b[n])
    if set(a) == set(b) and len(same) == len(a):
        ok = False
        lines.append(f"A/A all {len(a)} kernels identical between {la} and {lb}")
    return ok, lines


def check_runtime_arg_only(arms, kernels, markers, logs):
    """Hashes must match per --kernel; each arm's marker must be in its logs.
    markers: {label: str}; logs: {label: [text, ...]}."""
    lines, ok = [], True
    (la, a), (lb, b) = arms
    for k in kernels:
        ha, hb = a.get(k, set()), b.get(k, set())
        if not ha or not hb:
            ok = False
            lines.append(f"MISSING {k}: {la}={sorted(ha)} {lb}={sorted(hb)}")
        elif ha != hb:
            ok = False
            lines.append(f"DIFF {k}: {la}={sorted(ha)} {lb}={sorted(hb)} (runtime-arg-only expects same)")
        else:
            lines.append(f"OK {k}: same hashes {sorted(ha)} (runtime-arg-only)")
    for label in (la, lb):
        m = markers.get(label)
        texts = logs.get(label, [])
        if m is None or not texts:
            ok = False
            lines.append(f"MISSING marker/log for {label}")
        elif any(m in t for t in texts):
            lines.append(f"OK marker {label}: {m!r} found")
        else:
            ok = False
            lines.append(f"NO-MARKER {label}: {m!r} not in its log(s)")
    return ok, lines


def _kv(ap, specs, what):
    out = []
    for x in specs or []:
        if "=" not in x:
            ap.error(f"--{what} needs label=value")
        out.append(tuple(x.split("=", 1)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", required=True, help="label=path")
    ap.add_argument("--kernel", action="append", required=True)
    ap.add_argument("--runtime-arg-only", action="store_true",
                    help="knob is runtime-only: require same hashes + per-arm log markers")
    ap.add_argument("--marker", action="append", help="label=string (runtime-arg-only)")
    ap.add_argument("--log", action="append", help="label=path (runtime-arg-only)")
    args = ap.parse_args(argv)
    if len(args.arm) != 2 or any("=" not in x for x in args.arm):
        ap.error("need exactly two --arm label=path")
    arms = []
    for spec in args.arm:
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            ap.error(f"{path} not found")
        arms.append((label, kernel_hashes(path)))
    if args.runtime_arg_only:
        labels = {lab for lab, _ in arms}
        if len(labels) != 2:
            ap.error("--runtime-arg-only needs two distinct --arm labels")
        markers = dict(_kv(ap, args.marker, "marker"))
        logs = {}
        for lab, path in _kv(ap, args.log, "log"):
            if not os.path.isfile(path):
                ap.error(f"{path} not found")
            with open(path, errors="replace") as f:
                logs.setdefault(lab, []).append(f.read())
        if set(markers) != labels or set(logs) != labels:
            ap.error("--runtime-arg-only needs --marker and --log for each arm label")
        ma, mb = (markers[lab] for lab, _ in arms)
        if not ma or not mb:
            ap.error("--marker must not be empty")
        if ma in mb or mb in ma:
            ap.error("--marker strings must differ and neither may contain the other")
        ok, lines = check_runtime_arg_only(arms, args.kernel, markers, logs)
        print("\n".join(lines))
        print("JIT-HASH OK (runtime-arg-only)" if ok
              else "JIT-HASH FAIL: runtime-arg-only check failed")
        return 0 if ok else 1
    ok, lines = check(arms, args.kernel)
    print("\n".join(lines))
    print("JIT-HASH OK" if ok else "JIT-HASH FAIL: gate is A/A for the edited kernel(s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
