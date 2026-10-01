"""Keep the project_* / tile_assign_* stage-timer sub-buckets consistent across
render/host/stage_timers.h (Acc fields), the pybind stage_timings() dict in
render/host/render.cpp, and the _SUB_ORDER printout in render/run.py. A key
missing from any one of them silently prints 0.000 in TTW_TIMING.

    python3 tests/unit/test_stage_timer_keys.py      (or pytest)
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGES = ("project", "tile_assign")


def _acc_fields():
    src = (ROOT / "render/host/stage_timers.h").read_text()
    return set(re.findall(r"double ((?:%s)_\w+) = 0\.0;" % "|".join(STAGES), src))


def _binding_keys():
    src = (ROOT / "render/host/render.cpp").read_text()
    pairs = re.findall(r'd\["((?:%s)_\w+)"\] = a\.(\w+);' % "|".join(STAGES), src)
    for key, field in pairs:
        assert key == field, (key, field)
    return {k for k, _ in pairs}


def _run_py_keys():
    src = (ROOT / "render/run.py").read_text()
    block = src[src.index("_SUB_ORDER = {"):]
    block = block[:block.index("}") + 1]
    keys = set()
    for stage in STAGES:
        m = re.search(r'"%s": \[(.*?)\]' % stage, block, re.S)
        assert m, stage
        keys |= {f"{stage}_{k}" for k in re.findall(r'"(\w+)"', m.group(1))}
    return keys


def test_stage_timer_keys_consistent():
    acc, binding, run_py = _acc_fields(), _binding_keys(), _run_py_keys()
    assert acc, "no project_/tile_assign_ fields found in stage_timers.h"
    assert acc == binding, (acc ^ binding)
    assert acc == run_py, (acc ^ run_py)


_SUB_PREFIXES = ("sort_", "project_", "tile_assign_")


def _top_level(keys):
    return {k for k in keys if not k.startswith(_SUB_PREFIXES) and k != "view_total"}


def test_top_level_stage_keys_consistent():
    """The STAGES buckets (head ... tail, incl. task #90's `mat`) must match too."""
    acc = _top_level(re.findall(r"double (\w+) = 0\.0;",
                                (ROOT / "render/host/stage_timers.h").read_text()))
    binding = _top_level(k for k, f in re.findall(
        r'd\["(\w+)"\] = a\.(\w+);', (ROOT / "render/host/render.cpp").read_text())
        if k == f)
    src = (ROOT / "render/run.py").read_text()
    m = re.search(r"_STAGE_ORDER = \[(.*?)\]", src, re.S)
    assert m, "_STAGE_ORDER not found in render/run.py"
    run_py = set(re.findall(r'"(\w+)"', m.group(1)))
    assert "mat" in acc, "stage_timers.h has no `mat` bucket"
    assert acc == binding, (acc ^ binding)
    assert acc == run_py, (acc ^ run_py)


if __name__ == "__main__":
    test_stage_timer_keys_consistent()
    test_top_level_stage_keys_consistent()
    print(f"OK {len(_acc_fields())} sub-bucket keys + top-level stages consistent")
