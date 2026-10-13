"""Task #649: L-A (#639) and L-A' (#643) share the blend's after-hook; both default on."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _src(rel):
    return open(os.path.join(ROOT, rel)).read()


def test_defaults_on():
    env = _src("render/host/env_config.h")
    assert 'env_uint("GSPLAT_TT_LA_PREFETCH", 1u)' in env
    assert 'env_uint("GSPLAT_TT_SORT_IN_FOLD", 1u)' in env


def test_hook_runs_la_first_then_sort_in_fold():
    src = _src("render/host/render.cpp")
    m = re.search(r"blend_set_after_enqueue_hook\(\[\] \{(.*?)\n            \}\);", src, re.S)
    assert m, "fold after-hook not found"
    body = m.group(1)
    la, sif = body.find("tile_assign_fold_prefetch("), body.find("tile_assign_fold_enqueue_sort()")
    assert 0 <= la < sif, "L-A must run first; L-A' only when it did not"
    # a successful L-A prefetch returns before L-A' (no second sort enqueue)
    assert "return;" in body[la:sif]


def test_la_hook_sort_drains_before_regrow():
    # #552 rule (#643): a sort enqueued from the hook may regrow buffers a queued mat+blend reads.
    src = _src("render/host/tile_assign_device.cpp")
    m = re.search(r"auto enqueue_sort = \[&\]\(\) \{(.*?)\};", src, re.S)
    assert m and "/*drain_before_grow=*/prefetch" in m.group(1)
