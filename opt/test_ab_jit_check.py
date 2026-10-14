import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ab_jit_check as j  # noqa: E402


def _cache(tmp, name, kernels):
    for k, h in kernels.items():
        os.makedirs(tmp / name / "abc" / "kernels" / k / h)
    return str(tmp / name)


def test_differs(tmp_path):
    a = _cache(tmp_path, "a", {"blend": "111", "sort": "9"})
    b = _cache(tmp_path, "b", {"blend": "222", "sort": "9"})
    assert j.main(["--arm", f"base={a}", "--arm", f"cand={b}", "--kernel", "blend"]) == 0


def test_identical_edited_kernel_fails(tmp_path):
    a = _cache(tmp_path, "a", {"blend": "111", "sort": "9"})
    b = _cache(tmp_path, "b", {"blend": "111", "sort": "8"})
    assert j.main(["--arm", f"base={a}", "--arm", f"cand={b}", "--kernel", "blend"]) == 1


def test_all_identical_fails_even_if_kernel_missing_named(tmp_path):
    a = _cache(tmp_path, "a", {"blend": "111"})
    b = _cache(tmp_path, "b", {"blend": "111"})
    ok, lines = j.check([("a", j.kernel_hashes(a)), ("b", j.kernel_hashes(b))], ["blend"])
    assert not ok and any("all 1 kernels identical" in x for x in lines)


def test_missing_kernel_fails_and_listing_file(tmp_path):
    lst = tmp_path / "a.txt"
    lst.write_text("/x/ttmc-a/k/kernels/blend/111\n")
    b = _cache(tmp_path, "b", {"sort": "9"})
    assert j.main(["--arm", f"a={lst}", "--arm", f"b={b}", "--kernel", "blend"]) == 1


def _rao(tmp, ka, kb, la, lb, ma="cap=0", mb="cap=64"):
    a = _cache(tmp, "a", ka)
    b = _cache(tmp, "b", kb)
    (tmp / "a.log").write_text(la)
    (tmp / "b.log").write_text(lb)
    return j.main(["--runtime-arg-only", "--arm", f"base={a}", "--arm", f"cand={b}",
                   "--kernel", "blend", "--marker", f"base={ma}", "--marker", f"cand={mb}",
                   "--log", f"base={tmp / 'a.log'}", "--log", f"cand={tmp / 'b.log'}"])


def test_rao_pass_same_hash_markers_present(tmp_path):
    assert _rao(tmp_path, {"blend": "1"}, {"blend": "1"}, "x cap=0 y", "cap=64\n") == 0


def test_rao_fails_on_hash_diff(tmp_path):
    assert _rao(tmp_path, {"blend": "1"}, {"blend": "2"}, "cap=0", "cap=64") == 1


def test_rao_fails_on_missing_marker(tmp_path):
    assert _rao(tmp_path, {"blend": "1"}, {"blend": "1"}, "cap=0", "cap=0") == 1


def test_rao_fails_on_missing_kernel(tmp_path):
    assert _rao(tmp_path, {"sort": "1"}, {"sort": "1"}, "cap=0", "cap=64") == 1


def test_rao_requires_marker_and_log_per_arm(tmp_path, capsys):
    import pytest
    a = _cache(tmp_path, "a", {"blend": "1"})
    log = tmp_path / "a.log"
    log.write_text("x")
    with pytest.raises(SystemExit) as e:
        j.main(["--runtime-arg-only", "--arm", f"base={a}", "--arm", f"cand={a}",
                "--kernel", "blend", "--marker", "base=x", "--marker", "cand=y",
                "--log", f"base={log}"])
    assert e.value.code == 2
    assert "needs --marker and --log for each arm" in capsys.readouterr().err


import pytest


@pytest.mark.parametrize("ma,mb", [("", ""), ("", "cap=64"), ("cap=64", "cap=64"),
                                   ("cap=6", "cap=64"), ("cap=64", "cap=6")])
def test_rao_rejects_bad_markers(tmp_path, ma, mb):
    with pytest.raises(SystemExit) as e:
        _rao(tmp_path, {"blend": "1"}, {"blend": "1"}, "cap=64", "cap=64", ma, mb)
    assert e.value.code == 2


def test_rao_rejects_duplicate_arm_labels(tmp_path):
    a = _cache(tmp_path, "a", {"blend": "1"})
    (tmp_path / "a.log").write_text("cap=0")
    with pytest.raises(SystemExit) as e:
        j.main(["--runtime-arg-only", "--arm", f"base={a}", "--arm", f"base={a}",
                "--kernel", "blend", "--marker", "base=cap=0",
                "--log", f"base={tmp_path / 'a.log'}"])
    assert e.value.code == 2


def test_default_mode_unchanged_same_hash_fails(tmp_path):
    a = _cache(tmp_path, "a", {"blend": "1"})
    b = _cache(tmp_path, "b", {"blend": "1"})
    assert j.main(["--arm", f"base={a}", "--arm", f"cand={b}", "--kernel", "blend"]) == 1
