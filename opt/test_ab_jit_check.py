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
