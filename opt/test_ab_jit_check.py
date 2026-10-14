import os
import ab_jit_check as m


def _arm(root, h, elf):
    d = root / "kernels" / "k1" / h / "trisc0"
    d.mkdir(parents=True)
    (d / "trisc0.elf").write_bytes(elf)
    return str(root)


def _run(a, b):
    arms = [("a", m.kernel_hashes(a)), ("b", m.kernel_hashes(b))]
    return m.check(arms, ["k1"], [m.elf_md5s(a), m.elf_md5s(b)])


def test_same_hash_different_elf_passes(tmp_path):
    a = _arm(tmp_path / "a", "h1", b"one")
    b = _arm(tmp_path / "b", "h1", b"two")
    ok, lines = _run(a, b)
    assert ok and "ELFs differ" in lines[0] and "trisc0/trisc0.elf" in lines[0]


def test_same_hash_same_elf_fails(tmp_path):
    a = _arm(tmp_path / "a", "h1", b"one")
    b = _arm(tmp_path / "b", "h1", b"one")
    ok, lines = _run(a, b)
    assert not ok and lines[0].startswith("A/A")


def test_different_hash_ok(tmp_path):
    a = _arm(tmp_path / "a", "h1", b"one")
    b = _arm(tmp_path / "b", "h2", b"one")
    assert _run(a, b)[0]


def _swap_arms(tmp_path):
    for arm, names in (("a", ["shared", "fused"]), ("b", ["shared", "mb"])):
        for n in names:
            (tmp_path / arm / "kernels" / n / "h1").mkdir(parents=True)
    return str(tmp_path / "a"), str(tmp_path / "b")


def test_kernel_swap(tmp_path):
    a, b = _swap_arms(tmp_path)
    arms = [("a", m.kernel_hashes(a)), ("b", m.kernel_hashes(b))]
    assert not m.check(arms, ["fused", "mb"], [{}, {}])[0]
    ok, lines = m.check_swap(arms, ["fused", "mb"])
    assert ok and "only in a" in lines[0] and "differ: []" in lines[2]
    assert not m.check_swap(arms, ["shared"])[0]
    assert m.main(["--arm", f"a={a}", "--arm", f"b={b}", "--kernel", "fused",
                   "--allow-kernel-swap"]) == 0
