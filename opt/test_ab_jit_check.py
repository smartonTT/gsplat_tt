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
