"""Task #469: GSPLAT_TT_K2_TRISC (#274, K2 on the idle TRISCs) is on by default; =0 turns it off."""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "render/host/tile_assign_device.cpp"


def _trisc_enabled(env_value):
    """Evaluate the host's `trisc` condition for one env value (None = unset), diet and dual on."""
    m = re.search(r"const bool trisc = (.*?);", SRC.read_text(), re.S)
    assert m, "trisc condition not found"
    expr = m.group(1)
    expr = expr.replace("&&", " and ").replace("||", " or ").replace("!=", " != ").replace("==", " == ")
    expr = expr.replace("std::atoi(et)", "int(et)").replace("nullptr", "None")
    expr = re.sub(r"\bctx\.dual\b", "True", re.sub(r"\bdiet\b", "True", expr))
    return eval(expr, {}, {"et": env_value, "None": None, "True": True})  # noqa: S307


def test_default_on():
    assert _trisc_enabled(None)


def test_zero_off_one_on():
    assert not _trisc_enabled("0")
    assert _trisc_enabled("1")
