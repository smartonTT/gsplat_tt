"""render/run.py main() (task #292) ends the process with os._exit on every
path, so an uncaught render error never reaches native static teardown (the
tt-metal ShmResourceTracker / MeshWorkload destructors crash there after the
device was opened). No device: _main is stubbed and os._exit recorded.

    python3 tests/unit/test_run_error_exit.py      (or pytest)
"""
import contextlib
import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_run_back_to_back import _load_run  # noqa: E402


class _Exited(Exception):
    def __init__(self, rc):
        super().__init__(rc)
        self.rc = rc


def _run_main(body):
    mod = _load_run()
    mod._main = body
    real_exit = mod.os._exit

    def fake_exit(rc):
        raise _Exited(rc)

    err = io.StringIO()
    mod.os._exit = fake_exit
    try:
        with contextlib.redirect_stderr(err):
            mod.main()
    except _Exited as e:
        return e.rc, err.getvalue()
    finally:
        mod.os._exit = real_exit
    raise AssertionError("main() returned instead of calling os._exit")


def test_uncaught_error_exits_1_with_traceback():
    def body():
        raise RuntimeError("render_clean: device sort failed")
    rc, err = _run_main(body)
    assert rc == 1
    assert "Traceback" in err and "device sort failed" in err


def test_sys_exit_message_exits_1():
    def body():
        sys.exit("[run] --view-range 5:5 selects no views")
    rc, err = _run_main(body)
    assert rc == 1 and "selects no views" in err


def test_sys_exit_code_kept():
    def body():
        sys.exit(3)
    assert _run_main(body)[0] == 3


def test_normal_return_exits_0():
    assert _run_main(lambda: None)[0] == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
