import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nano-mle-test-mpl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def pytest_runtest_setup(item):
    # No test is allowed to construct a live language-model backend.
    from unittest.mock import patch

    item._no_live_backend = patch("nano_mle.agents.DSPyBackend.__init__",
                                  side_effect=AssertionError("Live model calls are forbidden in tests"))
    item._no_live_backend.start()


def pytest_runtest_teardown(item):
    item._no_live_backend.stop()
import tempfile
os.environ["NANO_MLE_MISSING_LIBRARIES"] = os.path.join(tempfile.mkdtemp(), "missing_libraries.json")
