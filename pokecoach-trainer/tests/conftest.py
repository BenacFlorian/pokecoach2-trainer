# ruff: noqa: E402
import os
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_PACKAGE_ROOT / "vendor" / "vgc-bench")

import pytest

from pokecoach_trainer.settings import clear_settings_cache
from pokecoach_trainer.upstream import vgc_bench_workdir


@pytest.fixture(autouse=True)
def _reset_settings():
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture(autouse=True)
def _vgc_cwd():
    with vgc_bench_workdir():
        yield
