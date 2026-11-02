"""Import VGC-Bench et collecte des versions d'exécution."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pokecoach_trainer.settings import PACKAGE_ROOT, get_settings

VGC_BENCH_COMMIT = "d79f9532947ac114dce1dda2456a590afcd375b2"
IMITATION_COMMIT = "56eb3b295da5d52b8f7277c6ba206913b19ab110"
POKE_ENV_COMMIT = "379a04628e42b2a2c94790c0788204b652f7ae1b"
REFERENCE_CHECKPOINT_SHA256 = "57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b"


@contextmanager
def vgc_bench_workdir() -> Iterator[Path]:
    """Place le CWD sur la racine vendor pour les fichiers `data/*.json`."""

    root = get_settings().vgc_bench_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VGC-Bench vendor absent: {root}")
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield root
    finally:
        os.chdir(previous)


def collect_runtime_versions() -> dict[str, str]:
    """Versions utiles pour manifestes et diagnostics."""

    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "trainer_package": (PACKAGE_ROOT / "pyproject.toml").exists() and "0.1.0" or "unknown",
        "vgc_bench_commit_expected": VGC_BENCH_COMMIT,
        "imitation_commit_expected": IMITATION_COMMIT,
        "poke_env_commit_expected": POKE_ENV_COMMIT,
    }
    try:
        out = subprocess.check_output(["uv", "--version"], text=True).strip()
        versions["uv"] = out
    except (OSError, subprocess.CalledProcessError):
        versions["uv"] = "unavailable"
    for mod, key in (
        ("torch", "torch"),
        ("stable_baselines3", "stable_baselines3"),
        ("imitation", "imitation"),
        ("poke_env", "poke_env"),
    ):
        try:
            m = __import__(mod)
            versions[key] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[key] = "not_installed"
    return versions


def import_policy_stack() -> None:
    """Force l'import des dépendances lourdes (tests d'environnement)."""

    with vgc_bench_workdir():
        import vgc_bench.src.policy  # noqa: F401
        import vgc_bench.src.policy_player  # noqa: F401
