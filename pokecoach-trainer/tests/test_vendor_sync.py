import hashlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches" / "logs2trajs-provenance.patch"
SYNC_SCRIPT = ROOT / "scripts" / "sync_vgc_bench.sh"
PATCHED_SOURCE = (
    Path(__file__).resolve().parents[1] / "vendor/vgc-bench/vgc_bench/logs2trajs.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_upstream_without_provenance(base: Path) -> Path:
    upstream = base / "upstream"
    vgc_dir = upstream / "vgc_bench"
    vgc_dir.mkdir(parents=True)
    target = vgc_dir / "logs2trajs.py"
    shutil_copy = PATCHED_SOURCE.read_text(encoding="utf-8")
    target.write_text(shutil_copy, encoding="utf-8")
    subprocess.run(
        ["patch", "-p1", "-d", str(upstream), "--reverse", "--batch"],
        input=PATCH.read_text(encoding="utf-8"),
        text=True,
        check=True,
    )
    assert "TaggedTrajectory" not in target.read_text(encoding="utf-8")
    (upstream / "pyproject.toml").write_text("[project]\nname='vgc-bench'\n", encoding="utf-8")
    return upstream


def test_vendor_sync_preserves_provenance_patch(tmp_path):
    if not PATCH.is_file():
        pytest.skip("patch provenance absent")

    upstream = _prepare_upstream_without_provenance(tmp_path)
    dest = tmp_path / "dest"

    def sync_once():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["rsync", "-a", "--delete", "--exclude", ".git", f"{upstream}/", f"{dest}/"],
            check=True,
        )
        subprocess.run(
            ["patch", "-p1", "-d", str(dest), "--forward", "--batch"],
            input=PATCH.read_text(encoding="utf-8"),
            text=True,
            check=True,
        )

    sync_once()
    target = dest / "vgc_bench" / "logs2trajs.py"
    h1 = _sha256(target)
    assert "TaggedTrajectory" in target.read_text(encoding="utf-8")

    sync_once()
    h2 = _sha256(target)
    assert h1 == h2


def test_sync_script_rejects_missing_source(tmp_path):
    env = {**subprocess.os.environ, "POKECOACH_VGC_BENCH_ROOT": str(tmp_path / "missing")}
    result = subprocess.run(
        ["bash", str(SYNC_SCRIPT)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
