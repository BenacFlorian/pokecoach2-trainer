"""Export de packages modèle et inventaire."""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from pokecoach_trainer import __version__
from pokecoach_trainer.compatibility import build_contract
from pokecoach_trainer.schemas import ExperimentConfig, ModelConfig, ModelManifest, TrainingResult
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import sha256_file, validate_id, write_json_atomic


def export_model(
    result: TrainingResult,
    config: ExperimentConfig,
    model_id: str | None = None,
) -> ModelManifest:
    """Publie un package vérifié sous `storage/models/{id}/`."""

    model_id = validate_id(model_id or result.run_id)
    contract = build_contract(config.model)
    ckpt = Path(result.final_checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    ckpt_hash = sha256_file(ckpt)

    out_dir = storage_path("models", model_id)
    if out_dir.exists():
        raise FileExistsError(f"modèle déjà publié: {model_id}")
    out_dir.mkdir(parents=True)
    pkg_dir = out_dir / "package"
    pkg_dir.mkdir()
    shutil.copy2(ckpt, pkg_dir / "model.zip")
    shutil.copy2(result.metrics_path, pkg_dir / "metrics.jsonl")

    manifest = ModelManifest(
        model_id=model_id,
        architecture=config.model,
        dataset_id=config.dataset_id,
        experiment_name=config.name,
        contract_fingerprint=contract.fingerprint,
        checkpoint_sha256=ckpt_hash,
        trainer_version=__version__,
        vgc_bench_commit=contract.vgc_bench_commit,
    )
    write_json_atomic(pkg_dir / "manifest.json", manifest.model_dump(mode="json"))
    (pkg_dir / "README.md").write_text(
        f"# Modèle {model_id}\n\nContrat: `{contract.fingerprint[:16]}…`\n",
        encoding="utf-8",
    )

    report = verify_reload(pkg_dir / "model.zip", pkg_dir / "manifest.json")
    if not report.ok:
        shutil.rmtree(out_dir)
        raise RuntimeError(f"reload échoué: {report.message}")

    archive = out_dir / f"{model_id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in pkg_dir.iterdir():
            zf.write(file, arcname=file.name)
    write_json_atomic(
        out_dir / "published.json", {"model_id": model_id, "trainer_reload": "passed"}
    )
    return manifest


def verify_reload(model_path: Path, manifest_path: Path):
    from pokecoach_trainer.schemas import ReloadReport

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pokecoach_trainer.reload_check",
            str(model_path),
            str(manifest_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        # to check : pourquoi ModelConfig est pas importé ?
        return ReloadReport(
            ok=False,
            checkpoint_sha256="",
            param_count=0,
            architecture=ModelConfig(),
            message=proc.stderr or proc.stdout or "reload_check failed",
        )
    import json

    data = json.loads(proc.stdout)
    return ReloadReport.model_validate(data)


def list_models() -> list[dict]:
    root = storage_path("models")
    if not root.is_dir():
        return []
    items = []
    for path in sorted(root.iterdir()):
        pub = path / "published.json"
        if pub.is_file():
            items.append(
                __import__(
                    "pokecoach_trainer.storage", fromlist=["read_json"]
                ).read_json(pub)
            )
    return items
