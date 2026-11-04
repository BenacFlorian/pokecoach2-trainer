"""Export de packages modèle et inventaire."""

from __future__ import annotations

import json
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


def architecture_from_checkpoint(checkpoint_path: Path) -> ModelConfig:
    """Lit l'architecture effective depuis un checkpoint SB3 (sans publier)."""

    from pokecoach_trainer.upstream import vgc_bench_workdir

    with vgc_bench_workdir():
        from stable_baselines3 import PPO
        from vgc_bench.src.policy import MaskedActorCriticPolicy

        loaded = PPO.load(str(checkpoint_path), device="cpu")
        policy = loaded.policy
        if not isinstance(policy, MaskedActorCriticPolicy):
            raise TypeError("policy attendue: MaskedActorCriticPolicy")
        pi = policy.pi_features_extractor
        return ModelConfig(
            d_model=int(policy.action_net.in_features),
            num_layers=int(pi.embed_layers),
            num_heads=int(pi.num_heads),
            embed_len=int(pi.embed_len),
            choose_on_teampreview=bool(policy.choose_on_teampreview),
        )


def _promote_staging(staging_root: Path, out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for item in staging_root.iterdir():
        dest = out_root / item.name
        if dest.exists():
            raise FileExistsError(f"destination déjà présente: {dest}")
        shutil.move(str(item), str(dest))


def export_model(
    result: TrainingResult,
    config: ExperimentConfig,
    model_id: str | None = None,
) -> ModelManifest:
    """Prépare sous staging, vérifie le reload CPU, puis publie sous `storage/models/{id}/`."""

    model_id = validate_id(model_id or result.run_id)
    ckpt = Path(result.final_checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    architecture = architecture_from_checkpoint(ckpt)
    contract = build_contract(architecture)
    ckpt_hash = sha256_file(ckpt)

    out_dir = storage_path("models", model_id)
    if out_dir.exists():
        raise FileExistsError(f"modèle déjà publié: {model_id}")

    staging_root = storage_path("staging", model_id)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    pkg_dir = staging_root / "package"
    pkg_dir.mkdir()

    try:
        shutil.copy2(ckpt, pkg_dir / "model.zip")
        shutil.copy2(result.metrics_path, pkg_dir / "metrics.jsonl")

        manifest = ModelManifest(
            model_id=model_id,
            architecture=architecture,
            dataset_id=config.dataset_id,
            experiment_name=config.name,
            contract_fingerprint=contract.fingerprint,
            checkpoint_sha256=ckpt_hash,
            trainer_version=__version__,
            vgc_bench_commit=contract.vgc_bench_commit,
            trainer_reload="pending",
            reserved_test="pending",
            pokecoach_compatibility="not_tested",
        )
        write_json_atomic(pkg_dir / "manifest.json", manifest.model_dump(mode="json"))
        (pkg_dir / "README.md").write_text(
            f"# Modèle {model_id}\n\nContrat: `{contract.fingerprint[:16]}…`\n",
            encoding="utf-8",
        )
        (pkg_dir / "test_report.json").write_text(
            json.dumps({"status": "reserved", "schema_version": 1}),
            encoding="utf-8",
        )

        report = verify_reload(pkg_dir / "model.zip", pkg_dir / "manifest.json")
        if not report.ok:
            raise RuntimeError(f"reload échoué: {report.message}")

        manifest = manifest.model_copy(update={"trainer_reload": "passed"})
        write_json_atomic(pkg_dir / "manifest.json", manifest.model_dump(mode="json"))

        archive = staging_root / f"{model_id}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(pkg_dir.iterdir()):
                zf.write(file, arcname=file.name)

        write_json_atomic(
            staging_root / "published.json",
            {
                "model_id": model_id,
                "checkpoint_sha256": ckpt_hash,
                "contract_fingerprint": contract.fingerprint,
                "trainer_reload": "passed",
                "reserved_test": "pending",
                "pokecoach_compatibility": "not_tested",
            },
        )
        _promote_staging(staging_root, out_dir)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)

    return manifest


def verify_reload(model_path: Path, manifest_path: Path):
    from pokecoach_trainer.schemas import ReloadReport

    try:
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
    except subprocess.TimeoutExpired as exc:
        return ReloadReport(
            ok=False,
            checkpoint_sha256="",
            param_count=0,
            architecture=ModelConfig(d_model=128, num_layers=2, num_heads=4, embed_len=32),
            message=f"reload_check timeout: {exc}",
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "reload_check failed").strip()
        return ReloadReport(
            ok=False,
            checkpoint_sha256="",
            param_count=0,
            architecture=ModelConfig(d_model=128, num_layers=2, num_heads=4, embed_len=32),
            message=detail,
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return ReloadReport(
            ok=False,
            checkpoint_sha256="",
            param_count=0,
            architecture=ModelConfig(d_model=128, num_layers=2, num_heads=4, embed_len=32),
            message=f"sortie reload_check non JSON: {exc}",
        )
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
