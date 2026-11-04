"""CLI du Trainer (diagnostic, données, entraînement, export)."""

from __future__ import annotations

import os

from pokecoach_trainer.settings import PACKAGE_ROOT, get_settings

# vgc_bench charge data/*.json relativement au CWD (vendor).
os.chdir(get_settings().vgc_bench_root.resolve())

import argparse
import json
import sys
from pathlib import Path

import yaml

from pokecoach_trainer import __version__
from pokecoach_trainer.archive_service import create_snapshot, import_legacy_logs
from pokecoach_trainer.artifact_service import export_model
from pokecoach_trainer.base_model_service import register_base_model
from pokecoach_trainer.collection_service import submit_collection
from pokecoach_trainer.dataset_service import (
    create_dataset,
    get_dataset,
    import_logs,
    inspect_decision,
    prepare_dataset,
    submit_preparation,
)
from pokecoach_trainer.jobs import get_job
from pokecoach_trainer.schemas import (
    BaseModelSource,
    CollectionConfig,
    DatasetRecipe,
    ExperimentConfig,
    InitializationConfig,
    ModelConfig,
    PrepareConfig,
    SnapshotRequest,
    TrainingConfig,
)
from pokecoach_trainer.settings import (
    ensure_storage_layout,
    get_settings,
    load_dotenv_explicit,
)
from pokecoach_trainer.storage import read_json, sha256_file
from pokecoach_trainer.training import train_bc
from pokecoach_trainer.upstream import (
    REFERENCE_CHECKPOINT_SHA256,
    collect_runtime_versions,
    import_policy_stack,
    vgc_bench_workdir,
)


def _cmd_doctor(args: argparse.Namespace) -> int:
    load_dotenv_explicit()
    versions = collect_runtime_versions()
    print(json.dumps(versions, indent=2))
    import_policy_stack()
    print("imports: OK")
    ckpt_path = args.reference_checkpoint
    if ckpt_path:
        path = Path(ckpt_path)
        if not path.is_file():
            print(f"checkpoint introuvable: {path}", file=sys.stderr)
            return 1
        digest = sha256_file(path)
        print(f"checkpoint sha256={digest}")
        if digest.lower() != REFERENCE_CHECKPOINT_SHA256.lower():
            print("AVERTISSEMENT: hash différent du manifeste PokeCoach de référence")
        with vgc_bench_workdir():
            from stable_baselines3 import PPO
            from vgc_bench.src.policy import MaskedActorCriticPolicy

            loaded = PPO.load(str(path), device="cpu")
            policy = loaded.policy
            assert isinstance(policy, MaskedActorCriticPolicy)
            pi = policy.pi_features_extractor
            print(
                json.dumps(
                    {
                        "policy_class": type(policy).__name__,
                        "d_model": policy.action_net.in_features,
                        "num_layers": pi.embed_layers,
                        "num_heads": pi.num_heads,
                        "embed_len": pi.embed_len,
                        "choose_on_teampreview": policy.choose_on_teampreview,
                        "device": "cpu",
                        "param_count": sum(p.numel() for p in policy.parameters()),
                    },
                    indent=2,
                )
            )
    return 0


def _load_dataset_recipe(path: Path) -> DatasetRecipe:
    path = _resolve_user_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data.get("dataset", data)
    return DatasetRecipe.model_validate(block)


def _cmd_dataset(args: argparse.Namespace) -> int:
    ensure_storage_layout()
    if args.dataset_cmd == "import":
        manifest = import_logs(args.path, args.dataset_id)
        print(manifest.model_dump_json(indent=2))
    elif args.dataset_cmd == "create":
        recipe = _load_dataset_recipe(Path(args.recipe))
        manifest = create_dataset(args.snapshot_id, recipe, dataset_id=args.dataset_id)
        print(manifest.model_dump_json(indent=2))
    elif args.dataset_cmd == "prepare":
        if args.sync_legacy:
            cfg = PrepareConfig(
                seed=args.seed,
                validation_ratio=args.validation_ratio,
                limit=args.limit,
                strict=args.strict,
                force=args.force,
            )
            manifest = prepare_dataset(args.dataset_id, cfg)
            print(manifest.model_dump_json(indent=2))
        else:
            record = submit_preparation(args.dataset_id)
            print(record.model_dump_json(indent=2))
    elif args.dataset_cmd == "inspect":
        payload = inspect_decision(
            args.dataset_id,
            args.index,
            replay_id=args.replay_id,
            player_side=args.player_side,
        )
        print(json.dumps(payload, indent=2))
    elif args.dataset_cmd == "show":
        print(get_dataset(args.dataset_id).model_dump_json(indent=2))
    return 0


def _resolve_user_path(path: Path) -> Path:
    """Chemins utilisateur relatifs au package (pas au CWD vendor)."""

    if path.is_absolute():
        return path
    candidate = PACKAGE_ROOT / path
    if candidate.is_file() or candidate.is_dir():
        return candidate
    parts = path.parts
    if parts and parts[0] == "pokecoach-trainer":
        stripped = PACKAGE_ROOT.joinpath(*parts[1:])
        if stripped.is_file() or stripped.is_dir():
            return stripped
    return path


def _load_experiment(path: Path) -> ExperimentConfig:
    path = _resolve_user_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    exp = data.get("experiment", data)
    training_raw = data.get("training", {})
    training = (
        TrainingConfig.model_validate(training_raw)
        if training_raw
        else TrainingConfig()
    )
    dataset = data.get("dataset", {})
    dataset_id = dataset.get("id") or dataset.get("dataset_id") or exp.get("dataset_id")
    init_raw = data.get("initialization")
    initialization = (
        InitializationConfig.model_validate(init_raw) if init_raw is not None else None
    )
    model_raw = data.get("model")
    if initialization is not None and model_raw:
        raise ValueError("model explicite incompatible avec initialization.architecture=inherit")
    model = ModelConfig.model_validate(model_raw) if model_raw else ModelConfig()
    return ExperimentConfig(
        name=exp.get("name", "experiment"),
        seed=exp.get("seed", 42),
        dataset_id=dataset_id,
        model=model,
        initialization=initialization,
        training=training,
    )


def _cmd_train(args: argparse.Namespace) -> int:
    config = _load_experiment(Path(args.config))
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else get_settings().storage_root / "runs" / config.name
    )
    if args.foreground:
        from pokecoach_trainer.training_service import preflight

        preflight(config)
    result = train_bc(config, run_dir)
    print(result.model_dump_json(indent=2))
    return 0


def _cmd_base_model(args: argparse.Namespace) -> int:
    ensure_storage_layout()
    source_path = _resolve_user_path(Path(args.source))
    payload = read_json(source_path) if source_path.suffix == ".json" else json.loads(
        source_path.read_text(encoding="utf-8")
    )
    source = BaseModelSource.model_validate(payload)
    manifest = register_base_model(args.path, source)
    print(manifest.model_dump_json(indent=2))
    return 0


def _load_collection_config(path: Path) -> CollectionConfig:
    path = _resolve_user_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data.get("collection", data)
    return CollectionConfig.model_validate(block)


def _load_snapshot_request(path: Path) -> SnapshotRequest:
    path = _resolve_user_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    block = data.get("snapshot", data)
    return SnapshotRequest.model_validate(block)


def _cmd_collection(args: argparse.Namespace) -> int:
    ensure_storage_layout()
    if args.collection_cmd == "start":
        config = _load_collection_config(Path(args.config))
        record = submit_collection(config)
        print(record.model_dump_json(indent=2))
    elif args.collection_cmd == "status":
        record = get_job(args.job_id)
        print(record.model_dump_json(indent=2))
    return 0


def _cmd_archive(args: argparse.Namespace) -> int:
    ensure_storage_layout()
    if args.archive_cmd == "snapshot":
        request = _load_snapshot_request(Path(args.config))

        def progress(msg: str) -> None:
            print(msg, file=sys.stderr, flush=True)

        snapshot = create_snapshot(request, progress_line=progress)
        print(
            json.dumps(
                {
                    "format_ids": snapshot.format_ids,
                    "replay_count": len(snapshot.refs),
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "snapshot_file": f"archive/snapshots/{snapshot.snapshot_id}.json",
                },
                indent=2,
            )
        )
        print(f"snapshot_id={snapshot.snapshot_id}")
    elif args.archive_cmd == "import":
        snapshot = import_legacy_logs(args.path)
        print(
            json.dumps(
                {
                    "format_ids": snapshot.format_ids,
                    "replay_count": len(snapshot.refs),
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "snapshot_file": f"archive/snapshots/{snapshot.snapshot_id}.json",
                },
                indent=2,
            )
        )
        print(f"snapshot_id={snapshot.snapshot_id}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    config = _load_experiment(Path(args.config))
    result_path = _resolve_user_path(Path(args.result))
    from pokecoach_trainer.schemas import TrainingResult

    result = TrainingResult.model_validate(
        json.loads(result_path.read_text(encoding="utf-8"))
    )
    manifest = export_model(result, config, model_id=args.model_id)
    print(manifest.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv_explicit()
    parser = argparse.ArgumentParser(prog="pokecoach-trainer")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Versions et chargement checkpoint")
    doctor.add_argument("--reference-checkpoint", dest="reference_checkpoint")

    ds = sub.add_parser("dataset", help="Gestion datasets")
    ds_sub = ds.add_subparsers(dest="dataset_cmd", required=True)
    imp = ds_sub.add_parser("import")
    imp.add_argument("--dataset-id", required=True)
    imp.add_argument("--path", required=True, help="Chemin relatif sous storage/inbox/")
    create = ds_sub.add_parser("create")
    create.add_argument("--snapshot-id", required=True)
    create.add_argument("--recipe", required=True)
    create.add_argument("--dataset-id", default=None)
    prep = ds_sub.add_parser("prepare")
    prep.add_argument("--dataset-id", required=True)
    prep.add_argument("--sync-legacy", action="store_true", help="prepare synchrone raw_logs")
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--validation-ratio", type=float, default=0.10)
    prep.add_argument("--limit", type=int, default=None)
    prep.add_argument(
        "--strict",
        action="store_true",
        help="échouer à la première erreur de parsing (diagnostic)",
    )
    prep.add_argument(
        "--force",
        action="store_true",
        help="refaire un prepare même si le dataset est déjà prêt",
    )
    ins = ds_sub.add_parser("inspect")
    ins.add_argument("--dataset-id", required=True)
    ins.add_argument("--index", type=int, default=0)
    ins.add_argument("--replay-id", default=None)
    ins.add_argument("--player-side", default=None)
    show = ds_sub.add_parser("show")
    show.add_argument("--dataset-id", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--run-dir", default=None)
    train.add_argument(
        "--foreground",
        action="store_true",
        help="préflight synchrone avant entraînement (diagnostic CLI)",
    )

    base_model = sub.add_parser("base-model", help="Registre des modèles parents")
    base_model_sub = base_model.add_subparsers(dest="base_model_cmd", required=True)
    reg = base_model_sub.add_parser("register")
    reg.add_argument("--path", required=True, help="Chemin relatif sous storage/inbox/")
    reg.add_argument(
        "--source",
        required=True,
        help="JSON de provenance (BaseModelSource)",
    )

    export = sub.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--result", required=True)
    export.add_argument("--model-id", default=None)

    collection = sub.add_parser("collection", help="Collecte Showdown")
    collection_sub = collection.add_subparsers(dest="collection_cmd", required=True)
    collection_start = collection_sub.add_parser("start")
    collection_start.add_argument("--config", required=True)
    collection_status = collection_sub.add_parser("status")
    collection_status.add_argument("--job-id", required=True)

    archive = sub.add_parser("archive", help="Archive brute et snapshots")
    archive_sub = archive.add_subparsers(dest="archive_cmd", required=True)
    archive_snapshot = archive_sub.add_parser("snapshot")
    archive_snapshot.add_argument("--config", required=True)
    archive_import = archive_sub.add_parser("import")
    archive_import.add_argument("--path", required=True, help="Chemin relatif sous storage/inbox/")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "dataset":
        return _cmd_dataset(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "base-model":
        return _cmd_base_model(args)
    if args.command == "export":
        return _cmd_export(args)
    if args.command == "collection":
        return _cmd_collection(args)
    if args.command == "archive":
        return _cmd_archive(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
