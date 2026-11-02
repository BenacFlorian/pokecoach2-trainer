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
from pokecoach_trainer.artifact_service import export_model
from pokecoach_trainer.dataset_service import (
    get_dataset,
    import_logs,
    inspect_decision,
    prepare_dataset,
)
from pokecoach_trainer.schemas import ExperimentConfig, ModelConfig, PrepareConfig
from pokecoach_trainer.settings import (
    ensure_storage_layout,
    get_settings,
    load_dotenv_explicit,
)
from pokecoach_trainer.storage import sha256_file
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


def _cmd_dataset(args: argparse.Namespace) -> int:
    ensure_storage_layout()
    if args.dataset_cmd == "import":
        manifest = import_logs(args.path, args.dataset_id)
        print(manifest.model_dump_json(indent=2))
    elif args.dataset_cmd == "prepare":
        cfg = PrepareConfig(
            seed=args.seed,
            validation_ratio=args.validation_ratio,
            limit=args.limit,
            strict=args.strict,
            force=args.force,
        )
        manifest = prepare_dataset(args.dataset_id, cfg)
        print(manifest.model_dump_json(indent=2))
    elif args.dataset_cmd == "inspect":
        print(json.dumps(inspect_decision(args.dataset_id, args.index), indent=2))
    elif args.dataset_cmd == "show":
        print(get_dataset(args.dataset_id).model_dump_json(indent=2))
    return 0


def _resolve_user_path(path: Path) -> Path:
    """Chemins utilisateur relatifs au package (pas au CWD vendor)."""

    if path.is_absolute():
        return path
    candidate = PACKAGE_ROOT / path
    if candidate.exists():
        return candidate
    return path


def _load_experiment(path: Path) -> ExperimentConfig:
    path = _resolve_user_path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    exp = data.get("experiment", data)
    model = ModelConfig.model_validate(data.get("model", {}))
    training = data.get("training", {})
    dataset = data.get("dataset", {})
    return ExperimentConfig(
        name=exp.get("name", "experiment"),
        seed=exp.get("seed", 42),
        dataset_id=dataset.get("id") or dataset.get("dataset_id"),
        model=model,
        training=training,
    )


def _cmd_train(args: argparse.Namespace) -> int:
    config = _load_experiment(Path(args.config))
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else get_settings().storage_root / "runs" / config.name
    )
    result = train_bc(config, run_dir)
    print(result.model_dump_json(indent=2))
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
    prep = ds_sub.add_parser("prepare")
    prep.add_argument("--dataset-id", required=True)
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
    show = ds_sub.add_parser("show")
    show.add_argument("--dataset-id", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--run-dir", default=None)

    export = sub.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--result", required=True)
    export.add_argument("--model-id", default=None)

    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "dataset":
        return _cmd_dataset(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "export":
        return _cmd_export(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
