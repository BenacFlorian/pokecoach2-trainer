"""Entraînement BC (sans HTTP) et validation isolée."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from pokecoach_trainer.base_model_service import (
    assert_dataset_contract_compatible,
    load_base_model,
    read_base_model_manifest,
)
from pokecoach_trainer.bc_adapter import (
    CompleteBatchBC,
    duplicate_policy,
    evaluate_nll,
)
from pokecoach_trainer.dataset_service import get_dataset
from pokecoach_trainer.errors import InvalidConfiguration
from pokecoach_trainer.schemas import ExperimentConfig, TrainingResult
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import sha256_file, write_json_atomic
from pokecoach_trainer.trajectory_dataset import load_trajectory_pickle
from pokecoach_trainer.upstream import vgc_bench_workdir

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Résout `auto` ou vérifie la disponibilité CUDA demandée."""

    if requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"device demandé indisponible: {requested}")
        return requested
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < (7, 5):
            logger.info(
                "device=auto → CPU (GPU CC %s.%s non supporté par cette build PyTorch)",
                major,
                minor,
            )
            return "cpu"
        return "cuda:0"
    logger.info("device=auto → CPU (CUDA indisponible)")
    return "cpu"


def count_policy_params(policy) -> int:
    return sum(p.numel() for p in policy.parameters())


def actor_weight_snapshot(policy) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in policy.state_dict().items()}


def build_ppo_container(model, device: str):
    """Instancie PPO + env minimal (conteneur SB3, sans `learn()`)."""

    from pokecoach_trainer.schemas import ModelConfig

    if not isinstance(model, ModelConfig):
        raise TypeError("ModelConfig requis")
    with vgc_bench_workdir():
        from poke_env.environment import SingleAgentWrapper
        from poke_env.player import RandomPlayer
        from vgc_bench.src.env import ShowdownEnv
        from vgc_bench.src.policy import MaskedActorCriticPolicy
        from vgc_bench.src.utils import format_map, set_global_seed

        set_global_seed(42)
        battle_format = format_map["mb"]
        env = ShowdownEnv(
            battle_format=battle_format,
            log_level=40,
            accept_open_team_sheet=True,
            start_listening=False,
            choose_on_teampreview=model.choose_on_teampreview,
        )
        opponent = RandomPlayer(start_listening=False)
        wrapped = SingleAgentWrapper(env, opponent)
        from stable_baselines3 import PPO

        ppo = PPO(
            MaskedActorCriticPolicy,
            wrapped,
            policy_kwargs=model.policy_kwargs(),
            device=device,
            n_steps=256,
            batch_size=64,
        )
    return ppo


def load_split_trajectories(dataset_id: str, split: str):
    directory = storage_path("datasets", dataset_id, split)
    files = sorted(directory.glob("*.pkl"))
    return [load_trajectory_pickle(p) for p in files]


def evaluate_validation(policy, demonstrations, device: str) -> float:
    """Délègue à `evaluate_nll` pour compatibilité des appelants historiques."""

    return evaluate_nll(policy, demonstrations, device).nll


def _non_embedding_actor_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> float:
    delta = 0.0
    for key, tensor in before.items():
        if "embed" in key:
            continue
        if (
            "action_net" in key
            or key.startswith("pi_features_extractor")
            or key.startswith("mlp_extractor")
        ):
            delta += float((after[key] - tensor).norm().item())
    return delta


def _require_initialization(config: ExperimentConfig) -> None:
    if config.initialization is None:
        raise InvalidConfiguration(
            "initialization.base_model_id requis pour le fine-tuning BC",
        )
    if config.initialization.architecture != "inherit":
        raise InvalidConfiguration("seul architecture=inherit est supporté")


def train_bc(config: ExperimentConfig, run_dir: Path) -> TrainingResult:
    """
    Fine-tune le parent enregistré via `CompleteBatchBC` (jamais `PPO.learn()`).

    Mesure NLL et compteurs après chaque retour de `train()`, pas dans la callback SB3.
    """

    _require_initialization(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.training.device)

    dataset_manifest = get_dataset(config.dataset_id)
    manifest_path = storage_path("datasets", config.dataset_id, "manifest.json")
    dataset_manifest_sha256 = sha256_file(manifest_path) if manifest_path.is_file() else None

    parent_id = config.initialization.base_model_id
    parent_manifest = read_base_model_manifest(parent_id)
    assert_dataset_contract_compatible(parent_manifest, dataset_manifest)

    train_demos = load_split_trajectories(config.dataset_id, "train")
    val_demos = load_split_trajectories(config.dataset_id, "validation")
    test_demos = load_split_trajectories(config.dataset_id, "test")
    if not train_demos:
        raise ValueError("split train vide")

    with vgc_bench_workdir():
        from vgc_bench.src.utils import set_global_seed

        set_global_seed(config.seed)

    parent_ckpt_path = storage_path("base_models", parent_id, parent_manifest.checkpoint_relpath)
    parent_digest_before = sha256_file(parent_ckpt_path)

    ppo = load_base_model(parent_id, device)
    policy = ppo.policy
    architecture = parent_manifest.architecture
    config.model = architecture

    before = actor_weight_snapshot(policy)
    param_before = count_policy_params(policy)

    rng = np.random.default_rng(config.seed)
    batch_size = config.training.batch_size

    bc = CompleteBatchBC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=rng,
        policy=policy,
        batch_size=batch_size,
        minibatch_size=batch_size,
        device=device,
        optimizer_kwargs={"lr": config.training.learning_rate},
    )
    bc.remember_trajectories(train_demos)
    bc.set_demonstrations(train_demos)

    effective_lr = bc.optimizer.param_groups[0]["lr"]
    logger.info("BC optimizer lr effectif=%s", effective_lr)

    metrics_path = run_dir / "metrics.jsonl"
    checkpoints: list[str] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    observed_steps: list[int] = []
    observed_transitions: list[int] = []

    for epoch in range(1, config.training.num_epochs + 1):
        t0 = datetime.now(timezone.utc)
        bc.remember_trajectories(train_demos)
        bc.set_demonstrations(train_demos)
        bc.train(n_epochs=1, progress_bar=False)
        steps = bc.last_train_stats.optimizer_steps
        transitions_epoch = bc.last_train_stats.transitions_seen
        if steps < 1:
            raise ValueError("aucun pas d'optimiseur observé")
        expected_steps = max(1, math.ceil(transitions_epoch / bc.batch_size))
        if steps != expected_steps:
            raise ValueError(
                f"pas observés {steps} != attendus {expected_steps} "
                f"(transitions={transitions_epoch}, batch={bc.batch_size})",
            )

        train_eval = evaluate_nll(policy, train_demos, device)
        if val_demos:
            val_eval = evaluate_nll(policy, val_demos, device)
            val_nll = val_eval.nll
        else:
            val_nll = float("nan")
        train_losses.append(train_eval.nll)
        val_losses.append(val_nll)
        observed_steps.append(steps)
        observed_transitions.append(transitions_epoch)

        ckpt = run_dir / f"epoch_{epoch}"
        ppo.save(str(ckpt))
        checkpoints.append(str(ckpt) + ".zip")
        line = {
            "epoch": epoch,
            "train_nll": train_eval.nll,
            "validation_nll": val_nll,
            "train_loss": train_eval.nll,
            "validation_loss": val_nll,
            "optimizer_steps": steps,
            "transitions_train": train_eval.transitions,
            "transitions_consumed": transitions_epoch,
            "learning_rate": effective_lr,
            "device": device,
            "param_count": param_before,
            "duration_s": (datetime.now(timezone.utc) - t0).total_seconds(),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        if not np.isfinite(train_eval.nll):
            raise ValueError("métrique non finie")
        if val_demos and not np.isfinite(val_nll):
            raise ValueError("métrique validation non finie")

    after = actor_weight_snapshot(policy)
    delta = 0.0
    for key in before:
        delta += float((after[key] - before[key]).norm().item())
    actor_delta = _non_embedding_actor_delta(before, after)
    if delta == 0.0 or actor_delta == 0.0:
        raise ValueError("preuve d'apprentissage absente (poids inchangés)")

    if sha256_file(parent_ckpt_path) != parent_digest_before:
        raise InvalidConfiguration("le fichier parent a été modifié pendant l'entraînement")

    final_ckpt = run_dir / "final"
    ppo.save(str(final_ckpt))
    final_path = str(final_ckpt) + ".zip"

    test_nll: float | None = None
    test_report_path: str | None = None
    if test_demos:
        test_eval = evaluate_nll(policy, test_demos, device)
        test_nll = test_eval.nll
        test_report = {
            "test_nll": test_eval.nll,
            "transitions": test_eval.transitions,
            "parent_checkpoint_sha256": parent_manifest.checkpoint_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "final_checkpoint": final_path,
        }
        test_report_path = str(run_dir / "test_report.json")
        write_json_atomic(Path(test_report_path), test_report)

    result = TrainingResult(
        run_id=run_dir.name,
        checkpoints=checkpoints,
        final_checkpoint=final_path,
        metrics_path=str(metrics_path),
        param_count=param_before,
        device=device,
        weight_delta_norm=delta,
        train_losses=train_losses,
        val_losses=val_losses,
        architecture=architecture,
        parent_base_model_id=parent_id,
        parent_checkpoint_sha256=parent_manifest.checkpoint_sha256,
        dataset_id=config.dataset_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        split_id=dataset_manifest.split_id,
        effective_learning_rate=effective_lr,
        observed_optimizer_steps=observed_steps,
        observed_transitions_per_epoch=observed_transitions,
        actor_delta_norm=actor_delta,
        test_nll=test_nll,
        test_report_path=test_report_path,
    )
    write_json_atomic(run_dir / "result.json", result.model_dump(mode="json"))
    return result
