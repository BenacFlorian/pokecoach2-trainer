"""Entraînement BC (sans HTTP) et validation isolée."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from imitation.algorithms.bc import BC
from imitation.data.types import Trajectory
from poke_env.environment import SingleAgentWrapper
from stable_baselines3 import PPO

from pokecoach_trainer.schemas import ExperimentConfig, ModelConfig, TrainingResult
from pokecoach_trainer.settings import storage_path
from pokecoach_trainer.storage import write_json_atomic
from pokecoach_trainer.trajectory_dataset import (
    _obs_arrays,
    load_trajectory_pickle,
)
from pokecoach_trainer.upstream import vgc_bench_workdir

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
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


def build_ppo_container(model: ModelConfig, device: str):
    """Instancie PPO + env minimal (conteneur SB3, sans `learn()`)."""

    with vgc_bench_workdir():
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
        ppo = PPO(
            MaskedActorCriticPolicy,
            wrapped,
            policy_kwargs=model.policy_kwargs(),
            device=device,
            n_steps=256,
            batch_size=64,
        )
    return ppo


def load_split_trajectories(dataset_id: str, split: str) -> list[Trajectory]:
    directory = storage_path("datasets", dataset_id, split)
    files = sorted(directory.glob("*.pkl"))
    return [load_trajectory_pickle(p) for p in files]


def duplicate_policy(policy):
    """Clone isolé compatible avec `max_norm` sur les embeddings (deepcopy échoue)."""

    from stable_baselines3.common.utils import ConstantSchedule

    clone = policy.__class__(
        policy.observation_space,
        policy.action_space,
        ConstantSchedule(3e-4),
        d_model=policy.action_net.in_features,
        choose_on_teampreview=policy.choose_on_teampreview,
        num_layers=policy.num_layers,
        num_heads=policy.num_heads,
        embed_len=policy.embed_len,
    )
    clone.load_state_dict({k: v.detach().clone() for k, v in policy.state_dict().items()})
    return clone


def evaluate_validation(
    policy,
    demonstrations: list[Trajectory],
    device: str,
) -> float:
    """
    Loss BC sur une copie isolée pour ne pas toucher la policy d'entraînement.
    """

    clone = duplicate_policy(policy)
    clone.eval()
    clone.actor_grad = False
    for param in clone.parameters():
        param.requires_grad = False
    losses: list[float] = []
    with torch.no_grad():
        for traj in demonstrations:
            obs = _obs_arrays(traj.obs)
            acts = torch.as_tensor(traj.acts, device=device)
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
            for step in range(acts.shape[0]):
                step_obs = {k: obs_t[k][step : step + 1] for k in obs_t}
                step_act = acts[step : step + 1]
                _, log_prob, _ = clone.evaluate_actions(step_obs, step_act)
                losses.append(-float(log_prob.mean().cpu()))
    del clone
    if not losses:
        raise ValueError("validation sans transitions")
    return float(np.mean(losses))


def train_bc(config: ExperimentConfig, run_dir: Path) -> TrainingResult:
    """Entraîne le Tiny via `imitation.BC` sans évaluation de combats."""

    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.training.device)
    train_demos = load_split_trajectories(config.dataset_id, "train")
    val_demos = load_split_trajectories(config.dataset_id, "validation")
    if not train_demos:
        raise ValueError("split train vide")

    with vgc_bench_workdir():
        from vgc_bench.src.utils import set_global_seed

        set_global_seed(config.seed)

    ppo = build_ppo_container(config.model, device)
    policy = ppo.policy
    before = actor_weight_snapshot(policy)
    param_before = count_policy_params(policy)

    rng = np.random.default_rng(config.seed)
    batch_size = min(config.training.batch_size, sum(len(t.acts) for t in train_demos))
    batch_size = max(batch_size, 1)

    bc = BC(
        observation_space=ppo.observation_space,
        action_space=ppo.action_space,
        rng=rng,
        policy=policy,
        batch_size=batch_size,
        device=device,
    )
    bc.set_demonstrations(train_demos)

    metrics_path = run_dir / "metrics.jsonl"
    checkpoints: list[str] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, config.training.num_epochs + 1):
        t0 = datetime.now(timezone.utc)
        bc.train(n_epochs=1)
        train_loss = evaluate_validation(policy, train_demos, device)
        val_loss = evaluate_validation(policy, val_demos, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        ckpt = run_dir / f"epoch_{epoch}"
        ppo.save(str(ckpt))
        checkpoints.append(str(ckpt) + ".zip")
        line = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "transitions_train": sum(len(t.acts) for t in train_demos),
            "device": device,
            "param_count": param_before,
            "duration_s": (datetime.now(timezone.utc) - t0).total_seconds(),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        if not np.isfinite(train_loss):
            train_loss = float("nan")
        if not np.isfinite(val_loss):
            val_loss = float("nan")

    after = actor_weight_snapshot(policy)
    delta = 0.0
    for key in before:
        delta += float((after[key] - before[key]).norm().item())
    if delta == 0.0:
        raise ValueError("preuve d'apprentissage absente (poids inchangés)")

    final_ckpt = run_dir / "final"
    ppo.save(str(final_ckpt))
    final_path = str(final_ckpt) + ".zip"
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
    )
    write_json_atomic(run_dir / "result.json", result.model_dump(mode="json"))
    return result
