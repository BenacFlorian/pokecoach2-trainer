"""Adaptateur BC : batches complets et évaluation NLL isolée."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from imitation.algorithms.bc import BC
from imitation.data import rollout, types
from stable_baselines3.common.utils import ConstantSchedule
from torch.utils import data as th_data

from pokecoach_trainer.schemas import EvaluationResult
from pokecoach_trainer.trajectory_dataset import _obs_arrays


@dataclass
class BCTrainStats:
    """Compteurs observés pendant une passe `train()`."""

    optimizer_steps: int = 0
    transitions_seen: int = 0


class CompleteBatchBC(BC):
    """
    BC avec chargeur `drop_last=False` et taille effective `min(batch_size, N)`.

    Conserve la boucle d'optimisation du fork `imitation` (dernier lot pondéré).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_train_stats = BCTrainStats()
        self._wrap_optimizer_step()

    def _wrap_optimizer_step(self) -> None:
        original = self.optimizer.step

        def counted_step(*args, **kwargs):
            self._last_train_stats.optimizer_steps += 1
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise ValueError("gradient non fini avant optimizer.step")
            return original(*args, **kwargs)

        self.optimizer.step = counted_step  # type: ignore[method-assign]

    def set_demonstrations(self, demonstrations) -> None:
        """Prépare un DataLoader sans `drop_last` ni wrapper à taille fixe."""

        trajectories = list(demonstrations)
        transitions = rollout.flatten_trajectories(trajectories)
        n_transitions = len(transitions)
        if n_transitions < 1:
            raise ValueError("demonstrations vides")
        effective = min(self.batch_size, n_transitions)
        effective = max(effective, 1)
        self.batch_size = effective
        self.minibatch_size = effective

        generator = torch.Generator()
        generator.manual_seed(int(self.rng.integers(0, 2**31 - 1)))
        loader = th_data.DataLoader(
            transitions,
            batch_size=effective,
            shuffle=True,
            drop_last=False,
            collate_fn=types.transitions_collate_fn,
            generator=generator,
        )
        self._demo_data_loader = loader

    def train(self, *args, **kwargs):
        self._last_train_stats = BCTrainStats()
        result = super().train(*args, **kwargs)
        cached = getattr(self, "_cached_trajectories", None)
        if cached is not None:
            self._last_train_stats.transitions_seen = sum(len(t.acts) for t in cached)
        return result

    def remember_trajectories(self, trajectories) -> None:
        """Mémorise les trajectoires pour le décompte des transitions consommées."""

        self._cached_trajectories = list(trajectories)

    @property
    def last_train_stats(self) -> BCTrainStats:
        return self._last_train_stats


def duplicate_policy(policy, device: str):
    """
    Clone la policy sur le device demandé (deepcopy incompatible avec `max_norm`).

    Retourne une instance indépendante avec les mêmes poids.
    """

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
    clone.to(device)
    return clone


def evaluate_nll(
    policy,
    demonstrations,
    device: str,
) -> EvaluationResult:
    """
    NLL moyenne (-log π(a|s)) sur une copie isolée, sans modifier la policy d'origine.
    """

    demos = list(demonstrations)
    if not demos:
        raise ValueError("évaluation sans transitions")

    clone = duplicate_policy(policy, device)
    clone.eval()
    clone.actor_grad = False
    for param in clone.parameters():
        param.requires_grad = False

    losses: list[float] = []
    transitions = 0
    with torch.no_grad():
        for traj in demos:
            obs = _obs_arrays(traj.obs)
            acts = torch.as_tensor(traj.acts, device=device)
            obs_t = {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
            for step in range(acts.shape[0]):
                step_obs = {k: obs_t[k][step : step + 1] for k in obs_t}
                step_act = acts[step : step + 1]
                _, log_prob, _ = clone.evaluate_actions(step_obs, step_act)
                lp = float(log_prob.mean().detach().cpu())
                loss = -lp
                if not np.isfinite(loss):
                    raise ValueError("NLL non finie")
                losses.append(loss)
                transitions += 1
    del clone
    return EvaluationResult(nll=float(np.mean(losses)), transitions=transitions)


def count_transitions(demonstrations) -> int:
    """Compte le nombre total de transitions dans des trajectoires imitation."""

    return sum(len(t.acts) for t in demonstrations)
