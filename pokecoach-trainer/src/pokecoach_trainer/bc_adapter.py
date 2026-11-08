"""Adaptateur BC : batches complets et évaluation NLL isolée."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from imitation.algorithms.bc import BC
from imitation.data import rollout, types
from stable_baselines3.common.utils import ConstantSchedule
from torch.utils import data as th_data

from pokecoach_trainer.schemas import EvaluationResult
from pokecoach_trainer.training_data import TransitionBatch
from pokecoach_trainer.trajectory_dataset import _obs_arrays


@dataclass
class BCTrainStats:
    """Compteurs observés pendant une passe `train()`."""

    optimizer_steps: int = 0
    transitions_seen: int = 0


class _TransitionBatchIterable:
    """Enveloppe un flux de batches pour `BC.train()` et compte les transitions."""

    def __init__(
        self,
        batches: Iterable[TransitionBatch],
        stats: BCTrainStats,
    ) -> None:
        self._batches = batches
        self._stats = stats

    def __iter__(self):
        for batch in self._batches:
            acts = batch["acts"]
            n = int(acts.shape[0])
            obs = batch["obs"]
            if len(obs) != n:
                raise ValueError("taille obs/acts incohérente")
            self._stats.transitions_seen += n
            yield batch


class CompleteBatchBC(BC):
    """
    BC avec chargeur `drop_last=False` et taille effective `min(batch_size, N)`.

    Conserve la boucle d'optimisation du fork `imitation` (dernier lot pondéré).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_train_stats = BCTrainStats()
        self._wrap_optimizer_step()
        self._streaming_batches = False
        self._expected_transitions: int | None = None
        self._cached_trajectories: list | None = None
        self._batch_source: Iterable[TransitionBatch] | None = None

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

        self._streaming_batches = False
        self._expected_transitions = None
        self._batch_source = None

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

    def set_transition_batches(
        self,
        batches: Iterable[TransitionBatch],
        *,
        expected_transitions: int,
        batch_size: int,
    ) -> None:
        """Installe un flux de batches (sans aplatir les trajectoires)."""

        if expected_transitions < 1 or batch_size < 1:
            raise ValueError("comptes batch invalides")
        effective = min(batch_size, expected_transitions)
        effective = max(effective, 1)
        self.batch_size = effective
        self.minibatch_size = effective
        self._streaming_batches = True
        self._expected_transitions = expected_transitions
        self._cached_trajectories = None
        self._batch_source = batches

    def train(self, *args, **kwargs):
        n_epochs = kwargs.get("n_epochs")
        n_batches = kwargs.get("n_batches")
        streaming = self._streaming_batches
        if streaming:
            if n_batches is not None:
                raise ValueError("n_batches non supporté en mode flux")
            if n_epochs is not None and n_epochs != 1:
                raise ValueError("n_epochs doit valoir 1 en mode flux")

        self._last_train_stats = BCTrainStats()
        if streaming:
            if self._batch_source is None:
                raise RuntimeError("flux sans source de batches")
            self._demo_data_loader = _TransitionBatchIterable(
                self._batch_source,
                self._last_train_stats,
            )

        result = super().train(*args, **kwargs)

        if streaming:
            expected = self._expected_transitions
            if expected is None:
                raise RuntimeError("mode flux sans expected_transitions")
            seen = self._last_train_stats.transitions_seen
            if seen != expected:
                raise ValueError(
                    f"transitions observées {seen} != attendues {expected}",
                )
            eff = self.batch_size
            expected_steps = max(1, math.ceil(expected / eff))
            steps = self._last_train_stats.optimizer_steps
            if steps != expected_steps:
                raise ValueError(
                    f"pas observés {steps} != attendus {expected_steps}",
                )
            return result

        cached = self._cached_trajectories
        if cached is not None:
            self._last_train_stats.transitions_seen = sum(len(t.acts) for t in cached)
        return result

    def remember_trajectories(self, trajectories) -> None:
        """Mémorise les trajectoires pour le décompte des transitions consommées."""

        if self._streaming_batches:
            return
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
    *,
    batch_size: int = 256,
) -> EvaluationResult:
    """
    NLL moyenne (-log π(a|s)) sur une copie isolée, sans modifier la policy d'origine.
    """

    from pokecoach_trainer.training_data import iter_transition_batches

    demos = list(demonstrations)
    if not demos:
        raise ValueError("évaluation sans transitions")

    batches = iter_transition_batches(demos, batch_size=batch_size)
    from pokecoach_trainer.bc_evaluation import evaluate_nll_batches

    return evaluate_nll_batches(policy, batches, device)


def count_transitions(demonstrations) -> int:
    """Compte le nombre total de transitions dans des trajectoires imitation."""

    return sum(len(t.acts) for t in demonstrations)
