"""Évaluation NLL par batches avec policy isolée."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from pokecoach_trainer.schemas import EvaluationResult
from pokecoach_trainer.training_data import TransitionBatch
from pokecoach_trainer.trajectory_dataset import _obs_arrays


def evaluate_nll_batches(
    policy,
    batches: Iterable[TransitionBatch],
    device: str,
) -> EvaluationResult:
    from pokecoach_trainer.bc_adapter import duplicate_policy

    total_neglogp = 0.0
    total_transitions = 0
    clone = None
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    devices = [device] if use_cuda else []
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            clone = duplicate_policy(policy, device)
            clone.eval()
            clone.actor_grad = False
            for param in clone.parameters():
                param.requires_grad_(False)

            with torch.no_grad():
                for batch in batches:
                    n = int(batch["acts"].shape[0])
                    if n == 0:
                        continue
                    obs = batch["obs"]
                    acts = torch.as_tensor(batch["acts"], device=device)
                    obs_dict = {
                        k: torch.as_tensor(v, device=device) for k, v in _obs_arrays(obs).items()
                    }
                    _, log_prob, _ = clone.evaluate_actions(obs_dict, acts)
                    log_prob = log_prob.reshape(-1)
                    if log_prob.numel() != n:
                        raise ValueError("log_prob incohérent avec la taille du batch")
                    loss = -log_prob.to(dtype=torch.float64)
                    if not torch.isfinite(loss).all():
                        raise ValueError("NLL non finie")
                    total_neglogp += float(loss.sum().item())
                    total_transitions += n
    finally:
        if clone is not None:
            del clone

    if total_transitions == 0:
        raise ValueError("évaluation sans transitions")
    return EvaluationResult(
        nll=total_neglogp / total_transitions,
        transitions=total_transitions,
    )
