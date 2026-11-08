"""Parité NLL scalaire / batches et comptage d'appels policy."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from pokecoach_trainer.bc_adapter import duplicate_policy, evaluate_nll
from pokecoach_trainer.bc_evaluation import evaluate_nll_batches
from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.training import build_ppo_container
from pokecoach_trainer.training_data import iter_transition_batches
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation
from imitation.data.types import Trajectory
from vgc_bench.src.utils import chunk_obs_len


def _demos(n: int, *, seed: int = 0) -> list:
    _ = seed
    obs = np.zeros((n + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((n, 2), dtype=np.int64)
    acts[:, 1] = 1
    traj = trajectory_to_imitation(
        Trajectory(obs=obs, acts=acts, infos=None, terminal=True),
    )
    return [traj]


def _scalar_oracle(policy, demonstrations, device: str) -> float:
    from pokecoach_trainer.trajectory_dataset import _obs_arrays

    clone = duplicate_policy(policy, device)
    clone.eval()
    clone.actor_grad = False
    total = 0.0
    count = 0
    with torch.no_grad():
        for traj in demonstrations:
            n = len(traj.acts)
            obs_map = _obs_arrays(traj.obs)
            for step in range(n):
                obs_dict = {
                    k: torch.as_tensor(v[step : step + 1], device=device)
                    for k, v in obs_map.items()
                }
                acts = torch.as_tensor(traj.acts[step : step + 1], device=device)
                _, log_prob, _ = clone.evaluate_actions(obs_dict, acts)
                loss = -log_prob.reshape(-1).to(dtype=torch.float64)
                total += float(loss.sum().item())
                count += 1
    del clone
    if count == 0:
        raise ValueError("évaluation sans transitions")
    return total / count


@pytest.mark.parametrize("batch_size", [1, 2, 32, 256])
def test_batched_nll_matches_scalar_oracle(batch_size: int):
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    demos = _demos(33, seed=11)
    ref = _scalar_oracle(ppo.policy, demos, "cpu")
    got = evaluate_nll(ppo.policy, demos, "cpu", batch_size=batch_size)
    assert got.transitions == 33
    assert got.nll == pytest.approx(ref, rel=1e-5, abs=1e-6)


def test_weighted_mean_fake_policy(monkeypatch):
    device = "cpu"
    batches = list(iter_transition_batches(_demos(5), batch_size=3))
    losses = [1.0, 1.0, 1.0, 9.0, 9.0]

    class FakePolicy:
        def __init__(self):
            self.calls = 0
            self.observation_space = None
            self.action_space = None
            self.choose_on_teampreview = True
            self.num_layers = 1
            self.num_heads = 4
            self.embed_len = 32
            self.action_net = MagicMock(in_features=64)

        def parameters(self):
            return []

        def state_dict(self):
            return {}

        def load_state_dict(self, _state):
            return None

        def to(self, _dev):
            return self

        def eval(self):
            return None

        actor_grad = False

        def evaluate_actions(self, _obs, acts):
            self.calls += 1
            n = int(acts.shape[0])
            idx = FakePolicy._cursor
            FakePolicy._cursor += n
            chunk = losses[idx : idx + n]
            log_prob = torch.tensor([-x for x in chunk], dtype=torch.float32)
            return None, log_prob, None

    FakePolicy._cursor = 0
    policy = FakePolicy()
    monkeypatch.setattr(
        "pokecoach_trainer.bc_adapter.duplicate_policy",
        lambda _p, _d: policy,
    )
    result = evaluate_nll_batches(policy, batches, device)
    assert result.nll == pytest.approx(4.2)
    assert result.transitions == 5
    assert policy.calls == math.ceil(5 / 3)


def test_evaluate_nll_policy_call_count_real(monkeypatch):
    model = ModelConfig(d_model=64, num_layers=1, num_heads=4, embed_len=32)
    ppo = build_ppo_container(model, "cpu")
    policy = ppo.policy
    calls = {"n": 0}
    original = policy.evaluate_actions

    def counted(obs, acts):
        calls["n"] += 1
        return original(obs, acts)

    policy.evaluate_actions = counted  # type: ignore[method-assign]
    monkeypatch.setattr(
        "pokecoach_trainer.bc_adapter.duplicate_policy",
        lambda _p, _d: policy,
    )
    demos = _demos(33)
    evaluate_nll(policy, demos, "cpu", batch_size=8)
    assert calls["n"] == math.ceil(33 / 8)
