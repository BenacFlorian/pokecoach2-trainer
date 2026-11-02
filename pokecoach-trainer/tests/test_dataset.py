import pickle

import numpy as np
import pytest
from imitation.data.types import Trajectory
from vgc_bench.src.utils import chunk_obs_len

from pokecoach_trainer.dataset_service import import_logs
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout, storage_path
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation


def _fake_traj(steps: int = 3) -> Trajectory:
    obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((steps, 2), dtype=np.int64)
    return Trajectory(obs=obs, acts=acts, infos=None, terminal=True)


def test_split_disjoint(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path / "storage"))
    clear_settings_cache()
    ensure_storage_layout()
    ds_id = "ds-test"
    inbox = storage_path("inbox")
    raw = {"r1": ["", "log1"], "r2": ["", "log2"], "r3": ["", "log3"]}
    import json

    (inbox / "logs.json").write_text(json.dumps(raw), encoding="utf-8")
    import_logs("logs.json", ds_id)
    train_dir = storage_path("datasets", ds_id, "train")
    val_dir = storage_path("datasets", ds_id, "validation")
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    for name in ("r1", "r2"):
        traj = trajectory_to_imitation(_fake_traj())
        with (train_dir / f"{name}.pkl").open("wb") as f:
            pickle.dump(traj, f)
    traj = trajectory_to_imitation(_fake_traj())
    with (val_dir / "r3.pkl").open("wb") as f:
        pickle.dump(traj, f)
    from pokecoach_trainer.dataset_service import _split_replay_ids

    train, val = _split_replay_ids(["a", "b", "c", "d"], seed=42, validation_ratio=0.25)
    assert set(train).isdisjoint(set(val))


def test_mask_blocks_illegal_target():
    traj = trajectory_to_imitation(_fake_traj())
    traj.acts[0, 0] = 50
    with pytest.raises(ValueError):
        from pokecoach_trainer.trajectory_dataset import assert_actions_allowed

        assert_actions_allowed(traj, "x")
