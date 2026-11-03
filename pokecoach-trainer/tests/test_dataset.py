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


def test_write_prepared_trajectories_split_by_replay_id(tmp_path, monkeypatch):
    from dataclasses import dataclass

    from pokecoach_trainer.dataset_service import _write_prepared_trajectories

    @dataclass
    class FakeTagged:
        replay_id: str
        player_side: str
        trajectory: object

    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path / "storage"))
    clear_settings_cache()
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    tagged = [
        FakeTagged("r1", "p1", _fake_traj()),
        FakeTagged("r1", "p2", _fake_traj()),
        FakeTagged("r2", "p1", _fake_traj()),
    ]
    train_files, val_files, errors = _write_prepared_trajectories(
        tagged, train_ids=["r1"], val_ids=["r2"], train_dir=train_dir, val_dir=val_dir
    )
    assert not errors
    assert len(train_files) == 2
    assert len(val_files) == 1
    assert {p.name for p in train_files} == {"r1_p1.pkl", "r1_p2.pkl"}
    assert val_files[0].name == "r2_p1.pkl"


def test_process_logs_with_provenance_keeps_metadata(monkeypatch):
    from concurrent.futures import Future

    from vgc_bench.logs2trajs import TaggedTrajectory, process_logs_with_provenance

    traj_a = _fake_traj()
    traj_b = _fake_traj(steps=2)

    class FakeExecutor:
        def submit(self, _fn, tag, _log, role, *_rest):
            fut: Future = Future()
            if tag == "slow" and role == "p1":
                result = traj_b
            elif tag == "fast" and role == "p2":
                result = traj_a
            else:
                result = None
            fut.set_result(result)
            return fut

    logs = {"slow": ("", "logslow"), "fast": ("", "logfast")}
    tagged = process_logs_with_provenance(logs, FakeExecutor(), None, False, False)
    by_key = {(t.replay_id, t.player_side): t.trajectory for t in tagged}
    assert by_key[("slow", "p1")] is traj_b
    assert by_key[("fast", "p2")] is traj_a


def test_prepare_dataset_runs_without_force(tmp_path, monkeypatch):
    import json

    from pokecoach_trainer.dataset_service import prepare_dataset
    from pokecoach_trainer.schemas import PrepareConfig
    from vgc_bench.logs2trajs import TaggedTrajectory

    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path / "storage"))
    clear_settings_cache()
    ensure_storage_layout()
    ds_id = "ds-prepare"
    inbox = storage_path("inbox")
    raw = {"r1": ["", "log1"], "r2": ["", "log2"], "r3": ["", "log3"], "r4": ["", "log4"]}
    (inbox / "logs_sv.json").write_text(json.dumps(raw), encoding="utf-8")
    import_logs("logs_sv.json", ds_id)

    fake_traj = _fake_traj()

    def fake_provenance(_subset, _executor, *_args):
        return [
            TaggedTrajectory("r1", "p1", fake_traj),
            TaggedTrajectory("r2", "p1", fake_traj),
            TaggedTrajectory("r3", "p1", fake_traj),
            TaggedTrajectory("r4", "p1", fake_traj),
        ]

    monkeypatch.setattr(
        "vgc_bench.logs2trajs.process_logs_with_provenance",
        fake_provenance,
    )
    class FakePool:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "pokecoach_trainer.dataset_service.ProcessPoolExecutor",
        lambda *a, **k: FakePool(),
    )

    manifest = prepare_dataset(ds_id, PrepareConfig(force=False, seed=42, validation_ratio=0.25))
    assert manifest.state.value == "ready"
    assert manifest.trajectories_train + manifest.trajectories_val == 4
    assert set(manifest.replay_ids_train).isdisjoint(set(manifest.replay_ids_val))
