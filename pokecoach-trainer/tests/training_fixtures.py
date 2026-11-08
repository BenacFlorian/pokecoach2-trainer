"""Fixtures communes pour les tests d'entraînement BC."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from imitation.data.types import Trajectory

from pokecoach_trainer.base_model_service import register_base_model
from pokecoach_trainer.compatibility import build_contract
from pokecoach_trainer.schemas import (
    BaseModelSource,
    DatasetManifest,
    DatasetState,
    ExperimentConfig,
    InitializationConfig,
    ModelConfig,
    TrainingConfig,
)
from pokecoach_trainer.settings import clear_settings_cache, ensure_storage_layout, storage_path
from pokecoach_trainer.storage import sha256_file, write_json_atomic
from pokecoach_trainer.training import build_ppo_container
from pokecoach_trainer.trajectory_dataset import trajectory_to_imitation
from vgc_bench.src.utils import chunk_obs_len


def install_storage_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()


def save_parent_checkpoint(
    tmp_path: Path,
    model: ModelConfig,
    *,
    base_model_id: str = "parent-test",
    inbox_relpath: str = "parent.ckpt",
) -> BaseModelSource:
    """Crée un parent SB3 en inbox et l'enregistre dans `base_models/`."""

    ppo = build_ppo_container(model, "cpu")
    inbox = storage_path("inbox", inbox_relpath)
    inbox.parent.mkdir(parents=True, exist_ok=True)
    stem = inbox.with_suffix("")
    ppo.save(str(stem))
    saved = Path(str(stem) + ".zip")
    if saved != inbox:
        shutil.move(str(saved), str(inbox))
    digest = sha256_file(inbox)
    source = BaseModelSource(
        base_model_id=base_model_id,
        label="test-fixture",
        expected_sha256=digest,
    )
    register_base_model(inbox_relpath, source)
    return source


def write_ready_dataset(
    dataset_id: str,
    model: ModelConfig,
    *,
    n_train: int = 4,
    n_val: int = 2,
    n_test: int = 1,
    steps: int = 3,
) -> None:
    """Écrit pickles train/val/test et un manifeste `ready` cohérent avec le parent."""

    transitions = 0

    def _demo_traj(steps: int) -> Trajectory:
        obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
        acts = np.zeros((steps, 2), dtype=np.int64)
        acts[:, 1] = 1
        return trajectory_to_imitation(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))

    for split, n in (("train", n_train), ("validation", n_val), ("test", n_test)):
        d = storage_path("datasets", dataset_id, split)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            traj = _demo_traj(steps)
            with (d / f"t{i}.pkl").open("wb") as f:
                pickle.dump(traj, f)
            if split == "train":
                transitions += steps

    contract = build_contract(model)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        state=DatasetState.ready,
        transitions_train=transitions,
        transitions_val=n_val * steps,
        transitions_test=n_test * steps,
        trajectories_train=n_train,
        trajectories_val=n_val,
        trajectories_test=n_test,
        encoding_contract_fingerprint=contract.fingerprint,
    )
    write_json_atomic(storage_path("datasets", dataset_id, "manifest.json"), manifest.model_dump(mode="json"))


def write_ready_dataset_with_index(
    dataset_id: str,
    model: ModelConfig,
    *,
    train_steps: list[int] | None = None,
    val_steps: list[int] | None = None,
    test_steps: list[int] | None = None,
    write_index: bool = False,
) -> None:
    """Dataset prêt avec un nombre de steps variable par fichier pickle."""

    train_steps = train_steps if train_steps is not None else [3, 3, 3, 3]
    val_steps = val_steps if val_steps is not None else [2, 2]
    test_steps = test_steps if test_steps is not None else [3]

    def _demo_traj(steps: int) -> Trajectory:
        obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
        acts = np.zeros((steps, 2), dtype=np.int64)
        acts[:, 1] = 1
        return trajectory_to_imitation(Trajectory(obs=obs, acts=acts, infos=None, terminal=True))

    index_rows: list[dict[str, object]] = []
    totals = {"train": 0, "validation": 0, "test": 0}
    counts = {"train": 0, "validation": 0, "test": 0}

    for split, steps_list in (
        ("train", train_steps),
        ("validation", val_steps),
        ("test", test_steps),
    ):
        d = storage_path("datasets", dataset_id, split)
        d.mkdir(parents=True, exist_ok=True)
        for i, n_steps in enumerate(steps_list):
            traj = _demo_traj(n_steps)
            fname = f"t{i}.pkl"
            with (d / fname).open("wb") as f:
                pickle.dump(traj, f)
            totals[split] += n_steps
            counts[split] += 1
            if write_index:
                index_rows.append(
                    {
                        "split": split,
                        "file": fname,
                        "file_sha256": sha256_file(d / fname),
                        "transitions": n_steps,
                    }
                )

    contract = build_contract(model)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        state=DatasetState.ready,
        transitions_train=totals["train"],
        transitions_val=totals["validation"],
        transitions_test=totals["test"],
        trajectories_train=counts["train"],
        trajectories_val=counts["validation"],
        trajectories_test=counts["test"],
        encoding_contract_fingerprint=contract.fingerprint,
    )
    write_json_atomic(storage_path("datasets", dataset_id, "manifest.json"), manifest.model_dump(mode="json"))
    if write_index:
        index_path = storage_path("datasets", dataset_id, "trajectory_index.jsonl")
        with index_path.open("w", encoding="utf-8") as handle:
            for row in index_rows:
                handle.write(json.dumps(row) + "\n")


def experiment_config(
    dataset_id: str,
    base_model_id: str,
    model: ModelConfig,
    **training_kwargs,
) -> ExperimentConfig:
    training = TrainingConfig(**training_kwargs) if training_kwargs else TrainingConfig()
    return ExperimentConfig(
        name="test-run",
        seed=42,
        dataset_id=dataset_id,
        model=model,
        initialization=InitializationConfig(base_model_id=base_model_id),
        training=training,
    )
