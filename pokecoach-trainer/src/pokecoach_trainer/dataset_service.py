"""Import, préparation et split de datasets replays."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from pokecoach_trainer.schemas import DatasetManifest, DatasetState, PrepareConfig
from pokecoach_trainer.settings import ensure_storage_layout, storage_path
from pokecoach_trainer.storage import (
    read_json,
    resolve_inbox_relative,
    sha256_file,
    validate_id,
    write_json_atomic,
)
from pokecoach_trainer.trajectory_dataset import (
    _obs_arrays,
    count_transitions,
    load_trajectory_pickle,
    trajectory_to_imitation,
)
from pokecoach_trainer.upstream import vgc_bench_workdir


def _manifest_path(dataset_id: str) -> Path:
    return storage_path("datasets", dataset_id, "manifest.json")


def import_logs(relative_path: str, dataset_id: str) -> DatasetManifest:
    """Enregistre un JSON inbox vers `raw_logs/{dataset_id}/`."""

    validate_id(dataset_id)
    ensure_storage_layout()
    src = resolve_inbox_relative(relative_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    size = src.stat().st_size
    settings_max = __import__(
        "pokecoach_trainer.settings", fromlist=["get_settings"]
    ).get_settings().max_inbox_bytes
    if size > settings_max:
        raise ValueError(f"fichier trop volumineux ({size} > {settings_max})")
    raw_dir = storage_path("raw_logs", dataset_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / src.name
    shutil.copy2(src, dest)
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        state=DatasetState.imported,
        source_relpath=str(dest.relative_to(storage_path())),
        source_sha256=sha256_file(dest),
    )
    write_json_atomic(_manifest_path(dataset_id), manifest.model_dump(mode="json"))
    return manifest


def _load_log_dict(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON de logs attendu (objet id -> contenu)")
    return data


def _dedupe_logs(raw: dict) -> tuple[dict[str, tuple[str, str]], list[dict]]:
    """Retourne le mapping attendu par `process_logs` et les rejets."""

    cleaned: dict[str, tuple[str, str]] = {}
    rejected: list[dict] = []
    content_index: dict[str, str] = {}
    for replay_id, value in raw.items():
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            ts, log_text = str(value[0]), str(value[1])
        elif isinstance(value, str):
            ts, log_text = "", value
        else:
            rejected.append({"replay_id": replay_id, "reason": "format_inconnu"})
            continue
        content_hash = hashlib.sha256(log_text.encode()).hexdigest()
        if replay_id in cleaned and cleaned[replay_id][1] != log_text:
            rejected.append({"replay_id": replay_id, "reason": "id_contradictoire"})
            continue
        if content_hash in content_index and content_index[content_hash] != replay_id:
            rejected.append(
                {
                    "replay_id": replay_id,
                    "reason": "duplicate_content",
                    "other_id": content_index[content_hash],
                }
            )
            continue
        content_index[content_hash] = replay_id
        cleaned[replay_id] = (ts, log_text)
    return cleaned, rejected


def _split_replay_ids(
    replay_ids: list[str], seed: int, validation_ratio: float
) -> tuple[list[str], list[str]]:
    ids = sorted(set(replay_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    if not ids:
        return [], []
    n_val = max(1, int(len(ids) * validation_ratio)) if len(ids) > 1 else 0
    if n_val >= len(ids):
        n_val = len(ids) - 1
    val = ids[:n_val]
    train = ids[n_val:]
    if not train or not val:
        raise ValueError("split train/validation vide")
    return train, val


def prepare_dataset(dataset_id: str, config: PrepareConfig) -> DatasetManifest:
    """Convertit les logs, contrôle le masque BC et écrit train/validation."""

    validate_id(dataset_id)
    manifest_path = _manifest_path(dataset_id)
    if manifest_path.is_file():
        existing = DatasetManifest.model_validate(read_json(manifest_path))
        if (
            not config.force
            and existing.state == DatasetState.ready
            and (existing.trajectories_train or existing.trajectories_val)
        ):
            raise FileExistsError(
                "dataset déjà prêt ; ne pas écraser (ajouter --force pour refaire)"
            )

    raw_dir = storage_path("raw_logs", dataset_id)
    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("aucun JSON dans raw_logs")
    log_path = json_files[0]
    raw_logs, rejected = _dedupe_logs(_load_log_dict(log_path))
    replay_ids = list(raw_logs.keys())
    if config.limit:
        replay_ids = replay_ids[: config.limit]
    subset = {k: raw_logs[k] for k in replay_ids}
    train_ids, val_ids = _split_replay_ids(replay_ids, config.seed, config.validation_ratio)

    out_root = storage_path("datasets", dataset_id)
    train_dir = out_root / "train"
    val_dir = out_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    if config.force:
        for split_dir in (train_dir, val_dir):
            for pkl in split_dir.glob("*.pkl"):
                pkl.unlink()
        from vgc_bench.logs2trajs import _init_worker_loop, process_logs

        with ProcessPoolExecutor(
            max_workers=1, initializer=_init_worker_loop
        ) as executor:
            trajs = process_logs(
                subset,
                executor,
                config.min_rating,
                config.only_winner,
                config.strict,
            )

    if not trajs:
        raise ValueError(
            "aucune trajectoire produite ; relancer avec strict=true pour la première erreur"
        )

    # Associer trajectoires aux replays de façon best-effort (ordre process_logs)
    traj_files_train: list[Path] = []
    traj_files_val: list[Path] = []
    errors: list[str] = []
    for i, traj in enumerate(trajs):
        traj = trajectory_to_imitation(traj)
        replay_guess = replay_ids[i % len(replay_ids)]
        try:
            from pokecoach_trainer.trajectory_dataset import assert_actions_allowed

            assert_actions_allowed(traj, replay_guess)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        split = "train" if replay_guess in train_ids else "validation"
        target_dir = train_dir if split == "train" else val_dir
        out = target_dir / f"{replay_guess}_{i:04d}.pkl"
        with out.open("wb") as f:
            pickle.dump(traj, f)
        if split == "train":
            traj_files_train.append(out)
        else:
            traj_files_val.append(out)

    if errors:
        raise ValueError("conversion bloquée par masque BC: " + "; ".join(errors[:3]))

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        state=DatasetState.ready,
        source_relpath=str(log_path.relative_to(storage_path())),
        source_sha256=sha256_file(log_path),
        format=log_path.stem.replace("logs_", ""),
        seed=config.seed,
        validation_ratio=config.validation_ratio,
        replay_ids_train=train_ids,
        replay_ids_val=val_ids,
        trajectories_train=len(traj_files_train),
        trajectories_val=len(traj_files_val),
        transitions_train=count_transitions(traj_files_train),
        transitions_val=count_transitions(traj_files_val),
        rejected=rejected,
        errors=errors,
    )
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def get_dataset(dataset_id: str) -> DatasetManifest:
    return DatasetManifest.model_validate(read_json(_manifest_path(dataset_id)))


def inspect_decision(dataset_id: str, index: int = 0) -> dict:
    """Affiche une transition réelle pour la CLI."""

    train_dir = storage_path("datasets", dataset_id, "train")
    files = sorted(train_dir.glob("*.pkl"))
    if not files:
        raise FileNotFoundError("aucune trajectoire train")
    traj = load_trajectory_pickle(files[index])
    if len(traj.acts) == 0:
        raise ValueError("trajectoire sans actions")
    obs = _obs_arrays(traj.obs)
    step = 0
    return {
        "file": files[index].name,
        "step": step,
        "actions": traj.acts[step].tolist(),
        "observation_shape": list(obs["observation"].shape),
        "mask_shape": list(obs["action_mask"].shape),
    }
