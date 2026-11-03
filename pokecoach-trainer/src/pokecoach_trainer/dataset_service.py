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


def _promote_staging(staging_root: Path, out_root: Path, manifest_path: Path) -> None:
    """Remplace train/validation et le manifeste par une préparation réussie."""

    for split in ("train", "validation"):
        dest = out_root / split
        src = staging_root / split
        if dest.exists():
            shutil.rmtree(dest)
        if src.exists():
            shutil.move(str(src), str(dest))
    staged_manifest = staging_root / "manifest.json"
    if staged_manifest.is_file():
        shutil.move(str(staged_manifest), str(manifest_path))


def _write_prepared_trajectories(
    tagged: list,
    train_ids: list[str],
    val_ids: list[str],
    train_dir: Path,
    val_dir: Path,
) -> tuple[list[Path], list[Path], list[str]]:
    """Écrit les pickles et retourne chemins + erreurs masque BC."""

    from pokecoach_trainer.trajectory_dataset import assert_actions_allowed

    train_set = set(train_ids)
    val_set = set(val_ids)
    traj_files_train: list[Path] = []
    traj_files_val: list[Path] = []
    errors: list[str] = []
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    for item in tagged:
        traj = trajectory_to_imitation(item.trajectory)
        replay_id = item.replay_id
        player_side = item.player_side
        try:
            assert_actions_allowed(traj, replay_id)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if replay_id in train_set:
            split_dir = train_dir
            bucket = traj_files_train
        elif replay_id in val_set:
            split_dir = val_dir
            bucket = traj_files_val
        else:
            errors.append(f"replay_id hors split: {replay_id}")
            continue
        out = split_dir / f"{replay_id}_{player_side}.pkl"
        with out.open("wb") as f:
            pickle.dump(traj, f)
        bucket.append(out)
    return traj_files_train, traj_files_val, errors


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
    staging_root = out_root / ".prepare_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    train_dir = staging_root / "train"
    val_dir = staging_root / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    from vgc_bench.logs2trajs import _init_worker_loop, process_logs_with_provenance

    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker_loop) as executor:
        tagged = process_logs_with_provenance(
            subset,
            executor,
            config.min_rating,
            config.only_winner,
            config.strict,
        )

    if not tagged:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise ValueError(
            "aucune trajectoire produite ; relancer avec strict=true pour la première erreur"
        )

    traj_files_train, traj_files_val, errors = _write_prepared_trajectories(
        tagged, train_ids, val_ids, train_dir, val_dir
    )

    if errors:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise ValueError("conversion bloquée par masque BC: " + "; ".join(errors[:3]))

    assert {x.replay_id for x in tagged if x.replay_id in train_ids} <= set(train_ids)
    assert {x.replay_id for x in tagged if x.replay_id in val_ids} <= set(val_ids)
    assert set(train_ids).isdisjoint(set(val_ids))

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
    write_json_atomic(staging_root / "manifest.json", manifest.model_dump(mode="json"))
    _promote_staging(staging_root, out_root, manifest_path)
    if staging_root.exists():
        shutil.rmtree(staging_root)
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
