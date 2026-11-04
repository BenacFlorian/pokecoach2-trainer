"""Import, préparation et split de datasets replays."""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pokecoach_trainer.schemas import (
    ConversionStatus,
    DatasetManifest,
    DatasetRecipe,
    DatasetState,
    FilterDecision,
    JobRecord,
    LegacyImportDatasetSource,
    PrepareConfig,
    SnapshotDatasetSource,
)
from pokecoach_trainer.settings import ensure_storage_layout, storage_path
from pokecoach_trainer.storage import (
    append_jsonl_atomic,
    canonical_json_sha256,
    new_opaque_id,
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


class PrepareJobLog:
    """Journal texte flushé dans worker.log (prepare)."""

    def __init__(self, job_id: str | None) -> None:
        self._path: Path | None = None
        if job_id:
            self._path = storage_path("jobs", job_id, "worker.log")
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        if self._path is None:
            return
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(f"{ts} {message}\n")
            handle.flush()


def _sync_prepare_job_progress(job_id: str | None, snapshot: dict[str, Any]) -> None:
    if not job_id:
        return
    try:
        from pokecoach_trainer.jobs import get_job, save_job

        record = get_job(job_id)
        record.progress = snapshot
        save_job(record)
    except Exception:
        return


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


def _dataset_dir(dataset_id: str) -> Path:
    return storage_path("datasets", dataset_id)


def _recipe_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / "recipe.json"


def _decisions_path(dataset_id: str) -> Path:
    return _dataset_dir(dataset_id) / "replay_decisions.jsonl"


def _recipe_sha256(recipe: DatasetRecipe) -> str:
    return canonical_json_sha256(recipe.model_dump(mode="json"))


def create_dataset(
    snapshot_id: str,
    recipe: DatasetRecipe,
    *,
    dataset_id: str | None = None,
) -> DatasetManifest:
    """
    Réserve un dataset immuable lié à un snapshot et une recette figée.

    Raises:
        FileExistsError: dataset déjà prêt.
    """

    from pokecoach_trainer.archive_service import read_snapshot

    ensure_storage_layout()
    ds_id = validate_id(dataset_id or new_opaque_id("ds-"))
    manifest_path = _manifest_path(ds_id)
    if manifest_path.is_file():
        existing = DatasetManifest.model_validate(read_json(manifest_path))
        if existing.state == DatasetState.ready:
            raise FileExistsError("dataset prêt non modifiable")
    snapshot = read_snapshot(snapshot_id)
    ds_dir = _dataset_dir(ds_id)
    ds_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(_recipe_path(ds_id), recipe.model_dump(mode="json"))
    write_json_atomic(
        ds_dir / "source.json",
        {
            "source": "snapshot",
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_sha256": snapshot.snapshot_sha256,
        },
    )
    manifest = DatasetManifest(
        dataset_id=ds_id,
        state=DatasetState.created,
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.snapshot_sha256,
        recipe=recipe,
        recipe_sha256=_recipe_sha256(recipe),
        seed=recipe.seed,
        target_replays=recipe.target_replays,
        validation_ratio=recipe.splits.validation,
    )
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def _record_decision(dataset_id: str, decision: FilterDecision) -> None:
    append_jsonl_atomic(_decisions_path(dataset_id), decision.model_dump(mode="json"))


def prepare_dataset(
    dataset_id: str,
    config: PrepareConfig | None = None,
    *,
    job_id: str | None = None,
) -> DatasetManifest:
    """
    Prépare un dataset legacy (import JSON) ou snapshot (worker).

    Pour un dataset `created`, passer `job_id` depuis le worker.
    """

    manifest_path = _manifest_path(dataset_id)
    if manifest_path.is_file():
        existing = DatasetManifest.model_validate(read_json(manifest_path))
        if existing.state in {DatasetState.created, DatasetState.preparing} and config is None:
            return _prepare_snapshot_dataset(dataset_id, job_id=job_id)
        if existing.state == DatasetState.imported and config is not None:
            return _prepare_legacy_raw_logs(dataset_id, config)
        if existing.state == DatasetState.ready and config is not None and not config.force:
            raise FileExistsError(
                "dataset déjà prêt ; ne pas écraser (ajouter --force pour refaire)"
            )
    if config is not None:
        return _prepare_legacy_raw_logs(dataset_id, config)
    raise ValueError("configuration PrepareConfig requise pour import legacy")


def _prepare_legacy_raw_logs(dataset_id: str, config: PrepareConfig) -> DatasetManifest:
    """Chemin historique raw_logs → train/validation (compatibilité palier 1)."""

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


def _prepare_snapshot_dataset(dataset_id: str, *, job_id: str | None) -> DatasetManifest:
    """Builder snapshot : filtres, sélection, conversion, splits et publication."""

    from pokecoach_trainer.archive_service import read_snapshot
    from pokecoach_trainer.compatibility import build_contract
    from pokecoach_trainer.conversion import convert_perspectives_parallel
    from pokecoach_trainer.dataset_filters import decide_game, decide_perspective
    from pokecoach_trainer.dataset_split import assign_splits, build_replay_groups, select_groups
    from pokecoach_trainer.errors import BlockingMaskError
    from pokecoach_trainer.replay_metadata import extract_replay_metadata
    from pokecoach_trainer.schemas import ModelConfig, ReplayMetadata

    validate_id(dataset_id)
    manifest_path = _manifest_path(dataset_id)
    manifest = DatasetManifest.model_validate(read_json(manifest_path))
    if manifest.state == DatasetState.ready:
        return manifest
    if manifest.state not in {DatasetState.created, DatasetState.preparing}:
        raise ValueError(f"état dataset incompatible: {manifest.state.value}")
    if not manifest.recipe or not manifest.snapshot_id:
        raise ValueError("recette ou snapshot manquant")

    recipe = manifest.recipe
    log = PrepareJobLog(job_id)
    log.write(
        f"prepare démarré dataset={dataset_id} snapshot={manifest.snapshot_id} "
        f"target_replays={recipe.target_replays}"
    )
    snapshot = read_snapshot(manifest.snapshot_id)
    total_refs = len(snapshot.refs)
    log.write(f"prepare snapshot chargé replays={total_refs}")
    out_root = _dataset_dir(dataset_id)
    staging_root = out_root / ".prepare_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    for sub in ("train", "validation", "test"):
        (staging_root / sub).mkdir(parents=True, exist_ok=True)

    manifest.state = DatasetState.preparing
    manifest.prepare_job_id = job_id
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))

    metas: dict[str, ReplayMetadata] = {}
    for ref in snapshot.refs:
        cached = snapshot.replay_metadata.get(ref.replay_id)
        if cached:
            metas[ref.replay_id] = ReplayMetadata.model_validate(cached)
        else:
            metas[ref.replay_id] = extract_replay_metadata(ref)

    groups = build_replay_groups(snapshot, metas)
    admissible_replays: set[str] = set()
    converted_tasks: list[tuple] = []
    refs_by_id = {ref.replay_id: ref for ref in snapshot.refs}
    blocking_error: BlockingMaskError | None = None

    for index, ref in enumerate(snapshot.refs, start=1):
        meta = metas[ref.replay_id]
        game_decision = decide_game(meta, recipe)
        _record_decision(dataset_id, game_decision)
        if not game_decision.accepted:
            continue
        perspectives_ok = False
        for side in ("p1", "p2"):
            decision = decide_perspective(meta, side, recipe)
            _record_decision(dataset_id, decision)
            if decision.accepted:
                perspectives_ok = True
                converted_tasks.append((ref, side))
        if perspectives_ok:
            admissible_replays.add(ref.replay_id)
        if total_refs and (index == 1 or index == total_refs or index % 150 == 0):
            log.write(
                f"prepare filtres {index}/{total_refs} "
                f"admissibles={len(admissible_replays)} "
                f"perspectives={len(converted_tasks)}"
            )
            _sync_prepare_job_progress(
                job_id,
                {
                    "phase": "filters",
                    "refs_scanned": index,
                    "refs_total": total_refs,
                    "admissible_replays": len(admissible_replays),
                },
            )

    log.write(
        f"prepare sélection candidats admissibles={len(admissible_replays)} "
        f"perspectives={len(converted_tasks)}"
    )
    selected_groups, selected_count, overshoot = select_groups(
        groups,
        seed=recipe.seed,
        target_replays=recipe.target_replays,
        admissible_replays=admissible_replays,
    )
    selected_replay_ids = set()
    for group in selected_groups:
        for rid in group.replay_ids:
            if rid in admissible_replays:
                selected_replay_ids.add(rid)

    ungrouped_bo3 = sum(
        1
        for meta in metas.values()
        if meta.match_type == "bo3" and meta.series_link_status != "confirmed"
    )
    assignments = assign_splits(
        selected_groups,
        recipe,
        selected_replay_ids=selected_replay_ids,
        ungrouped_bo3=ungrouped_bo3,
    )

    if not assignments.trainable or selected_count < recipe.target_replays:
        log.write(
            f"prepare insuffisant selected={selected_count} "
            f"target={recipe.target_replays} reason={assignments.reason_code!r}"
        )
        manifest.state = DatasetState.insufficient_data
        manifest.selected_replays = selected_count
        manifest.overshoot_replays = overshoot
        manifest.series_leakage_guarantee = assignments.series_leakage_guarantee
        manifest.ungrouped_bo3_count = assignments.ungrouped_bo3_count
        manifest.errors = [
            assignments.reason_code or "target_not_reached",
            f"selected={selected_count} target={recipe.target_replays}",
        ]
        write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        if staging_root.exists():
            shutil.rmtree(staging_root)
        return manifest

    contract = build_contract(ModelConfig())
    tasks = [
        (refs_by_id[ref.replay_id], side)
        for ref, side in converted_tasks
        if ref.replay_id in selected_replay_ids
    ]
    log.write(
        f"prepare sélection retenue replays={selected_count} "
        f"(train={len(assignments.train)} val={len(assignments.validation)} "
        f"test={len(assignments.test)}) conversion={len(tasks)} perspectives"
    )
    from vgc_bench.logs2trajs import _init_worker_loop

    traj_index: list[dict] = []
    split_dirs = {
        "train": staging_root / "train",
        "validation": staging_root / "validation",
        "test": staging_root / "test",
    }
    replay_to_split: dict[str, str] = {}
    for rid in assignments.train:
        replay_to_split[rid] = "train"
    for rid in assignments.validation:
        replay_to_split[rid] = "validation"
    for rid in assignments.test:
        replay_to_split[rid] = "test"

    try:
        with ProcessPoolExecutor(max_workers=1, initializer=_init_worker_loop) as executor:
            log.write(f"prepare conversion démarrée ({len(tasks)} perspectives)")
            results = convert_perspectives_parallel(tasks, executor, contract=contract)
    finally:
        pass

    converted_n = sum(1 for r in results if r.status == ConversionStatus.converted)
    rejected_n = sum(1 for r in results if r.status == ConversionStatus.rejected)
    error_n = sum(1 for r in results if r.status == ConversionStatus.error)
    log.write(
        f"prepare conversion terminée converted={converted_n} "
        f"rejected={rejected_n} errors={error_n}"
    )
    train_files: list[Path] = []
    val_files: list[Path] = []
    test_files: list[Path] = []

    for result in results:
        if result.status == ConversionStatus.error and result.reason_code in {
            "blocking_mask_first_target",
            "blocking_mask_second_target",
            "action_index_out_of_bounds",
            "blocking_mask_error",
        }:
            blocking_error = BlockingMaskError(
                result.reason_code or "blocking_mask_error",
                details=result.details,
            )
            break
        if result.status != ConversionStatus.converted or result.trajectory is None:
            continue
        split_name = replay_to_split.get(result.replay_id)
        if not split_name:
            continue
        out = split_dirs[split_name] / f"{result.replay_id}_{result.player_side}.pkl"
        with out.open("wb") as handle:
            pickle.dump(result.trajectory, handle)
        digest = sha256_file(out)
        traj_index.append(
            {
                "replay_id": result.replay_id,
                "player_side": result.player_side,
                "raw_sha256": result.raw_sha256,
                "split": split_name,
                "file": out.name,
                "file_sha256": digest,
                "transitions": len(result.trajectory.acts),
                "converter_revision": result.converter_revision,
            }
        )
        if split_name == "train":
            train_files.append(out)
        elif split_name == "validation":
            val_files.append(out)
        else:
            test_files.append(out)

    if blocking_error:
        log.write(f"prepare échec blocking: {blocking_error.message}")
        manifest.state = DatasetState.failed
        manifest.errors = [blocking_error.message]
        write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise blocking_error

    write_json_atomic(staging_root / "split_assignments.json", assignments.model_dump(mode="json"))
    index_path = staging_root / "trajectory_index.jsonl"
    for row in traj_index:
        append_jsonl_atomic(index_path, row)

    manifest = DatasetManifest(
        dataset_id=dataset_id,
        state=DatasetState.ready,
        snapshot_id=manifest.snapshot_id,
        snapshot_sha256=manifest.snapshot_sha256,
        recipe=recipe,
        recipe_sha256=manifest.recipe_sha256,
        seed=recipe.seed,
        validation_ratio=recipe.splits.validation,
        replay_ids_train=assignments.train,
        replay_ids_val=assignments.validation,
        replay_ids_test=assignments.test,
        trajectories_train=len(train_files),
        trajectories_val=len(val_files),
        trajectories_test=len(test_files),
        transitions_train=count_transitions(train_files) if train_files else 0,
        transitions_val=count_transitions(val_files) if val_files else 0,
        transitions_test=count_transitions(test_files) if test_files else 0,
        selected_replays=selected_count,
        target_replays=recipe.target_replays,
        overshoot_replays=overshoot,
        split_id=assignments.split_id,
        series_leakage_guarantee=assignments.series_leakage_guarantee,
        ungrouped_bo3_count=assignments.ungrouped_bo3_count,
    )
    write_json_atomic(staging_root / "manifest.json", manifest.model_dump(mode="json"))
    _promote_staging_extended(staging_root, out_root, manifest_path)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    log.write(
        f"prepare terminé state=ready selected={manifest.selected_replays} "
        f"trajectories train={manifest.trajectories_train} "
        f"val={manifest.trajectories_val} test={manifest.trajectories_test} "
        f"transitions train={manifest.transitions_train}"
    )
    _sync_prepare_job_progress(
        job_id,
        {
            "phase": "done",
            "state": "ready",
            "selected_replays": manifest.selected_replays,
            "trajectories_train": manifest.trajectories_train,
        },
    )
    return manifest


def _promote_staging_extended(staging_root: Path, out_root: Path, manifest_path: Path) -> None:
    for split in ("train", "validation", "test"):
        dest = out_root / split
        src = staging_root / split
        if dest.exists():
            shutil.rmtree(dest)
        if src.exists() and any(src.iterdir()):
            shutil.move(str(src), str(dest))
    for name in ("split_assignments.json", "trajectory_index.jsonl"):
        src = staging_root / name
        if src.is_file():
            shutil.move(str(src), str(out_root / name))
    staged_manifest = staging_root / "manifest.json"
    if staged_manifest.is_file():
        shutil.move(str(staged_manifest), str(manifest_path))


def submit_preparation(dataset_id: str) -> JobRecord:
    """Soumet un job prepare pour un dataset snapshot ou legacy import."""

    from pokecoach_trainer.job_service import submit_job
    from pokecoach_trainer.schemas import DatasetJobConfig, JobKind

    manifest = get_dataset(dataset_id)
    if manifest.state == DatasetState.ready:
        from pokecoach_trainer.errors import Conflict

        raise Conflict("dataset prêt non modifiable", details={"dataset_id": dataset_id})
    if manifest.state == DatasetState.preparing:
        from pokecoach_trainer.errors import Conflict

        raise Conflict("préparation déjà en cours", details={"dataset_id": dataset_id})

    source_path = _dataset_dir(dataset_id) / "source.json"
    if source_path.is_file():
        source_data = read_json(source_path)
    else:
        source_data = {"source": "snapshot", "snapshot_id": manifest.snapshot_id}

    if source_data.get("source") == "legacy_import":
        dataset_source = LegacyImportDatasetSource.model_validate(source_data)
    else:
        if not manifest.snapshot_id:
            raise ValueError("snapshot_id manquant")
        dataset_source = SnapshotDatasetSource(snapshot_id=manifest.snapshot_id)

    payload = DatasetJobConfig(
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        recipe=manifest.recipe,
    )
    record = submit_job(JobKind.prepare, payload)
    manifest.state = DatasetState.preparing
    manifest.prepare_job_id = record.job_id
    write_json_atomic(_manifest_path(dataset_id), manifest.model_dump(mode="json"))
    return record


def reserve_legacy_import_dataset(
    inbox_relpath: str,
    recipe: DatasetRecipe,
    *,
    dataset_id: str | None = None,
) -> DatasetManifest:
    """Réserve un dataset legacy ; snapshot créé dans le worker prepare."""

    ensure_storage_layout()
    ds_id = validate_id(dataset_id or new_opaque_id("ds-"))
    ds_dir = _dataset_dir(ds_id)
    ds_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(_recipe_path(ds_id), recipe.model_dump(mode="json"))
    write_json_atomic(
        ds_dir / "source.json",
        LegacyImportDatasetSource(inbox_relpath=inbox_relpath, recipe=recipe).model_dump(
            mode="json"
        ),
    )
    manifest = DatasetManifest(
        dataset_id=ds_id,
        state=DatasetState.created,
        recipe=recipe,
        recipe_sha256=_recipe_sha256(recipe),
        seed=recipe.seed,
        target_replays=recipe.target_replays,
    )
    write_json_atomic(_manifest_path(ds_id), manifest.model_dump(mode="json"))
    return manifest


def list_rejections(
    dataset_id: str,
    *,
    offset: int = 0,
    limit: int = 100,
    stage: str | None = None,
    reason_code: str | None = None,
) -> tuple[list[dict], int]:
    """Liste paginée des décisions persistées."""

    path = _decisions_path(dataset_id)
    if not path.is_file():
        return [], 0
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if stage and row.get("stage") != stage:
            continue
        if reason_code and row.get("reason_code") != reason_code:
            continue
        rows.append(row)
    total = len(rows)
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)
    return rows[offset : offset + limit], total


def run_prepare_job(job_id: str) -> None:
    """Exécute la préparation dataset dans le worker (legacy import ou snapshot)."""

    from datetime import datetime, timezone

    from pokecoach_trainer.archive_service import import_legacy_logs
    from pokecoach_trainer.jobs import get_job, release_job_lock, save_job
    from pokecoach_trainer.schemas import DatasetJobConfig, JobStatus, LegacyImportDatasetSource

    record = get_job(job_id)
    cfg = record.config
    if not isinstance(cfg, DatasetJobConfig):
        raise ValueError("job prepare sans DatasetJobConfig")
    dataset_id = cfg.dataset_id
    plog = PrepareJobLog(job_id)
    plog.write(f"prepare worker job={job_id} dataset={dataset_id}")
    manifest = get_dataset(dataset_id)
    try:
        if isinstance(cfg.dataset_source, LegacyImportDatasetSource):
            snapshot = import_legacy_logs(cfg.dataset_source.inbox_relpath)
            manifest.snapshot_id = snapshot.snapshot_id
            manifest.snapshot_sha256 = snapshot.snapshot_sha256
            write_json_atomic(_manifest_path(dataset_id), manifest.model_dump(mode="json"))
        prepare_dataset(dataset_id, job_id=job_id)
        record.status = JobStatus.completed
        record.finished_at = datetime.now(timezone.utc)
        refreshed = get_dataset(dataset_id)
        if refreshed.state == DatasetState.insufficient_data:
            record.result_ref = refreshed.state.value
        elif refreshed.state == DatasetState.ready:
            record.result_ref = refreshed.state.value
        else:
            record.status = JobStatus.failed
            record.error = refreshed.errors[0] if refreshed.errors else refreshed.state.value
    except Exception as exc:
        plog.write(f"prepare worker échec: {exc}")
        record.status = JobStatus.failed
        record.error = str(exc)
        manifest = get_dataset(dataset_id)
        manifest.state = DatasetState.failed
        manifest.errors = [str(exc)]
        write_json_atomic(_manifest_path(dataset_id), manifest.model_dump(mode="json"))
        save_job(record)
        release_job_lock(job_id)
        raise
    plog.write(
        f"prepare worker terminé status={record.status.value} "
        f"result_ref={record.result_ref!r}"
    )
    save_job(record)
    release_job_lock(job_id)


def get_dataset(dataset_id: str) -> DatasetManifest:
    return DatasetManifest.model_validate(read_json(_manifest_path(dataset_id)))


def inspect_decision(
    dataset_id: str,
    index: int = 0,
    *,
    replay_id: str | None = None,
    player_side: str | None = None,
) -> dict:
    """Affiche une transition réelle pour la CLI."""

    train_dir = storage_path("datasets", dataset_id, "train")
    if replay_id:
        pattern = f"{replay_id}_{player_side or 'p1'}.pkl"
        matches = list(train_dir.glob(pattern))
        if not matches and player_side is None:
            matches = list(train_dir.glob(f"{replay_id}_*.pkl"))
        if not matches:
            raise FileNotFoundError("trajectoire introuvable pour ce replay")
        target = matches[0]
    else:
        files = sorted(train_dir.glob("*.pkl"))
        if not files:
            raise FileNotFoundError("aucune trajectoire train")
        target = files[index]
    traj = load_trajectory_pickle(target)
    if len(traj.acts) == 0:
        raise ValueError("trajectoire sans actions")
    obs = _obs_arrays(traj.obs)
    step = 0
    return {
        "file": target.name,
        "step": step,
        "actions": traj.acts[step].tolist(),
        "observation_shape": list(obs["observation"].shape),
        "mask_shape": list(obs["action_mask"].shape),
    }
