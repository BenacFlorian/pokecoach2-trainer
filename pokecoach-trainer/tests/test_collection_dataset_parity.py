from datetime import datetime, timezone

from pokecoach_trainer.archive_catalog import catalog_connection
from pokecoach_trainer.archive_service import archive_replay, create_campaign_snapshot
from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.replay_metadata import extract_replay_metadata
from pokecoach_trainer.replay_policy import evaluate_replay_policy
from pokecoach_trainer.schemas import ArchiveSource, CollectionConfig, ReplayFilterConfig
from pokecoach_trainer.settings import clear_settings_cache
from replay_log_fixtures import build_log

SOURCE = "gen9championsvgc2026regmc"


def test_campaign_snapshot_paginates_beyond_1000(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"], max_selected_replays=2000))
    campaign_id = view.job.job_id
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    with catalog_connection(tmp_path) as conn:
        conn.execute("BEGIN")
        for i in range(1001):
            rid = f"{SOURCE}-{i:04d}"
            conn.execute(
                """
                INSERT INTO replays(replay_id, source_format, uploadtime, first_published_at)
                VALUES (?, ?, 1, 1)
                """,
                (rid, SOURCE),
            )
            conn.execute(
                """
                INSERT INTO replay_versions(
                    replay_id, content_sha256, log_sha256, object_relpath,
                    provider_format, uploadtime, provenance, fetched_at, conflict
                ) VALUES (?, ?, ?, ?, ?, 1, 't', 1, 0)
                """,
                (rid, f"{i:064x}"[:64], f"{i:064x}"[:64], f"objects/{rid}/x.json", SOURCE),
            )
            conn.execute(
                """
                INSERT INTO campaign_replays(
                    campaign_id, replay_id, decision, reason, filter_sha256, version_content_sha256
                ) VALUES (?, ?, 'selected', 'selected', 'fp', ?)
                """,
                (campaign_id, rid, f"{i:064x}"[:64]),
            )
        conn.commit()
    snap = create_campaign_snapshot(campaign_id)
    assert len(snap.refs) == 1001


def test_snapshot_keeps_pinned_version(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path)
    view = store.start_campaign(CollectionConfig(format_ids=["mc"]))
    rid = f"{SOURCE}-pin"
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    first = archive_replay(rid, {"id": rid, "uploadtime": 1, "log": "|turn|1|\n", "format": SOURCE}, source)
    with catalog_connection(tmp_path) as conn:
        conn.execute(
            """
            INSERT INTO campaign_replays(
                campaign_id, replay_id, decision, reason, filter_sha256, version_content_sha256
            ) VALUES (?, ?, 'selected', 'selected', 'fp', ?)
            """,
            (view.job.job_id, rid, first.content_sha256),
        )
        conn.commit()
    archive_replay(rid, {"id": rid, "uploadtime": 2, "log": "|turn|2|\n", "format": SOURCE}, source)
    snap = create_campaign_snapshot(view.job.job_id)
    assert len(snap.refs) == 1
    assert snap.refs[0].content_sha256 == first.content_sha256


def test_filtered_snapshot_refs_are_policy_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    filters = ReplayFilterConfig()
    store = CollectionStore(tmp_path)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=5, dataset_filters=filters)
    )
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    refs = []
    for i in range(3):
        rid = f"{SOURCE}-par{i}"
        log = build_log(p1_name=f"Alice{i}", p1_rating=1300, p2_rating=900)
        ref = archive_replay(
            rid, {"id": rid, "uploadtime": 1, "log": log, "format": SOURCE}, source
        )
        refs.append(ref)
        with catalog_connection(tmp_path) as conn:
            conn.execute(
                """
                INSERT INTO campaign_replays(
                    campaign_id, replay_id, decision, reason, filter_sha256, version_content_sha256
                ) VALUES (?, ?, 'selected', 'selected', 'fp', ?)
                """,
                (view.job.job_id, rid, ref.content_sha256),
            )
            conn.commit()
    snap = create_campaign_snapshot(view.job.job_id)
    assert len(snap.refs) == 3
    hashes = [ref.log_sha256 for ref in snap.refs]
    assert len(hashes) == len(set(hashes))
    for ref in snap.refs:
        meta = extract_replay_metadata(ref)
        result = evaluate_replay_policy(meta, filters)
        assert result.accepted
        assert result.accepted_sides


def test_snapshot_excludes_pending_and_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    store = CollectionStore(tmp_path)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], dataset_filters=ReplayFilterConfig())
    )
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    selected_id = f"{SOURCE}-sel"
    pending_id = f"{SOURCE}-pend"
    rejected_id = f"{SOURCE}-rej"
    selected = archive_replay(
        selected_id,
        {"id": selected_id, "uploadtime": 1, "log": build_log(p1_name="S"), "format": SOURCE},
        source,
    )
    pending = archive_replay(
        pending_id,
        {"id": pending_id, "uploadtime": 1, "log": build_log(p1_name="P"), "format": SOURCE},
        source,
    )
    rejected = archive_replay(
        rejected_id,
        {"id": rejected_id, "uploadtime": 1, "log": build_log(p1_name="R"), "format": SOURCE},
        source,
    )
    with catalog_connection(tmp_path) as conn:
        conn.execute(
            """
            INSERT INTO campaign_replays(
                campaign_id, replay_id, decision, reason, filter_sha256, version_content_sha256
            ) VALUES (?, ?, 'selected', 'selected', 'fp', ?)
            """,
            (view.job.job_id, selected_id, selected.content_sha256),
        )
        conn.execute(
            """
            INSERT INTO campaign_replays(
                campaign_id, replay_id, decision, reason, filter_sha256, version_content_sha256
            ) VALUES (?, ?, 'pending_publication', 'selected', 'fp', ?)
            """,
            (view.job.job_id, pending_id, pending.content_sha256),
        )
        conn.execute(
            """
            INSERT INTO campaign_replays(
                campaign_id, replay_id, decision, reason, filter_sha256, version_content_sha256
            ) VALUES (?, ?, 'rejected', 'ots_missing', 'fp', ?)
            """,
            (view.job.job_id, rejected_id, rejected.content_sha256),
        )
        conn.commit()
    snap = create_campaign_snapshot(view.job.job_id)
    assert [ref.replay_id for ref in snap.refs] == [selected_id]


def test_prepare_snapshot_uses_same_filters_with_mocked_conversion(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from pokecoach_trainer.dataset_service import create_dataset, prepare_dataset
    from pokecoach_trainer.schemas import ConversionResult, ConversionStatus, DatasetRecipe
    from pokecoach_trainer.settings import ensure_storage_layout

    monkeypatch.setenv("TRAINER_STORAGE_ROOT", str(tmp_path))
    clear_settings_cache()
    ensure_storage_layout()
    filters = ReplayFilterConfig(min_completed_turns_forfeit=2)
    store = CollectionStore(tmp_path)
    view = store.start_campaign(
        CollectionConfig(format_ids=["mc"], max_selected_replays=10, dataset_filters=filters)
    )
    source = ArchiveSource(provenance="t", fetched_at=datetime.now(timezone.utc))
    refs = []
    for i in range(6):
        rid = f"{SOURCE}-prep{i}"
        p2_rating = 900 if i == 0 else 1300
        log = build_log(p1_name=f"P{i}", p2_name=f"Q{i}", p1_rating=1300, p2_rating=p2_rating)
        ref = archive_replay(
            rid, {"id": rid, "uploadtime": 1, "log": log, "format": SOURCE}, source
        )
        refs.append(ref)
        with catalog_connection(tmp_path) as conn:
            conn.execute(
                """
                INSERT INTO campaign_replays(
                    campaign_id, replay_id, decision, reason, filter_sha256, version_content_sha256
                ) VALUES (?, ?, 'selected', 'selected', 'fp', ?)
                """,
                (view.job.job_id, rid, ref.content_sha256),
            )
            conn.commit()
    snap = create_campaign_snapshot(view.job.job_id)
    recipe = DatasetRecipe(
        target_replays=3,
        min_completed_turns_forfeit=2,
        splits={"train": 0.80, "validation": 0.10, "test": 0.10},
    )
    manifest = create_dataset(snap.snapshot_id, recipe, dataset_id="ds-parity")
    requested: list[tuple[str, str]] = []

    def fake_iter(tasks, _executor, **_kwargs):
        from types import SimpleNamespace

        for ref, side in tasks:
            requested.append((ref.replay_id, side))
            yield ConversionResult(
                replay_id=ref.replay_id,
                raw_sha256=ref.content_sha256,
                player_side=side,
                status=ConversionStatus.converted,
                trajectory=SimpleNamespace(acts=[0, 1, 2]),
            )

    @contextmanager
    def fake_pool(**_kwargs):
        yield object()

    monkeypatch.setattr("pokecoach_trainer.conversion.iter_converted_perspectives", fake_iter)
    monkeypatch.setattr("pokecoach_trainer.conversion_runtime.conversion_pool", fake_pool)
    monkeypatch.setattr(
        "pokecoach_trainer.prepare_resources.resolve_conversion_workers",
        lambda *a, **k: 1,
    )
    prepared = prepare_dataset(manifest.dataset_id)
    assert prepared.state.value == "ready"
    assert ("gen9championsvgc2026regmc-prep0", "p2") not in requested
    assert ("gen9championsvgc2026regmc-prep0", "p1") in requested
    decisions = (tmp_path / "datasets" / "ds-parity" / "replay_decisions.jsonl").read_text(
        encoding="utf-8"
    )
    assert "ots_missing" not in decisions
    assert "inactivity_loss" not in decisions
