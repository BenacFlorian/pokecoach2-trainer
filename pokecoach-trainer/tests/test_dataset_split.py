import hashlib

from pokecoach_trainer.dataset_split import (
    assign_splits,
    build_replay_groups,
    select_groups,
)
from pokecoach_trainer.schemas import ArchiveRef, ArchiveSnapshot, DatasetRecipe, ReplayGroup, ReplayMetadata


def _meta(replay_id: str, *, log_sha256: str, linked: list[str] | None = None) -> ReplayMetadata:
    return ReplayMetadata(
        replay_id=replay_id,
        raw_sha256=log_sha256,
        log_sha256=log_sha256,
        format_id="gen9championsvgc2026regmc",
        series_link_status="confirmed" if linked else "unknown",
        linked_replay_ids=linked or [],
    )


def _snapshot(refs: list[ArchiveRef]) -> ArchiveSnapshot:
    return ArchiveSnapshot(
        snapshot_id="snap-test",
        snapshot_sha256="sha",
        format_ids=["gen9championsvgc2026regmc"],
        refs=refs,
    )


def test_alias_and_link_merge_single_group():
    refs = [
        ArchiveRef(
            replay_id="a",
            content_sha256="c1",
            log_sha256="logsame",
            format_id="fmt",
            object_relpath="o/a",
        ),
        ArchiveRef(
            replay_id="b",
            content_sha256="c2",
            log_sha256="logsame",
            format_id="fmt",
            object_relpath="o/b",
        ),
    ]
    metas = {
        "a": _meta("a", log_sha256="logsame", linked=["b"]),
        "b": _meta("b", log_sha256="logsame", linked=["a"]),
    }
    groups = build_replay_groups(_snapshot(refs), metas)
    assert len(groups) == 1
    assert set(groups[0].replay_ids) == {"a", "b"}


def test_select_groups_deterministic():
    groups = [
        ReplayGroup(group_id="g1", replay_ids=["g1"]),
        ReplayGroup(group_id="g2", replay_ids=["g2"]),
        ReplayGroup(group_id="g3", replay_ids=["g3"]),
    ]
    first, count, _ = select_groups(
        groups, seed=42, target_replays=2, admissible_replays={"g1", "g2", "g3"}
    )
    second, count2, _ = select_groups(
        groups, seed=42, target_replays=2, admissible_replays={"g1", "g2", "g3"}
    )
    assert [g.group_id for g in first] == [g.group_id for g in second]
    assert count == count2 == 2


def test_assign_splits_three_non_empty():
    recipe = DatasetRecipe(target_replays=3)
    groups = [
        ReplayGroup(group_id=f"g{i}", replay_ids=[f"g{i}"]) for i in range(1, 4)
    ]
    selected = {f"g{i}" for i in range(1, 4)}
    assignments = assign_splits(groups, recipe, selected_replay_ids=selected)
    assert assignments.trainable
    assert assignments.replay_counts["train"] >= 1
    assert assignments.replay_counts["validation"] >= 1
    assert assignments.replay_counts["test"] >= 1


def test_group_sort_key_uses_sha256_not_builtin_hash():
    key = hashlib.sha256("42:g1".encode()).hexdigest()
    from pokecoach_trainer.dataset_split import _group_sort_key

    assert _group_sort_key(42, "g1") == key
