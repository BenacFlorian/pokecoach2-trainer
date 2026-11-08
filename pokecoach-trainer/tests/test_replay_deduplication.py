from pokecoach_trainer.replay_deduplication import duplicate_replay_ids
from pokecoach_trainer.schemas import ArchiveRef


def _ref(replay_id: str, log_sha256: str) -> ArchiveRef:
    return ArchiveRef(
        replay_id=replay_id,
        content_sha256=f"c-{replay_id}",
        log_sha256=log_sha256,
        format_id="gen9championsvgc2026regmc",
        object_relpath=f"objects/{replay_id}/c.json",
    )


def test_two_ids_same_log_one_alias():
    refs = [_ref("b-id", "same"), _ref("a-id", "same")]
    aliases = duplicate_replay_ids(refs, {"a-id", "b-id"})
    assert aliases == {"b-id": "a-id"}


def test_inadmissible_first_leaves_second_canonical():
    refs = [_ref("bad", "same"), _ref("good", "same")]
    aliases = duplicate_replay_ids(refs, {"good"})
    assert aliases == {}


def test_distinct_logs_no_alias():
    refs = [_ref("a", "h1"), _ref("b", "h2")]
    aliases = duplicate_replay_ids(refs, {"a", "b"})
    assert aliases == {}
