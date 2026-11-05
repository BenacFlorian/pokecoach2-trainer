"""Conversion bornée, provenance et erreurs infrastructure."""

from __future__ import annotations

from concurrent.futures import Future

import pytest

from pokecoach_trainer.conversion import (
    ConversionInfrastructureError,
    iter_converted_perspectives,
)
from pokecoach_trainer.schemas import ArchiveRef, ConversionResult, ConversionStatus
from prepare_process_helpers import archive_ref


def test_in_flight_is_bounded(monkeypatch):
    tasks = [(archive_ref(f"r{i}", str(i)), "p1") for i in range(12)]
    observed: list[int] = []
    import pokecoach_trainer.conversion as conv_mod

    real_wait = conv_mod.wait

    def tracking_wait(fs, **kwargs):
        observed.append(len(fs))
        return real_wait(fs, **kwargs)

    monkeypatch.setattr(conv_mod, "wait", tracking_wait)

    class SlowExecutor:
        _max_workers = 2

        def submit(self, _fn, ref, side, _c, _s):
            fut: Future = Future()
            fut.set_result(
                ConversionResult(
                    replay_id=ref.replay_id,
                    raw_sha256=ref.content_sha256,
                    player_side=side,  # type: ignore[arg-type]
                    status=ConversionStatus.rejected,
                    reason_code="conversion_empty",
                )
            )
            return fut

    list(
        iter_converted_perspectives(
            tasks,
            SlowExecutor(),
            seed=1,
            max_in_flight=4,
        )
    )
    assert observed
    assert max(observed) <= 4


def test_completion_order_preserves_full_identity():
    ref_a = archive_ref("a", "1")
    ref_b = archive_ref("b", "2")
    results = []

    class OrderExecutor:
        _max_workers = 2

        def submit(self, _fn, ref, side, _c, _s):
            fut: Future = Future()
            fut.set_result(
                ConversionResult(
                    replay_id=ref.replay_id,
                    raw_sha256=ref.content_sha256,
                    player_side=side,  # type: ignore[arg-type]
                    status=ConversionStatus.rejected,
                    reason_code="conversion_empty",
                )
            )
            return fut

    for result in iter_converted_perspectives(
        [(ref_b, "p2"), (ref_a, "p1")],
        OrderExecutor(),
        seed=99,
        max_in_flight=2,
    ):
        results.append(result)
    assert {(r.replay_id, r.raw_sha256, r.player_side) for r in results} == {
        ("b", "2", "p2"),
        ("a", "1", "p1"),
    }


def test_same_replay_two_hashes_do_not_overwrite_future():
    ref_v1 = archive_ref("same", "hash1")
    ref_v2 = archive_ref("same", "hash2")

    class DualExecutor:
        _max_workers = 2

        def submit(self, _fn, ref, side, _c, _s):
            fut: Future = Future()
            fut.set_result(
                ConversionResult(
                    replay_id=ref.replay_id,
                    raw_sha256=ref.content_sha256,
                    player_side=side,  # type: ignore[arg-type]
                    status=ConversionStatus.rejected,
                    reason_code="conversion_empty",
                )
            )
            return fut

    results = list(
        iter_converted_perspectives(
            [(ref_v1, "p1"), (ref_v2, "p1")],
            DualExecutor(),
            seed=3,
        )
    )
    assert len(results) == 2
    assert {r.raw_sha256 for r in results} == {"hash1", "hash2"}


def test_duplicate_full_identity_is_rejected():
    ref = archive_ref("dup", "x")

    class OneShotExecutor:
        _max_workers = 1

        def submit(self, *_a, **_k):
            fut: Future = Future()
            fut.set_result(None)
            return fut

    with pytest.raises(ValueError, match="dupliquée"):
        list(
            iter_converted_perspectives(
                [(ref, "p1"), (ref, "p1")],
                OneShotExecutor(),
                seed=1,
            )
        )


def test_broken_process_pool_aborts():
    ref = archive_ref("x", "1")

    class BrokenExecutor:
        _max_workers = 1

        def submit(self, *_a, **_k):
            fut: Future = Future()
            fut.set_exception(RuntimeError("pool broken"))
            return fut

    with pytest.raises(ConversionInfrastructureError, match="broken_process_pool"):
        list(iter_converted_perspectives([(ref, "p1")], BrokenExecutor(), seed=1))
