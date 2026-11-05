"""Helpers partagés pour tests prepare multiprocessing."""

from __future__ import annotations

import os
from concurrent.futures import Future
from contextlib import contextmanager

from pokecoach_trainer.schemas import ArchiveRef, ConversionResult, ConversionStatus


class FakeConversionExecutor:
    """Executor minimal compatible avec iter_converted_perspectives."""

    def __init__(self, *, max_workers: int = 2) -> None:
        self._max_workers = max_workers
        self.submit_calls = 0

    def submit(self, fn, ref, side, contract, task_seed):
        self.submit_calls += 1
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


@contextmanager
def fake_conversion_pool(*, workers: int = 2, **_kwargs):
    yield FakeConversionExecutor(max_workers=workers)


def fake_converted_iter(tasks, executor, *, contract=None, seed=42, max_in_flight=None):
    for ref, side in tasks:
        yield ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side,  # type: ignore[arg-type]
            status=ConversionStatus.converted,
            reason_code="converted",
            trajectory=_fake_traj_for_tests(),
        )


def _fake_traj_for_tests():
    import numpy as np
    from imitation.data.types import Trajectory
    from vgc_bench.src.utils import chunk_obs_len

    steps = 2
    obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((steps, 2), dtype=np.int64)
    return Trajectory(obs=obs, acts=acts, infos=None, terminal=True)


def archive_ref(replay_id: str, sha: str = "abc") -> ArchiveRef:
    return ArchiveRef(
        replay_id=replay_id,
        content_sha256=sha,
        log_sha256="log",
        format_id="fmt",
        object_relpath=f"objects/{replay_id}",
    )
