from concurrent.futures import Future

import numpy as np
import pytest
from imitation.data.types import Trajectory
from vgc_bench.src.utils import chunk_obs_len

from pokecoach_trainer.conversion import convert_perspectives_parallel
from pokecoach_trainer.schemas import ArchiveRef, ConversionResult, ConversionStatus
from vgc_bench.logs2trajs import process_logs_with_provenance


def _fake_traj(steps: int = 2) -> Trajectory:
    obs = np.zeros((steps + 1, 12 * chunk_obs_len), dtype=np.float32)
    acts = np.zeros((steps, 2), dtype=np.int64)
    return Trajectory(obs=obs, acts=acts, infos=None, terminal=True)


def test_process_logs_with_provenance_order_independent(monkeypatch):
    traj_a = _fake_traj()
    traj_b = _fake_traj(steps=1)

    class FakeExecutor:
        def submit(self, _fn, tag, _log, role, *_rest):
            fut: Future = Future()
            if tag == "slow" and role == "p1":
                fut.set_result(traj_b)
            elif tag == "fast" and role == "p2":
                fut.set_result(traj_a)
            else:
                fut.set_result(None)
            return fut

    logs = {"slow": ("", "logslow"), "fast": ("", "logfast")}
    tagged = process_logs_with_provenance(logs, FakeExecutor(), None, False, False)
    mapping = {(t.replay_id, t.player_side): t.trajectory for t in tagged}
    assert mapping[("slow", "p1")] is traj_b
    assert mapping[("fast", "p2")] is traj_a


def test_convert_perspectives_parallel_keeps_task_order(monkeypatch):
    ref_a = ArchiveRef(
        replay_id="a",
        content_sha256="1",
        log_sha256="l1",
        format_id="fmt",
        object_relpath="o/a",
    )
    ref_b = ArchiveRef(
        replay_id="b",
        content_sha256="2",
        log_sha256="l2",
        format_id="fmt",
        object_relpath="o/b",
    )

    class FakeExecutor:
        _max_workers = 2

        def submit(self, _fn, ref, side, _contract, _seed):
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

    results = convert_perspectives_parallel(
        [(ref_b, "p2"), (ref_a, "p1")],
        FakeExecutor(),
    )
    assert [r.replay_id for r in results] == ["b", "a"]
    assert all(r.status == ConversionStatus.rejected for r in results)
