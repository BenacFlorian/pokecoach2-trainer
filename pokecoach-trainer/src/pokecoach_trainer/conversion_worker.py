"""Travail sérialisable exécuté dans les workers de conversion."""

from __future__ import annotations

import atexit
import hashlib
import os
import random
from typing import Any

from pokecoach_trainer.archive_service import read_archived_object
from pokecoach_trainer.compatibility import ModelContract
from pokecoach_trainer.schemas import ArchiveRef, ConversionResult, ConversionStatus

_WORKER_TIMEOUT_S: float = 120.0
_WORKER_FINALIZED = False


def _stable_task_seed(recipe_seed: int, replay_id: str, raw_sha256: str, side: str) -> int:
    digest = hashlib.sha256(
        f"{recipe_seed}:{replay_id}:{raw_sha256}:{side}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def initialize_conversion_worker(
    storage_root: str,
    vendor_root: str,
    timeout_s: float,
) -> None:
    global _WORKER_TIMEOUT_S, _WORKER_FINALIZED

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["TRAINER_STORAGE_ROOT"] = storage_root
    os.chdir(vendor_root)

    from pokecoach_trainer.settings import clear_settings_cache

    clear_settings_cache()

    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    from vgc_bench.logs2trajs import _init_worker_loop

    _init_worker_loop()
    _WORKER_TIMEOUT_S = timeout_s
    _WORKER_FINALIZED = False

    def _finalize_worker() -> None:
        global _WORKER_FINALIZED
        if _WORKER_FINALIZED:
            return
        _WORKER_FINALIZED = True
        try:
            from vgc_bench import logs2trajs

            loop = getattr(logs2trajs, "_READER_LOOP", None)
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

    atexit.register(_finalize_worker)


def convert_archive_task(
    ref: ArchiveRef,
    side: str,
    contract: ModelContract,
    task_seed: int,
) -> ConversionResult:
    """Convertit une perspective à partir de l'archive (sans pool imbriqué)."""

    from pokecoach_trainer.conversion import result_from_vendor_trajectory
    from vgc_bench.logs2trajs import process_log

    side_norm = side if side in {"p1", "p2"} else side.lower()
    if side_norm not in {"p1", "p2"}:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side="p1",
            status=ConversionStatus.error,
            reason_code="invalid_side",
            details={"side": side},
        )

    random.seed(task_seed)
    try:
        import numpy as np

        np.random.seed(task_seed % (2**32))
    except Exception:
        pass

    env = read_archived_object(ref.replay_id, ref.content_sha256)
    provider = env.get("provider") or {}
    log = provider.get("log")
    if not isinstance(log, str):
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.error,
            reason_code="missing_log",
        )

    try:
        traj = process_log(
            ref.replay_id,
            log,
            side_norm,
            None,
            False,
            read_timeout_s=_WORKER_TIMEOUT_S,
        )
    except TimeoutError:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.error,
            reason_code="conversion_timeout",
            details={"timeout_s": _WORKER_TIMEOUT_S},
        )
    except Exception as exc:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.error,
            reason_code="conversion_exception",
            details={"error": str(exc)},
        )

    return result_from_vendor_trajectory(
        ref,
        side_norm,
        traj,
        contract=contract,
    )


def task_seed_for_perspective(
    recipe_seed: int,
    ref: ArchiveRef,
    side: str,
) -> int:
    return _stable_task_seed(recipe_seed, ref.replay_id, ref.content_sha256, side)
