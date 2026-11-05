"""Conversion replay→trajectoire avec provenance et contrôles contractuels."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any

from pokecoach_trainer.archive_service import read_archived_object
from pokecoach_trainer.compatibility import ModelContract, build_contract
from pokecoach_trainer.conversion_worker import convert_archive_task, task_seed_for_perspective
from pokecoach_trainer.errors import BlockingMaskError
from pokecoach_trainer.schemas import (
    ArchiveRef,
    ConversionResult,
    ConversionStatus,
    ModelConfig,
)
from pokecoach_trainer.trajectory_dataset import (
    trajectory_to_imitation,
    validate_trajectory_contract,
)

_CONVERTER_REVISION = "conversion/1"
_BLOCKING_REASONS = frozenset(
    {
        "blocking_mask_first_target",
        "blocking_mask_second_target",
        "action_index_out_of_bounds",
        "blocking_mask_error",
        "conversion_timeout",
    }
)
_INFRA_REASONS = frozenset({"conversion_timeout", "conversion_exception"})


class ConversionInfrastructureError(RuntimeError):
    """Échec pool/timeout empêchant la publication."""

    def __init__(self, reason_code: str, details: dict | None = None) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.details = details or {}


def result_from_vendor_trajectory(
    ref: ArchiveRef,
    side_norm: str,
    traj: Any,
    *,
    contract: ModelContract,
) -> ConversionResult:
    if traj is None:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.rejected,
            reason_code="conversion_empty",
        )

    prepared = trajectory_to_imitation(traj)
    try:
        validate_trajectory_contract(
            prepared,
            contract,
            replay_id=ref.replay_id,
            player_side=side_norm,
        )
    except BlockingMaskError as exc:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.error,
            reason_code=exc.code,
            details=exc.details,
        )
    except ValueError as exc:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.rejected,
            reason_code="trajectory_invalid",
            details={"error": str(exc)},
        )

    return ConversionResult(
        replay_id=ref.replay_id,
        raw_sha256=ref.content_sha256,
        player_side=side_norm,  # type: ignore[arg-type]
        status=ConversionStatus.converted,
        reason_code="converted",
        trajectory=prepared,
        converter_revision=_CONVERTER_REVISION,
        encoding_contract=contract.fingerprint,
    )


def _full_identity(ref: ArchiveRef, side: str) -> tuple[str, str, str]:
    side_norm = side if side in {"p1", "p2"} else side.lower()
    return ref.replay_id, ref.content_sha256, side_norm


def iter_converted_perspectives(
    tasks: Iterable[tuple[ArchiveRef, str]],
    executor: ProcessPoolExecutor,
    *,
    contract: ModelContract | None = None,
    seed: int,
    max_in_flight: int | None = None,
) -> Iterator[ConversionResult]:
    """Soumission bornée ; résultats dans l'ordre de complétion."""

    active_contract = contract or build_contract(ModelConfig())
    task_list = list(tasks)
    if not task_list:
        return

    workers_guess = getattr(executor, "_max_workers", 1) or 1
    window = max_in_flight if max_in_flight is not None else max(2, 2 * workers_guess)

    pending: dict[Any, tuple[int, ArchiveRef, str]] = {}
    identities: set[tuple[str, str, str]] = set()
    task_iter = enumerate(task_list)

    def _submit_until_full() -> None:
        while len(pending) < window:
            try:
                index, (ref, side) = next(task_iter)
            except StopIteration:
                break
            ident = _full_identity(ref, side)
            if ident in identities:
                raise ValueError(f"tâche dupliquée: {ident}")
            identities.add(ident)
            task_seed = task_seed_for_perspective(seed, ref, side)
            fut = executor.submit(
                convert_archive_task,
                ref,
                side,
                active_contract,
                task_seed,
            )
            pending[fut] = (index, ref, side)

    _submit_until_full()
    while pending:
        done, _ = wait(tuple(pending.keys()), return_when=FIRST_COMPLETED)
        for fut in done:
            index, ref, side = pending.pop(fut)
            try:
                result = fut.result()
            except Exception as exc:
                raise ConversionInfrastructureError(
                    "broken_process_pool",
                    {"error": str(exc), "index": index},
                ) from exc
            expected = _full_identity(ref, side)
            actual = (result.replay_id, result.raw_sha256, result.player_side)
            if actual != expected:
                raise ValueError(
                    f"provenance incohérente attendu={expected} obtenu={actual}"
                )
            yield result
            if result.status == ConversionStatus.error and result.reason_code in _BLOCKING_REASONS:
                if result.reason_code in _INFRA_REASONS:
                    raise ConversionInfrastructureError(
                        result.reason_code,
                        result.details,
                    )
                raise BlockingMaskError(
                    result.reason_code,
                    details={
                        **(result.details or {}),
                        "replay_id": result.replay_id,
                        "player_side": result.player_side,
                    },
                )
            _submit_until_full()


def convert_perspective(
    ref: ArchiveRef,
    side: str,
    *,
    executor: ProcessPoolExecutor | None = None,
    envelope: dict[str, Any] | None = None,
    contract: ModelContract | None = None,
) -> ConversionResult:
    """
    Convertit une perspective sans filtres métier Elo/gagnant.

    Returns:
        ConversionResult avec trajectoire seulement si status=converted.
    """

    from vgc_bench.logs2trajs import _init_worker_loop, process_log

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

    env = (
        envelope
        if envelope is not None
        else read_archived_object(ref.replay_id, ref.content_sha256)
    )
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

    owns_pool = executor is None
    pool = executor
    if pool is None:
        pool = ProcessPoolExecutor(max_workers=1, initializer=_init_worker_loop)
    try:
        future = pool.submit(process_log, ref.replay_id, log, side_norm, None, False)
        traj = future.result()
    except Exception as exc:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.error,
            reason_code="conversion_exception",
            details={"error": str(exc)},
        )
    finally:
        if owns_pool and pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    if traj is None:
        return ConversionResult(
            replay_id=ref.replay_id,
            raw_sha256=ref.content_sha256,
            player_side=side_norm,  # type: ignore[arg-type]
            status=ConversionStatus.rejected,
            reason_code="conversion_empty",
        )

    active = contract or build_contract(ModelConfig())
    return result_from_vendor_trajectory(ref, side_norm, traj, contract=active)


def convert_perspectives_parallel(
    tasks: list[tuple[ArchiveRef, str]],
    executor: ProcessPoolExecutor,
    *,
    contract: ModelContract | None = None,
    seed: int = 42,
) -> list[ConversionResult]:
    """Adaptateur : retourne les résultats dans l'ordre des tâches d'entrée."""

    if not tasks:
        return []
    index_map = {_full_identity(ref, side): i for i, (ref, side) in enumerate(tasks)}
    ordered: list[ConversionResult | None] = [None] * len(tasks)
    for result in iter_converted_perspectives(
        tasks,
        executor,
        contract=contract,
        seed=seed,
    ):
        key = (result.replay_id, result.raw_sha256, result.player_side)
        ordered[index_map[key]] = result
    return [r for r in ordered if r is not None]


def validate_trajectory(result: ConversionResult, contract: ModelContract) -> None:
    """Contrôle contractuel d'un résultat converti ; lève si invalide."""

    if result.status != ConversionStatus.converted or result.trajectory is None:
        raise ValueError("résultat non converti")
    validate_trajectory_contract(
        result.trajectory,
        contract,
        replay_id=result.replay_id,
        player_side=result.player_side,
    )
