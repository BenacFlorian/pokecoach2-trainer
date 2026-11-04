"""Conversion replay→trajectoire avec provenance et contrôles contractuels."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

from pokecoach_trainer.archive_service import read_archived_object
from pokecoach_trainer.compatibility import ModelContract, build_contract
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

    prepared = trajectory_to_imitation(traj)
    contract = contract or build_contract(ModelConfig())
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


def convert_perspectives_parallel(
    tasks: list[tuple[ArchiveRef, str]],
    executor: ProcessPoolExecutor,
    *,
    contract: ModelContract | None = None,
) -> list[ConversionResult]:
    """
    Convertit plusieurs perspectives ; l'ordre de retour suit l'ordre des tâches.

    Les résultats restent identifiés par replay_id et côté même si le pool termine
    dans un ordre différent.
    """

    from vgc_bench.logs2trajs import process_log

    futures: dict[tuple[str, str], Any] = {}
    for ref, side in tasks:
        env = read_archived_object(ref.replay_id, ref.content_sha256)
        provider = env.get("provider") or {}
        log = provider.get("log", "")
        fut = executor.submit(process_log, ref.replay_id, log, side, None, False)
        futures[(ref.replay_id, side)] = (fut, ref)

    results: list[ConversionResult] = []
    for ref, side in tasks:
        fut, ref_obj = futures[(ref.replay_id, side)]
        try:
            traj = fut.result()
        except Exception as exc:
            results.append(
                ConversionResult(
                    replay_id=ref.replay_id,
                    raw_sha256=ref.content_sha256,
                    player_side=side,  # type: ignore[arg-type]
                    status=ConversionStatus.error,
                    reason_code="conversion_exception",
                    details={"error": str(exc)},
                )
            )
            continue
        if traj is None:
            results.append(
                ConversionResult(
                    replay_id=ref.replay_id,
                    raw_sha256=ref.content_sha256,
                    player_side=side,  # type: ignore[arg-type]
                    status=ConversionStatus.rejected,
                    reason_code="conversion_empty",
                )
            )
            continue
        prepared = trajectory_to_imitation(traj)
        active_contract = contract or build_contract(ModelConfig())
        try:
            validate_trajectory_contract(
                prepared,
                active_contract,
                replay_id=ref.replay_id,
                player_side=side,
            )
        except BlockingMaskError as exc:
            results.append(
                ConversionResult(
                    replay_id=ref.replay_id,
                    raw_sha256=ref.content_sha256,
                    player_side=side,  # type: ignore[arg-type]
                    status=ConversionStatus.error,
                    reason_code=exc.code,
                    details=exc.details,
                )
            )
            continue
        except ValueError as exc:
            results.append(
                ConversionResult(
                    replay_id=ref.replay_id,
                    raw_sha256=ref.content_sha256,
                    player_side=side,  # type: ignore[arg-type]
                    status=ConversionStatus.rejected,
                    reason_code="trajectory_invalid",
                    details={"error": str(exc)},
                )
            )
            continue
        results.append(
            ConversionResult(
                replay_id=ref.replay_id,
                raw_sha256=ref.content_sha256,
                player_side=side,  # type: ignore[arg-type]
                status=ConversionStatus.converted,
                reason_code="converted",
                trajectory=prepared,
                converter_revision=_CONVERTER_REVISION,
                encoding_contract=active_contract.fingerprint,
            )
        )
    return results


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
