"""Autorisations HTTP globales et backoff fournisseur."""

from __future__ import annotations

import random
import secrets
from email.utils import parsedate_to_datetime
from typing import Literal

from pokecoach_trainer.collection_store import CollectionStore
from pokecoach_trainer.collection_types import PermitResult, RequestPermit, TaskLease
from pokecoach_trainer.errors import BudgetReached, StaleLease


def acquire_request_permit(
    store: CollectionStore,
    lease: TaskLease,
    kind: Literal["search", "fetch"],
    *,
    jitter: random.Random | None = None,
) -> PermitResult:
    jitter = jitter or random.Random(0)
    if not store.verify_lease_for_permit(lease):
        return PermitResult(permit=None, wait_s=0.0, stop_reason="stale_lease")
    attempt = store.get_attempt_for_permit(lease.attempt_id)
    if attempt["requests_consumed"] >= attempt["max_requests"]:
        return PermitResult(permit=None, wait_s=0.0, stop_reason="max_requests")
    if store._clock.wall() >= attempt["deadline_at"]:
        return PermitResult(permit=None, wait_s=0.0, stop_reason="deadline")
    limits = store.host_limit_state()
    now_mono = store._clock.monotonic()
    now_wall = store._clock.wall()
    if limits["boot_id"] != store._clock.boot_id():
        store.sync_host_limit_boot()
        limits = store.host_limit_state()
        now_mono = store._clock.monotonic()
    pause_until = limits["provider_pause_until_wall"]
    if pause_until is not None and now_wall < float(pause_until):
        return PermitResult(permit=None, wait_s=float(pause_until) - now_wall, stop_reason=None)
    next_slot = float(limits["next_slot_mon"])
    if next_slot > now_mono + 3600:
        next_slot = now_mono
        store.update_host_limit_slot(next_slot)
    if now_mono < next_slot:
        return PermitResult(permit=None, wait_s=next_slot - now_mono, stop_reason=None)
    rate = float(limits["effective_rate"])
    spacing = 1.0 / rate if rate > 0 else 1.0
    permit_id = secrets.token_hex(8)
    store.reserve_request_permit(
        attempt_id=lease.attempt_id, task_id=lease.task_id, kind=kind, permit_id=permit_id
    )
    store.update_host_limit_slot(now_mono + spacing + jitter.uniform(0, 1.0))
    return PermitResult(
        permit=RequestPermit(
            permit_id=permit_id,
            attempt_id=lease.attempt_id,
            task_id=lease.task_id,
            kind=kind,
        ),
        wait_s=0.0,
        stop_reason=None,
    )


def record_request_result(
    store: CollectionStore,
    permit: RequestPermit,
    *,
    status_code: int | None,
    elapsed_s: float,
    retry_after: str | None = None,
) -> None:
    state = "completed"
    if status_code is None:
        state = "uncertain"
    elif status_code == 429:
        state = "completed"
        limits = store.host_limit_state()
        pause = store._clock.wall() + _retry_after_seconds(retry_after, permit, store)
        new_rate = max(0.1, float(limits["configured_rate"]) / 2.0)
        store.set_provider_pause(pause, new_rate)
    elif status_code >= 500:
        state = "uncertain"
    store.finish_request_permit(permit.permit_id, state=state, status_code=status_code)


def _retry_after_seconds(
    header: str | None, permit: RequestPermit, store: CollectionStore
) -> float:
    if header:
        try:
            if header.strip().isdigit():
                return float(header)
            dt = parsedate_to_datetime(header)
            return max(0.0, dt.timestamp() - store._clock.wall())
        except (ValueError, TypeError, OverflowError):
            pass
    attempt = store.get_attempt_for_permit(permit.attempt_id)
    used = int(attempt["requests_consumed"])
    return min(2 ** max(0, used - 1), 60.0)
