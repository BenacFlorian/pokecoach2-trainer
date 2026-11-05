"""Erreurs métier typées pour l'API et les services."""

from __future__ import annotations

from typing import Any


class TrainerError(Exception):
    """Erreur métier avec code stable et détails structurés."""

    code: str = "trainer_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class NotFound(TrainerError):
    code = "not_found"


class Conflict(TrainerError):
    code = "conflict"


class InvalidConfiguration(TrainerError):
    code = "invalid_configuration"


class ContractMismatch(TrainerError):
    code = "contract_mismatch"


class BlockingMaskError(TrainerError):
    code = "blocking_mask_error"


class StaleLease(TrainerError):
    code = "stale_lease"


class BudgetReached(TrainerError):
    code = "budget_reached"


class ProviderUnavailable(TrainerError):
    code = "provider_unavailable"


class ProviderRateLimited(TrainerError):
    code = "provider_rate_limited"


class ProviderInvalidPayload(TrainerError):
    code = "provider_invalid_payload"


class StorageIncompatible(TrainerError):
    code = "storage_incompatible"
