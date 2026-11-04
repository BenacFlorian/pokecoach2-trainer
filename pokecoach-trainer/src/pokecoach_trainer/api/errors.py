"""Gestion des erreurs métier pour FastAPI."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from pokecoach_trainer.errors import (
    BlockingMaskError,
    Conflict,
    ContractMismatch,
    InvalidConfiguration,
    NotFound,
    TrainerError,
)


def trainer_error_status(exc: TrainerError) -> int:
    if isinstance(exc, NotFound):
        return 404
    if isinstance(exc, Conflict):
        return 409
    if isinstance(exc, (InvalidConfiguration, ContractMismatch, BlockingMaskError)):
        return 422
    return 500


async def trainer_error_handler(_request: Request, exc: TrainerError) -> JSONResponse:
    return JSONResponse(status_code=trainer_error_status(exc), content=exc.to_dict())
