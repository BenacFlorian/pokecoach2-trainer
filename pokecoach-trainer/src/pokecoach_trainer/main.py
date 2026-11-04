"""Application FastAPI du Trainer."""

from __future__ import annotations

import os

from pokecoach_trainer.settings import get_settings

os.chdir(get_settings().vgc_bench_root.resolve())

from contextlib import asynccontextmanager

from fastapi import FastAPI

from pokecoach_trainer.api.archive import router as archive_router
from pokecoach_trainer.api.collection import router as collection_router
from pokecoach_trainer.api.datasets import router as datasets_router
from pokecoach_trainer.api.errors import trainer_error_handler
from pokecoach_trainer.api.health import router as health_router
from pokecoach_trainer.api.models import router as models_router
from pokecoach_trainer.api.training import router as training_router
from pokecoach_trainer.errors import TrainerError
from pokecoach_trainer.jobs import reconcile_jobs
from pokecoach_trainer.settings import load_dotenv_explicit


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_dotenv_explicit()
    reconcile_jobs()
    yield
    reconcile_jobs()


app = FastAPI(title="PokeCoach Trainer", version="0.1.0", lifespan=lifespan)
app.add_exception_handler(TrainerError, trainer_error_handler)
app.include_router(health_router)
app.include_router(collection_router, prefix="/collection", tags=["collection"])
app.include_router(archive_router, prefix="/archive", tags=["archive"])
app.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
app.include_router(training_router, prefix="/training", tags=["training"])
app.include_router(models_router, prefix="/models", tags=["models"])
