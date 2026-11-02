"""Routes de santé."""

from fastapi import APIRouter

from pokecoach_trainer import __version__
from pokecoach_trainer.upstream import collect_runtime_versions

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "trainer_version": __version__, "versions": collect_runtime_versions()}
