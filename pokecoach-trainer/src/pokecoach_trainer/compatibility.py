"""Empreinte canonique du contrat modèle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from pokecoach_trainer.schemas import ModelConfig
from pokecoach_trainer.settings import get_settings
from pokecoach_trainer.upstream import (
    IMITATION_COMMIT,
    POKE_ENV_COMMIT,
    VGC_BENCH_COMMIT,
)


@dataclass(frozen=True)
class ModelContract:
    fingerprint: str
    vgc_bench_commit: str
    imitation_commit: str
    poke_env_commit: str
    act_len: int
    mask_rules: dict


def _canonical_json(data: object) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def build_contract(model: ModelConfig) -> ModelContract:
    """Calcule l'empreinte à partir des sources, règles de masque et architecture explicite."""

    if model is None:
        raise TypeError("ModelConfig requis ; aucune architecture Tiny implicite")
    root = get_settings().vgc_bench_root
    with root.joinpath("vgc_bench/src/utils.py").open("rb") as f:
        utils_hash = hashlib.sha256(f.read()).hexdigest()
    with root.joinpath("vgc_bench/src/policy.py").open("rb") as f:
        policy_hash = hashlib.sha256(f.read()).hexdigest()
    from pokecoach_trainer.upstream import vgc_bench_workdir

    with vgc_bench_workdir():
        from vgc_bench.src.utils import act_len

    payload = {
        "vgc_bench_commit": VGC_BENCH_COMMIT,
        "imitation_commit": IMITATION_COMMIT,
        "poke_env_commit": POKE_ENV_COMMIT,
        "policy_sha256": policy_hash,
        "utils_sha256": utils_hash,
        "act_len": act_len,
        "architecture": model.model_dump(),
        "bc_synthetic_mask": {
            "forbidden_ranges_per_mon": [[47, 87]],
            "layout": "two_halves_of_length_act_len",
        },
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return ModelContract(
        fingerprint=digest,
        vgc_bench_commit=VGC_BENCH_COMMIT,
        imitation_commit=IMITATION_COMMIT,
        poke_env_commit=POKE_ENV_COMMIT,
        act_len=act_len,
        mask_rules=payload["bc_synthetic_mask"],
    )


def contracts_compatible(left: ModelContract, right: ModelContract) -> bool:
    """
    Vérifie l'égalité sémantique de deux contrats (empreinte versionnée).

    Une permutation de tables à dimensions identiques produit une empreinte différente
    via les hashes des sources `policy.py` / `utils.py` et l'architecture explicite.
    """

    return (
        left.fingerprint == right.fingerprint
        and left.act_len == right.act_len
        and left.vgc_bench_commit == right.vgc_bench_commit
    )
