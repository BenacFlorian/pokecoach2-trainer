"""Politique de filtrage replay partagée entre collecte et dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

from pokecoach_trainer.dataset_filters import decide_game, decide_perspective
from pokecoach_trainer.replay_metadata import REPLAY_METADATA_VERSION
from pokecoach_trainer.schemas import DatasetRecipe, FilterDecision, ReplayFilterConfig, ReplayMetadata

FILTER_POLICY_VERSION = "replay_policy/1"

_FILTER_FIELD_NAMES = frozenset(ReplayFilterConfig.model_fields.keys())


def filters_from_recipe(recipe: DatasetRecipe) -> ReplayFilterConfig:
    data = recipe.model_dump(mode="json", include=_FILTER_FIELD_NAMES)
    return ReplayFilterConfig.model_validate(data)


def replay_filter_fingerprint(filters: ReplayFilterConfig) -> str:
    payload = {
        "policy": FILTER_POLICY_VERSION,
        "parser": REPLAY_METADATA_VERSION,
        "filters": filters.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayEligibility:
    accepted: bool
    reason_code: str
    accepted_sides: tuple[Literal["p1", "p2"], ...]
    decisions: tuple[FilterDecision, ...]


def evaluate_replay_policy(
    metadata: ReplayMetadata,
    filters: ReplayFilterConfig,
) -> ReplayEligibility:
    game = decide_game(metadata, filters)
    decisions: list[FilterDecision] = [game]
    if not game.accepted:
        return ReplayEligibility(
            accepted=False,
            reason_code=game.reason_code,
            accepted_sides=(),
            decisions=tuple(decisions),
        )
    accepted_sides: list[Literal["p1", "p2"]] = []
    for side in ("p1", "p2"):
        persp = decide_perspective(metadata, side, filters)
        decisions.append(persp)
        if persp.accepted:
            accepted_sides.append(side)  # type: ignore[arg-type]
    if not accepted_sides:
        return ReplayEligibility(
            accepted=False,
            reason_code="no_eligible_perspective",
            accepted_sides=(),
            decisions=tuple(decisions),
        )
    return ReplayEligibility(
        accepted=True,
        reason_code="replay_accepted",
        accepted_sides=tuple(accepted_sides),
        decisions=tuple(decisions),
    )
