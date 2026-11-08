"""Filtrage facultatif date/équipes avant publication (pur, sans I/O réseau)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pokecoach_trainer.collection_types import CollectionEvaluation, SelectionDecision, TeamSummary
from pokecoach_trainer.replay_log import preview_team_tuple
from pokecoach_trainer.schemas import ArchiveRef, CollectionConfig, TeamFilter
from pokecoach_trainer.storage import canonical_json_sha256

_PARSER_VERSION = "collection_filters/2"


def load_team_filter(path: Path) -> TeamFilter:
    """Charge `{"teams": [[six ids], ...]}` depuis un fichier JSON."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data.keys()) != {"teams"}:
        raise ValueError("JSON d'équipes invalide: clé teams attendue seule")
    return TeamFilter.model_validate(data)


def filter_fingerprint(config: CollectionConfig) -> str:
    """Empreinte des filtres utilisateur actifs (date/équipe + profil dataset éventuel)."""

    if config.dataset_filters is None:
        payload = {
            "upload_after": config.upload_after.isoformat() if config.upload_after else None,
            "upload_before": config.upload_before.isoformat() if config.upload_before else None,
            "team_filter": (
                config.team_filter.model_dump(mode="json") if config.team_filter else None
            ),
        }
    else:
        from pokecoach_trainer.replay_policy import replay_filter_fingerprint

        payload = {
            "business": replay_filter_fingerprint(config.dataset_filters),
            "upload_after": config.upload_after.isoformat() if config.upload_after else None,
            "upload_before": config.upload_before.isoformat() if config.upload_before else None,
            "team_filter": (
                config.team_filter.model_dump(mode="json") if config.team_filter else None
            ),
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def extract_team_summary(payload: dict[str, Any]) -> TeamSummary:
    """Extrait les six espèces initiales par joueur depuis le log preview."""

    log = payload.get("log")
    content_sha256 = canonical_json_sha256(
        {k: payload[k] for k in ("id", "uploadtime", "log", "format") if k in payload}
    )
    if not isinstance(log, str) or not log:
        return TeamSummary(
            p1=None, p2=None, parser_version=_PARSER_VERSION, content_sha256=content_sha256
        )
    lines = log.splitlines()
    p1 = preview_team_tuple(lines, "p1")
    p2 = preview_team_tuple(lines, "p2")
    return TeamSummary(
        p1=p1,
        p2=p2,
        parser_version=_PARSER_VERSION,
        content_sha256=content_sha256,
    )


def _team_matches(filter_team: list[str], known: tuple[str, ...]) -> bool:
    return tuple(sorted(filter_team)) == known


def evaluate_selection(
    config: CollectionConfig,
    *,
    uploadtime: int | None,
    summary: TeamSummary | None,
) -> SelectionDecision:
    """Applique les filtres utilisateur `[after, before)` et équipes (OU)."""

    fp = filter_fingerprint(config)
    has_date = config.upload_after is not None or config.upload_before is not None
    if has_date:
        if uploadtime is None:
            return SelectionDecision(selected=False, reason="date_unknown", filter_sha256=fp)
        if config.upload_after is not None and uploadtime < int(config.upload_after.timestamp()):
            return SelectionDecision(selected=False, reason="outside_period", filter_sha256=fp)
        if config.upload_before is not None and uploadtime >= int(config.upload_before.timestamp()):
            return SelectionDecision(selected=False, reason="outside_period", filter_sha256=fp)

    if config.team_filter is None:
        return SelectionDecision(selected=True, reason="selected", filter_sha256=fp)

    if summary is None:
        return SelectionDecision(selected=False, reason="team_unknown", filter_sha256=fp)

    for team in config.team_filter.teams:
        if summary.p1 is not None and _team_matches(team, summary.p1):
            return SelectionDecision(selected=True, reason="selected", filter_sha256=fp)
        if summary.p2 is not None and _team_matches(team, summary.p2):
            return SelectionDecision(selected=True, reason="selected", filter_sha256=fp)

    if summary.p1 is None or summary.p2 is None:
        return SelectionDecision(selected=False, reason="team_unknown", filter_sha256=fp)
    return SelectionDecision(selected=False, reason="team_mismatch", filter_sha256=fp)


def evaluate_collection_candidate(
    config: CollectionConfig,
    ref: ArchiveRef,
    envelope: dict[str, Any],
) -> CollectionEvaluation:
    """Date/équipe puis politique dataset (sans I/O)."""

    from pokecoach_trainer.replay_metadata import extract_replay_metadata
    from pokecoach_trainer.replay_policy import evaluate_replay_policy

    provider = envelope.get("provider") or {}
    summary = extract_team_summary(provider)
    uploadtime = provider.get("uploadtime")
    ts = int(uploadtime) if uploadtime is not None else None
    date_decision = evaluate_selection(config, uploadtime=ts, summary=summary)
    if not date_decision.selected:
        return CollectionEvaluation(
            selection=date_decision, metadata=None, eligibility=None, summary=summary
        )
    meta = extract_replay_metadata(ref, cached=envelope)
    if config.dataset_filters is None:
        return CollectionEvaluation(
            selection=date_decision, metadata=meta, eligibility=None, summary=summary
        )
    eligibility = evaluate_replay_policy(meta, config.dataset_filters)
    if not eligibility.accepted:
        return CollectionEvaluation(
            selection=SelectionDecision(
                selected=False,
                reason=eligibility.reason_code,
                filter_sha256=date_decision.filter_sha256,
            ),
            metadata=meta,
            eligibility=eligibility,
            summary=summary,
        )
    return CollectionEvaluation(
        selection=SelectionDecision(
            selected=True, reason="selected", filter_sha256=date_decision.filter_sha256
        ),
        metadata=meta,
        eligibility=eligibility,
        summary=summary,
    )
