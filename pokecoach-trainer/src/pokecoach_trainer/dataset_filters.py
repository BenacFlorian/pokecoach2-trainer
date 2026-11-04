"""Filtres métier dataset (décisions traçables, archive intacte)."""

from __future__ import annotations

from pokecoach_trainer.schemas import DatasetRecipe, FilterDecision, FilterStage, ReplayMetadata


def _rating_for_side(meta: ReplayMetadata, side: str) -> int | None:
    return meta.p1_rating if side == "p1" else meta.p2_rating


def _username_for_side(meta: ReplayMetadata, side: str) -> str | None:
    return meta.p1_username if side == "p1" else meta.p2_username


def decide_game(meta: ReplayMetadata, recipe: DatasetRecipe) -> FilterDecision:
    """Décision au niveau partie (format, fin, OTS, équipes, BO)."""

    base = dict(replay_id=meta.replay_id, player_side=None, sources=["replay_metadata"])
    if meta.match_type == "bo1" and not recipe.include_bo1:
        return FilterDecision(
            accepted=False,
            stage=FilterStage.game,
            reason_code="bo1_excluded",
            details={"match_type": meta.match_type},
            **base,
        )
    if meta.match_type == "bo3" and not recipe.include_bo3:
        return FilterDecision(
            accepted=False,
            stage=FilterStage.game,
            reason_code="bo3_excluded",
            details={"match_type": meta.match_type},
            **base,
        )
    if recipe.exclude_inactivity_losses and meta.termination == "inactivity":
        return FilterDecision(
            accepted=False,
            stage=FilterStage.game,
            reason_code="inactivity_loss",
            **base,
        )
    if meta.termination == "tie":
        return FilterDecision(
            accepted=False,
            stage=FilterStage.game,
            reason_code="draw_unsupported",
            **base,
        )
    if meta.termination == "unknown":
        return FilterDecision(
            accepted=False,
            stage=FilterStage.game,
            reason_code="termination_unknown",
            **base,
        )
    if meta.termination == "forfeit":
        if meta.completed_turns is None:
            return FilterDecision(
                accepted=False,
                stage=FilterStage.game,
                reason_code="forfeit_turn_count_unknown",
                **base,
            )
        if recipe.include_forfeits and meta.completed_turns < recipe.min_completed_turns_forfeit:
            return FilterDecision(
                accepted=False,
                stage=FilterStage.game,
                reason_code="forfeit_turns_insufficient",
                details={
                    "completed_turns": meta.completed_turns,
                    "required": recipe.min_completed_turns_forfeit,
                },
                **base,
            )
        if not recipe.include_forfeits:
            return FilterDecision(
                accepted=False,
                stage=FilterStage.game,
                reason_code="forfeit_excluded",
                **base,
            )
    if recipe.require_ots and not (meta.ots_p1 and meta.ots_p2):
        return FilterDecision(
            accepted=False,
            stage=FilterStage.game,
            reason_code="ots_missing",
            details={"ots_p1": meta.ots_p1, "ots_p2": meta.ots_p2},
            **base,
        )
    if recipe.require_complete_teams and not (meta.team_complete_p1 and meta.team_complete_p2):
        return FilterDecision(
            accepted=False,
            stage=FilterStage.game,
            reason_code="team_incomplete",
            details={
                "team_complete_p1": meta.team_complete_p1,
                "team_complete_p2": meta.team_complete_p2,
            },
            **base,
        )
    return FilterDecision(
        accepted=True,
        stage=FilterStage.game,
        reason_code="game_accepted",
        **base,
    )


def decide_perspective(meta: ReplayMetadata, side: str, recipe: DatasetRecipe) -> FilterDecision:
    """Décision au niveau perspective (Elo individuel, gagnant seulement)."""

    game = decide_game(meta, recipe)
    if not game.accepted:
        return game.model_copy(update={"stage": FilterStage.perspective})

    rating = _rating_for_side(meta, side)
    base = dict(replay_id=meta.replay_id, player_side=side, sources=["replay_metadata"])
    if rating is None:
        if recipe.unknown_elo == "exclude_perspective":
            if meta.average_rating is not None:
                return FilterDecision(
                    accepted=False,
                    stage=FilterStage.perspective,
                    reason_code="elo_unknown_no_individual",
                    details={"average_rating": meta.average_rating},
                    **base,
                )
            return FilterDecision(
                accepted=False,
                stage=FilterStage.perspective,
                reason_code="elo_unknown",
                **base,
            )
    else:
        if recipe.min_elo is not None and rating < recipe.min_elo:
            return FilterDecision(
                accepted=False,
                stage=FilterStage.perspective,
                reason_code="elo_below_min",
                details={"rating": rating, "min_elo": recipe.min_elo},
                **base,
            )
        if recipe.max_elo is not None and rating > recipe.max_elo:
            return FilterDecision(
                accepted=False,
                stage=FilterStage.perspective,
                reason_code="elo_above_max",
                details={"rating": rating, "max_elo": recipe.max_elo},
                **base,
            )
    if recipe.only_winners:
        username = _username_for_side(meta, side)
        if meta.winner and username and meta.winner != username:
            return FilterDecision(
                accepted=False,
                stage=FilterStage.perspective,
                reason_code="only_winners_loser",
                details={"winner": meta.winner, "username": username},
                **base,
            )
    return FilterDecision(
        accepted=True,
        stage=FilterStage.perspective,
        reason_code="perspective_accepted",
        **base,
    )
