"""Agrégats de progression collecte."""

from __future__ import annotations

from typing import Any

from pokecoach_trainer.archive_catalog import catalog_connection
from pokecoach_trainer.collection_store import CollectionStore


def collection_progress(store: CollectionStore, campaign_id: str) -> dict[str, Any]:
    """Vue bornée de progression pour polling API/CLI."""

    view = store.get_campaign(campaign_id)
    with catalog_connection(store._root) as conn:
        pending = conn.execute(
            """
            SELECT COUNT(*) AS n FROM download_tasks
            WHERE state IN ('pending', 'leased', 'retry_wait')
            """
        ).fetchone()["n"]
        rejected = conn.execute(
            """
            SELECT reason, COUNT(*) AS n FROM campaign_replays
            WHERE campaign_id=? AND decision='rejected'
            GROUP BY reason
            """,
            (campaign_id,),
        ).fetchall()
        selected = conn.execute(
            """
            SELECT COUNT(*) AS n FROM campaign_replays
            WHERE campaign_id=? AND decision='selected'
            """,
            (campaign_id,),
        ).fetchone()["n"]
        windows_open = conn.execute(
            """
            SELECT COUNT(*) AS n FROM windows w
            JOIN campaign_windows cw ON cw.window_id = w.window_id
            WHERE cw.campaign_id=? AND w.state NOT IN ('complete', 'blocked')
            """,
            (campaign_id,),
        ).fetchone()["n"]
        attempt = conn.execute(
            "SELECT * FROM attempts WHERE attempt_id=?", (view.attempt_id,)
        ).fetchone()
    return {
        "phase": view.phase,
        "coverage_incomplete": view.coverage_incomplete,
        "counters": {
            **view.counters,
            "selected_replays": int(selected),
            "pending_downloads": int(pending),
            "requests_consumed": int(attempt["requests_consumed"]) if attempt else 0,
            "new_published": int(attempt["new_published"]) if attempt else 0,
        },
        "rejected_by_reason": {row["reason"]: int(row["n"]) for row in rejected},
        "discovery_windows_open": int(windows_open),
        "effective_rate": None,
        "eta_s": None if int(pending) == 0 else None,
    }
