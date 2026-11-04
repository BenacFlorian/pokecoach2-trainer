"""Regroupement, sélection déterministe et splits train/validation/test."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Literal

from pokecoach_trainer.schemas import (
    ArchiveSnapshot,
    DatasetRecipe,
    ReplayGroup,
    ReplayMetadata,
    SplitAssignments,
)
from pokecoach_trainer.storage import new_opaque_id


def _union_find(parent: dict[str, str], a: str, b: str) -> None:
    root_a = _find(parent, a)
    root_b = _find(parent, b)
    if root_a != root_b:
        parent[root_b] = root_a


def _find(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def build_replay_groups(
    snapshot: ArchiveSnapshot,
    metas: dict[str, ReplayMetadata],
) -> list[ReplayGroup]:
    """
    Composantes connexes : aliases (hash de log) et liens BO3 confirmés.

    Les groupes incluent les replays rejetés plus tard ; identifiant canonique =
    plus petit replay_id de la composante.
    """

    refs_by_id = {ref.replay_id: ref for ref in snapshot.refs}
    log_hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for ref in snapshot.refs:
        log_hash_to_ids[ref.log_sha256].append(ref.replay_id)

    parent: dict[str, str] = {ref.replay_id: ref.replay_id for ref in snapshot.refs}
    for ids in log_hash_to_ids.values():
        if len(ids) > 1:
            base = min(ids)
            for other in ids:
                if other != base:
                    _union_find(parent, base, other)

    for replay_id, meta in metas.items():
        if meta.series_link_status != "confirmed":
            continue
        for linked in meta.linked_replay_ids:
            if linked in refs_by_id and linked != replay_id:
                _union_find(parent, replay_id, linked)

    components: dict[str, set[str]] = defaultdict(set)
    for replay_id in refs_by_id:
        root = _find(parent, replay_id)
        components[root].add(replay_id)

    groups: list[ReplayGroup] = []
    for root, members in sorted(components.items(), key=lambda item: item[0]):
        ordered = sorted(members)
        group_id = ordered[0]
        alias_ids = sorted({rid for rid in ordered if rid != group_id})
        series_id = None
        for rid in ordered:
            meta = metas.get(rid)
            if meta and meta.series_id:
                series_id = meta.series_id
                break
        groups.append(
            ReplayGroup(
                group_id=group_id,
                replay_ids=ordered,
                alias_ids=alias_ids,
                series_id=series_id,
            )
        )
    return groups


def _group_sort_key(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()


def select_groups(
    groups: list[ReplayGroup],
    *,
    seed: int,
    target_replays: int,
    admissible_replays: set[str],
) -> tuple[list[ReplayGroup], int, int]:
    """
    Sélectionne des groupes entiers jusqu'à la cible.

    Returns:
        (groupes retenus, replays comptés, dépassement)
    """

    ordered = sorted(groups, key=lambda g: _group_sort_key(seed, g.group_id))
    selected: list[ReplayGroup] = []
    count = 0
    for group in ordered:
        if count >= target_replays:
            break
        playable = [rid for rid in group.replay_ids if rid in admissible_replays]
        if not playable:
            continue
        selected.append(group)
        count += 1
        if count >= target_replays:
            break
    overshoot = max(0, count - target_replays)
    return selected, count, overshoot


def assign_splits(
    groups: list[ReplayGroup],
    recipe: DatasetRecipe,
    *,
    selected_replay_ids: set[str],
    ungrouped_bo3: int = 0,
) -> SplitAssignments:
    """
    Algorithme `group_greedy_v1` : affectation entière par déficit de ratio.

    Raises:
        ValueError: si moins de trois groupes non vides après affectation.
    """

    split_id = new_opaque_id("split-")
    if len(groups) < 3:
        return SplitAssignments(
            split_id=split_id,
            trainable=False,
            reason_code="split_groups_insufficient",
            series_leakage_guarantee=ungrouped_bo3 == 0,
            ungrouped_bo3_count=ungrouped_bo3,
        )

    ratios = recipe.splits
    total_replays = len(selected_replay_ids)
    targets = {
        "train": ratios.train * total_replays,
        "validation": ratios.validation * total_replays,
        "test": ratios.test * total_replays,
    }
    counts = {"train": 0, "validation": 0, "test": 0}
    group_to_split: dict[str, Literal["train", "validation", "test"]] = {}
    split_names: list[Literal["train", "validation", "test"]] = [
        "train",
        "validation",
        "test",
    ]

    ordered_groups = sorted(
        groups,
        key=lambda g: (-len(g.replay_ids), _group_sort_key(recipe.seed, g.group_id)),
    )
    remaining = list(ordered_groups)
    empty_splits = 3

    while remaining:
        if empty_splits > 0 and len(remaining) == empty_splits:
            for split_name, group in zip(split_names, remaining, strict=False):
                group_to_split[group.group_id] = split_name
                counts[split_name] += len(
                    [rid for rid in group.replay_ids if rid in selected_replay_ids]
                )
            break
        group = remaining.pop(0)
        size = len([rid for rid in group.replay_ids if rid in selected_replay_ids])
        deficits = {
            name: targets[name] - counts[name]
            for name in split_names
        }
        chosen = max(split_names, key=lambda name: (deficits[name], -split_names.index(name)))
        group_to_split[group.group_id] = chosen
        counts[chosen] += size
        if counts[chosen] > 0 and empty_splits:
            empty_splits -= 1

    if min(counts.values()) <= 0:
        return SplitAssignments(
            split_id=split_id,
            trainable=False,
            reason_code="split_groups_insufficient",
            group_to_split=group_to_split,
            replay_counts=counts,
            series_leakage_guarantee=ungrouped_bo3 == 0,
            ungrouped_bo3_count=ungrouped_bo3,
        )

    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    for group in groups:
        split = group_to_split.get(group.group_id)
        if not split:
            continue
        for rid in group.replay_ids:
            if rid not in selected_replay_ids:
                continue
            if split == "train":
                train_ids.append(rid)
            elif split == "validation":
                val_ids.append(rid)
            else:
                test_ids.append(rid)

    return SplitAssignments(
        split_id=split_id,
        train=sorted(train_ids),
        validation=sorted(val_ids),
        test=sorted(test_ids),
        group_to_split=group_to_split,
        replay_counts=counts,
        trainable=True,
        series_leakage_guarantee=ungrouped_bo3 == 0,
        ungrouped_bo3_count=ungrouped_bo3,
    )
