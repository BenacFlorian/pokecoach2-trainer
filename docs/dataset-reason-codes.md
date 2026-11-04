# Codes de rejet dataset (recette `natural`)

Les filtres émettent des `FilterDecision` ; l'archive brute n'est jamais modifiée.

## Niveau partie (`stage=game`)

| Code | Signification |
| --- | --- |
| `game_accepted` | Partie admissible au niveau global |
| `bo1_excluded` / `bo3_excluded` | Type de match exclu par la recette |
| `inactivity_loss` | Défaite par inactivité |
| `draw_unsupported` | Égalité (non convertie en V1) |
| `termination_unknown` | Fin non classifiable |
| `forfeit_turn_count_unknown` | Abandon sans nombre de tours clos fiable |
| `forfeit_turns_insufficient` | Abandon avant le seuil de tours terminés |
| `forfeit_excluded` | Abandons désactivés dans la recette |
| `ots_missing` | Feuille ouverte inexploitable pour au moins une équipe |
| `team_incomplete` | Moins de six identités de Pokémon repérables |

## Niveau perspective (`stage=perspective`)

| Code | Signification |
| --- | --- |
| `perspective_accepted` | Perspective admissible |
| `elo_below_min` / `elo_above_max` | Elo individuel hors bornes |
| `elo_unknown` | Rating individuel absent |
| `elo_unknown_no_individual` | Seul un rating moyen est disponible |
| `only_winners_loser` | Perdant exclu car `only_winners=true` |

## Sélection et conversion

| Code | Signification |
| --- | --- |
| `not_selected_target_reached` | Candidat non traité après atteinte de la cible |
| `conversion_empty` | Convertisseur sans trajectoire |
| `conversion_exception` | Erreur technique de conversion |
| `trajectory_invalid` | Forme de trajectoire invalide |
| `blocking_mask_first_target` / `blocking_mask_second_target` | Action humaine interdite par le masque BC (bloquant) |
| `action_index_out_of_bounds` | Indice d'action hors bornes |

## Publication

| Code | Signification |
| --- | --- |
| `split_groups_insufficient` | Moins de trois groupes non vides pour splitter |
| `target_not_reached` | Moins de replays exploitables que `target_replays` |
