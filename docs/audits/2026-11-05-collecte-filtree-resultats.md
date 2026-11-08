# Collecte filtrée par profil dataset — résultats (2026-11-05)

## Environnement

- Répertoire : `pokecoach2-trainer/pokecoach-trainer/`
- Python / outil : `uv run --locked`
- Stockage de production **non modifié** (aucune migration, purge ou collecte 100k)

## Profil appliqué

```yaml
min_elo: 1100
max_elo: null
unknown_elo: exclude_perspective
only_winners: false
require_ots: true
require_complete_teams: true
include_bo1: true
include_bo3: true
include_forfeits: true
min_completed_turns_forfeit: 2
exclude_inactivity_losses: true
deduplicate: true
```

Parseur `replay_metadata/2`, politique `replay_policy/1`. Recette figée : `configs/dataset-mc-collectable.yaml`, campagne : `configs/collection-mc-100k-valid.yaml`.

## Commandes et résultats

| Commande | Résultat |
| --- | --- |
| Suite ciblée du plan (policy, log, metadata, admission, workers, snapshot, HTTP quota=3, etc.) | OK |
| `uv run --locked pytest -q --tb=line` | OK, **375 passed, 5 skipped**, ~319 s |
| `git diff --check` | OK |
| `uv run --locked ruff check src tests` | **194** diagnostics (I001 / E501 surtout). Aucune régression fonctionnelle associée. Le dépôt n’était pas ruff-clean avant ce chantier. |

Scénario HTTP local (`tests/test_collection_filtered_http.py`, serveur `tests/collection_http_server.py`) :

- mélange OTS absente / Elo trop bas / inactivité / 3 valides
- 1 worker, pause/reprise, puis 2 workers
- quota = 3 → exactement 3 objets publiés, 3 rejets, snapshot de 3 refs

Reprise / crash (hooks `before_rename`, `after_rename_before_catalog`, `after_catalog_before_finalize`) : un objet, une sélection, zéro claim de hash `pending`.

## Audit lecture seule — 200 objets du stock réel

- Provenance : `pokecoach-trainer/storage/archive/catalog.sqlite3` (v1, 104 367 versions, 4 campagnes `stopped`)
- Accès : SQLite `mode=ro` + lecture JSON, **sans écriture catalogue**, **sans migration v2**
- Échantillon : 200 premières versions par `ORDER BY replay_id`
- Durée moyenne d’extraction + politique : **0,8 ms** / objet (0,15 s au total)

| Décision | n |
| --- | --- |
| `ots_missing` | 195 |
| `no_eligible_perspective` | 3 |
| `replay_accepted` | 2 (4 perspectives) |
| fichiers manquants / erreurs | 0 |

Cet échantillon **n’est pas** un rendement global : il est ordonné par identifiant, pas tiré au hasard, et ne représente pas les 104k. Le chiffre utilisateur « ~25 % valides » n’est ni confirmé ni infirmé ici.

## Limites de preuve

- Admissible ≠ convertible. Le test de parité dataset mocke la conversion.
- Les budgets de `collection-mc-100k-valid.yaml` (`max_requests: 200000`) ne garantissent pas 100k retenus si le taux d’acceptation est bas.
- Aucune collecte Showdown réelle n’a été lancée pendant ce chantier.
- Catalogue de production toujours en schéma v1 ; la migration 1→2 n’a été exercée que sur des copies temporaires de tests.

## Commandes après validation (volontaires)

```bash
uv run --locked pokecoach-trainer collection start --config configs/collection-mc-100k-valid.yaml
uv run --locked pokecoach-trainer collection status --job-id JOB_ID
```

Snapshot : `POST /collection/jobs/{job_id}/snapshot`, puis

```bash
uv run --locked pokecoach-trainer dataset create \
  --snapshot-id SNAPSHOT_ID \
  --recipe configs/dataset-mc-collectable.yaml \
  --dataset-id DATASET_ID
```
