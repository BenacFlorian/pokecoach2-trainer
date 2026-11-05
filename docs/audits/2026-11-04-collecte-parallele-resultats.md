# Résultats collecte parallèle (2026-11-04)

## Environnement

- Répertoire : `pokecoach2-trainer/pokecoach-trainer/`
- Suite unitaire : `uv run --locked pytest -q`
- Intégration : `uv run --locked pytest tests/test_collection_integration.py -q`
- Lint : `uv run --locked ruff check src tests`
- Benchmark synthétique :
  `uv run --locked python scripts/benchmark_collection.py --sizes 20000 --workers 1 8 --output /tmp/collection-benchmark.json`

## Mesures réelles Showdown

Non exécutées dans cette passe (harness HTTP local uniquement).

## Synthétique

Le script `scripts/benchmark_collection.py` insère des métadonnées par lots de 1 000 via `insert_replay_version` (sans gros logs) et mesure latences d’insertion / lookup.

## Vérification (2026-11-04, passe finale)

- `uv run --locked pytest -q` : **OK** (suite complète, ~3,7 min)
- Tests ajoutés en fin de passe : isolation snapshots campagne, dataset figé pendant croissance archive, intégration parallèle corrigée (racine stockage par run)

## Tests d’intégration

- `test_parallel_pipeline_matches_serial_unique_set` : 5 IDs, 1 vs 4 workers, même ensemble retenu.
- `test_restart_after_crash_preserves_pipeline` : reprise de lease téléchargement.
- `test_rate_limit_and_budget_hold_under_process_races` : 429 respecte `max_requests`.
- `test_snapshot_and_compute_coexist_with_collection` : verrous compute + snapshot campagne.

## Limites connues

- Benchmark 2 M non lancé automatiquement (durée / espace disque).
- Couverture fenêtres : matérialisation simplifiée pour backfill multi-formats.
- Mesure Showdown réelle (1 000 requêtes / 10 min) : à planifier séparément.
