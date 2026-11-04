# Audit M-C — compatibilité et baseline (lot A, tâche 0)

**Date :** 2026-11-03  
**Résultat technique :** PASSED (baseline locale)  
**Validation humaine lot A :** en attente

## Dépôts et commits

| Dépôt | HEAD | État working tree |
| --- | --- | --- |
| pokecoach2-trainer | `f4afd00` fix(dataset) | +1 commit local non poussé ; docs plan/spec non suivis |
| pokecoach2-python | `6fae866` | `simulation_budgets.py` modifié ; `engine-runtime/vendor/` non suivi |
| pokecoach2-angular | `48f2297` | propre |

## Environnement

- Devbox : scripts `setup-vendor`, `test`, `check`, `start` à la racine Git trainer.
- Python/uv : environnement verrouillé sous `pokecoach-trainer/` (`uv run --locked`).
- VGC-Bench attendu : commit `d79f9532947ac114dce1dda2456a590afcd375b2` (voir `upstream.py`).

## Baseline tests (timeout 120 s)

Commande :

```bash
cd pokecoach-trainer && uv run --locked pytest tests/test_dataset.py tests/test_jobs.py tests/test_health.py -q
```

- **Code de sortie :** 0  
- **Processus :** terminé sans blocage (7 tests).

Suite complète après lot B (tâches 2–3) :

```bash
uv run --locked pytest tests/ -q
```

- **Code de sortie :** 0 (38 tests).

## Acquis dataset (ne pas réécrire)

| Correction | État | Preuve |
| --- | --- | --- |
| Préparation sans `force=True` | implémenté | `test_prepare_dataset_runs_without_force` |
| Provenance replay/côté | implémenté | `TaggedTrajectory`, `process_logs_with_provenance`, `_write_prepared_trajectories` |

Patch vendor : `patches/logs2trajs-provenance.patch` + réapplication dans `scripts/sync_vgc_bench.sh`.

## Vendor et synchronisation

- Trainer : `pokecoach-trainer/vendor/vgc-bench` gitignoré ; source rsync depuis `pokecoach2-python/engine-runtime/vendor/vgc-bench`.
- Risque : `rsync --delete` écrase les adaptations locales si le patch n’est pas réappliqué — mitigé par patch versionné (tâche 2).

## Problèmes préexistants / à suivre

1. **Jobs historiques** : réconciliation simplifiée ; flock remplace le fichier `.training.lock` (lot B tâche 3 partielle).
2. **Entraînement** : checkpoint M-B non chargé par défaut ; Tiny implicite retiré de `build_contract()` — fine-tuning lot D à faire.
3. **pytest long** : la suite complète ~11 s ; pas de fuite bloquante observée sur la baseline ci-dessus.
4. **PokeCoach bridge** : factory encore orientée M-B ; lot E tâche 15.

## Prochaines preuves lot A (tâche 1)

- Charger le checkpoint parent réel (profil / `REFERENCE_CHECKPOINT_PATH`).
- Formats Showdown M-C source vs moteur : compléter `mc-runtime.json`.
- Conversion d’une perspective M-C réelle vs contrat encodage.
