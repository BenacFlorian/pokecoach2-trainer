# Audit — entraînement BC en flux (2026-11-08)

## Périmètre mesuré

- Machine : environnement de développement du dépôt (CPU uniquement pour cette exécution).
- Dataset : `ds-mb-mc-collectable-80k-v1`, sous-ensemble **8 premiers fichiers train** (76 transitions), lecture seule.
- Parent : `vgcbench-mb-reference`.
- Options : `batch_size=32`, `evaluation_batch_size=256`, `shuffle_buffer_mib=256`, sampler `file_buffer_shuffle/1`.

## Résultats (`/tmp/pokecoach-training-streaming-benchmark.json`)

| Métrique | Valeur |
| --- | ---: |
| Transitions (sous-ensemble) | 76 |
| Appels policy (éval batchée) | 1 |
| NLL batchée | 3.516 |
| Lecture 8 fichiers | ~1 ms |
| Évaluation batchée | ~119 ms |
| Passe shuffle (1 epoch logique) | ~5 ms |

## Limites

- Pas d’exécution CUDA sur cette machine (mesure GPU : non exécutée).
- Les durées CPU sur 76 transitions ne prédisent ni le temps des ~1,05 M transitions train ni le coût cloud.
- Aucun pickle, manifeste ou parent modifié pendant le benchmark.
