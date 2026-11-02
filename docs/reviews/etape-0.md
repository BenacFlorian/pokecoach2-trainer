# Revue étape 0 — Audit et contrat

| Champ | Valeur |
| --- | --- |
| Résultat technique | **PASSED** (audit documentaire ; chargement BC non exécuté dans ce dépôt) |
| Validation humaine | **en attente** |

## Fichiers à lire (ordre)

1. `docs/audits/compatibilite-initiale.md` — tableau compatibilité et révisions figées.
2. `docs/comprendre-le-bc.md` — glossaire et rôles BC/PPO.
3. `../pokecoach2-python/engine-runtime/manifest.json` — checkpoint de référence (lecture croisée).

## Commandes exécutées

```text
cd ../pokecoach2-python/engine-runtime/vendor/vgc-bench && git rev-parse HEAD && git status --short && git submodule status
git ls-remote …/imitation.git fix-build
git ls-remote …/poke-env.git vgc-bench
```

## Preuves

- VGC-Bench `d79f953…`, Showdown `913da360…`, imitation `56eb3b29…`, poke-env `379a0462…`.
- SHA checkpoint référence `57f5edca…` (manifeste PokeCoach).

## Écarts / décisions

- Format de logs HF : choix final reporté à l’étape 3 après inspection JSON.
- `open-spiel` : hors parcours BC (patch vendor prévu).
- Étapes 8–9 (bridge PokeCoach) : hors dépôt `pokecoach2-trainer` (consigne utilisateur session).

## Questions ouvertes

- Confirmer régulation et fichier logs HF après premier téléchargement (étape 3).
