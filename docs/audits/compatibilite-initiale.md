# Audit de compatibilité initiale — Palier 1 BC Tiny

Date : 2026-11-02. Périmètre : figer le contrat Trainer ↔ PokeCoach avant installation du package.

## Révisions Git constatées

| Composant | Chemin | HEAD / commit | État local |
| --- | --- | --- | --- |
| VGC-Bench (copie PokeCoach) | `../pokecoach2-python/engine-runtime/vendor/vgc-bench` | `d79f9532947ac114dce1dda2456a590afcd375b2` | `?? uv.lock` non suivi |
| Showdown (sous-module VGC-Bench) | `…/pokemon-showdown` | `913da3602a3aa1db79f9fdc5d5222eaf8d39569d` | branche `remotes/origin/vgc-bench` |
| Fork `imitation` | branche `fix-build` | `56eb3b295da5d52b8f7277c6ba206913b19ab110` | résolu via `git ls-remote` |
| Fork `poke-env` | branche `vgc-bench` | `379a04628e42b2a2c94790c0788204b652f7ae1b` | résolu via `git ls-remote` |
| Manifeste moteur PokeCoach | `../pokecoach2-python/engine-runtime/manifest.json` | commit VGC aligné | checkpoint ref SHA-256 documenté |

## Checkpoint de référence (policy historique)

| Champ | Valeur |
| --- | --- |
| Fichier amont | `results/saves_bc/seed1/100.zip` |
| Hugging Face | `cameronangliss/vgc-bench-models` |
| SHA-256 attendu (manifeste PokeCoach) | `57f5edcab415cf6ccc1b6231923c8b66d3b1b7249b6b2562b97531471e4ca60b` |
| Chargement bridge | `PPO.load(path, device).policy` dans `policy_player.set_policy` |
| Vérification fichier | `verify_checkpoint` (SHA-256) ; **ne prouve pas** la désérialisation SB3 |

Architecture historique figée dans le code amont (`AttentionExtractor`) : `d_model=256` (via policy_kwargs à l’entraînement BC), `embed_layers=3`, `num_heads=4`, `embed_len=32`, `choose_on_teampreview` selon checkpoint.

## Tableau identique / différent / conséquence / correction minimale

| Élément | Statut | Conséquence | Correction minimale (palier 1) |
| --- | --- | --- | --- |
| `act_len` = 107, `action_map` | Identique (source unique VGC-Bench) | Même espace d’actions | Copier VGC-Bench au commit ci-dessus ; patch policy configurable versionné |
| Dimensions observation (`chunk_obs_len`, tables abilities/items/moves) | Identique si même commit + `data/*.json` | Mismatch → logits incohérents | Même vendor + empreinte contrat (étape 5) |
| Masque légal en jeu (`DoublesEnv` / joueur) | Identique côté inférence | — | Réutiliser `policy_player` + env amont |
| Masque BC entraînement (`pretrain.py`) | Synthétique : ones sauf `[:,47:87]` et seconde moitié | Cibles humaines peuvent être « interdites » par le masque loss | Valider chaque transition (étape 3) ; documenter dans `comprendre-le-bc.md` |
| Masque conditionnel 2e action (`_update_mask`) | Identique | Paire d’actions illégale → erreur BC | Tests masques étape 2–4 |
| Team Preview | `choose_on_teampreview` dans policy | Preview imposée PokeCoach vs policy | Tiny avec `true` ; tests intégration PokeCoach (hors repo Trainer) |
| Manifeste moteur vs manifeste modèle Trainer | Différent | PokeCoach ne connaît pas encore contrat Trainer | Manifeste export Trainer (étape 5) ; extension bridge (étape 8, dépôt PokeCoach) |
| Deux joueurs même checkpoint | Identique aujourd’hui dans `battle_session.py` | N’accepte pas Tiny vs historique | `VGC_OPPONENT_*` dans PokeCoach (étape 8, hors repo Trainer) |
| Formats VGC (`format_map`) | `ma` / `mb` → reg 2026 M-A / M-B | Dataset et Showdown doivent aligner | Choisir **un** format après inspection logs HF ; ne pas supposer M-B seul |
| Dépendance `open-spiel` | Déclarée dans pyproject VGC-Bench | Install peut échouer | Exclue du parcours BC (`eval.py` seulement) ; patch vendor documenté |
| `POKECOACH_SHOWDOWN_SERVER` | défaut `ws://127.0.0.1:8090/...` | Port réel à vérifier localement | Devbox Node pour Showdown si tests live |

## Format de dataset retenu (proposition à valider à l’étape 3)

- **Source** : Hugging Face `cameronangliss/vgc-battle-logs` ou archive SV, **un fichier JSON** par régulation compatible avec `logs2trajs.py` au commit figé.
- **Sélection** : intersection format parseur / équipes PokeCoach / serveur Showdown ; commencer par quelques replays puis 200–1000.
- **Révision dataset** : hash du fichier brut + commit VGC-Bench + liste patches.

## Description canonique du contrat (empreinte étape 5)

1. Commit VGC-Bench + SHA patches `policy.py` (layers/heads/embed).
2. Commits Git `poke-env` + `imitation` + versions `torch`, `stable_baselines3`.
3. Ordre des features observation (layout `utils.py` + fichiers `data/`).
4. `act_len`, table `action_map`, règles `_update_mask`.
5. Sémantique masque BC synthétique (plages 47:87 par Pokémon).
6. Hyperparamètres policy sauvegardés dans zip SB3 (`policy_kwargs`).

Fichiers sources pour hash : `vgc_bench/src/{policy,utils,policy_player,env}.py`, `data/*.json`, patches `patches/*.patch`.

## Outils système proposés (Devbox)

- `uv` (Python 3.13.x précis après `uv python pin`)
- Optionnel ultérieur : `nodejs` + `npm` pour Showdown vendor (étape 1 diagnostic / combats)

## Limites de cet audit

Inspection de sources et commits uniquement — pas de preuve de chargement BC ni de conversion logs à cette date (réservé étapes 1 et 3).
