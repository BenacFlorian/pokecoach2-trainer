# PokeCoach Trainer — Plan d’implémentation du palier 1 BC Tiny

> **Pour l’agent qui exécutera ce plan :** utiliser `superpowers:executing-plans`, étape par étape. Les cases suivent l’avancement technique ; chaque étape se termine par une **pause obligatoire pour revue humaine**. Ne jamais commencer l’étape suivante sans validation explicite de l’utilisateur. La rédaction de ce plan n’autorise pas son implémentation.

**Objectif :** entraîner un petit modèle par imitation de replays humains, depuis une API locale, puis utiliser son checkpoint dans PokeCoach pour terminer un combat et reproduire la preuve après redémarrage.

**Architecture :** un Trainer Python indépendant : Devbox à la racine du dépôt fournit uv ; uv gère le projet empaquetable `pokecoach-trainer/`, son environnement `.venv` et ses dépendances verrouillées. Le package `src/pokecoach_trainer/` expose une API **FastAPI**, servie par Uvicorn, et une CLI. Un worker Python distinct réutilise les services d’entraînement et la copie contrôlée de VGC-Bench. Données, états, métriques et modèles sur disque ; intégration finale dans le bridge VGC-Bench déjà présent dans PokeCoach.

**Technologies :** Devbox, uv, Python 3.13 comme version de départ, package Python construit avec Hatchling, FastAPI, Uvicorn, Pydantic, PyTorch, Stable-Baselines3, les forks `imitation` et `poke-env`, Pokémon Showdown, pytest et Ruff. Les versions exactes seront figées après l’audit dans `devbox.lock`, `.python-version` et `uv.lock`.

**Spécification :** [PokeCoach Palier 1 — Trainer BC Tiny](../specs/step/PokeCoach_Palier_1_Trainer_BC_Tiny.md).

**Date :** 2 novembre 2026. **Statut :** plan à relire ; aucune étape d’implémentation exécutée par ce document.

**Révision demandée :** Devbox et uv sont obligatoires pour initialiser et utiliser le Trainer ; FastAPI est le framework HTTP retenu. Cette consigne précise les exemples d’installation de la spécification : la recette du Trainer passera par uv, sans installation manuelle par pip.

## 1. Comment nous allons travailler ensemble

Le but est que tu puisses comprendre et valider le code au fur et à mesure, même sans expérience d’entraînement de modèles. Nous construirons d’abord les éléments observables séparément : charger un modèle, inspecter un exemple, entraîner, exporter, puis piloter par HTTP et jouer dans PokeCoach.

### Règle de pause

À la fin de **chaque étape numérotée de 0 à 9**, l’agent doit :

1. Présenter les fichiers modifiés et proposer un ordre de lecture court, avec liens vers les fonctions importantes.
2. Expliquer ce que fait le code, avec un exemple concret et le vocabulaire nouveau.
3. Donner les commandes réellement exécutées, leurs résultats et les limites. Un test ignoré n’est pas un test réussi.
4. Fournir une démonstration adaptée à l’étape : sortie CLI, exemple JSON, métriques ou trace de partie.
5. Noter les écarts à la spécification et les décisions prises.
6. **S’arrêter et attendre ta validation du code.** Les corrections demandées restent dans cette étape ; elles doivent être revérifiées avant une nouvelle revue.

Un « continue » autorise l’étape suivante, jusqu’à sa propre pause. Une question ou une absence de réponse ne vaut pas approbation. Un test vert ne remplace pas ta revue. Si un point bloquant apparaît au milieu d’une étape, présenter les éléments établis et le choix à résoudre, sans empiler les étapes suivantes.

Garder un compte rendu par étape dans `docs/reviews/etape-<N>.md` : résultat technique (`PASSED`, `FAILED` ou `BLOCKED`), preuves, fichiers à lire, questions résolues, validation humaine (`en attente` ou réponse explicite). Un petit commit par étape validée est recommandé ; ne pas inclure les modifications préexistantes ni les artefacts lourds.

### Exigences de lisibilité et de commentaires

- Docstrings en français pour les fonctions ajoutées : rôle, entrées/sorties, effets sur disque ou sur les poids, erreurs importantes.
- Commenter les **raisons** des choix : pourquoi le split est fait par partie, pourquoi le masque est synthétique, pourquoi PPO ne réalise pas ici l’apprentissage.
- Dans le chemin d’apprentissage, indiquer les formes réelles des tenseurs : observations, deux actions, masque, probabilités et loss. Les dimensions sont déduites du code figé, jamais inventées.
- Distinguer explicitement `train()` du réseau, calcul de loss, rétropropagation et mise à jour par l’optimiseur. Expliquer ce que délègue notre code à `imitation.BC`.
- Expliquer dans la validation pourquoi `eval()` et l’absence de gradients sont nécessaires ; restaurer ensuite le mode d’entraînement.
- Conserver les commentaires amont utiles. Ajouter des notes ciblées aux modifications VGC-Bench, sans traduire ou reformater tout le dépôt.
- Préférer de petites fonctions nommées à une abstraction générique. Pas de framework maison d’entraînement.
- Faire évoluer `docs/comprendre-le-bc.md` avec un exemple réel anonymisé et un glossaire : replay, trajectoire, transition, observation, action, policy, batch, époque, loss, gradient, optimiseur, checkpoint, inférence.

**Une étape n’est pas prête pour revue si le code fonctionne mais reste opaque.**

## 2. Point de départ constaté dans l’espace de travail

Cette inspection porte sur les fichiers locaux ; elle ne constitue pas une validation de l’environnement, des dépendances installées ou d’un entraînement réel.

| Élément | Constat et conséquence |
| --- | --- |
| `pokecoach2-trainer/` | Dépôt Git autonome, avec `step.md` et la spécification ; pas encore de code Trainer observé. Ne pas créer un second dépôt à l’intérieur. |
| `step.md` | Ancienne feuille de route séparant FastAPI et architecture configurable en paliers ultérieurs. Pour ce plan, la spécification détaillée fait référence : les deux font partie du palier 1. |
| VGC-Bench local | `../pokecoach2-python/engine-runtime/vendor/vgc-bench/`, HEAD observé : `d79f9532947ac114dce1dda2456a590afcd375b2`. Point de départ recommandé, à confirmer à l’étape 0. |
| Showdown de ce VGC-Bench | HEAD observé : `913da3602a3aa1db79f9fdc5d5222eaf8d39569d`. Vérifier également le gitlink attendu et les modifications locales. |
| Dépendances amont | `poke-env` et `imitation` sont référencés par branches de forks dans le `pyproject.toml`. Résoudre et enregistrer des commits exacts ; ces noms de branches ne suffisent pas à figer l’environnement. |
| `pretrain.py` amont | Utilise bien `imitation.BC`, un conteneur PPO, un masque synthétique et un découpage inadapté aux petits datasets ; évaluations de combats coûteuses présentes. |
| Bridge PokeCoach | `VgcBenchEngineAdapter` → sous-processus JSONL → `pokecoach_vgc_bridge` → `PolicyPlayer`. Réutiliser ce chemin. |
| Recherche du code VGC-Bench | `vgc_source.py` accepte `POKECOACH_VGC_BENCH_ROOT`. Utile pour tester une révision compatible sans reconstruire un second encodeur. |
| Chargement et manifeste | `vgc_real.py` vérifie le fichier/hash avant d’annoncer `checkpointLoaded` ; ce signal seul ne prouve pas la désérialisation effective de la policy. Le manifeste moteur existant n’est pas encore le manifeste de modèle exigé ici. |
| Tests existants | Preview, premier tour, combat complet et contraintes dans `../pokecoach2-python/tests/integration/`. Certains chemins utilisent des fixtures : les distinguer de la preuve réelle. |

Fichiers PokeCoach à lire en priorité à l’audit :

- `engine-runtime/manifest.json` et `docs/audits/vgc-bench.md` ;
- `engine-runtime/src/pokecoach_vgc_bridge/{vgc_source,vgc_real,battle_session,battle_runner,checkpoint}.py` ;
- `src/pokecoach_python/engine/{vgc_bench_adapter,check}.py` et `src/pokecoach_python/core/config.py` ;
- `tests/integration/test_vgc_{preview,first_turn,full_battle,constraints}_real.py` ;
- `engine-runtime/vendor/vgc-bench/vgc_bench/{pretrain,logs2trajs}.py` et `src/{policy,policy_player,env,utils}.py`.

Ces chemins sont relatifs à `../pokecoach2-python/`. Les modifications locales existantes, notamment les fichiers de lock, doivent être inventoriées et conservées.

## 3. Contraintes globales et choix proposés

- **Devbox + uv obligatoires** : Devbox fournit les outils système ; uv choisit Python, crée `.venv`, résout/verrouille les dépendances et exécute le package. Versionner `devbox.json`, `devbox.lock`, `pyproject.toml`, `.python-version` et `uv.lock` ; ignorer `.devbox/`, `.venv/`, `.env` et les données locales.
- Déclarer les dépendances et les sources locales/Git dans `pyproject.toml`, y compris VGC-Bench et ses forks figés ; utiliser `uv sync --locked` pour reproduire l’environnement. Aucun `pip install` complémentaire non représenté dans le verrou.
- **API FastAPI** : application dans `src/pokecoach_trainer/main.py`, routeurs dédiés, schémas Pydantic, cycle de vie `lifespan` pour la récupération des jobs, Uvicorn pour le lancement. Ajouter FastAPI/Uvicorn dès l’initialisation ; les routes métier arrivent à l’étape 7.
- **BC uniquement** : les poids sont ajustés par `imitation.algorithms.bc.BC`. PPO fournit le conteneur et la sauvegarde ; ne pas appeler `ppo.learn()`.
- Tiny : `d_model=128`, `num_layers=2`, `num_heads=4`, `embed_len=32`, `choose_on_teampreview=true`, `seed=42`, `batch_size=32`, `num_epochs=3`, `device=auto`.
- Conserver les valeurs historiques : largeur de référence 256, 3 couches, 4 têtes, embeddings de 32. Aucun changement de l’encodage ou de l’espace d’actions pour réduire la taille.
- Vérifier `d_model > 0`, `num_layers >= 1`, `num_heads >= 1`, `d_model % num_heads == 0`, ainsi que la validité des autres paramètres exposés.
- Même régulation pour les données et la démonstration. **Ne pas choisir automatiquement M-B** parce que les tests PokeCoach actuels la citent : vérifier l’intersection logs disponibles / parseur / moteur / PokeCoach / équipes.
- Viser 200 à 1 000 replays réels ; commencer la conversion sur quelques replays pour comprendre le format. Le petit essai de diagnostic ne remplace pas le dataset de démonstration.
- Pas de scraping, RL, recherche d’hyperparamètres, nouvelle régulation, dashboard Angular, Redis, Celery ou base de données dans le Trainer.
- FastAPI sur `127.0.0.1`, un seul worker d’entraînement actif, écritures atomiques sur disque et lancement par liste d’arguments, jamais par commande shell libre.
- Showdown de développement accessible seulement localement ; vérifier réellement son adresse d’écoute. Ne pas supposer que `--no-security` implique une restriction réseau.
- Pickles générés localement depuis les logs validés ; checkpoints issus d’une source connue. Pas d’import arbitraire de pickles ou de poids par HTTP.
- Utiliser une copie contrôlée de VGC-Bench au commit compatible, accompagnée de patches versionnés et de leur empreinte. Éviter deux variantes indépendantes de la policy : appliquer la même révision corrigée aux deux environnements et tester le contrat.
- Commencer par la CLI, qui sera ensuite appelée par le worker HTTP. Une seule implémentation d’apprentissage.
- Proposition de simplification : l’import HTTP accepte un fichier JSON préalablement déposé dans `storage/inbox/`, via un chemin relatif borné. L’upload est optionnel dans la spécification et reste hors de cette première implémentation.
- Conserver les évaluations de combats à `0` pour le smoke test. Toute valeur non nulle doit être effectivement implémentée ou refusée explicitement, jamais ignorée.
- La validation BC doit laisser la policy d’entraînement, ses gradients et son optimiseur strictement inchangés. Appliquer la procédure de copie isolée de l’étape 4 : `eval()` et `no_grad()` seuls ne suffisent pas avec la policy amont.
- Pour les combats d’acceptation, notre joueur utilise le Tiny exporté et l’adversaire utilise le checkpoint historique validé à l’étape 1. Les deux modèles ont des chemins et des empreintes distincts ; le même Tiny chargé pour les deux joueurs ne valide pas ce scénario.

### Cinq cas à surveiller particulièrement

| Cas | Comportement attendu | Étape de test |
| --- | --- | --- |
| Même replay présent plusieurs fois ou sous deux perspectives | Déduplication et split par identité de partie, aucune fuite train/validation. | 3 |
| Action du second Pokémon interdite après celle du premier | Contrôle de la paire sous le masque conditionnel réellement utilisé par la loss ; erreur exploitable. | 2–4 |
| Dataset plus petit qu’un batch ou dernier batch incomplet | Aucune disparition silencieuse des données, nombre de transitions exact, aucun batch vide. | 4 |
| Deux lancements simultanés, crash API ou PID réutilisé | Verrou interprocessus, identité du worker vérifiable, état récupérable ; pas de second entraînement concurrent. | 6 |
| ZIP chargeable mais mappings différents, ou sélection de preview imposée | Refus de contrat incompatible ; preuve que le Tiny choisit réellement son quartet et joue ensuite. | 8–9 |

## 4. Organisation des fichiers et des revues

Arborescence relative à la racine Git `pokecoach2-trainer/`, avec Devbox à la racine et un sous-projet uv. Les fichiers sont **à créer progressivement**, pas à générer tous dès le départ. Le sous-dossier `pokecoach-trainer/` est un projet Python, pas un deuxième dépôt Git.

```text
devbox.json                      # uv et scripts setup/start/test/check/cli/build
devbox.lock                      # Résolution reproductible de l’outillage Devbox
.gitignore
.env.example                     # Configuration locale documentée, sans secrets
README.md                        # Entrée : devbox run setup, start, test, etc.
patches/                         # Modifications amont reproductibles
docs/{comprendre-le-bc.md,audits/,reviews/,reports/}
pokecoach-trainer/
  pyproject.toml                 # Dépendances, sources uv, Hatchling, entrée CLI
  uv.lock                        # Verrou des dépendances Python et révisions Git
  .python-version                # Python 3.13, version précise après audit
  README.md                      # Présentation du package ; recette à la racine
  .venv/                         # Créé par uv, ignoré par Git
  vendor/vgc-bench/              # Source figée, licences et sous-module Showdown
  configs/tiny.yaml              # Expérience de démonstration
  src/pokecoach_trainer/
    __init__.py
    schemas.py                   # Configuration, états et manifestes typés
    settings.py                  # Configuration et chemins explicites du Trainer
    storage.py                   # Chemins bornés, IDs, JSON atomique
    upstream.py                  # Imports/révision VGC-Bench et environnement de policy
    compatibility.py             # Description canonique et empreintes du contrat
    dataset_service.py           # Import, sélection, conversion, split, manifeste
    trajectory_dataset.py        # Transitions et contrôles du masque BC
    training.py                  # BC et validation, sans dépendre de HTTP
    artifact_service.py          # ZIP SB3, manifeste, package et inventaire
    reload_check.py              # Vérification CPU dans un nouveau processus
    cli.py                       # Diagnostic, données, entraînement, export
    jobs.py                      # États persistants, verrou et récupération
    training_service.py          # Préflight et lancement du worker
    worker.py                    # Entrée du processus enfant
    main.py                      # FastAPI, routeurs et cycle de vie
    api/{__init__,health,datasets,training,models}.py
  tests/                         # Tests ajoutés avec chaque étape
  storage/                       # Ignoré : inbox, raw_logs, datasets, jobs, models
```

**Convention pour la suite du plan :** `src/`, `tests/`, `vendor/`, `configs/`, `storage/`, `pyproject.toml`, `uv.lock` et `.python-version` désignent des chemins sous `pokecoach-trainer/`. `docs/`, `patches/`, `devbox.json`, `devbox.lock`, `.gitignore`, `.env.example`, `step.md` et le README principal restent à la racine Git. Les chemins PokeCoach restent relatifs à son propre dépôt, comme indiqué précédemment.

### Initialisation et commandes de référence

À réaliser à l’étape 1, après validation de l’audit. Depuis la racine Git existante :

```bash
devbox init
devbox add uv
devbox shell
uv init --app --package --python 3.13 --vcs none pokecoach-trainer
cd pokecoach-trainer
```

Configurer ensuite Hatchling comme backend, le package `src/pokecoach_trainer` et l’entrée `[project.scripts]` `pokecoach-trainer = "pokecoach_trainer.cli:main"`. Ajouter avec uv les dépendances utiles, dont `fastapi`, `uvicorn[standard]`, `pydantic-settings` et le groupe `dev` (`pytest`, `httpx`, `ruff`). Déclarer les sources VGC-Bench/forks retenues à l’audit, générer `uv.lock` et vérifier `uv sync --locked`. Le choix des versions doit respecter les dépendances de VGC-Bench retenues à l’audit.

Les scripts Devbox ci-dessous se lancent depuis la racine Git. Chacun entre explicitement dans `pokecoach-trainer/` avant sa commande uv ; transmettre les arguments supplémentaires de la CLI et de pytest sans les interpréter comme du shell.

| Commande publique | Commande dans le sous-projet uv | Disponible à partir de |
| --- | --- | --- |
| `devbox run setup` | `uv sync --locked` après mise en place de la source VGC-Bench figée | Étape 1 |
| `devbox run start` | `uv run --locked uvicorn pokecoach_trainer.main:app --host 127.0.0.1 --port 8010 --workers 1` | Étape 1 : API minimale ; étape 7 : API métier |
| `devbox run test -- <fichiers/options>` | `uv run --locked pytest <fichiers/options>` | Étape 1 |
| `devbox run check` | `uv run --locked ruff check src tests` | Étape 1 |
| `devbox run cli -- <arguments>` | `uv run --locked pokecoach-trainer <arguments>` | Étape 1, commandes complétées progressivement |
| `devbox run build` | `uv build` avec le backend Hatchling déclaré | Étape 1 |
| `devbox run reload-check -- <arguments>` | `uv run --locked python -m pokecoach_trainer.reload_check <arguments>` | Étape 5 |

Le démarrage de l’API n’active pas le rechargement automatique pendant les entraînements. Définir les chemins de configuration, stockage et vendor dans `settings.py` ; charger le `.env` racine par un chemin explicite, pas selon le répertoire courant. Dans l’IDE, sélectionner l’interpréteur `pokecoach-trainer/.venv/bin/python`.

L’API et la CLI sont lancées avec uv. Les sous-processus de calcul et de reload utilisent ensuite **`sys.executable` de cet environnement**, avec `-m pokecoach_trainer.worker` ou `-m pokecoach_trainer.reload_check` : cela garantit le même Python sans relancer de résolution/installation pendant un job. Tester les imports et les chemins depuis un autre répertoire courant.

Les identifiants restent opaques et validés. Les chemins d’artefacts proviennent des services, pas des corps de requêtes. Les interfaces ci-dessous constituent les choix proposés du wrapper ; les signatures internes des forks seront vérifiées à l’audit avant adaptation.

| Étape | Résultat que tu pourras examiner | Pause |
| --- | --- | --- |
| 0 | Audit, différences, choix du format et des versions | Validation du socle |
| 1 | Socle Devbox/uv, package Python, FastAPI minimal et modèle historique chargé | Validation de l’installation |
| 2 | Tiny configurable, plus petit et rechargeable | Validation du réseau |
| 3 | Dataset réel inspectable et split propre | Validation des données |
| 4 | Premier entraînement CLI, métriques et poids modifiés | Validation de l’apprentissage |
| 5 | Package exportable et rechargement isolé | Validation de l’artefact |
| 6 | Worker, logs, états et gestion des interruptions | Validation des processus |
| 7 | API locale complète, entraînement lancé par HTTP | Validation de l’API |
| 8 | Chargement réel dans le bridge PokeCoach | Validation de l’intégration |
| 9 | Combats complets, redémarrage, traces et rapport | Validation du palier 1 |

## 5. Étapes d’implémentation

Pour les étapes de code : écrire d’abord les tests de comportement indiqués, constater leur échec pertinent, implémenter le minimum commenté, puis les rejouer. Les fixtures rapides facilitent le développement ; elles ne remplacent ni les replays réels, ni l’apprentissage réel, ni les combats finaux.

Les commandes `devbox run cli -- ...` ci-dessous sont des **interfaces à implémenter**, pas des commandes déjà disponibles. Les commandes `devbox run` s’exécutent depuis la racine Git du Trainer ; leurs scripts utilisent uv dans le sous-projet `pokecoach-trainer/`. Les commandes uv directes se lancent dans ce sous-projet, depuis `devbox shell`. Les identifiants d’exemple deviennent les IDs réellement produits lors de l’exécution.

### Étape 0 — Auditer et figer le contrat de compatibilité

**À comprendre :** un checkpoint ne contient pas seulement des poids utilisables partout ; les mêmes nombres doivent représenter les mêmes observations et actions dans les deux applications.

**Fichiers :** créer `docs/audits/compatibilite-initiale.md`, commencer `docs/comprendre-le-bc.md`. Lire les fichiers listés en section 2. Ne pas modifier le moteur à cette étape.

- [ ] Relever les commits, les modifications locales, les interpréteurs, les locks, les dépendances effectivement utilisées par le bridge et le chemin exact de chargement dans `PolicyPlayer`.
- [ ] Identifier les versions Devbox/uv/Python à figer et les outils système utiles à Showdown, notamment Node.js/npm, à fournir via Devbox lorsqu’ils sont nécessaires.
- [ ] Comparer les interfaces d’observation, les tables d’identifiants, l’espace d’actions, les masques, les contraintes entre actions et le traitement de Team Preview.
- [ ] Identifier le fichier de logs candidat et sa régulation ; inspecter son schéma et confirmer sa disponibilité à l’exécution. Fixer ensuite un fichier et une révision de dataset, sans suivre implicitement `main`.
- [ ] Vérifier les équipes compatibles et le serveur Showdown réellement utilisé par PokeCoach ; distinguer le Showdown à la racine de l’espace de travail de celui de VGC-Bench.
- [ ] Documenter le format commun retenu, les versions à résoudre, les patches nécessaires et les différences du manifeste moteur actuel. Si aucun format commun n’est établi, marquer l’étape bloquée, sans inventer une compatibilité.
- [ ] Définir une description canonique du contrat : ordre des features et tables, dimensions, indices d’actions, règles de preview et de masque ; identifier les fichiers/fonctions dont l’empreinte complétera cette description.

**Vérification :** commandes Git `rev-parse HEAD`, `status --short`, `submodule status`, lecture des locks et comparaison des sources ; conserver sorties utiles et chemins. Une inspection de sources ne vaut pas test de chargement.

**Livrable de revue :** tableau « identique / différent / conséquence / correction minimale », format choisi et liste des révisions. **PAUSE 0 : tu valides les fondations avant toute installation ou modification du modèle.**

### Étape 1 — Initialiser avec Devbox et uv, démarrer FastAPI et charger la référence

**À comprendre :** Devbox fournit l’outillage de développement ; uv gère Python, les dépendances et l’environnement du Trainer. Le package contient la CLI, l’API FastAPI et les services partagés. Séparer cet environnement de PokeCoach évite qu’une dépendance d’entraînement casse l’application.

**Fichiers à la racine Git :** créer `devbox.json`, `devbox.lock`, `.gitignore`, `.env.example`, `README.md` et `docs/audits/versions.json`. **Dans le sous-projet uv :** créer `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `src/pokecoach_trainer/{__init__,settings,upstream,cli,main}.py`, `src/pokecoach_trainer/api/{__init__,health}.py`, `tests/test_environment.py`, `tests/test_health.py`, la copie `vendor/vgc-bench/` et la documentation de ses patches à la racine.

**Interfaces :** `collect_runtime_versions() -> dict[str, str]`, `pokecoach_trainer.cli:main` pour la commande empaquetée, `pokecoach_trainer.main:app` pour FastAPI ; commande `devbox run cli -- doctor --reference-checkpoint <archive-fiable.zip>`.

- [ ] Initialiser Devbox et le sous-projet avec `uv init --app --package` selon la recette de la section 4, sans créer un second dépôt Git ; configurer Hatchling et l’entrée CLI réelle, sans laisser le « Hello world » du générateur.
- [ ] Configurer Python 3.13 via uv et `.python-version` ; déclarer les dépendances, la source locale VGC-Bench et les commits de ses forks dans `pyproject.toml`, puis générer `uv.lock`. Enregistrer les versions Devbox/uv et les révisions exactes de `poke-env` et `imitation`.
- [ ] Créer les scripts Devbox `setup`, `start`, `test`, `check`, `cli` et `build` décrits en section 4. `setup` doit préparer la source figée nécessaire avant `uv sync --locked` ; documenter la mise à jour volontaire du lock avec uv, distincte de l’installation reproductible.
- [ ] Ajouter une API FastAPI minimale : `/health`, `/docs`, routeur `health`, `lifespan` initial et configuration typée. `/health` confirme le fonctionnement de l’API ; il ne prétend pas qu’un modèle ou Showdown est chargé. Aucun job ni route d’entraînement à ce stade.
- [ ] Documenter l’interpréteur IDE et les chemins ; vérifier `devbox run setup`, la commande CLI et la construction du package avec `devbox run build`. Vérifier que le package contient `pokecoach_trainer` ; les sources vendor et artefacts restent accompagnés d’une recette explicite, sans embarquer les poids dans la wheel.
- [ ] Préserver licences et provenance. Si `open-spiel` bloque, démontrer son absence d’usage dans ce parcours puis documenter précisément son exclusion ; ne pas mettre à niveau les dépendances au hasard.
- [ ] Garantir les imports `torch`, `imitation`, `stable_baselines3`, `poke_env`, `vgc_bench.src.policy` et `policy_player`, y compris dans un nouveau processus lancé depuis un autre répertoire courant.
- [ ] Télécharger le checkpoint de référence depuis sa source connue, enregistrer provenance et SHA-256, et le charger effectivement sur CPU.
- [ ] Faire afficher au diagnostic la classe de policy, l’architecture, les dimensions, le device et les révisions. Démarrer Showdown localement si requis par le test, avec accès réseau restreint vérifié.
- [ ] Exécuter les tests amont pertinents pour la policy et l’encodage ; consigner les incompatibilités avant tout patch fonctionnel.

**Vérification :** `devbox run setup`, `devbox run check`, `devbox run test -- tests/test_environment.py tests/test_health.py -q`, `devbox run build`, puis `devbox run start` et requêtes à `/health` et `/docs`. Effectuer également le diagnostic CPU et les tests amont sélectionnés. Résultat attendu : environnement reproduit depuis les locks, package construit, FastAPI accessible localement, imports et chargement réels réussis, révisions exactes enregistrées. Consigner que les locks restent inchangés après les commandes de reproduction.

**PAUSE 1 :** revue de `devbox.json`, `pyproject.toml`, de l’arborescence `src/`, du démarrage FastAPI et du diagnostic. Aucun entraînement à ce stade.

### Étape 2 — Rendre la policy configurable et expliquer le Tiny

**À comprendre :** largeur, couches et têtes changent la capacité du réseau ; elles ne changent pas le sens des entrées et des sorties.

**Fichiers :** modifier `vendor/vgc-bench/vgc_bench/src/policy.py`, enregistrer le patch dans `patches/`, créer `src/pokecoach_trainer/schemas.py`, `configs/tiny.yaml`, `tests/test_policy_config.py` et `tests/test_policy_masks.py`.

**Interfaces :** `ModelConfig` porte les cinq paramètres de policy ; `AttentionExtractor` et `MaskedActorCriticPolicy` acceptent `num_layers=3`, `num_heads=4`, `embed_len=32`, en conservant les arguments historiques.

- [ ] Tester les configurations `256/3/4` et `128/2/4`, les valeurs invalides et des configurations où couches **et têtes** diffèrent des valeurs historiques, par exemple `128/1/2`.
- [ ] Transmettre les paramètres aux extracteurs actor et critic ; mapper `num_layers` vers l’attribut interne existant sans réécrire le Transformer.
- [ ] Faire enregistrer les paramètres nécessaires dans la sauvegarde SB3 ; vérifier ce mécanisme dans la version installée, y compris le chemin de reconstruction de la policy.
- [ ] Comparer le nombre de paramètres, les espaces d’observations/actions et les dimensions des sorties. Tiny doit avoir strictement moins de paramètres.
- [ ] Tester des actions interdites, une paire dont la seconde action dépend de la première et la sélection de preview ; préserver les contraintes amont et détecter un masque sans action disponible.
- [ ] Sauvegarder/recharger les modèles neufs, comparer architecture et sorties en mode déterministe sur les mêmes observations ; recharger aussi le checkpoint historique de l’étape 1.

**Vérification :** `devbox run test -- tests/test_policy_config.py tests/test_policy_masks.py -q`. Le test `128/1/2` empêche de croire que les têtes sont bien sauvegardées simplement parce que Tiny utilise la valeur historique 4.

**Livrable pédagogique :** un exemple commenté « observation → embeddings → Transformer → probabilités masquées → deux actions », avec dimensions observées. **PAUSE 2 : tu valides le patch et les tests avant de traiter les données.**

### Étape 3 — Préparer et inspecter un mini-dataset réel

**À comprendre :** un replay devient une ou deux trajectoires, puis des décisions à imiter. Les deux points de vue d’une même partie ne doivent jamais se retrouver dans des splits différents.

**Fichiers :** créer `src/pokecoach_trainer/storage.py`, `src/pokecoach_trainer/dataset_service.py`, `src/pokecoach_trainer/trajectory_dataset.py`, `tests/test_dataset.py`, `tests/test_storage.py` ; adapter `vendor/vgc-bench/vgc_bench/logs2trajs.py` aux chemins explicites, sans dupliquer son encodeur.

**Interfaces :** `import_logs(relative_path: str) -> DatasetManifest` ; `prepare_dataset(dataset_id: str, config: PrepareConfig) -> DatasetManifest`. Définir ces types dans `src/pokecoach_trainer/schemas.py` : origine, révision/hash, format, seed, IDs, état, splits, compteurs et erreurs. `PrepareConfig` utilise `seed=42`, `validation_ratio=0.10` et une limite de replays.

- [ ] Télécharger un fichier réel identifié à l’étape 0 dans `storage/inbox/` ; limiter sa taille et valider le JSON avant conversion. Fixer la limite à 100 Mio par défaut, configurable localement.
- [ ] Sélectionner avant conversion, avec seed fixe, des parties du seul format retenu. Dédupliquer par ID de replay et détecter les duplications de contenu ; refuser un ID associé à des contenus contradictoires.
- [ ] Convertir d’abord quelques parties pour inspection, puis 200 à 1 000 selon disponibilité. Conserver les IDs et perspectives, le compte rendu de chaque échec et les hashes des entrées.
- [ ] Contrôler les observations finies, dimensions, indices d’actions, nombre réel de décisions et convention observation terminale / nombre d’actions attendue par `imitation`.
- [ ] Documenter le masque synthétique amont, notamment les plages `47:87` par Pokémon dans la copie inspectée ; les confirmer contre les mappings du commit retenu. Vérifier les cibles et les contraintes conditionnelles des deux actions avec la même logique que la loss.
- [ ] En cas de cible interdite par ce masque, bloquer la préparation et produire replay/transition/action concernés. Ne pas supprimer silencieusement ces exemples, ni forcer le bit cible à « autorisé ».
- [ ] Répartir les parties converties entre train et validation de manière déterministe, garder toutes leurs perspectives ensemble, et refuser un dataset final dont l’un des splits est vide.
- [ ] Écrire un manifeste immuable après succès : source/révision/hash, configuration de conversion, patches, replays retenus/rejetés, trajectoires et transitions par split. Une nouvelle préparation ne doit pas écraser un dataset prêt.
- [ ] Ajouter une commande d’inspection affichant une décision réelle et les limites de reconstruction depuis les logs, notamment les informations de preview inconnues ; ne pas inventer de labels.

**Vérification :** `devbox run test -- tests/test_dataset.py tests/test_storage.py -q` ; cas doublons, perspectives, longueurs variables, mauvais format, NaN, conversion partielle, collision d’ID, `../`, chemins absolus et symlink sortant de `storage/`. Les IDs train/validation doivent être disjoints et reproductibles.

**Démonstration :** `devbox run cli -- dataset prepare --dataset-id <id> --limit 300 --seed 42 --validation-ratio 0.10`, puis `devbox run cli -- dataset inspect --dataset-id <id>`. Si le volume visé est impossible, consigner l’écart avant de poursuivre.

**PAUSE 3 :** revue du convertisseur adapté, du manifeste et d’un exemple humain réellement transformé.

### Étape 4 — Entraîner le Tiny par CLI et montrer les poids qui changent

**À comprendre :** la policy prédit des actions ; la loss mesure son écart aux décisions humaines ; BC calcule les gradients et modifie les poids. La validation mesure sur des parties jamais utilisées pour ces mises à jour.

**Fichiers :** créer `src/pokecoach_trainer/training.py`, `tests/test_training.py`, compléter `trajectory_dataset.py`, `schemas.py`, `cli.py` et `configs/tiny.yaml`.

**Interfaces :** `train_bc(config: ExperimentConfig, run_dir: Path) -> TrainingResult` ; `ExperimentConfig` compose dataset ID, nom, seed, `ModelConfig` et `TrainingConfig`. `TrainingResult` fournit checkpoints, métriques, device effectif, nombre de paramètres et preuve de modification des poids.

- [ ] Créer les tests d’un minuscule dataset à trajectoires de longueurs différentes, dont un dataset plus petit que `batch_size=32` et un dernier lot incomplet.
- [ ] Charger des transitions via les outils `imitation` appropriés ; adapter la taille effective des lots au fork installé sans perdre d’exemples. Supprimer la logique `div_frac=0.01` pour ce parcours et compter les transitions réellement consommées.
- [ ] Initialiser PPO avec la policy configurable et les mêmes espaces VGC-Bench ; confier uniquement `ppo.policy` à `imitation.BC`. Ne pas faire générer de nouvelles parties pour entraîner.
- [ ] Fixer toutes les graines utiles. Pour `device=auto`, vérifier le fonctionnement du device et basculer vers CPU si nécessaire, en journalisant la raison ; un device explicitement demandé mais indisponible produit une erreur claire.
- [ ] Exécuter un court smoke test CLI, puis trois époques sur le dataset réel. Chaque époque parcourt le train, sans validation dans les mises à jour ; désactiver les évaluations de combat répétées.
- [ ] Exposer `train_loss` et `validation_loss` comparables : même objectif BC et mêmes coefficients explicités, moyenne pondérée par transitions ; détailler aussi la composante d’imitation si la loss inclut entropie ou régularisation. Calculer la validation sur une copie isolée selon la procédure ci-dessous, sans changer le mode ni les poids de la policy d’entraînement.
- [ ] Instrumenter la preuve d’apprentissage : pas d’optimiseur non nuls, gradients finis et variation mesurée de paramètres de l’acteur. Vérifier que les poids ne changent pas pendant le passage de validation.
- [ ] Écrire logs horodatés et `metrics.jsonl` : époque, pertes, transitions, durée, paramètres, device, mémoire CPU et GPU si disponible. Refuser toute loss non finie ou métrique de validation absente.
- [ ] Sauvegarder un `.zip` SB3 par époque et un final via `ppo.save(...)`, avec écriture temporaire puis publication atomique. Documenter qu’un checkpoint de policy n’est pas encore une reprise exacte de l’état de l’optimiseur BC.

**Précision obligatoire — validation réellement sans effet sur l’entraînement.** Dans `vgc_bench/src/policy.py`, `get_logits()` utilise `torch.enable_grad()` quand `actor_grad=True`, et les embeddings de capacités, objets et attaques utilisent `max_norm`. Le premier peut réactiver la construction d’un graphe sous `no_grad()` ; les seconds peuvent renormaliser leurs poids pendant un simple passage avant, même sous `eval()` et `no_grad()`. Ne pas conclure que l’absence d’appel à l’optimiseur suffit à préserver le modèle.

Procédure retenue pour ce palier, à isoler dans `evaluate_validation(...)` dans `training.py` :

1. Terminer tous les pas d’optimiseur de l’époque, y compris celui du dernier batch incomplet, avant de lancer la validation. Si `BC.train(n_epochs=1)` est utilisé, attendre son retour ; ne pas supposer qu’un callback de fin d’époque intervient après le dernier pas dans le fork installé.
2. Créer une copie indépendante de la policy avec `copy.deepcopy(policy)`. Aucun paramètre ni buffer de la copie ne doit partager son stockage avec l’original. Ne pas attacher cette copie à l’optimiseur BC.
3. Passer **la copie** en mode `eval()`, fixer son attribut `actor_grad=False` et désactiver `requires_grad` sur ses paramètres. Exécuter son évaluation dans `torch.no_grad()`.
4. Utiliser le même calculateur de loss BC et les mêmes coefficients qu’à l’entraînement, en lui passant explicitement la copie et les actions humaines. Le chemin attendu est `evaluate_actions()` ; ne pas appeler `forward()` pour calculer la loss, car ce chemin sélectionne des actions et force actuellement `actor_grad=True`.
5. Conserver `max_norm` sur la copie : sa renormalisation interne reste conforme à VGC-Bench, mais ne peut pas modifier la policy entraînée. Cette transformation de la copie n’est pas une mise à jour par apprentissage. Ne pas désactiver globalement `max_norm` ni modifier les règles d’inférence pour contourner le problème.
6. Agréger les métriques par nombre de transitions puis libérer la copie, y compris en cas d’exception. Ne jamais sauvegarder la copie de validation, recopier son état vers la policy entraînée, appeler `backward()` ou effectuer un pas d’optimiseur pendant ce passage. Les checkpoints restent produits à partir de la policy originale.

Tests obligatoires dans `tests/test_training.py`, à faire échouer avant l’implémentation :

- [ ] `test_validation_preserves_training_state_with_max_norm` : placer volontairement au-dessus de `max_norm` une ligne d’embedding réellement utilisée par une observation de validation. Cloner les tenseurs du `state_dict` original avant l’appel (un simple dictionnaire de références ne suffit pas), puis vérifier leur égalité exacte après validation, ainsi que les gradients existants, l’état de l’optimiseur, les modes des modules et `actor_grad`. Ce cas doit détecter une évaluation accidentelle sur l’original.
- [ ] `test_validation_has_no_autograd_or_optimizer_step` : vérifier que les logits et la loss de validation ont `requires_grad=False`, qu’aucun appel à `backward()` ou `optimizer.step()` n’a lieu et que la loss est finie. Vérifier également que les stockages de la copie sont distincts de ceux de l’original.
- [ ] `test_validation_failure_preserves_training_state` : provoquer une erreur sur un batch de validation, vérifier l’état original inchangé et l’erreur propagée ; un entraînement suivant doit encore produire des gradients et modifier ses poids normalement.
- [ ] `test_validation_runs_after_final_partial_batch_update` : avec un dernier batch incomplet, vérifier que les poids copiés pour la validation incluent sa mise à jour et que le checkpoint final provient de la policy originale.

**Vérification :** `devbox run test -- tests/test_training.py -q`, puis `devbox run cli -- train --config configs/tiny.yaml`. Résultat attendu : trois époques réelles, métriques finies, poids de l’acteur modifiés, aucune transition perdue et fichiers de sauvegarde présents. La baisse monotone de loss n’est pas un critère de réussite.

**PAUSE 4 :** revue commentée de la boucle, de la validation et des métriques. Tu dois pouvoir identifier qui calcule la loss, qui modifie les poids et qui sauvegarde.

### Étape 5 — Exporter un package vérifiable et le recharger sur CPU

**À comprendre :** le fichier SB3 porte le modèle ; son manifeste permet de savoir avec quel code, quelles données et quel encodage l’utiliser.

**Fichiers :** créer `src/pokecoach_trainer/compatibility.py`, `src/pokecoach_trainer/artifact_service.py`, `src/pokecoach_trainer/reload_check.py`, `tests/test_model_reload.py`, `tests/test_artifacts.py` ; compléter les schémas et la CLI.

**Interfaces :** `build_contract() -> ModelContract` ; `export_model(result: TrainingResult, config: ExperimentConfig) -> ModelManifest` ; `verify_reload(model_path: Path, manifest_path: Path) -> ReloadReport`. Exposer le contrôle manuel par `devbox run reload-check -- ...`. Dans le parcours d’export, lancer ce même module dans un **nouveau processus** avec `[sys.executable, "-m", "pokecoach_trainer.reload_check", ...]`.

- [ ] Calculer les empreintes du contrat défini à l’étape 0 à partir de JSON canonique, tables ordonnées et sources pertinentes. Inclure la sémantique des masques ; un simple hash des dimensions ou du nom de classe est insuffisant.
- [ ] Produire tous les champs du manifeste demandé par la spécification : architecture, dataset, format, schémas, commits, versions et SHA-256. Ajouter révision du Trainer, empreinte des patches et configuration complète de l’expérience ; aucune valeur `TO_FILL` dans un artefact publiable.
- [ ] Recharger le Tiny sur CPU dans un nouveau processus avec les imports réels ; vérifier l’architecture, les espaces, les poids, une prédiction masquée et l’égalité des sorties déterministes à tolérance numérique documentée.
- [ ] Tester un ZIP tronqué, un hash incorrect, des métadonnées incohérentes et un processus de reload en échec ou dépassant le timeout ; aucun ne doit publier de modèle validé.
- [ ] Construire un package contenant `model.zip`, `manifest.json`, `metrics.json` et `README.md`. Conserver l’archive SB3 intacte à l’intérieur.
- [ ] Publier uniquement après les vérifications ; distinguer `trainer_reload=passed` de `pokecoach_compatibility=not_tested`. Cette seconde preuve viendra aux étapes 8–9.

**Vérification :** `devbox run test -- tests/test_model_reload.py tests/test_artifacts.py -q`, puis export et reload CLI du modèle réellement entraîné. Extraire le package dans un autre répertoire et vérifier à nouveau les fichiers et empreintes.

**PAUSE 5 :** revue du manifeste, du code d’empreinte et du rapport de rechargement. Un package exporté n’est pas encore une preuve de compatibilité PokeCoach.

### Étape 6 — Lancer le même entraînement dans un worker supervisé

**À comprendre :** l’API gère une demande ; un autre processus réalise le calcul. Un fichier d’état permet de retrouver ce qui s’est passé après un redémarrage.

**Fichiers :** créer `src/pokecoach_trainer/jobs.py`, `src/pokecoach_trainer/training_service.py`, `src/pokecoach_trainer/worker.py`, `tests/test_jobs.py`, `tests/test_worker.py` ; réutiliser `training.py` et `artifact_service.py` sans copier leur logique.

**Interfaces :** `submit_training(config: ExperimentConfig) -> JobRecord`, `get_job(job_id: str) -> JobRecord`, `reconcile_jobs() -> None` ; lancement `[sys.executable, "-m", "pokecoach_trainer.worker", "--job-id", job_id]`. Le worker lit une configuration validée déjà enregistrée sous `storage/jobs/<id>/`.

- [ ] Définir les transitions `queued → running → completed|failed|interrupted`, le rôle du worker et les timestamps. `completed` exige export et reload validés. Les états d’échec conservent traceback, cause et derniers logs.
- [ ] Préparer le préflight : dataset prêt, dépendances, espace disque, device et serveur/ports si nécessaires. Refuser avant lancement une configuration inutilisable.
- [ ] Réserver atomiquement l’unique slot d’entraînement avec un verrou interprocessus ; une seconde demande concurrente est refusée, pas mise dans une file illimitée.
- [ ] Lancer via `sys.executable` de l’environnement uv et une liste d’arguments validés ; transmettre seulement l’ID de job, fixer le répertoire de travail et rediriger stdout/stderr vers ses logs. Vérifier que le worker utilise le même environnement que la CLI/API ; ne lancer aucune synchronisation de dépendances pendant le calcul.
- [ ] Utiliser un verrou conservé par le worker et une identité avec jeton de job et heure de démarrage, pas seulement un PID. Écrire états et progression atomiquement ; prévoir le réapage des processus enfants.
- [ ] Au redémarrage : conserver un worker identifié encore vivant ; marquer `interrupted` un job abandonné, y compris un `queued` dont le lancement n’a jamais abouti. Aucune reprise automatique des poids pour ce palier.
- [ ] Tester deux demandes simultanées, erreur de lancement, crash du worker, arrêt de l’API pendant le calcul, PID réutilisé et fichier de métriques avec dernière ligne incomplète. Les lectures restent possibles et le slot finit par être libéré.
- [ ] Documenter l’arrêt manuel sûr du worker et son état final. `cancelled` et une route d’annulation restent optionnels ; s’ils sont ajoutés, les tester sans tuer un processus étranger.

**Vérification :** `devbox run test -- tests/test_jobs.py tests/test_worker.py -q` avec de vrais petits sous-processus pour les tests de cycle de vie, puis un job utilisant le BC réel de l’étape 4.

**PAUSE 6 :** revue du diagramme d’états, du lancement et d’une démonstration de récupération après interruption.

### Étape 7 — Exposer FastAPI et refaire le parcours par HTTP

**À comprendre :** HTTP déclenche et consulte exactement le service validé en CLI ; il ne doit pas contenir une autre boucle d’apprentissage.

**Fichiers :** compléter `src/pokecoach_trainer/main.py` et `src/pokecoach_trainer/api/health.py` créés à l’étape 1 ; créer `src/pokecoach_trainer/api/{datasets,training,models}.py`, `tests/test_training_api.py`, `tests/test_dataset_api.py`, `tests/test_models_api.py` ; compléter le README et les exemples de requêtes.

- [ ] Étendre l’application FastAPI initiale avec les routeurs métier via `include_router`, les requêtes Pydantic et la récupération des jobs dans `lifespan`. Garder `/health` et `/docs`, puis lancer avec `devbox run start` (Uvicorn cible `pokecoach_trainer.main:app`, sur `127.0.0.1:8010`, avec un seul worker HTTP).
- [ ] Implémenter `POST /datasets/import`, `POST /datasets/{id}/prepare` et `GET /datasets/{id}`. L’import est borné à `storage/inbox/` ; validation de taille, IDs et chemins même si le service est local.
- [ ] Exécuter la conversion potentiellement longue dans un processus distinct utilisant le même mécanisme persistant, avec un état dataset consultable. Retourner `202` pour la préparation ; borner les ressources et éviter de lancer préparation et entraînement simultanément dans ce POC.
- [ ] Implémenter `POST /training/jobs` avec `202` et ID/statut immédiats ; `409` si le slot est occupé, `422` pour la configuration invalide et `404` pour un ID absent. Renvoyer l’état réellement persisté, sans fabriquer un `queued` si le job a déjà échoué.
- [ ] Implémenter `GET /training/jobs/{id}` et `/logs?tail=100`, avec borne de 1 à 1 000 lignes ; exposer progression, configuration, métriques, erreurs et device effectif.
- [ ] Implémenter `GET /models`, `/models/{id}` et `/models/{id}/download` ; ne lister que les artefacts publiés après vérification, télécharger le package complet.
- [ ] Tester absence de fichiers, chemins hostiles, symlinks, JSON trop volumineux/malformé, dataset non prêt, training concurrent, logs absents/partiels et checkpoint non validé. `/health` doit répondre pendant un job actif et après redémarrage de l’API.
- [ ] Exécuter le parcours réel par HTTP : import, préparation, lancement de **trois époques**, polling, logs, téléchargement et reload CPU du package téléchargé. Conserver requêtes et réponses sans données sensibles.

**Vérification :** `devbox run test -- tests/test_training_api.py tests/test_dataset_api.py tests/test_models_api.py -q`, puis la recette HTTP documentée. Les tests de routes peuvent utiliser un petit worker de test ; la démonstration finale de cette étape utilise réellement BC et les données humaines.

**PAUSE 7 :** revue des routes et démonstration depuis `/docs` ou `curl`. Le checkpoint téléchargé à cette étape est celui utilisé pour la preuve finale.

### Étape 8 — Intégrer le checkpoint dans le bridge PokeCoach

**À comprendre :** réussir `PPO.load` dans le Trainer ne garantit pas que PokeCoach construit les mêmes entrées ou interprète les mêmes actions.

**Fichiers PokeCoach à modifier seulement si nécessaire :** `engine-runtime/src/pokecoach_vgc_bridge/{vgc_real,battle_session,policy,checkpoint,vgc_source}.py`, le chemin de lancement/configuration et sa documentation. Créer `engine-runtime/tests/test_model_compatibility.py` et côté Trainer `tests/test_pokecoach_compatibility.py`.

**Précision obligatoire — deux modèles distincts pour les deux joueurs.** Le code inspecté de `battle_session.py` appelle actuellement `p1.set_policy(checkpoint_path, device)` puis `p2.set_policy(checkpoint_path, device)`. Changer uniquement `VGC_CHECKPOINT_PATH` ferait donc jouer Tiny contre Tiny. Pour le scénario d’acceptation, retenir un adversaire utilisant **le checkpoint historique déjà chargé et vérifié à l’étape 1** ; aucun joueur heuristique supplémentaire n’est nécessaire.

- [ ] Ajouter une configuration locale explicite pour l’adversaire : `VGC_OPPONENT_CHECKPOINT_PATH`, `VGC_OPPONENT_CHECKPOINT_SHA256` et `VGC_OPPONENT_CHECKPOINT_ID`. Ces noms sont à implémenter dans la configuration PokeCoach, sa transmission au bridge et `.env.example` ; ils ne désignent pas des variables déjà prises en charge.
- [ ] Propager un `opponent_checkpoint_path` distinct jusqu’à la création de session, pour les chemins preview, premier tour et combat complet. Charger notre checkpoint dans `p1` et celui de référence dans `p2`, chacun dans sa propre instance de policy, afin de permettre les architectures Tiny et historique différentes.
- [ ] Vérifier les deux fichiers, leurs hashes et leur compatibilité avec le même format/encodeur. Le modèle historique suit la procédure de compatibilité historique prévue ci-dessous ; il ne doit pas être dispensé de vérification de provenance.
- [ ] Conserver, si nécessaire, le comportement historique des simulations existantes lorsque la configuration adversaire est absente. En revanche, la recette et le test d’acceptation Tiny doivent exiger une configuration adversaire complète et deux hashes différents ; ils échouent avant le combat si elle manque ou si les deux fichiers désignent les mêmes poids. Aucun repli silencieux vers Tiny comme adversaire dans cette recette.
- [ ] Enregistrer séparément notre modèle et celui de l’adversaire : rôle, ID, SHA-256 complet, architecture et révision/contrat utilisé. Associer ces informations aux joueurs réellement construits et les conserver jusqu’aux traces persistées, sans se limiter aux valeurs demandées dans la configuration.
- [ ] Ajouter `test_distinct_checkpoints_are_loaded_for_each_player` dans `engine-runtime/tests/test_model_compatibility.py` : vérifier la transmission des deux chemins et les architectures chargées (`128/2/4` pour notre Tiny, architecture constatée à l’étape 1 pour la référence). Ajouter les cas adversaire absent dans la recette d’acceptation, hash incorrect, mêmes poids sous deux noms de fichiers et modèle incompatible ; chacun doit échouer explicitement.

- [ ] Installer/désigner dans le bridge la même révision corrigée de la policy, via son mécanisme existant de source. Conserver la classe/import attendus par l’archive ; ne pas charger un `state_dict` Tiny dans une instance historique précréée.
- [ ] Réutiliser `VGC_CHECKPOINT_PATH`, `VGC_CHECKPOINT_SHA256` et `ANALYSIS_CHECKPOINT_ID` ; vérifier leur transmission aux variables internes du bridge. Corriger les exemples d’extension `.pt` si nécessaire pour documenter le `.zip` SB3.
- [ ] Distinguer manifeste moteur et manifeste du modèle exporté, définir explicitement leur chargement et vérifier le hash/contrat avant d’accepter le modèle. Conserver le chargement de référence historique ; une exception historique documentée ne doit pas permettre à un Tiny sans manifeste de contourner les contrôles.
- [ ] Exiger un chargement réel de la policy avant de déclarer le modèle chargé ; rendre les erreurs de version/architecture/contrat visibles, sans message générique masquant leur cause.
- [ ] Comparer dans les deux environnements l’encodage, les masques légaux `DoublesEnv.get_action_mask`, les indices et ordres produits sur des états `DoubleBattle` représentatifs : preview, tour normal, switch et remplacement forcé. Ajouter un test où les dimensions sont identiques mais un mapping est permuté : il doit être refusé.
- [ ] Vérifier que les chemins PokeCoach de preview imposée ne remplacent pas le choix du Tiny dans la démonstration. Tester le choix réel d’un quartet par la policy, puis son utilisation aux tours suivants.
- [ ] Enregistrer ID, hash complet, architecture et version de contrat de chacun des deux modèles dans la provenance des simulations, en réutilisant le mécanisme existant.
- [ ] Rejouer les tests de référence historiques et les nouveaux tests de compatibilité. La dépendance à un chemin de développement doit être explicite ; la recette finale doit être reproductible à partir des révisions enregistrées.

**Vérification :** tests du bridge dans son environnement isolé, puis `tests/test_pokecoach_compatibility.py` lançant effectivement le bridge avec le checkpoint HTTP téléchargé. Un processus neuf doit charger Tiny et le checkpoint historique ; les manifestes incompatibles doivent être refusés clairement.

**PAUSE 8 :** revue des seuls changements nécessaires dans PokeCoach, du contrat et de la provenance. Pas de refonte de son moteur ou de son interface.

### Étape 9 — Prouver le combat complet et livrer la recette

**À comprendre :** une défaite est acceptable. Le succès est que le modèle entraîné prenne les décisions, que Showdown les accepte et que PokeCoach conserve une partie terminée.

**Fichiers :** compléter `tests/test_pokecoach_compatibility.py`, les tests réels PokeCoach existants, `README.md`, `docs/comprendre-le-bc.md`, `docs/reports/palier-1.md` et un rapport JSON de preuves.

- [ ] Sélectionner les deux équipes du format validé à l’étape 0. Configurer explicitement notre joueur avec le Tiny issu du job HTTP et l’adversaire avec le checkpoint historique de l’étape 1, via le chemin distinct ajouté à l’étape 8. Refuser la recette si la configuration adversaire est absente, incomplète ou désigne les mêmes poids que Tiny.
- [ ] Désactiver explicitement fixture/stub et vérifier, depuis les deux joueurs chargés puis dans la trace persistée, les IDs, hashes et architectures des deux modèles. Notre joueur doit être le Tiny exporté, l’adversaire la référence historique ; un simple affichage des valeurs de configuration n’est pas une preuve suffisante.
- [ ] Exécuter un combat depuis le choix réel de Team Preview jusqu’au résultat terminal. Capturer commandes, rejets Showdown, exceptions, tours, switches et preuve de fin ; appliquer un timeout borné qui marque un échec au lieu de laisser un worker bloqué.
- [ ] Vérifier zéro ordre rejeté et zéro repli silencieux vers une autre policy/une action de secours après erreur. Un moteur qui corrige une action illégale ne constitue pas la preuve demandée.
- [ ] S’assurer qu’un switch est effectivement observé dans la campagne de validation ; si le premier match n’en contient pas, ajouter un scénario compatible ciblé et conserver tous les essais, sans masquer les échecs.
- [ ] Passer par le parcours normal de sauvegarde PokeCoach et ouvrir les traces dans l’affichage existant. Un simple test du bridge en mémoire ne valide pas cette persistance.
- [ ] Arrêter et relancer les processus PokeCoach concernés, notamment bridge/worker ; recharger **le même fichier Tiny et le même checkpoint de référence adverse** sans réentraînement et terminer un autre combat. Consigner nouveaux PID, hashes inchangés pour chaque rôle et nouveaux IDs de parties. Les décisions n’ont pas à être identiques si l’inférence est stochastique.
- [ ] Produire le rapport : versions/commits/patches, origine et taux de conversion, dimensions et paramètres, device, losses, preuve de mises à jour et de validation sans mutation du modèle entraîné, job HTTP, chemins/hashes des deux modèles associés à leur rôle, reload CPU, tests de contrat, IDs et traces des deux exécutions.
- [ ] Donner un résultat global `PASSED` uniquement si tous les critères obligatoires sont démontrés ; sinon `FAILED` ou `BLOCKED` avec le critère non satisfait, les traces et l’action restante.
- [ ] Finaliser la recette : installation, données, entraînement CLI/HTTP, suivi, téléchargement, configuration PokeCoach, commandes de reproduction, arrêt des services et limites connues. Mettre `step.md` en cohérence avec ce premier palier, sans supprimer sa vision des paliers ultérieurs.

**Vérification :** tests réels avec `POKECOACH_VGC_REAL=1`, sans `POKECOACH_BRIDGE_FIXTURE`, et recette PokeCoach depuis le matchup jusqu’aux traces sauvegardées. Ne pas compter les tests réels ignorés faute de dépendances comme des validations.

**PAUSE 9 :** revue du rapport, des deux exécutions et du README. **Tu valides alors le palier complet.**

## 6. Checklist de livraison

- [ ] Initialisation avec Devbox et uv : Devbox à la racine, sous-projet uv empaquetable, layout `src/`, Hatchling, environnement et dépendances verrouillés.
- [ ] `devbox run setup`, `start`, `test`, `check`, `cli` et `build` fonctionnent ; FastAPI/Uvicorn et les workers utilisent l’environnement uv prévu.
- [ ] Socle et versions approuvés aux pauses 0–1 ; source VGC-Bench/Showdown et patches reproductibles.
- [ ] Policy configurable, Tiny plus petit, ancien checkpoint chargeable et architecture conservée après reload.
- [ ] Dataset réel, split par replay, statistiques exactes, aucune cible interdite par le masque BC.
- [ ] Trois époques de BC effectif, losses finies, poids d’acteur modifiés par l’entraînement ; validation sur copie indépendante avec tests d’absence de gradients et de modification de l’état original, y compris avec `max_norm`.
- [ ] Export SB3 et package complet, manifeste renseigné et recharge CPU dans un nouveau processus.
- [ ] API locale disponible avec `/docs`, statuts, logs, métriques, modèles et téléchargement ; entraînement réellement lancé par HTTP.
- [ ] Un seul entraînement actif, interruptions récupérables et chemins bornés.
- [ ] PokeCoach charge le Tiny avec le contrat correct et enregistre sa provenance.
- [ ] L’adversaire charge le checkpoint historique distinct ; les chemins, hashes, architectures et rôles des deux modèles sont vérifiés et conservés dans les traces, puis retrouvés après redémarrage.
- [ ] Partie complète réelle avec choix de preview, tours, switches, résultat et traces consultables.
- [ ] Même checkpoint réutilisé après redémarrage, avec nouvelle preuve de partie complète.
- [ ] Code commenté, guide pédagogique, recette reproductible, rapport d’écarts et validations humaines enregistrées.

**Limites assumées :** faible volume de données, masque BC synthétique documenté, absence de promesse sur le niveau de jeu, pas de reprise exacte d’un entraînement interrompu, pas d’exposition publique. Aucun de ces points ne dispense de la preuve finale dans PokeCoach.

---

## Addendum (2026-11-02) — étapes 8–9 réorientées

Plan détaillé d’implémentation : [`2026-11-02-pokecoach-chargement-modele-unique.md`](2026-11-02-pokecoach-chargement-modele-unique.md).

**Changement de scénario d’acceptation :** un fichier profil `vgc_model.yaml` (`VGC_MODEL_PROFILE`) regroupe chemin checkpoint + architecture (ou `trainer_package_dir`). **Même modèle sur p1 et p2** — pas de duel obligatoire Tiny vs checkpoint historique à l’étape 9.

**Livré côté PokeCoach :** `VgcModelProfile`, `policy_factory`, `battle_session` + `runtime` via env, recette [`docs/recettes/pokecoach-charger-modele.md`](../recettes/pokecoach-charger-modele.md), rapport [`docs/reviews/palier-1-etapes-8-9.md`](../reviews/palier-1-etapes-8-9.md).

**Reste pour clôturer le palier :** combat complet réel (`POKECOACH_VGC_REAL=1`), traces persistées, redémarrage avec le même profil — voir checklist étape 9 ci-dessus (adaptée au profil unique).
