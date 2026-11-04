# PokeCoach Trainer — Plan d’implémentation collecte, datasets et fine-tuning M-B → M-C

> **Pour l’agent chargé de l’implémentation :** utiliser `superpowers:executing-plans` pour exécuter les tâches dans l’ordre. Cocher uniquement les actions réellement effectuées. Respecter les pauses A à E : ne pas commencer le lot suivant sans validation explicite de l’utilisateur. La validation de la spec et la rédaction de ce plan n’autorisent pas son exécution.

**Objectif :** collecter manuellement des replays M-C depuis FastAPI, publier un dataset `natural` d’environ 300 parties exploitables, poursuivre le BC du checkpoint M-B sans changer son architecture, puis prouver son utilisation dans PokeCoach après export et redémarrage.

**Architecture :** faire évoluer le Trainer existant : trois services de collecte, préparation et entraînement ; une API FastAPI ; un seul processus de travail long actif ; fichiers locaux et manifestes versionnés. Les interfaces CLI et HTTP partagent les mêmes services. Le bridge PokeCoach continue à utiliser son profil de modèle unique.

**Technologies :** Devbox, uv, Python de l’environnement verrouillé, FastAPI/Uvicorn, Pydantic, PyTorch, Stable-Baselines3, forks `imitation` et `poke-env`, VGC-Bench et Showdown figés, pytest et Ruff.

**Spec de référence :** [Collecte, datasets et fine-tuning BC M-B → M-C](../specs/2026-11-03-collecte-datasets-finetuning-mc.md). Lire ce document et la spec avant de modifier le code. La spec validée prime sur les anciens plans Tiny.

**Date :** 3 novembre 2026. **Statut :** plan à exécuter ultérieurement ; aucune tâche ci-dessous réalisée par sa rédaction. Les noms de nouvelles interfaces sont des décisions d’implémentation ; les formats Showdown exacts doivent être établis au lot A.

## 1. Règles communes

- « Petit entraînement » signifie peu de données et trois époques de démonstration. Charger les poids M-B ; ne pas construire un Tiny `128/2/4` à leur place.
- Conserver architecture, dimensions, tables et sens des actions du checkpoint parent. Un changement nécessaire de ce contrat bloque le fine-tuning et demande une décision de périmètre.
- BC est réalisé par `imitation.algorithms.bc.BC`. Ne jamais appeler `PPO.learn()` ; PPO sert de conteneur et de format de sauvegarde.
- Dataset par défaut : M-C, `natural`, 300 replays exploitables, seed 42, Elo individuel ≥ 1100, sans maximum, Elo inconnu exclu par perspective, gagnants et perdants, OTS et six membres identifiables des deux équipes, BO1 et BO3.
- Abandons acceptés après trois **tours terminés** ; inactivité exclue. Égalités et fins inconnues archivées mais exclues de cette version du dataset.
- Ne pas rejeter une partie valide à cause d’une série BO3 inconnue/incomplète. Conserver les liens explicites ; grouper les séries confirmées pour les splits ; déclarer la limite des autres.
- Split cible 80/10/10 par nombre de replays, avec groupes indivisibles et trois splits non vides. Aucune perspective, alias ou série confirmée commune entre splits.
- Fine-tuning par défaut : seed 42, batch 32, trois époques, learning rate `1e-4`, évaluations de combats par époque à 0, device `auto`. Valeurs configurables, effectives et tracées.
- Collecte manuelle, brute et incrémentale : aucun filtre métier destructif ; aucun lancement automatique du Builder ou du BC.
- FastAPI sur `127.0.0.1:8010`, un seul worker HTTP ; un seul job long, tous types confondus. Showdown de développement accessible localement seulement.
- Pas d’Angular nouveau, de scheduler, Redis, Celery, base de données serveur, RL, distillation, quotas d’équipes ou variante `diverse`.
- Préserver les données et modifications préexistantes. Ne pas lancer une synchronisation vendor destructive avant d’avoir sauvegardé ses adaptations sous forme reproductible.
- Pickles produits localement et checkpoints de confiance uniquement. Les routes prennent des IDs validés ; aucun chemin libre de poids, aucune URL de téléchargement libre.
- Docstrings et commentaires ajoutés en français : rôle, entrées/sorties, erreurs et effets sur disque/poids. Expliquer les choix difficiles, pas chaque ligne Python.
- Distinguer tests à fixtures, tests réels et tests ignorés. Un affichage `100%` suivi d’un processus bloqué n’est pas une commande terminée avec succès.

### Contrat de revue humaine

À chaque pause A–E : écrire `docs/reviews/mc-lot-<lettre>.md`, présenter les fichiers à lire dans l’ordre, un exemple concret, les commandes et codes de sortie, les preuves et limites, puis attendre la réponse de l’utilisateur. Une question ne vaut pas validation ; un « continue » autorise le lot suivant uniquement. Corriger les retours dans le lot courant et revérifier ce qui a changé.

Le compte rendu contient deux champs distincts : résultat technique (`PASSED`, `FAILED`, `BLOCKED`) et validation humaine (`en attente` ou réponse explicite). La réussite des tests ne vaut pas approbation du code. Un petit commit par tâche stable est recommandé, avec sélection explicite des fichiers ; ne pas committer les données, poids, secrets ni modifications d’autrui. La pause de lot reste obligatoire même si ses tâches sont committées.

### Cinq risques à suivre pendant la revue

| Condition | Résultat imposé | Tâches de preuve |
| --- | --- | --- |
| Réinstallation/synchronisation vendor | Les correctifs provenance et policy survivent ; version inconnue refusée. | 0, 2, 8 |
| Deux soumissions, crash entre réservation et démarrage, PID réutilisé | Un seul calcul, état récupérable, aucun worker vivant déclaré mort. | 3, 6, 10 |
| Alias de replay reliant deux séries, conversion terminée dans un ordre différent | Regroupement transitif et sélection/splits identiques, sans décalage d’identité. | 8, 9 |
| Dataset de 1, 31, 32 ou 33 transitions ; callback de fin d’époque | Toutes les transitions consommées ; dernier pas effectué avant validation/sauvegarde. | 12, 13 |
| Même dimension mais tables permutées, ou package incomplet avec valeurs Tiny par défaut | Import refusé explicitement, jamais réparation silencieuse des poids ou métadonnées. | 1, 11, 14, 15 |

## 2. État initial et conventions de lecture

Les chemins `src/`, `tests/`, `configs/`, `vendor/`, `storage/`, `pyproject.toml` et `uv.lock` sont relatifs à **`pokecoach2-trainer/pokecoach-trainer/`**. `docs/`, `scripts/`, `patches/`, `devbox.json` et le README principal sont à la racine Git **`pokecoach2-trainer/`**. Les chemins préfixés `PokeCoach:` sont relatifs à **`pokecoach2-python/`**, un autre dépôt.

Constats de préparation du plan, à confirmer au démarrage :

- `prepare_dataset()` appelle déjà la conversion sans dépendre de `force` ; ne pas refaire cette correction.
- `TaggedTrajectory` et `process_logs_with_provenance()` relient déjà replay/perspective aux résultats. `_write_prepared_trajectories()` s’appuie sur ces identités. Conserver ces acquis.
- Le script `scripts/sync_vgc_bench.sh` utilise `rsync --delete` depuis le vendor voisin ; il ne suffit pas pour garantir la conservation de tous les patches locaux.
- `schemas.py` fournit déjà `ModelConfig`, `PrepareConfig`, `ExperimentConfig`, `DatasetManifest`, `JobRecord`, `ModelManifest` et `TrainingResult` : les faire évoluer, pas créer un modèle concurrent dans les routes.
- `train_bc()` crée actuellement PPO depuis `config.model`, utilise M-B en dur et ne transmet pas le learning rate à BC. Son compteur de transitions est théorique.
- Le fork `imitation` installé utilise `drop_last=True` dans son chargeur par défaut ; son wrapper d’itérable exige des batches égaux. Mettre seulement `drop_last=False` dans un chargeur ensuite réemballé ne suffit pas.
- `BC.train(n_epochs=1)` effectue le pas résiduel après la boucle ; sa callback de fin d’époque peut être trop tôt. Valider et sauvegarder **après le retour de `train()`**.
- `duplicate_policy()` existe ; la déplacer sur le device avant évaluation, contrôler les mutations `max_norm` et mesurer un vrai changement par optimiseur.
- Les jobs utilisent un fichier de verrou, des `PIPE` non drainés et une réconciliation trop large. Ces défauts doivent être traités avant les nouveaux jobs.
- Le bridge possède `model_profile.py`, `policy_factory.py` et `VGC_MODEL_PROFILE`. Sa factory construit encore un environnement M-B ; son lecteur d’architecture utilise des défauts Tiny si des champs manquent.

Lors de la mise à jour précédente de la spec, cinq tests dataset ont affiché un succès, mais le processus a dû être interrompu à la fermeture. Établir une nouvelle baseline avec code de sortie ; ne pas inscrire « suite verte » à partir de ce seul affichage.

### Carte des fichiers

Créer les fichiers seulement à la tâche qui en a besoin. Ne pas réorganiser tout le projet.

| Fichiers | Responsabilité | Tâches |
| --- | --- | --- |
| `docs/audits/mc-compatibilite.md`, `docs/audits/mc-runtime.json` | Sources vérifiées, formats et contrat, disponibilité réelle | 0–1 |
| `patches/README.md`, `patches/*.patch`, `scripts/sync_vgc_bench.sh` | Reproduction des sources et adaptations | 2, 8, 12 |
| `src/pokecoach_trainer/schemas.py`, `settings.py`, `storage.py` | Types communs, chemins, fichiers atomiques | 2 |
| `src/pokecoach_trainer/formats.py`, `configs/formats.json` | Registre de formats issus de l’audit et contrôle de disponibilité | 2 |
| `src/pokecoach_trainer/jobs.py`, `worker.py`, nouveau `job_service.py` | Cycle de vie commun et lancement des trois types de jobs | 3 |
| nouveaux `archive_service.py`, `collection_client.py`, `collection_service.py` | Objets bruts, snapshots, HTTP Showdown et reprise | 4–6 |
| nouveaux `replay_metadata.py`, `dataset_filters.py`, `conversion.py`, `dataset_split.py` | Métadonnées, admissibilité, provenance de conversion, groupes/splits | 7–9 |
| `dataset_service.py`, `trajectory_dataset.py` | Orchestration Builder, publication et contrôles de transitions | 8–10 |
| nouveau `base_model_service.py`, `compatibility.py` | Contrat disponible avant le Builder, puis registre parent et architecture effective | 1–2, 11 |
| nouveau `bc_adapter.py`, `training.py`, `training_service.py` | BC existant adapté aux petits lots, évaluation et fine-tuning | 12–13 |
| `artifact_service.py`, `reload_check.py` | Export, contrôle isolé et publication | 14 |
| `main.py`, `cli.py`, `api/{collection,archive,datasets,training,models}.py` | Brancher les services, contrats HTTP et CLI | 6, 10, 13–14 |
| PokeCoach: `engine-runtime/src/pokecoach_vgc_bridge/{model_profile,policy_factory,showdown_teams,battle_session,battle_runner,vgc_real}.py` | Contrat, régulation et chargement réel | 15–16 |

Les nouveaux modules sont sous `src/pokecoach_trainer/`. Une signature donnée ci-dessous est le contrat cible ; adapter ses appelants et tests existants dans la même tâche. Conserver les anciennes interfaces seulement si leur comportement reste sûr et explicite.

### Commandes et méthode de travail

Depuis la racine Git Trainer, après vérification des scripts à la tâche 0 :

```bash
devbox run test -- tests/test_dataset.py -q
devbox run check
devbox run start
devbox run cli -- --help
```

Les sous-commandes CLI nouvelles de ce plan sont **à implémenter**, pas disponibles aujourd’hui. Les scripts doivent transmettre les arguments à uv sans les réinterpréter. Si le `cd` de l’`init_hook` fait doubler le sous-dossier, corriger les scripts avec une racine déterminée explicitement. Commande directe de diagnostic, depuis `devbox shell` et le sous-projet uv : `uv run --locked pytest tests/test_dataset.py -q`.

Pour chaque tâche de code : écrire le test nommé, l’exécuter et constater un échec pertinent, implémenter la modification, rejouer les tests concernés et Ruff. Une erreur de dépendance ne constitue pas un test rouge du comportement. Ne pas reconstruire la suite entière à chaque petite modification ; exécuter la régression du lot avant sa pause. Fixer un timeout aux sous-processus de test et vérifier leur code de sortie.

## 3. Lot A — Audit M-C et point de départ réel

### Tâche 0 — Établir la baseline sans écraser l’existant

**Fichiers :** lire les fichiers listés en section 2, les locks, le README, les tests dataset ; créer `docs/audits/mc-compatibilite.md`. Aucun reset, installation forcée ou lancement d’entraînement.

**Entrées / sortie :** état des deux dépôts et environnements → rapport avec commits, fichiers modifiés, commandes et problèmes préexistants.

- [ ] Relever `git status --short`, HEAD et état des sous-modules dans les deux dépôts et leurs vendors ; enregistrer les versions Python/uv/Devbox et les commits effectivement installés des forks.
- [ ] Lire les commandes Devbox avant de les exécuter. Vérifier les imports et la baseline dataset avec un timeout externe de 120 secondes ; vérifier séparément `/health` et les tests jobs actuels.
- [ ] Si le processus pytest reste vivant après ses résultats, identifier les ressources non fermées (threads/boucles/players/executors) et consigner la cause et la tâche de correction. Ne pas masquer le problème par `os._exit()`, un `kill` dans le test ou une réussite forcée.
- [ ] Comparer les vendors Trainer/PokeCoach et les patches ; sauvegarder les différences utiles dans l’inventaire avant toute future synchronisation. Identifier où résident les correctifs de provenance.
- [ ] Documenter que les deux corrections dataset sont acquises ; relever uniquement leurs régressions éventuelles, sans les réécrire par principe.

**Vérification :** rapport lisible, commandes avec codes de sortie ou timeout explicite, aucun fichier de code modifié par cette tâche. Un problème d’environnement non résolu interdit de prétendre avoir testé les tâches suivantes.

### Tâche 1 — Établir le contrat M-C et les données de référence

**Fichiers :** compléter `docs/audits/mc-compatibilite.md`, créer `docs/audits/mc-runtime.json` et `tests/fixtures/mc/README.md`. Lire `compatibility.py`, les tables amont, `scrape_logs.py`, `logs2trajs.py`, `policy.py`, `policy_player.py`, les formats Showdown et le profil PokeCoach.

**Entrées / sortie :** checkpoint M-B de confiance, moteurs et sources réelles → rapport `status`, formats source BO1/BO3 exacts, format moteur, architecture complète, contrat, hashes et références de fixtures.

- [ ] Identifier le checkpoint parent réellement souhaité via le profil et les fichiers existants ; enregistrer source, révision de téléchargement et SHA-256. Ne pas affirmer qu’il a appris exclusivement sur M-B.
- [ ] Charger effectivement ce checkpoint sur CPU dans l’environnement isolé ; relever largeur, couches, têtes, embedding, preview, espaces et nombre de paramètres. Les têtes absentes des métadonnées doivent venir du code historique audité, pas de la forme d’une matrice.
- [ ] Vérifier les identifiants M-C réels dans le fournisseur et le moteur figé. Enregistrer URLs/révisions consultées et exemples de recherche/téléchargement ; ne pas inventer `format_map["mc"]` comme preuve de support.
- [ ] Examiner quelques replays M-C réels : rating individuel, OTS, preview, fin, abandon/inactivité, références BO3. Archiver les preuves brutes localement ; pour les fixtures versionnées, conserver une provenance et anonymiser sans détruire les marqueurs utiles.
- [ ] Faire convertir une perspective réelle et comparer son encodage avec le checkpoint ; vérifier les tables ordonnées, dimensions, masque conditionnel et indices. Si une espèce/mécanique exige une nouvelle table, consigner le blocage du contrat.
- [ ] Vérifier deux équipes et la capacité du Showdown local à accepter le format. Relever les adaptations minimales requises côté bridge ; aucun remplacement silencieux de M-C par M-B.
- [ ] Définir la description canonique à calculer en tâche 11 : ordre des features, tables chargées à l’exécution, espaces, preview, masques et empreintes des sources pertinentes. L’ancien hash limité à quelques fichiers ne prouve pas tout cela.

**Vérification :** rapport avec cases `établi / à adapter / bloquant` et exemples réels. Les limites de recherche réseau sont consignées comme telles. En cas de blocage de contrat, la collecte peut être préparée après accord de périmètre, mais pas de dataset `ready` ni de fine-tuning M-C prétendument compatible.

**PAUSE A — Validation du socle.** Présenter ce qui a été réellement chargé/converti, les révisions et le format exact retenu. Attendre l’accord avant le lot B.

## 4. Lot B — Archive et collecte manuelle par FastAPI

### Tâche 2 — Figer les patches, types et règles de stockage

**Fichiers :** modifier `scripts/sync_vgc_bench.sh`, `patches/README.md`, `devbox.json`, `schemas.py`, `storage.py`, `settings.py`, `compatibility.py` ; créer `formats.py`, `configs/formats.json`, `tests/test_vendor_sync.py`, `tests/test_mc_schemas.py`, `tests/test_contract_mc.py` ; compléter `tests/test_storage.py`. Modifier `pyproject.toml`/`uv.lock` seulement pour les dépendances utilisées : notamment HTTPX doit être une dépendance d’exécution du collecteur, pas seulement un extra de test.

**Interfaces :** `resolve_format(format_id: str, *, require_conversion: bool) -> FormatSupport` dans `formats.py` ; `build_contract(model: ModelConfig) -> ModelContract` dans `compatibility.py` ; étendre `write_json_atomic()` et les résolveurs bornés existants. `FormatSupport` contient IDs source/moteur, régulation, `collection_supported`, `conversion_supported`, empreinte de contrat et références d’audit ; aucun drapeau vrai sans preuve. `ModelContract` porte la description canonique, ses empreintes, les dimensions/mappings et les révisions établies au lot A.

- [ ] Ajouter `test_vendor_sync_preserves_provenance_patch` : synchroniser deux fois une source de test figée dans un répertoire temporaire ; mêmes hashes, présence de `TaggedTrajectory`, patch policy conservé. Tester qu’une base inconnue est refusée. Ne pas tester `rsync --delete` sur le vendor de travail.
- [ ] Versionner les patches contre une base identifiée et les réappliquer de façon vérifiée. Inclure les adaptations existantes, les épingles et l’exclusion documentée de dépendances. Une synchronisation ne doit pas dépendre d’un voisin modifié non identifié.
- [ ] Définir les types ci-dessous dans `schemas.py`, avec `extra="forbid"` pour les requêtes et validation des bornes. Versionner les documents persistés (`schema_version=2` pour le nouveau contrat Trainer). Les JSON HTTP restent des JSON usuels ; valider les nombres/booleans sans conversions ambiguës.

| Type | Champs minimaux et règles |
| --- | --- |
| `CollectionConfig` | `format_ids`, dates d’upload facultatives UTC, `max_new_replays=5000`, `max_requests=10000`, `max_duration_s=7200`, `request_interval_s=1`, `timeout_s=30`, `max_attempts=5` |
| `DatasetRecipe` | Tous les champs YAML §6.1 de la spec ; `target_replays>=1`, ratios strictement positifs de somme 1, Elo min ≤ max si max défini, `variant="natural"`, plafond équipe nul |
| `SnapshotRequest` | Formats exacts, bornes d’upload facultatives, sélection explicite de versions si conflit |
| `DatasetJobConfig` | `dataset_id`, source discriminée `snapshot` ou `legacy_import` ; cette dernière porte seulement le chemin inbox borné et la recette validée |
| `InitializationConfig` | `mode="finetune"`, `base_model_id` obligatoire, `architecture="inherit"` |
| `ExperimentConfig` | `name`, `seed`, `dataset_id`, `initialization`, `training` ; aucune création implicite de `ModelConfig` Tiny |
| `TrainingConfig` | `method="bc"`, device validé, batch/époques ≥1, learning rate fini >0 par défaut `1e-4`, `evaluation_battles_per_epoch=0` ; refuser autre valeur non implémentée |
| `JobRecord` | ID, `kind` parmi collection/prepare/training, config typée discriminée par kind, état, dates UTC, PID, token, identité de démarrage, progression, résultat référencé, erreur |
| `DatasetManifest` | État cible §9, snapshot/hash, recette/hash, contrat, split ID, statistiques, job ID, limites BO3 et manifestes de fichiers |

- [ ] Choisir des plafonds administratifs locaux pour les budgets : par défaut égaux aux limites de collecte ci-dessus, relevables dans la configuration serveur ; une requête ne peut les dépasser. Intervalle ≥1 seconde, tentatives de 1 à 5 par défaut ; dates inversées et formats non enregistrés refusés.
- [ ] Normaliser `min_rating`/`only_winner` vers `min_elo`/`only_winners` avant validation ; accepter deux valeurs identiques, refuser les divergences. Refuser `force=true` sur le nouveau Builder avec instruction de créer un ID ; accepter `force=false` comme compatibilité sans effet.
- [ ] Implémenter dès maintenant le calcul canonique défini en tâche 1, pour que le Builder dispose d’un contrat avant la tâche 11 : tables réellement chargées, ordre des features, espaces/indices, règles de preview et masques, sources d’encodage/conversion des actions. Séparer `encoding_fingerprint` du contrat complet incluant architecture/provenance ; supprimer l’architecture Tiny implicite de `build_contract()`. Tester tables permutées à dimensions égales et sérialisation canonique stable. La tâche 11 réutilisera ce code pour le parent réellement chargé.
- [ ] Lire les anciens manifestes sans version comme documents historiques version 1 via un adaptateur explicite ; ne pas les réécrire ni inventer leur snapshot/test split. Une préparation selon ce plan crée un nouvel ID. Les anciens jobs sont reconnus comme entraînements historiques pour la réconciliation ; leur configuration n’est pas relancée implicitement. Tester leur consultation et les erreurs de migration lisibles.
- [ ] Définir erreurs métier typées avec `code`/`details` dans un nouveau `errors.py` : `NotFound`, `Conflict`, `InvalidConfiguration`, `ContractMismatch`, `BlockingMaskError`. Les routes les mapperont ; ne pas transformer une exception Python inattendue en erreur utilisateur `422`.
- [ ] Ajouter les racines archive/base_models/staging aux settings ; réutiliser les résolveurs contre `../`, absolus, IDs invalides et symlinks sortants. Publier via fichiers temporaires sur le même système de fichiers, flush et renommage atomique ; ne pas afficher les chemins absolus dans les réponses publiques.

**Tests :** `test_mc_schemas.py` vérifie les valeurs exactes de la spec, alias contradictoires, NaN/infini, option diverse/plafond équipe refusés, configuration d’architecture contradictoire et formats inconnus. `test_storage.py` couvre les sorties de racine et fichiers temporaires invisibles.

**Commande :** `devbox run test -- tests/test_vendor_sync.py tests/test_mc_schemas.py tests/test_storage.py tests/test_contract_mc.py -q` ; puis `devbox run check`. Attendu : code 0 et sources/locks inchangés après reproduction, hors mise à jour volontaire tracée.

### Tâche 3 — Réparer les jobs avant de leur confier la collecte

**Fichiers :** `jobs.py`, `worker.py`, `training_service.py`, `main.py` ; créer `job_service.py`, `tests/test_worker_lifecycle.py` ; compléter `tests/test_jobs.py`.

**Interfaces :** `submit_job(kind: JobKind, config: JobConfig) -> JobRecord`, `run_job(job_id: str) -> None`, `read_job_logs(job_id: str, tail: int) -> list[str]`. Conserver `get_job()` et `reconcile_jobs()`. `submit_training()` devient un adaptateur vers ce mécanisme ; pas de second verrou spécifique à l’entraînement.

- [ ] Écrire `test_only_one_long_job_across_types` avec deux vrais petits sous-processus concurrents ; exactement une soumission acceptée et une `Conflict`.
- [ ] Remplacer le fichier-marqueur supprimable par un verrou système local interprocessus. Sur le Linux du projet, utiliser `fcntl.flock` non bloquant sur un inode stable, sans supprimer le fichier à la libération. Le parent garde le descripteur jusqu’à son héritage par le worker via `pass_fds` ; le worker le garde jusqu’à la fin. Documenter la dépendance POSIX.
- [ ] Écrire config et état `queued` avant lancement ; passer `[sys.executable, "-m", "pokecoach_trainer.worker", "--job-id", id, ...]`, sans shell. Les informations de descripteur/jeton sont internes, jamais des paramètres HTTP. Le worker écrit lui-même `running` avec PID et heure de création : le parent ne doit pas écraser un état terminal si le worker finit immédiatement.
- [ ] Vérifier le préflight léger avant lancement ; répéter les vérifications de fichiers/contrat dans le worker. Ne pas charger Torch, désérialiser un checkpoint ou convertir un dataset dans la requête HTTP.
- [ ] Rediriger stdout/stderr vers `worker.log`. Persister traceback, état et dates **avant** propagation de l’erreur. Réaper les enfants sans bloquer les requêtes ; un arrêt de l’API ne tue pas le worker ni son verrou.
- [ ] Réconcilier avec verrou et identité du processus (PID, heure de démarrage et token de job). Une réservation encore verrouillée pendant le lancement n’est pas un orphelin. Un PID étranger n’est ni conservé comme worker ni tué. Si une incohérence est indécidable, refuser un nouveau calcul et exposer le diagnostic au lieu de libérer aveuglément.
- [ ] Couvrir `queued` abandonné, `Popen` en échec, worker brutalement arrêté, restart API avec worker vivant, PID réutilisé, gros stdout et dernière ligne JSONL tronquée. Une erreur d’un état historique corrompu est rapportée ; elle ne doit pas effacer tous les autres jobs.
- [ ] Lire les logs par fin de fichier avec `tail` entre 1 et 1000 ; ignorer seulement une dernière ligne de métriques incomplète, pas une corruption au milieu. Ajouter un heartbeat/progrès horodaté ; il aide au diagnostic mais ne remplace pas l’identité du processus.

**Commande :** `devbox run test -- tests/test_jobs.py tests/test_worker_lifecycle.py -q`. Attendu : processus de test terminés, aucun zombie/slot résiduel, erreur persistée et worker vivant conservé après restart.

### Tâche 4 — Construire l’archive brute et ses snapshots

**Fichiers :** créer `archive_service.py`, `tests/test_archive.py`, `tests/test_archive_snapshots.py` ; compléter `schemas.py` et `storage.py`.

**Interfaces :** `archive_replay(replay_id: str, payload: dict, source: ArchiveSource) -> ArchiveRef`, `create_snapshot(request: SnapshotRequest) -> ArchiveSnapshot`, `read_snapshot(snapshot_id: str) -> ArchiveSnapshot`, `import_legacy_logs(relative_path: str) -> ArchiveSnapshot`. `ArchiveRef` contient replay ID, hash brut, hash du log, format et référence relative. `ArchiveSource` porte provenance/date de récupération/révision du collecteur. `ArchiveSnapshot` contient ID, hash canonique, références exactes et version des métadonnées.

- [ ] Tester qu’un replay faible Elo, sans OTS ou avec espèce ambiguë est archivé intact. Le service ne dépend pas de `dataset_filters.py`.
- [ ] Écrire `archive/objects/<id>/<sha256>.json` : enveloppe versionnée avec réponse fournisseur intégrale et provenance. Définir le hash du contenu fournisseur en JSON canonique UTF-8, sans date locale ; définir séparément le hash du log exact. L’heure de téléchargement ne doit pas créer une nouvelle version à chaque import.
- [ ] Dédupliquer même ID/hash ; conserver deux versions d’un ID différent en contenu et marquer le conflit. Refuser la discordance entre ID demandé et réponse, rapporter les objets invalides sans les indexer comme prêts.
- [ ] Maintenir l’index JSONL reconstruisible à partir des objets publiés ; une interruption entre objet et index est réparable. Ignorer les fichiers temporaires ; journaliser la corruption d’un objet, jamais l’indexer aveuglément.
- [ ] Produire les snapshots sur références triées et métadonnées BO3 figées. Un conflit de versions sans sélection explicite interdit le snapshot concerné. Une collecte ultérieure ou une réextraction ne change pas son hash.
- [ ] À ce lot, conserver les métadonnées directement disponibles et marquer les liens non extraits `unknown`. Le parseur complet de la tâche 7 enrichira les **nouveaux** snapshots ; il ne réécrira jamais les snapshots de cette démonstration initiale.
- [ ] Importer le format historique `id -> [timestamp, log]` depuis un chemin borné à inbox, maximum 100 Mio par défaut. Détecter les clés contradictoires avec un décodeur conservant les paires avant création du dictionnaire ; métadonnées absentes à `unknown`.
- [ ] Tester restauration d’index, import malformé/trop gros, snapshot inchangé, version explicitement choisie et liens BO3 pointant vers des replays absents. Ne pas désérialiser de pickle à cette étape.

**Commande :** `devbox run test -- tests/test_archive.py tests/test_archive_snapshots.py -q`. Démonstration : deux imports du même JSON ne doublent pas les objets ; le snapshot référence leur contenu exact.

### Tâche 5 — Collecter avec pagination, limites et reprise

**Fichiers :** créer `collection_client.py`, `collection_service.py`, `tests/test_collection.py`, `tests/test_collection_resume.py` ; ajouter les types `SearchPage`, `CollectionCursor`, `CollectionResult` dans `schemas.py`.

**Interfaces :** `ShowdownReplayClient.search(format_id: str, cursor: dict | None) -> SearchPage`, `fetch(replay_id: str) -> dict`, `collect_replays(config: CollectionConfig, job_id: str) -> CollectionResult`. Le curseur externe est la représentation exacte vérifiée au lot A, pas un numéro de page supposé. `CollectionResult` expose compteurs, limites de couverture, erreurs et `stop_reason`.

- [ ] Écrire les tests avec transport HTTP simulé, horloge/sommeil injectables ; aucune attente réelle de 30 secondes pour tester les retries. Couvrir page vide, page répétée, doublon, IDs au même timestamp, 429 avec `Retry-After`, 5xx, timeout et JSON invalide.
- [ ] Reprendre les endpoints réellement audités de VGC-Bench ; fixer les hôtes côté service et vérifier toute redirection. Appliquer les bornes de date à l’upload, sans appeler cela une date certaine du combat.
- [ ] Persister les découvertes et leur état dans un journal de collecte avant d’avancer le curseur. Séparer progression historique et recherche de nouveautés ; conserver les téléchargements en attente même si la page suivante a été atteinte.
- [ ] Compter **toutes** les requêtes, recherches et retries compris ; limiter à un téléchargement simultané et au moins une seconde entre requêtes. Appliquer 30 secondes de timeout et cinq tentatives au maximum ; respecter `Retry-After` ou s’arrêter si l’attente dépasserait le budget restant.
- [ ] Reprendre les IDs en attente, ignorer les versions déjà archivées lors d’une reprise normale. Une réimportation de version différente est explicite, pas un téléchargement répété implicite.
- [ ] S’arrêter à 5000 nouveaux replays, 10000 requêtes ou 7200 secondes par défaut. Produire `budget_reached` avec le plafond précis ; pages répétées/frontière non parcourable produisent un motif distinct et `coverage_incomplete=true`, pas « archive complète ».
- [ ] Classer les erreurs par ID et réponse observée ; ne pas convertir toute erreur temporaire en suppression définitive. Simuler crash après découverte, après objet écrit et avant curseur ; la relance récupère chaque cas.
- [ ] Brancher le handler `collection` dans `run_job()` ; sauvegarder configuration effective, progression et résultat. Ne pas lancer de dataset automatiquement.

**Commandes :** `devbox run test -- tests/test_collection.py tests/test_collection_resume.py -q`. Puis une collecte réelle bornée à dix nouveaux objets, après accord du lot A, avec source et limites conservées. Les tests réseau simulés ne prouvent pas les capacités de pagination du fournisseur.

### Tâche 6 — Livrer les routes de collecte et d’archive

**Fichiers :** créer `api/collection.py`, `api/archive.py`, `tests/test_collection_api.py`, `tests/test_archive_api.py` ; compléter `main.py`, `cli.py`, README et `configs/collection-mc.yaml`.

**Interfaces CLI à ajouter :** `collection start --config <yaml>`, `collection status --job-id <id>`, `archive snapshot --config <yaml>`, `archive import --path <chemin-inbox>`. `start` soumet le même job que HTTP et rend l’ID ; il ne lance pas une seconde boucle directe.

- [ ] Brancher `POST /collection/jobs`, `GET /collection/jobs/{id}`, `/logs`, `GET /archive/replays`, `POST /archive/snapshots`, `GET /archive/snapshots/{id}`. La liste accepte `offset>=0`, `limit` 1–1000 (100 par défaut), format/date et retourne une pagination stable par ID/version.
- [ ] Mapper les erreurs typées : ID absent 404, slot occupé 409, mauvaise config 422. Un ID d’un autre type de job ne doit pas être retourné par une route de collecte.
- [ ] Ne pas interroger Showdown pendant `POST /collection/jobs` ; réponse 202 après réservation/lancement. Tester via worker factice bloqué que la réponse et `/health` restent disponibles avant sa libération.
- [ ] Sérialiser les petites mutations d’archive/snapshot avec un verrou de métadonnées distinct du slot long. Pendant collecte, figer seulement des objets entièrement publiés ; ne pas inclure une ligne d’index partielle. Les imports lourds seront mis en worker en tâche 10.
- [ ] Tester configuration hostile, logs bornés, erreurs après lancement consultables, reprise et restart API sans tuer la collecte. Vérifier `/docs` et la transmission des arguments CLI.
- [ ] Démontrer une collecte manuelle HTTP, l’inventaire et un snapshot ; conserver les requêtes/réponses et fichiers de preuve sous le rapport de lot, sans poids ni archive brute volumineuse dans Git.

**Commande :** `devbox run test -- tests/test_collection_api.py tests/test_archive_api.py -q`, puis tests du lot B et `devbox run check`.

**PAUSE B — Validation collecte/archive.** Montrer un replay brut qui sera rejeté par les filtres mais reste archivé, une reprise sans doublon et un snapshot stable. Expliquer les curseurs et les limites de couverture.

## 5. Lot C — Dataset natural fiable et inspectable

### Tâche 7 — Extraire les métadonnées et appliquer les filtres

**Fichiers :** créer `replay_metadata.py`, `dataset_filters.py`, `tests/test_replay_metadata.py`, `tests/test_dataset_filters.py` ; compléter les fixtures et `schemas.py`.

**Interfaces :** `extract_replay_metadata(ref: ArchiveRef) -> ReplayMetadata`, `decide_game(meta: ReplayMetadata, recipe: DatasetRecipe) -> FilterDecision`, `decide_perspective(meta: ReplayMetadata, side: str, recipe: DatasetRecipe) -> FilterDecision`. `FilterDecision` porte `accepted`, `stage`, `reason_code`, détails et sources ; `ReplayMetadata` conserve séparément faits observés et valeurs inconnues.

- [ ] Écrire les assertions de référence : ratings 1300/1000 ⇒ p1 seule ; 1100 inclus ; rating moyen présent sans rating individuel ⇒ perspective inconnue exclue ; perdant de 1300 accepté avec `only_winners=false`.
- [ ] Parser les ratings individuels, les deux feuilles OTS et six identités distinctes selon les marqueurs constatés en A. Une chaîne `showteam` présente mais inexploitable ne suffit pas ; pas d’identité inventée pour Illusion.
- [ ] Classifier victoire normale/abandon/inactivité/égalité/fin inconnue avec sources des messages. Compter les tours effectivement clos selon le protocole audité, pas `max(turn)` ; ordre du marqueur de résolution et de l’abandon pris en compte.
- [ ] Tester abandon après deux tours refusé, trois accepté, entrée au tour 3 sans résolution refusée, compteur indéterminable ⇒ `forfeit_turn_count_unknown`, inactivité exclue même dans une longue partie, égalité/fin inconnue exclues avec leurs codes.
- [ ] Extraire `match_type`, `series_id`, `game_number`, liens, source et statut. Les ressemblances joueurs/équipes/dates ne confirment rien. Les séries inconnues/incomplètes restent admissibles ; un conflit conserve ses sources sans regroupement inventé.
- [ ] Brancher cet extracteur lors de la création des nouveaux snapshots et enregistrer sa version. Créer un nouveau snapshot pour le dataset du lot C afin de figer les liens BO3 extraits ; ne pas enrichir un ancien snapshot en place. Le Builder reprend ses relations figées, pas celles découvertes par une collecte ultérieure.
- [ ] Documenter les codes de rejet dans `docs/dataset-reason-codes.md` ; les filtres émettent des décisions, ne suppriment aucun objet de l’archive. La recette seule contrôle les filtres métier, y compris les options non utilisées par défaut.

**Commande :** `devbox run test -- tests/test_replay_metadata.py tests/test_dataset_filters.py -q`. Démonstration pédagogique : expliquer pourquoi chaque perspective de trois replays réels est admissible ou non.

### Tâche 8 — Enrichir la conversion existante et contrôler les cibles

**Fichiers :** créer `conversion.py`, `tests/test_conversion_provenance.py`, `tests/test_trajectory_validation.py` ; modifier `trajectory_dataset.py`, enveloppe de `vendor/vgc-bench/vgc_bench/logs2trajs.py` et patch correspondant ; conserver `tests/test_dataset.py`.

**Interfaces :** `convert_perspective(ref: ArchiveRef, side: str) -> ConversionResult`, `validate_trajectory(result: ConversionResult, contract: ModelContract) -> None`. `ConversionResult` reprend exactement les champs §6.5 de la spec ; succès avec trajectoire, rejet/erreur sans trajectoire. `BlockingMaskError` interrompt la construction.

- [ ] Conserver `TaggedTrajectory` et l’association future → replay/perspective. Ajouter une interface de résultats détaillés incluant `None` et exceptions ; faire déléguer l’ancienne interface à la nouvelle pour conserver les appelants existants, sans attribution par position.
- [ ] Après les filtres du Builder, neutraliser explicitement les filtres métier redondants de `process_log` (Elo/gagnant), tout en gardant les contrôles techniques nécessaires. Une exception doit garder son contexte ; ne pas compter silencieusement tous les échecs comme zéro trajectoire.
- [ ] Tester un executor retournant dans l’ordre inverse, p1/p2 pour un replay, une seule perspective et erreur entre deux succès. Les IDs, côtés et raisons restent exacts. Prouver aussi la persistance du patch après synchronisation.
- [ ] Vérifier les formes réelles d’observation et la convention `len(obs)=len(acts)+1`, observations finies, actions entières et bornées **avant** indexation, états terminaux cohérents et nombre non nul de transitions.
- [ ] Construire le masque synthétique existant, puis appeler la logique de `_update_mask` de la policy pour contrôler la seconde cible conditionnée par la première. Si cette méthode doit être exposée, ajouter un wrapper pur amont partagé, pas une copie divergente des règles. Ne pas exécuter une policy aléatoire pour déterminer la validité.
- [ ] Tester indice négatif, indice hors borne, demi-masque sans cible, paire interdite après action du premier Pokémon. L’erreur contient replay, côté, transition, deux actions et masque pertinent ; elle bloque la publication entière, même si d’autres trajectoires étaient valides.
- [ ] Fermer les ressources de conversion/process pools et boucles dédiées dans `finally`. Corriger ici une fuite de ces ressources établie en tâche 0 ; vérifier que pytest se termine vraiment.

**Commandes :** `devbox run test -- tests/test_dataset.py tests/test_conversion_provenance.py tests/test_trajectory_validation.py -q`. Puis convertir quelques replays M-C réels et inspecter une observation/action ; aucune assertion sur le niveau de jeu.

### Tâche 9 — Sélectionner 300 parties, splitter par groupes et publier

**Fichiers :** créer `dataset_split.py`, `tests/test_dataset_split.py`, `tests/test_dataset_publication.py` ; modifier `dataset_service.py`, `schemas.py`.

**Interfaces :** `create_dataset(snapshot_id: str, recipe: DatasetRecipe) -> DatasetManifest`, `prepare_dataset(dataset_id: str, *, job_id: str) -> DatasetManifest`, `build_replay_groups(snapshot: ArchiveSnapshot) -> list[ReplayGroup]`, `assign_splits(groups: list[ReplayGroup], recipe: DatasetRecipe) -> SplitAssignments`. Les anciens appelants de `prepare_dataset(id, PrepareConfig)` sont migrés dans cette tâche et la suivante ; recette enregistrée à la création, pas modifiable à la préparation.

- [ ] Construire les composantes connexes d’aliases (hash de log exact) et liens BO3 confirmés ; utiliser des identifiants canoniques stables. Les liens explicites vers une partie absente peuvent relier deux parties présentes ; ils ne créent pas de fichier fictif. Un conflit de série ne constitue pas un lien confirmé.
- [ ] Choisir un représentant déterministe par classe de doublons, garder tous les IDs alias. Former les groupes avant de retirer les parties rejetées, afin qu’un maillon rejeté ne casse pas une série connue ; ne convertir que les perspectives admissibles.
- [ ] Définir l’ordre de sélection par tri des hashes SHA-256 de la chaîne canonique `(seed, group_id)`, avec ID en départage ; ne pas utiliser `hash()` Python. Traiter les groupes dans cet ordre ; la concurrence de conversion ne change pas l’ordre des décisions de sélection.
- [ ] Atteindre `target_replays` avec des replays ayant ≥1 perspective valide ; terminer le groupe courant. Compter le dépassement. Ne pas compter deux perspectives ou aliases comme deux replays. Les groupes suivants sont non sélectionnés, sans prétendre connaître leur qualité.
- [ ] Affecter les groupes retenus entiers par un algorithme documenté `group_greedy_v1` : trier par taille décroissante puis clé seed/ID ; choisir le split avec le plus grand déficit `ratio*cible_effective - compte_actuel`. Si le nombre de groupes restants égale le nombre de splits vides, réserver ces groupes aux splits vides. Départager par ordre train/validation/test. Moins de trois groupes non vides ⇒ résultat non entraînable, motif `split_groups_insufficient`.
- [ ] Tester mêmes snapshot/recette/seed ⇒ mêmes groupes, sélection, affectations et compteurs ; BO3 incomplet accepté ; aliases reliant deux séries ⇒ un seul groupe ; ratios calculés sur replays et non transitions. Ne pas exiger exactement 80/10/10 quand les groupes l’empêchent.
- [ ] Écrire dans une zone temporaire recette/source/décisions/index/affectations et pickles. L’index de trajectoires contient replay/côté/hash brut/split/fichier/hash de fichier/longueur/révision convertisseur. Contrôler les fichiers réellement écrits, pas seulement les listes en mémoire.
- [ ] Publier le manifeste `ready` en dernier après déplacement des sorties sous l’ID réservé ; toutes les lectures d’entraînement exigent `ready` et les hashes. Au crash, un répertoire partiel est `interrupted`/non entraînable. Ne jamais remplacer un répertoire déjà `ready` ni supprimer ses anciens splits.
- [ ] Si cible non atteinte, publier le rapport `insufficient_data` sans rendre les sorties entraînables. Aucun abaissement de filtre, collecte ou réduction de cible automatiques. Les erreurs bloquantes de masque/contrat donnent `failed` avec preuve ; les échecs ordinaires de conversion autorisent les candidats suivants.
- [ ] Tester l’immuabilité, crash avant publication, fichiers altérés, splits réellement disjoints et trois ensembles non vides. Exposer `series_leakage_guarantee=false` et le nombre de parties non groupables lorsqu’il le faut.

**Commande :** `devbox run test -- tests/test_dataset_split.py tests/test_dataset_publication.py -q`. Démonstration : manifeste, décisions et fichiers d’un petit snapshot inspectable ; ne pas le présenter comme le dataset de 300 de démonstration.

### Tâche 10 — Brancher le Builder au worker et à l’API

**Fichiers :** `dataset_service.py`, `job_service.py`, `worker.py`, `api/datasets.py`, `cli.py` ; créer `tests/test_dataset_api.py`, `configs/dataset-mc-natural.yaml` ; compléter README.

**Interfaces :** `submit_preparation(dataset_id: str) -> JobRecord` ; CLI `dataset create --snapshot-id <id> --recipe <yaml>`, `dataset prepare --dataset-id <id>`, `dataset inspect --dataset-id <id> --replay-id <id>`. `prepare` soumet le handler `prepare` du mécanisme commun.

- [ ] Créer `POST /datasets` avec snapshot/recette, puis `POST /datasets/{id}/prepare` en 202, `GET /datasets/{id}`, `/rejections` et `/logs`. Réutiliser les règles de pagination de l’archive ; les décisions filtrent `stage` et `reason_code`.
- [ ] Persister la relation job/dataset avant calcul ; état dataset `preparing`, puis `ready`, `insufficient_data`, `failed` ou `interrupted`. Un job `completed` avec données insuffisantes n’est pas un dataset prêt ; les réponses exposent les deux états.
- [ ] Garder `POST /datasets/import` pour un JSON inbox historique avec recette. Réserver immédiatement dataset/job IDs, retourner 202 et exécuter l’import+snapshot+préparation dans le handler `prepare` (source discriminée `legacy_import`). Le `snapshot_id` est nul jusqu’à sa création, puis persisté et consultable ; ne pas fabriquer un snapshot avant lecture. Documenter cette réponse asynchrone, nécessaire pour ne pas parser 100 Mio dans HTTP.
- [ ] Toute seconde préparation, collecte ou formation concurrente retourne 409. Un dataset `ready` est non modifiable ; une nouvelle recette crée un autre ID, avec `parent_dataset_id` optionnel.
- [ ] Tester import malformé/trop gros, chemins et symlinks, config inconnue, nouveau job sur ID inexistant, double appel, crash et récupération des deux états. S’assurer qu’une corruption de dataset ne charge aucun pickle avant contrôle du manifeste.
- [ ] Construire le dataset réel de 300 par HTTP. Si insuffisant, présenter le rapport et attendre une collecte manuelle ou une cible réduite explicitement demandée. Conserver le snapshot exact du dataset prêt pour les lots D–E.

**Commandes :** `devbox run test -- tests/test_dataset_api.py -q`, puis régression des tâches 7–9 et `devbox run check`. `/health` doit répondre pendant une vraie conversion ; le code de sortie du worker doit être récupéré.

**PAUSE C — Validation des données.** Montrer replays/perspectives/transitions, rejets, séries inconnues, affectation réelle des splits et une décision humaine encodée. Vérifier avec l’utilisateur le résultat `ready` ou les raisons d’insuffisance.

## 6. Lot D — Fine-tuning effectif depuis M-B

### Tâche 11 — Enregistrer le parent et comparer les contrats

**Fichiers :** créer `base_model_service.py`, `tests/test_base_models.py` ; compléter `tests/test_contract_mc.py`, `compatibility.py`, `schemas.py`, `cli.py`.

**Interfaces :** `register_base_model(relative_path: str, source: BaseModelSource) -> BaseModelManifest`, `load_base_model(base_model_id: str, device: str) -> PPO`, `describe_policy(policy) -> ModelConfig`, `build_contract(model: ModelConfig) -> ModelContract`. Retirer la valeur par défaut Tiny de `build_contract()` ; l’architecture effective est obligatoire.

- [ ] Proposer la CLI `base-model register --path <archive-inbox> --source <provenance-json>`. Exiger une origine et un hash attendus de confiance avant désérialisation ; copier l’archive intacte dans le registre local, sans URL arbitraire HTTP.
- [ ] Extraire l’architecture réelle ; tous ses champs sont renseignés et comparés aux métadonnées auditées. Une métadonnée historique manquante est complétée seulement par la révision historique démontrée au lot A et cette décision est tracée.
- [ ] Réutiliser le contrat canonique de la tâche 2 avec l’architecture effectivement chargée ; vérifier son égalité avec le contrat audité puis celui du dataset. Garder `contract_fingerprint` comme champ de compatibilité versionné du package. Ne pas prétendre qu’une différence textuelle sans effet vaut automatiquement une différence sémantique ; documenter ce que couvre chaque empreinte.
- [ ] Charger le parent par `PPO.load(..., device=...)` ; comparer les paramètres et buffers au contenu source avant tout forward. Refuser chargement partiel, classe inattendue, format SB3 invalide ou hash divergent. Un test à tables permutées avec dimensions identiques doit refuser la compatibilité.
- [ ] Refuser parent/dataset incompatibles avant le premier pas BC ; le parent M-B est admissible pour M-C si le contrat établi est le même. Ne pas exiger que son libellé de régulation source soit M-C.
- [ ] Enregistrer manifestes immuables parent, hash original, poids initiaux, architecture, nombre de paramètres et résultat de chargement. Le registre parent ne figure pas dans `/models` des artefacts entraînés.

**Commande :** `devbox run test -- tests/test_base_models.py tests/test_contract_mc.py -q`, puis enregistrement réel du parent audité. Attendu : aucune architecture Tiny implicite et aucune modification de l’archive source.

### Tâche 12 — Garantir batches complets et évaluation isolée

**Fichiers :** créer `bc_adapter.py`, `tests/test_bc_batches.py`, `tests/test_evaluation_isolation.py` ; modifier `training.py`, `trajectory_dataset.py` ; compléter `patches/README.md` si un patch du fork est indispensable.

**Interfaces :** `CompleteBatchBC(BC)` avec `set_demonstrations(demonstrations) -> None` ; `duplicate_policy(policy, device: str)` ; `evaluate_nll(policy, demonstrations, device: str) -> EvaluationResult` (`nll`, `transitions`). `evaluate_validation()` délègue à cette fonction pour ses anciens appelants.

- [ ] Tester 1, 31, 32, 33 transitions et trajectoires de longueurs variables. Avec batch effectif `min(32,N)` et `minibatch_size=batch_size`, chaque transition apparaît une fois par époque, `ceil(N/batch_size)` pas, pas de batch vide. Pour N=33, observer les deux pas avant le retour de `train(n_epochs=1)`.
- [ ] Dans la sous-classe minimale, remplacer uniquement la préparation du chargeur : aplatir avec les outils du fork, collate `DictObs` approprié, `drop_last=False`, mélange déterministe. Affecter le chargeur à l’extension attendue par BC sans repasser par le wrapper exigeant une taille fixe. Documenter l’usage de son attribut protégé et la révision compatible ; conserver la boucle `BC.train`, le calculateur de loss et l’optimiseur du fork.
- [ ] Laisser le fork appliquer sa pondération du dernier lot incomplet, en la documentant ; ne pas ajouter de padding, répétition ou suppression d’exemples. Si une autre version du fork rend l’adaptateur impossible, ajouter un patch étroit versionné et les mêmes tests, jamais une boucle BC maison non contrôlée.
- [ ] Reconstruire une copie de la policy avec ses paramètres effectifs, charger strictement, déplacer sur device puis `eval()`, `actor_grad=False` et gradients désactivés. Calculer `-log_prob` via `evaluate_actions()` sous `no_grad()`, avec moyenne pondérée par transitions.
- [ ] Évaluer des copies indépendantes pour train et validation/test afin que la renormalisation de la copie sur train n’influence pas la copie de validation. Ne jamais sauvegarder ces copies.
- [ ] Tester identité exacte des poids/buffers, gradients existants, modes et optimiseur originaux avant/après évaluation et après erreur injectée. Forcer un embedding au-delà de `max_norm` dans le test pour prouver l’isolation. Tester le device CPU, et GPU si disponible sans le compter comme testé s’il est ignoré.
- [ ] Refuser entrée vide, NLL non finie et mauvaise dimension ; fermer/libérer les ressources d’évaluation. Aucune lecture du split test dans la boucle d’époque.

**Commande :** `devbox run test -- tests/test_bc_batches.py tests/test_evaluation_isolation.py -q`. Ajouter au guide pédagogique une explication de `train()` réseau, gradients, optimiseur, `eval()` et copie isolée.

### Tâche 13 — Entraîner, tracer les vrais pas et évaluer le final

**Fichiers :** `training.py`, `training_service.py`, `schemas.py`, `worker.py`, `api/training.py`, `cli.py` ; créer `tests/test_finetuning.py`, `tests/test_training_api.py`, `configs/finetune-mb-mc.yaml` ; compléter `tests/test_training.py`.

**Interfaces :** garder `train_bc(config: ExperimentConfig, run_dir: Path) -> TrainingResult` et `submit_training(config) -> JobRecord`. `TrainingResult` ajoute architecture effective, parent/hash, dataset/hash/split, métriques BC/NLL, pas/transitions mesurés et rapport de test réservé. La CLI charge le YAML de la spec en normalisant `experiment`, `initialization`, `training` vers l’unique `ExperimentConfig` utilisé par HTTP.

- [ ] Tester que le premier état chargé est exactement le parent et qu’aucun `build_ppo_container(ModelConfig())` n’initialise le fine-tuning. Interdire `PPO.learn()` dans le test ; architecture contradictoire refusée.
- [ ] Charger le dataset prêt et ses manifestes/hashes, puis le parent. Fixer les graines de l’expérience après les constructions qui les réinitialisent ; ne pas laisser le `set_global_seed(42)` codé en dur remplacer la seed demandée.
- [ ] Utiliser les espaces sauvegardés du PPO si leur équivalence avec M-C est prouvée ; si un environnement est nécessaire, utiliser le format moteur vérifié du dataset et fermer ses ressources. Aucun besoin de jouer de nouvelles parties pour le BC.
- [ ] Créer une seule instance `CompleteBatchBC` et un seul optimiseur neuf pour le job : `optimizer_kwargs={"lr": config.training.learning_rate}`. Journaliser la valeur lue dans les groupes d’optimiseur ; ne pas reconstruire BC entre les époques.
- [ ] Instrumenter le calculateur de loss par délégation, sans changer l’objectif : vérifier finitude avant backward, relever loss BC, NLL/entropie/régularisation et coefficients effectifs. Ajouter un contrôle des gradients avant chaque `optimizer.step` (hook ou adaptateur minimal) ; un gradient non fini empêche le pas.
- [ ] Après chaque retour de `bc.train(n_epochs=1)`, mesurer NLL train/validation via tâche 12, relever les compteurs réellement observés et sauvegarder l’époque via fichier temporaire puis renommage. Pas de validation/sauvegarde dans la callback trop précoce de fin d’époque.
- [ ] Mesurer un delta sur des paramètres d’acteur hors embeddings `max_norm`, nommés dans le rapport. Exiger au moins un vrai pas et delta non nul ; la renormalisation seule ne prouve pas l’apprentissage. Vérifier que le parent sur disque reste identique.
- [ ] Écrire JSONL strict sans NaN/infini : époque, BC loss/coefficients, train/validation NLL, transitions et pas, durée, device, mémoire et checkpoint. Le mode `auto` vérifie réellement le device ; fallback CPU justifié, device demandé indisponible ⇒ échec clair.
- [ ] Le final est la dernière époque configurée, donc la troisième pour la démonstration. Fixer ce checkpoint avant toute lecture du split test ; évaluer le test une seule fois à la fin et lier le rapport aux hashes. `num_epochs` différent reste configurable ; ne pas chercher une époque inexistante 3.
- [ ] Tester lr effectif, optimiseur conservé, poids réellement modifiés, source intacte, erreur NaN/gradient avant publication, aucun accès test avant fin, absence de mutation par évaluation et jobs en échec persistés.
- [ ] Brancher le nouveau corps `POST /training/jobs` et la CLI `train --config <yaml>` au worker commun. Préflight léger HTTP, validation complète worker. Au lot D, garder le résultat comme run vérifié ; la publication finale et le passage `completed` après export/reload seront validés en tâche 14, sans publier un faux modèle entre-temps.

**Commandes :** `devbox run test -- tests/test_training.py tests/test_finetuning.py tests/test_training_api.py -q`. Effectuer ensuite un entraînement réel trois époques en CLI de diagnostic (`train --config ... --foreground`, option réservée à la CLI, même fonction métier) ; le parcours HTTP complet sera la preuve du lot E.

**PAUSE D — Validation de l’apprentissage.** Montrer parent/architecture, une loss BC et une NLL, le learning rate effectif, le nombre de pas, un delta de poids hors embeddings et le test final réservé. Aucune promesse de gain de niveau.

## 7. Lot E — Package, PokeCoach et démonstration HTTP complète

### Tâche 14 — Exporter puis recharger avant publication

**Fichiers :** `artifact_service.py`, `reload_check.py`, `schemas.py`, `training_service.py`, `api/models.py` ; créer `tests/test_artifacts.py`, `tests/test_model_reload.py`, `tests/test_models_api.py`.

**Interfaces :** conserver `export_model(result, config, model_id=None) -> ModelManifest` et `verify_reload(model_path, manifest_path) -> ReloadReport`. L’architecture vient de `TrainingResult` et du modèle, jamais d’un `config.model` par défaut.

- [ ] Préparer sous un répertoire temporaire : `model.zip`, `manifest.json`, `metrics.jsonl`, `test_report.json`, `README.md`. Le manifeste reprend les champs complets §10 de la spec, avec `schema_version`, parent, contrats, provenance, versions et hashes réels.
- [ ] Conserver `architecture`, `checkpoint_sha256`, `contract_fingerprint` consommés par PokeCoach ; exiger tous leurs champs. Ajouter `trainer_reload`, `reserved_test`, `pokecoach_compatibility` séparés ; avant les combats, ce dernier vaut `not_tested`.
- [ ] Lancer le reload CPU dans un nouveau processus avec `sys.executable`, timeout 120 secondes configurable localement. stdout contient un seul rapport JSON ; logs ailleurs. Capturer erreurs de timeout, sortie non JSON, code non nul et import absent.
- [ ] Vérifier hash, classe, tous les paramètres d’architecture, espaces/contrat, égalité des poids avant forward et prédiction masquée sur la même observation de contrôle. Comparer les sorties déterministes à tolérance explicite (initialement `atol=1e-6`, `rtol=1e-5` sur CPU ; tout changement justifié et tracé).
- [ ] Créer le ZIP de téléchargement, puis publier le répertoire final et le marqueur `published.json` seulement après succès. ZIP tronqué, hash incorrect, architecture incomplète ou reload échoué ne figurent ni dans la liste ni dans les routes de téléchargement.
- [ ] Marquer le job d’entraînement `completed` uniquement après export/reload/test final valides ; sinon `failed` avec traceback et run de diagnostic conservé. Tester la nouvelle requête d’entraînement à trois époques jusqu’à son état terminal.
- [ ] Tester extraction dans un autre répertoire et reload CPU depuis le package téléchargé ; verrou de job libéré seulement après cette chaîne d’export. Les preuves PokeCoach ultérieures sont conservées dans un rapport lié au hash, sans modifier les poids publiés.

**Commande :** `devbox run test -- tests/test_artifacts.py tests/test_model_reload.py tests/test_models_api.py tests/test_training_api.py -q`. Aucun manifeste publiable ne contient `TO_FILL`, une architecture déduite d’un défaut ou une réussite PokeCoach non démontrée.

### Tâche 15 — Faire respecter le contrat par le profil PokeCoach

**Fichiers PokeCoach :** `engine-runtime/src/pokecoach_vgc_bridge/{model_profile,policy_factory,showdown_teams,battle_session,battle_runner,vgc_real}.py`, tests `test_model_profile.py`, `test_policy_factory.py`, `test_encoder_parity.py` ; créer `engine-runtime/tests/test_mc_model_contract.py`. Modifier la configuration/transmission de provenance dans `src/pokecoach_python/engine/` uniquement si nécessaire.

**Interfaces :** étendre `VgcModelProfile` avec régulation cible, formats source/moteur et contrat ; `load_policy_from_profile()` garde une policy entièrement chargée comme retour. Ajouter `validate_profile_for_matchup(profile, regulation: str) -> None` avant création des joueurs.

- [ ] Lire le manifeste du package sans complétion Tiny implicite. Refuser hash absent/faux, architecture incomplète, version de manifeste non supportée et contrat incompatible. Conserver un parcours historique explicite et testé, sans que ses tolérances deviennent un fallback pour les nouveaux packages.
- [ ] Vérifier le contrat calculé sur le code réellement chargé par le bridge ; utiliser la même description canonique que le Trainer, avec test de parité. Si du code de calcul est partagé via une petite adaptation vendor, la versionner ; ne pas installer tout le Trainer dans le bridge pour cela.
- [ ] Propager M-C de la configuration du matchup au format moteur. Remplacer le M-B codé en dur de la factory ; relier le mapping aux identifiants audités. Pas de migration générale du moteur ni de modification des espaces.
- [ ] Charger strictement le checkpoint complet avant d’annoncer `checkpointLoaded`. Un package nouveau invalide ne doit pas tomber dans le fallback manuel `policy.pth` qui masque une erreur de contrat. Tester architecture égale mais têtes/mapping différents.
- [ ] Utiliser le même profil pour les deux joueurs si souhaité, dans deux instances indépendantes ; ne pas imposer deux modèles distincts. Vérifier la preview réelle sans quartet ni lead imposés dans la preuve finale.
- [ ] Comparer encodage, masque légal, action et ordre sur des états `DoubleBattle` représentatifs : preview, tour normal, switch et remplacement forcé. Conserver les contraintes des deux actions ; zéro repli silencieux après erreur.
- [ ] Tracer modèle/hash complet, architecture, contrat et régulation dans les résultats de simulation persistés. Les contrôles doivent porter sur le modèle chargé, pas seulement le YAML de configuration.

**Commandes dans l’environnement bridge existant :** depuis `PokeCoach: engine-runtime/`, `uv run --locked pytest tests/test_model_profile.py tests/test_policy_factory.py tests/test_encoder_parity.py tests/test_mc_model_contract.py -q` via Devbox. Confirmer au lot A que ce launcher utilise bien le venv du bridge ; ne pas lancer ces tests avec le venv Trainer. Rejouer également les tests unitaires de configuration de profil de l’application.

### Tâche 16 — Exécuter la recette réelle et livrer les preuves

**Fichiers :** créer `tests/test_mc_pipeline_real.py`, `docs/reports/mc-v1.md`, `docs/reports/mc-v1.json`, `configs/vgc_model_mc.example.yaml` ; compléter README et `docs/comprendre-le-bc.md`. Côté PokeCoach, étendre les tests réels de combat/persistance existants plutôt que créer un second simulateur.

**Interfaces de preuve :** rapport avec verdict par critère, commandes/codes de sortie, IDs de jobs/dataset/snapshot/modèles/matches, hashes, métriques et chemins des traces. Les artefacts lourds restent dans le stockage local ignoré par Git.

- [ ] Documenter la recette reproductible depuis les locks et patches : démarrage local, collecte HTTP manuelle, snapshot, création/préparation dataset, enregistrement local du parent, entraînement HTTP, polling/logs, téléchargement et profil PokeCoach. Tous les IDs d’exemple sont remplacés par les IDs obtenus, pas des noms magiques supposés présents.
- [ ] Exécuter le parcours HTTP avec trois époques réelles et le dataset de démonstration prêt. Une reprise de collecte utilise ses curseurs ; ne pas prétendre qu’une recherche publique dynamique reproduit les mêmes replays sans le snapshot archivé.
- [ ] Télécharger et extraire le package ; recharger dans un nouveau processus CPU. Enregistrer ce hash comme unique checkpoint de la preuve PokeCoach.
- [ ] Exécuter un matchup M-C par le parcours normal PokeCoach, avec deux équipes validées : sélection réelle du quartet par la policy, tours normaux et résultat terminal. Désactiver explicitement fixtures/stubs et imposer un timeout borné documenté.
- [ ] Capturer rejets Showdown, exceptions, actions de secours et switches. Exiger zéro ordre rejeté et aucun repli silencieux ; une action réparée ne valide pas ce critère. Si aucun switch n’apparaît, ajouter un scénario ciblé compatible et conserver les essais précédents.
- [ ] Vérifier les traces persistées et leur ouverture dans l’affichage existant. Un retour du bridge en mémoire seul ne suffit pas. Relever IDs de parties et provenance effective.
- [ ] Arrêter/recréer les processus PokeCoach/bridge concernés, vérifier de nouveaux PID, puis terminer un second combat avec le même hash de checkpoint et une nouvelle trace. Aucun réentraînement entre les deux ; décisions stochastiques éventuellement différentes autorisées.
- [ ] Produire le rapport JSON et le rapport lisible : conversion, perspectives/transitions/splits, séries inconnues, budgets/couverture, poids parent/enfant, losses, pas BC, CPU reload, parité de contrat et deux combats. Un test absent/ignoré ou une trace manquante interdit un verdict global `PASSED`.
- [ ] Relire la matrice ci-dessous et le guide pédagogique. Si cible réduite explicitement, distinguer réussite technique et écart à 300. Documenter commandes d’arrêt, reprise de collecte, erreurs et limites, notamment absence de reprise exacte de l’optimiseur BC après interruption.

**Vérification :** tests ciblés réels et recette PokeCoach, avec `POKECOACH_VGC_REAL=1` et sans `POKECOACH_BRIDGE_FIXTURE`. Le test Trainer réel doit échouer si les prérequis demandés sont absents, ou être explicitement ignoré hors campagne ; un skip n’est jamais converti en preuve positive.

**PAUSE E — Validation de la V1.** Présenter le package téléchargé, les deux parties complètes après redémarrage et le rapport. La victoire n’est pas un critère ; une partie valide perdue peut réussir la compatibilité.

## 8. Matrice de couverture et checklist finale

| Spec | Réalisation | Preuve attendue |
| --- | --- | --- |
| §1–3 : périmètre, acquis et corrections | 0–2, 8 | Acquis conservés ; versions établies ; aucun nouveau Tiny aléatoire |
| §4 : trois services, un worker, disque | 2–3, 6, 10, 13 | Une seule place interprocessus et consultations disponibles |
| §5 : archive, reprise et snapshots | 4–6 | Bruts conservés, retries/budgets, snapshots stables, limites visibles |
| §6.1–6.4 : filtres et BO3 | 7 | Admissibilité par partie/perspective ; BO3 inconnu non rejeté |
| §6.5 : conversion, provenance et masque | 8 | Résultats identifiés ; cible interdite bloquante |
| §6.6–6.7 : sélection, splits, immuabilité | 9–10 | 300 exploitables ou insuffisance explicite ; aucun croisement réel |
| §7 : contrat M-C | 1, 11, 15 | Dimensions et sémantique vérifiées ; aucun fallback M-B |
| §8.1–8.2 : parent et architecture | 11, 13 | Poids de départ exacts, optimiseur neuf et lr effectif |
| §8.3–8.4 : BC, validation et test | 12–13 | Pas réels, batches complets, zéro mutation par évaluation |
| §9 : API, jobs, erreurs | 3, 6, 10, 13–14 | Réponses rapides, erreurs consultables, restart sûr |
| §10 : export et PokeCoach | 14–16 | CPU reload isolé et deux combats réels persistés |
| §11–12 : preuves et livraison | Chaque tâche, 16 | Tests ciblés, rapports de lots et résultat final justifié |

- [ ] Pauses A, B, C, D et E approuvées explicitement, dans l’ordre.
- [ ] Commandes Devbox/uv reproductibles ; patches vendor conservés après synchronisation.
- [ ] Archive brute, snapshots, dataset et parent enregistrés avec provenance complète.
- [ ] Tous les filtres et limites BO3 conformes ; trois splits contrôlés sur les fichiers réels.
- [ ] BC réel depuis M-B ; architecture et contrat conservés ; métriques finies et acteur modifié.
- [ ] Job lancé par HTTP terminé après export/reload ; package téléchargé vérifié.
- [ ] Même modèle utilisé en M-C dans deux exécutions PokeCoach séparées par redémarrage.
- [ ] Rapport final complet, code commenté et recette compréhensible pour une personne débutant en entraînement.

**Consigne finale à l’implémenteur :** ne pas déclarer la V1 finie après un export ou des tests unitaires. La fin exige les preuves réelles du lot E et la revue humaine correspondante.
