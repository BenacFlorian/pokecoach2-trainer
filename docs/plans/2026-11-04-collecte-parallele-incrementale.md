# Collecte parallèle et incrémentale — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Lire la spec et ce plan avant de modifier le code. Ce document prépare l'implémentation ; sa rédaction n'a pas lancé les travaux de code.

**Goal:** Collecter sur une machine jusqu'à environ deux millions de replays accessibles, avec découverte et téléchargements parallèles, archive commune, déduplication, reprise et mises à jour incrémentales.

**Architecture:** Un job public représente une campagne, supervisée dans un processus détaché de l'API, avec deux pools de processus : découverte et téléchargement. SQLite coordonne les réservations, curseurs, budgets et publications ; les objets JSON restent sur disque. Les datasets et modèles continuent d'utiliser des données figées pendant la collecte.

**Tech Stack:** Python ≥ 3.13, SQLite via `sqlite3`, HTTPX synchrone, Pydantic 2, FastAPI, argparse, `subprocess`, `fcntl` sous Linux, pytest et Ruff, environnement uv/Devbox existant. Pas de Redis, Celery, ORM ou nouveau service réseau.

**Spec:** [Collecte parallèle et incrémentale — spec validée](../specs/2026-11-04-collecte-parallele-incrementale.md).

## Contraintes globales

- Une seule machine et disque local ; une campagne active par archive, plusieurs processus workers. Un seul job de calcul long (préparation ou entraînement) peut coexister avec elle.
- Archive neuve : **pas de migration JSONL → SQLite, pas d'import des anciens curseurs/jobs, pas de commande de purge automatique**. L'import explicite de fichiers de replays reste une fonctionnalité distincte à conserver.
- Le nettoyage demandé a déjà été exécuté le 4 novembre : 29 360 objets archivés, index/snapshots et copies de logs de `raw_logs`/`inbox` supprimés. Les modèles, datasets préparés, runs, smoke et historiques de jobs ont été conservés. **Ne pas répéter ce nettoyage lors de l'implémentation.** Les anciennes références vers les sources supprimées peuvent être inutilisables.
- Dimensionnement : 20 000, 100 000, 1 000 000 puis **2 000 000** d'IDs ; mémoire des workers bornée par les lots, aucune lecture intégrale du catalogue pour traiter un ID.
- WAL avec SQLite 3.51.3 ou ultérieure, ou backport officiel corrigé 3.44.6/3.50.7. Vérifier la bibliothèque réellement chargée par Python, pas seulement le binaire `sqlite3` du shell.
- Objets conservant le contrat `archive/objects/<replay-id>/<content-sha256>.json` ; unicité logique `(replay_id, content_sha256)` ; snapshots et datasets immuables.
- Pas de filtre Elo, OTS, résultat ou durée dans la collecte. Conserver les contrôles d'identité/format Champions existants et la logique métier du Dataset Builder.
- **Filtres date et équipes facultatifs, désactivés par défaut.** `format_ids: [mc]` seul collecte toutes les compositions et tout l'historique accessible du format, dans les budgets. Date et équipes se cumulent par ET quand présents ; les équipes configurées s'alternent par OU. `backfill` est le mode par défaut ; `incremental` exige une sélection explicite.
- Un téléchargement pour inspection est autorisé ; un replay rejeté n'est pas publié dans l'archive ni ajouté à la sélection de la campagne. Index technique minimal autorisé ; aucun corps JSON/log conservé pour un rejet. Objets déjà archivés par d'autres campagnes conservés, mais sélection par campagne obligatoire.
- `max_selected_replays` devient le nom canonique du plafond de replays retenus ; accepter `max_new_replays` comme alias legacy. Compter séparément les IDs retenus par campagne, les objets nouvellement archivés, les objets réutilisés, les rejets et les GET. Un rejet rend sa place de sélection, sans rembourser la requête réseau.
- Budgets et débit globaux, retries compris ; le nombre de workers ne multiplie pas le plafond de requêtes. Zéro transaction SQL ouverte pendant HTTP, attente réseau ou backoff.
- Valeurs proposées dans la spec pour le YAML parallèle : découverte 2, téléchargement 8, fenêtres 7 jours, file haute 10 000, 2 requêtes/s, 2 000 000 replays retenus maximum, 3 000 000 requêtes maximum, 86 400 s, timeout 30 s, 5 tentatives, réservation 120 s, heartbeat 15 s.
- Ancien YAML sans nouveaux paramètres : 1 worker de découverte + 1 de téléchargement, intervalle converti en débit global. Ne pas multiplier par huit la charge réseau d'une ancienne config implicitement.
- Commentaires/messages opérateur en français ; erreurs HTTP métier existantes conservées. Ne pas masquer un bug Python sous une erreur de configuration.
- Tests avec serveur simulé et racines temporaires ; aucun test automatique ne collecte Showdown ou ne touche le vrai `storage/`.

## Points de revue à couvrir explicitement

1. Deux soumissions simultanées sans borne haute explicite : même campagne malgré deux instants différents de soumission — tâche 4.
2. Crash après autorisation d'un GET, avant l'envoi/acquittement : plafond conservateur, aucune requête « gratuite » à la reprise — tâche 6.
3. Horloge système reculant, machine suspendue, PID réutilisé : ne jamais créer deux propriétaires valides ou des workers immortels — tâches 5 et 10.
4. Disque plein ou fichier publié avant rollback SQL : aucune tâche faussement archivée, reprise ciblée — tâche 7.
5. Redémarrage API pendant une préparation legacy qui importe des replays : verrou de calcul conservé et publication cohérente avec la collecte — tâches 10 et 12.

Ces cinq cas ont des tests dédiés ci-dessous ; ils ne sont pas laissés à une revue finale intuitive.

---

## 0. Mode d'emploi pour l'agent implémenteur

Tous les chemins du plan sont relatifs à la racine Git **`pokecoach2-trainer/`**, et non au workspace parent. Les noms `src/...` et `tests/...` dans les commandes sont relatifs au sous-projet **`pokecoach-trainer/`**. Les chemins complets sont donnés dans l'inventaire.

Travailler dans l'ordre des tâches. Chaque tâche suit : test rouge pertinent → changement minimal → tests ciblés verts → inspection du diff → commit limité à la tâche. Un échec d'import d'une dépendance externe n'est pas un test rouge du comportement. Ne pas transformer ce plan en une réécriture du trainer ou du vendor.

Commandes de référence depuis `pokecoach2-trainer/pokecoach-trainer/`, dans l'environnement uv disponible :

```bash
uv run --locked pytest tests/test_collection.py -q
uv run --locked ruff check src tests
uv run --locked pokecoach-trainer --help
```

Le `conftest.py`, `cli.py`, `main.py` et `worker.py` actuels changent le CWD vers le vendor. Utiliser des chemins absolus calculés depuis `__file__` ou les settings ; ne pas déduire le stockage du CWD du worker. Résoudre les chemins CLI fournis par l'utilisateur avant le changement de CWD, ou retirer ce changement global du parcours collecte. Ne pas importer Torch/VGC-Bench dans chaque worker de collecte : importer ces dépendances seulement dans les parcours qui les utilisent.

Les scripts Devbox actuels ne transmettent pas tous les arguments comme leur usage le suggère. Pour les tests ciblés, utiliser les commandes uv ci-dessus depuis le bon dossier. Si Devbox est le seul moyen d'obtenir uv, entrer dans son environnement, puis vérifier `pwd` avant la commande.

Pour chaque commit prévu ci-dessous : `git add` uniquement les fichiers de la tâche, puis `git -c user.name=benacFlorian -c user.email=florianbenac46@gmail.com commit --author="benacFlorian <florianbenac46@gmail.com>" -m "…"`. Ne pas modifier l'identité Git globale, ne pas committer `storage/`, ne pas pousser automatiquement.

### Cartographie des fichiers

| Action | Fichier sous `pokecoach-trainer/` | Responsabilité |
| --- | --- | --- |
| Modifier | `src/pokecoach_trainer/schemas.py` | Configuration et contrats API publics ; conserver les modèles archive/dataset existants. |
| Créer | `src/pokecoach_trainer/collection_types.py` | Petits types internes explicités plus bas, sans import ML. |
| Créer | `src/pokecoach_trainer/collection_filters.py` | Chargement JSON, normalisation des équipes, extraction depuis le payload et décision avant publication. |
| Créer | `src/pokecoach_trainer/archive_catalog.py` | Connexion SQLite, schéma, requêtes archive indexées, maintenance. |
| Créer | `src/pokecoach_trainer/collection_store.py` | Transactions de campagnes, tentatives, tâches, réservations et états. |
| Créer | `src/pokecoach_trainer/collection_limits.py` | Autorisations HTTP, budgets et ralentissement global. |
| Créer | `src/pokecoach_trainer/collection_publication.py` | Fichiers temporaires, intentions durables, publication et récupération. |
| Créer | `src/pokecoach_trainer/collection_discovery.py` | Fenêtres, pagination, couverture, mode incrémental. |
| Créer | `src/pokecoach_trainer/collection_workers.py` | Boucles des deux rôles, heartbeat et délais interruptibles. |
| Créer | `src/pokecoach_trainer/collection_supervisor.py` | Propriétaire de campagne, processus enfants, scale/stop/reprise. |
| Créer | `src/pokecoach_trainer/collection_metrics.py` | Vues de progression et agrégats bornés. |
| Modifier | `src/pokecoach_trainer/collection_client.py` | Une tentative HTTP par appel, résultat typé et politique de pagination auditée. |
| Modifier | `src/pokecoach_trainer/collection_service.py` | Façade start/status/commandes ; retirer l'ancienne boucle séquentielle. |
| Modifier | `src/pokecoach_trainer/archive_service.py` | API existante adossée au catalogue ; import, liste, snapshots. |
| Modifier | `src/pokecoach_trainer/jobs.py`, `job_service.py`, `worker.py`, `main.py` | Slots distincts, transmission des verrous et réconciliation correcte. |
| Modifier | `src/pokecoach_trainer/cli.py`, `api/collection.py`, `settings.py`, `errors.py` | Exposition et initialisation, sans duplication de logique. |
| Vérifier/adapter | `src/pokecoach_trainer/dataset_service.py`, `training_service.py` | Libération correcte du verrou transmis ; erreurs avant sauvegarde ; appels archive. |
| Modifier | `configs/collection-mc.yaml`, `configs/smoke/collection-mc.yaml`, `README.md` | Config parallèle, exemple incrémental et mode opératoire. |
| Créer | `scripts/benchmark_collection.py` | Mesures synthétiques explicites, hors tests par défaut. |

### Types internes à fixer en tâche 2

Définir des dataclasses immuables dans `collection_types.py`. Ne pas remplacer les schémas Pydantic publics par ces dataclasses.

| Type | Champs minimaux |
| --- | --- |
| `SearchItem` | `replay_id: str`, `uploadtime: int`, `source_format: str` |
| `DiscoveryPage` | `items: tuple[SearchItem, ...]`, `next_before: int | None`, `exhausted: bool`, `blocked_reason: str | None` |
| `WindowBounds` | `source_format: str`, `start: int`, `end: int` ; UTC epoch, intervalle `[start, end)`. |
| `TaskLease` | `kind: Literal['download','discovery']`, `task_id: str`, `campaign_id: str`, `attempt_id: str`, `worker_id: str`, `token: str`, `expires_at: float`, `replay_id: str | None`, `window: WindowBounds | None`, `cursor_before: int | None`, `attempts_used: int` |
| `ClaimResult` | `lease: TaskLease | None`, `wait_s: float`, `reason: str | None` ; ne pas confondre file vide et campagne terminée. |
| `RequestPermit` | `permit_id: str`, `attempt_id: str`, `task_id: str`, `kind: Literal['search','fetch']` |
| `PermitResult` | `permit: RequestPermit | None`, `wait_s: float`, `stop_reason: str | None` |
| `PreparedReplay` | `ref: ArchiveRef`, `temp_relpath: str`, `envelope: dict[str, Any]` |
| `PublicationResult` | `ref: ArchiveRef`, `new_replay: bool`, `new_version: bool` |
| `WorkerIdentity` | `worker_id: str`, `pid: int`, `boot_id: str`, `process_start_ticks: int`, `token: str` |
| `TeamSummary` | `p1: tuple[str, ...] | None`, `p2: tuple[str, ...] | None`, `parser_version: str`, `content_sha256: str` ; une équipe connue contient exactement six identités initiales triées. |
| `SelectionDecision` | `selected: bool`, `reason: Literal['selected','team_mismatch','team_unknown','outside_period','date_unknown']`, `filter_sha256: str` |
| `CampaignView` | `job: JobRecord`, `phase: str`, `attempt_id: str`, `effective_config: CollectionConfig`, `counters: dict[str, int]`, `stop_reason: str | None`, `coverage_incomplete: bool` |
| `Clock` | Protocole `wall() -> float`, `monotonic() -> float`, `boot_id() -> str` ; injectable dans limites/réservations/supervision. |

`CollectionStore(root: Path, *, clock: Clock)` ouvre une connexion courte par opération ; les éventuels helpers internes prenant une connexion ne sont pas des API publiques. Ne pas partager un objet connexion à travers un fork, un thread heartbeat ou des appels concurrents.

---

## Tâche 1 — Établir la baseline et le contrat fournisseur

**Fichiers :** créer `docs/audits/2026-11-04-collecte-parallele-baseline.md` et `pokecoach-trainer/tests/fixtures/collection/search_contract.json` ; lire les fichiers de la cartographie. Modifier l'environnement uv uniquement si nécessaire pour SQLite corrigé.

**Interfaces :** aucune interface applicative produite ; livrer les commandes réelles, versions, révision Git et le contrat qui servira à la tâche 8.

- [ ] Relever `git status --short`, `git rev-parse HEAD`, `uv run --locked python -c "import sys, sqlite3; print(sys.version); print(sqlite3.sqlite_version)"`. Conserver les changements utilisateur. Si SQLite n'est pas corrigé, établir puis documenter l'environnement Python corrigé avant les tests de concurrence ; ne pas désactiver le contrôle de version.
- [ ] Exécuter `uv run --locked pytest tests/test_collection.py tests/test_collection_resume.py tests/test_archive.py tests/test_archive_snapshots.py tests/test_collection_api.py tests/test_worker_lifecycle.py -q`. Enregistrer les échecs préexistants sans les attribuer aux futures modifications.
- [ ] Lire la [documentation de recherche officielle](https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md#replay-search) et, si nécessaire, le code serveur amont. Vérifier avec au plus dix GET de recherche, séquentiels, espacés d'au moins une seconde : inclusion/exclusion de `before`, ordre des timestamps, recouvrement de la frontière, réponse vide et taille de page. Ne pas télécharger une nouvelle archive pour cet audit.
- [ ] Sauvegarder une petite fixture de recherche et sa provenance/date, sans grands logs. Si le réseau est indisponible, documenter le contrat non confirmé et bloquer uniquement la validation réelle ; les tests simulés restent possibles. Aucune logique de saut de timestamp ne sera inventée pour faire passer les tests.
- [ ] Documenter que l'ancien code `before + 1` et le découpage temporel doivent être réexaminés. L'audit doit fournir une règle de continuation vérifiée, ou la décision explicite de signaler `coverage_incomplete` aux frontières non résolubles.
- [ ] Commit : `docs: establish parallel collection baseline and search contract`.

**Terminé si :** baseline et bibliothèque SQLite identifiées, contrat documenté avec ses limites, aucune donnée réelle modifiée.

## Tâche 2 — Types, configuration et compatibilité YAML

**Fichiers :** modifier `schemas.py`, `errors.py` ; créer `collection_types.py`, `tests/test_collection_config.py` ; adapter `tests/test_mc_schemas.py` si ses assertions portent sur les anciennes limites remplacées.

**Interfaces produites :** types de la table ; `normalize_collection_config(config: CollectionConfig, *, now: datetime) -> CollectionConfig` et `collection_request_fingerprint(config: CollectionConfig) -> str` dans `collection_service.py` (fonctions pures, sans lancement). Ajouter `StaleLease`, `BudgetReached`, `ProviderUnavailable`, `ProviderRateLimited`, `ProviderInvalidPayload` dans `errors.py`.

- [ ] Écrire `test_legacy_config_keeps_global_rate` : ancien YAML avec intervalle 1 → débit 1, pools 1/1. Écrire `test_interval_and_rate_rejected_even_when_equal` : deux paramètres explicitement présents → erreur Pydantic ; ne pas confondre le défaut implicite avec un champ fourni.
- [ ] Écrire `test_dates_are_normalized_not_relabelled` : `2026-11-04T10:00:00+02:00` devient 08:00 UTC ; une date naïve devient UTC ; dates inversées et fenêtre de durée nulle refusées. Tester NaN/infini, budgets nuls, limites négatives et formats inconnus.
- [ ] Écrire `test_basic_config_has_no_optional_filters` : absence de dates/équipes → `upload_after is None`, borne utilisateur `upload_before is None`, `team_filter is None`, `mode == 'backfill'`. La fin technique figée reste dans les bornes effectives séparées. `test_optional_filters_are_independent` couvre les quatre combinaisons et une seule borne de date. Aucun paramètre absent ne prend une valeur issue d'une ancienne campagne.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_config.py -q` et constater l'échec attendu.
- [ ] Étendre `CollectionConfig` avec les champs de la spec plus `incremental_overlap_hours=48`, `reinspect=False`, `shutdown_grace_s=60`. Défauts API conservateurs : pools 1/1, débit effectif 1/s ; les valeurs 2/8 et 2/s appartiennent au YAML parallèle explicite. Plafonds techniques proposés : découverte 1..8, téléchargement 1..64, débit fini 0 < x ≤ 20, fenêtres 1..30 jours, file 100..100000, timeout 1..120 s, tentatives 1..5. Budgets entiers positifs ; réservations 30..3600 s et heartbeat ≥1 s avec `heartbeat * 3 <= lease_duration`.
- [ ] Normaliser les formats en liste triée sans doublons, les dates en UTC, puis résoudre la borne haute effective au premier démarrage. Le fingerprint de **requête** conserve le marqueur `upload_before=None` : deux appels identiques ne divergent pas parce que `now` a changé. Stocker séparément requête normalisée, config effective et bornes figées. Les changements ultérieurs de pool ne modifient pas le fingerprint initial.
- [ ] Définir `TeamFilter(teams: list[list[str]])` (chaque composition exactement six IDs Showdown non vides) et `CollectionConfig.team_filter: TeamFilter | None = None`. Liste externe vide refusée, doublons de compositions normalisés ; ne pas transformer une composition en ensemble perdant sa multiplicité. Le chargement de `teams_file` dans le loader CLI avant validation de `CollectionConfig` sera implémenté à la tâche 2 bis ; l'API n'accepte que `team_filter` en contenu JSON. Fournir fichier et contenu simultanément est refusé. Les IDs triés et le contenu canonique participent au fingerprint, pas le chemin du fichier. Le contenu figé est persisté, jamais relu à la reprise.
- [ ] Accepter `max_new_replays` en entrée comme alias de `max_selected_replays` ; refuser les deux simultanément. Sortie canonique avec le nouveau nom et README explicitant la sémantique. Le compteur historique `new_replays` de résultat reste le nombre réellement nouvellement archivé ; ajouter `selected_replays` et `reused_replays`, sans réétiqueter les anciens résultats JSON.
- [ ] Définir `ResumeCollectionRequest(new_attempt: bool=False, budgets: CollectionBudgets | None=None)`, `CollectionBudgets(max_selected_replays: int, max_requests: int, max_duration_s: int)`, `ScaleCollectionRequest(discovery_workers: int, download_workers: int)` et `RetryFailedRequest(max_attempts: int=5, replay_ids: list[str] | None=None)` dans `schemas.py`.
- [ ] Rejouer `tests/test_collection_config.py tests/test_mc_schemas.py tests/test_formats_champions.py`, puis Ruff sur les fichiers modifiés. Commit : `feat: define parallel collection contracts and config`.

## Tâche 2 bis — Filtrage facultatif avant enregistrement

**Fichiers :** créer `collection_filters.py`, `tests/test_collection_filters.py`, `tests/fixtures/collection/teams.json` et de petits logs de preview versionnés ; modifier le loader `_load_collection_config` dans `cli.py`. Ce module est pur, ne télécharge rien et n'écrit aucun replay.

**Interfaces produites :**

```python
load_team_filter(path: Path) -> TeamFilter
extract_team_summary(payload: dict[str, Any]) -> TeamSummary
evaluate_selection(config: CollectionConfig, *, uploadtime: int | None,
                   summary: TeamSummary | None) -> SelectionDecision
```

- [ ] Écrire `test_no_team_filter_accepts_unknown_composition`, `test_match_p1_or_p2_ignores_order_and_sets`, `test_unknown_other_side_does_not_cancel_known_match`, `test_four_revealed_pokemon_are_not_a_team`, `test_distinct_starting_forms_do_not_match`, `test_team_and_dates_are_combined`, `test_absent_date_filter_does_not_reject_missing_date` et `test_empty_or_invalid_json_is_rejected`.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_filters.py tests/test_collection_config.py -q` et constater l'échec avant implémentation.
- [ ] Charger le JSON exact `{"teams": [[...six IDs...], [...six IDs...]]}`, refuser autres clés et identifiants non canoniques (chaîne ASCII minuscule alphanumérique non vide). Résoudre le chemin par rapport au YAML, pas au CWD modifié par le vendor. Les espèces/formes sont comparées par ID normalisé ; pas de résolution implicite de noms français ou surnoms. Enregistrer le contenu et son hash en configuration de campagne.
- [ ] Extraire la composition initiale depuis les six entrées de preview par joueur, avant les tours/transformations. Vérifier les variantes de protocole avec les fixtures et sources locales avant d'écrire le parseur : `_team_complete()` actuel attend des slots `p1a..p1f` et n'est pas une extraction de six espèces fiable. Ne pas recopier cette regex ni assimiler `switch`/Pokémon joués à un roster complet. Ne pas exiger OTS : une preview complète suffit. Si l'identité exacte des six n'est pas déterminable, retourner `None` pour ce côté, sans inventer sa forme.
- [ ] Évaluer les bornes fournies utilisateur comme `[after, before)` ; une borne seule est permise. Si une date est nécessaire mais manque, décision `date_unknown` ; si aucun filtre temporel utilisateur n'est fourni, ne pas exclure pour ce seul motif (les contrôles de pagination restent séparés). Si pas de filtre d'équipe, court-circuiter l'extraction/validation de composition. Sinon : p1 ou p2 correspondant → selected ; aucune correspondance et un côté inconnu → team_unknown ; deux côtés connus non correspondants → team_mismatch.
- [ ] Rejouer tests filtres/config. Commit : `feat: define optional date and team selection before archival`.

**Dépendances :** nécessite la tâche 2 ; les tâches 3 et suivantes doivent intégrer ces décisions. Le plan compte désormais **15 tâches**, la numérotation 2 bis évitant de casser les renvois existants.

## Tâche 3 — Catalogue SQLite et initialisation non destructive

**Fichiers :** créer `archive_catalog.py`, `tests/test_archive_catalog.py` ; modifier `settings.py`. Schéma SQL versionné dans `archive_catalog.py`, sans outil de migration supplémentaire.

**Interfaces produites :** `initialize_catalog(root: Path) -> None`, `catalog_connection(root: Path) -> ContextManager[sqlite3.Connection]`, `catalog_has_replay(root: Path, replay_id: str, content_sha256: str | None = None) -> bool`, `catalog_versions(root: Path, replay_id: str) -> list[ArchiveRef]`.

Schéma initial, `PRAGMA user_version=1` ; clés étrangères actives et contraintes `CHECK` pour les états. Timestamps UTC persistants en secondes ; ajouter les informations boot/monotonic pour les échéances locales (tâche 5).

| Table | Clé / informations à persister |
| --- | --- |
| `replays` | `replay_id PRIMARY KEY`, `source_format`, `uploadtime`, `first_published_at` ; créée seulement lors de publication validée. |
| `replay_versions` | `publication_seq INTEGER PRIMARY KEY`, ID/hash unique, `log_sha256`, chemin relatif, format fournisseur, uploadtime, provenance et date ; FK vers `replays`. |
| `campaigns` | ID public `job-*`, fingerprint, config demandée/effective, phase, `active_slot UNIQUE` nullable (1 si active), attempt courant, pools désirés, superviseur/heartbeat, raison d'arrêt. |
| `attempts` | ID, campagne, budgets, début/deadline, autorisations HTTP consommées, IDs sélectionnés dans cette tentative, nouveautés publiées et places réservées, fin/raison. |
| `windows` | ID, format exact, bornes, génération de scan, état, curseur, dernière signature, attempts du curseur, prochaine tentative, propriétaire/token/expiration ; unicité format/bornes/génération. |
| `campaign_windows` | Campagne/fenêtre unique : rattachement pour statistiques et reprise. |
| `coverage` | Segments réellement explorés, format, génération et preuve de terminaison ; trous/bloquages séparés et persistants. |
| `download_tasks` | `replay_id PRIMARY KEY`, format/date, état incluant `examined_only`, propriétaire/token/expiration, attempts durables, retry_at, dernière erreur, réservation de place/attempt. Un rejet d'équipe ne rend pas cette tâche indisponible pour toujours. |
| `replay_summaries` | ID unique découvert, format/date, compositions connues par côté, parser_version, hash observé ; aucune FK exigeant un objet archivé, aucun corps de replay. Garder aussi les IDs jamais téléchargés pour réévaluer la couverture. |
| `campaign_replays` | Campagne/ID unique, `decision`, raison, filter_sha256, version retenue nullable, attempt de première sélection et attributaire de première publication ; une entrée de suivi rejetée n'appartient pas aux résultats sélectionnés. |
| `workers` | ID, campagne/attempt, rôle, PID, boot/start ticks/token, état, heartbeat. |
| `publication_intents` | ID replay/hash unique, token, attempt, chemins temporaire/final, métadonnées nécessaires, phase ; seulement tant que récupération nécessaire. |
| `request_permits` | ID, attempt, tâche et rôle, état réservé/envoyé/terminé/incertain, code HTTP, dates ; compacter après agrégation. |
| `host_limits` | Hôte unique, débit configuré/effectif, prochain créneau, pause globale et repères d'horloge. |
| `metric_buckets` | Agrégats par minute/attempt/rôle, bornés ; compteurs durables et latences cumulées. |

- [ ] Écrire `test_empty_init_is_idempotent_and_preserves_other_roots` : initialiser deux fois, aucun replay, fichier modèle/dataset témoin inchangé. `test_legacy_state_is_not_imported` : index/journaux legacy présents → aucune ingestion automatique ; si objets legacy non vides sans catalogue, erreur explicite de stockage incompatible, aucun effacement.
- [ ] Écrire `test_duplicate_version_is_rejected` et `test_catalog_lookup_is_indexed` : contrainte unique et `EXPLAIN QUERY PLAN` montrant une recherche par index pour un ID. `test_unfixed_sqlite_is_rejected` teste la fonction de contrôle avec versions injectées, sans supposer celle du développeur.
- [ ] Exécuter `uv run --locked pytest tests/test_archive_catalog.py -q` avant implémentation.
- [ ] Implémenter l'initialisation avec verrou fichier de bootstrap et transaction ; vérifier version, `foreign_keys=ON`, WAL effectivement obtenu, `busy_timeout=5000`, `synchronous=FULL`. Une version de schéma inconnue échoue avec message ; aucun `DROP TABLE` automatique.
- [ ] Indexer réservations par état/prochaine tentative, expiration, rattachements et liste archive par format/date/ID. Ne pas utiliser `SELECT *` ou `COUNT(*)` sur toute la file à chaque GET ; maintenir les compteurs nécessaires aux transactions.
- [ ] Rejouer les tests et Ruff. Commit : `feat: initialize transactional replay catalog`.

## Tâche 4 — Campagnes, tentatives et commandes persistantes

**Fichiers :** créer `collection_store.py`, `tests/test_collection_campaigns.py` ; utiliser les types et tables des tâches 2–3.

**Interfaces produites (`CollectionStore`) :**

```python
start_campaign(config: CollectionConfig) -> CampaignView
get_campaign(campaign_id: str) -> CampaignView
command(campaign_id: str, action: Literal['pause', 'stop', 'resume'],
        *, new_budgets: CollectionBudgets | None = None) -> CampaignView
scale(campaign_id: str, *, discovery_workers: int, download_workers: int) -> CampaignView
retry_failed(campaign_id: str, request: RetryFailedRequest) -> int
```

- [ ] Écrire `test_simultaneous_identical_start_without_upper_bound_returns_one_campaign` avec deux processus/barrière et même racine temporaire : un ID et une borne effective figée. `test_different_active_config_conflicts` : une seule campagne active ; aucun worker supplémentaire.
- [ ] Écrire `test_pause_resume_keeps_attempt_budget` et `test_new_attempt_requires_explicit_budgets` : compteurs inchangés pour pause/reprise ; après budget atteint, sans nouveaux budgets → `Conflict`, avec budgets → nouvel attempt, mêmes tâches et compteurs cumulés conservés.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_campaigns.py -q` et constater les échecs.
- [ ] Implémenter sous `BEGIN IMMEDIATE` la décision rejoindre/créer/refuser, avec `active_slot=1`. Phases actives : `queued`, `running`, `pausing`, `paused`, `stopping` ; terminales : `completed`, `budget_reached`, `stopped`, `failed`, `interrupted`. Une pause conserve le slot ; stop/fin le libère après extinction des workers.
- [ ] Une commande HTTP/CLI écrit une intention ; le superviseur effectue la transition. `stop` répété est sans effet ; `resume` d'une campagne déjà running retourne son état. Une tentative ne peut être remplacée avant la fin/réconciliation des réservations de la précédente. La durée est une échéance murale depuis le début, pause incluse, non un compteur réinitialisé par pause.
- [ ] À la création, parcourir par lots les IDs/metadonnées persistés compatibles avec formats/dates, y compris les anciens rejets, pour appliquer la sélection courante. Une couverture historique déjà complète évite le GET des pages, pas cette réévaluation. Un filtre d'équipe différent participe au fingerprint ; une reprise conserve l'ancien contenu figé. Tester `test_new_filter_revisits_prior_rejections_without_rediscovery` et `test_modified_team_file_does_not_change_resume`.
- [ ] `retry_failed` ne réinitialise pas l'historique : enregistrer un nouveau cycle autorisé, conserver attempts cumulées et erreurs. Opération seulement sur campagne paused/stopped/failed/budget_reached ; ne lance pas implicitement un worker. IDs omis → sélection SQL des tâches échouées rattachées à la campagne.
- [ ] Rejouer les tests ; commit : `feat: persist collection campaigns and execution attempts`.

## Tâche 5 — File durable et réservations exclusives

**Fichiers :** modifier `collection_store.py` ; créer `tests/test_collection_leases.py`.

**Interfaces produites (`CollectionStore`) :**

```python
enqueue_replays(campaign_id: str, items: Sequence[SearchItem]) -> int
claim_download(campaign_id: str, worker_id: str) -> ClaimResult
claim_window(campaign_id: str, worker_id: str) -> ClaimResult
renew_lease(lease: TaskLease) -> TaskLease
release_lease(lease: TaskLease, *, retry_at: float | None = None) -> None
fail_lease(lease: TaskLease, *, permanent: bool, unavailable: bool, error: str) -> None
expired_leases(campaign_id: str, *, limit: int = 100) -> list[TaskLease]
record_selection(lease: TaskLease, decision: SelectionDecision,
                 summary: TeamSummary | None) -> None
```

- [ ] Écrire `test_eight_processes_claim_each_id_once` : 1 000 IDs, huit processus, ensemble de tâches attribuées sans doublon. `test_overlapping_discovery_enqueues_once` : même page deux fois et déjà-archivés → taille de file correcte.
- [ ] Écrire `test_stale_token_cannot_renew_release_or_finish`, `test_attempts_survive_expiry`, `test_boot_change_expires_old_lease`, `test_clock_rollback_does_not_create_second_owner`. Injecter l'horloge ; ne pas faire dormir les tests 120 s.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_leases.py -q` avant implémentation.
- [ ] Implémenter chaque réservation dans une transaction courte : sélectionner une tâche éligible, réserver sa place de sélection, écrire propriétaire + nouveau token aléatoire + échéance puis retourner la tâche. La prise d'une tâche est un `UPDATE` conditionnel, jamais un simple `SELECT` suivi d'une mise à jour hors transaction.
- [ ] Utiliser l'horloge monotone commune à un boot pour l'expiration locale ; stocker aussi l'échéance UTC pour affichage/reprise. Au nouveau boot, les anciens workers ne sont plus vivants. Revalider identité et jeton avant toute libération/publication. Tous les délais d'attente sont interruptibles par commande/heartbeat.
- [ ] Réserver une place de sélection avant de traiter un ID pour la campagne : `selected_in_attempt + reserved < max_selected_replays`. Consommer la place uniquement lors du premier rattachement retenu, même si l’objet existe déjà ; la rendre après rejet. Transférer/réconcilier une place abandonnée avant réattribution, sans la compter deux fois. Si la place est occupée par une intention de publication, récupérer celle-ci avant de rendre la place libre.
- [ ] `record_selection` vérifie le token, persiste uniquement la décision et le résumé technique ; rejet → `examined_only`, libération de place sans incrément de sélection. Une décision favorable reste provisoire jusqu'à publication/rattachement atomique. Un objet déjà présent est évalué localement et rattaché avec une place réservée, sans GET ni double incrément. Une erreur réseau ne doit pas se transformer en rejet d'équipe. Tester `test_rejection_releases_selection_slot_but_not_request_budget`.
- [ ] Ne pas réinitialiser attempts à chaque `claim`. Les incrémenter avant chaque tentative HTTP autorisée (tâche 6), pas à chaque polling. Pour un téléchargement, le plafond porte sur l'ID et son cycle d'inspection : une réinspection nécessaire après un rejet correctement traité démarre un nouveau cycle, avec compteurs cumulés conservés ; une simple expiration ou un échec ne réinitialise jamais le cycle. Une tâche réellement échouée requiert toujours `retry-failed` pour rouvrir son plafond. Pour la découverte, il porte sur **la page au curseur courant** : une page validée remet le compteur de retries de page à zéro dans la transaction qui avance le curseur, tout en conservant le total de requêtes de la fenêtre. Les états permanents et `unavailable` ne sont pas réclamés automatiquement.
- [ ] Rejouer les tests ; commit : `feat: coordinate replay and discovery task leases`.

## Tâche 6 — Limiteur global, budgets et client à tentative unique

**Fichiers :** créer `collection_limits.py`, `tests/test_collection_limits.py`, `tests/test_collection_http.py` ; modifier `collection_client.py`, `collection_store.py`, tests HTTP existants de `test_collection.py`.

**Interfaces produites :**

```python
# collection_limits.py
acquire_request_permit(store: CollectionStore, lease: TaskLease,
                       kind: Literal['search', 'fetch']) -> PermitResult
record_request_result(store: CollectionStore, permit: RequestPermit,
                      *, status_code: int | None, elapsed_s: float,
                      retry_after: str | None = None) -> None
# collection_client.py, méthodes ShowdownReplayClient
search_once(source_format: str, *, before: int) -> DiscoveryPage
fetch_once(replay_id: str) -> dict[str, Any]
```

- [ ] Écrire `test_global_rate_across_eight_workers` avec fausse horloge, plafond 2/s et burst 1 : intervalles d'autorisation ≥0,5 s ; recherche et fetch se partagent le même compteur. `test_retry_also_consumes_budget` : maximum 5 autorisations, retry inclus, sixième refusée.
- [ ] Écrire `test_crash_after_permit_does_not_refund_request`, `test_parallel_new_replay_budget_never_overshoots`, `test_deadline_prevents_new_requests`, `test_retry_after_date_and_seconds_pause_every_worker`, `test_invalid_retry_after_uses_backoff` et `test_stale_lease_cannot_take_permit`.
- [ ] Exécuter les deux nouveaux fichiers de tests et constater l'échec.
- [ ] Dans une transaction, vérifier campagne/attempt/lease/attempts/durée, pause fournisseur et prochain créneau ; si attente nécessaire, retourner `wait_s` sans garder de verrou ; si autorisé, consommer le budget, incrémenter attempts de tâche et créer le permis. Aucun crédit restitué après crash ambigu : afficher séparément autorisations consommées, requêtes confirmées et incertaines. Le compteur budgétaire reste conservateur.
- [ ] Éviter une boucle de centaines de transactions/s quand aucun créneau n'est disponible : attendre le délai retourné, par tranches permettant heartbeat/stop. Au changement de boot, conserver la pause UTC fournisseur et reconstruire les repères monotones. Un recul d'horloge ne doit jamais augmenter le débit autorisé.
- [ ] Implémenter la régulation 429 : pause commune au moins jusqu'au `Retry-After`, débit effectif divisé par deux avec plancher `min(débit configuré, 0.1)`, remontée de 10 % après chaque minute sans 429, bornée au débit configuré. Backoff fallback `min(2 ** (attempts_used - 1), 60)` + jitter injectable 0..1 s, jamais en dessous d'un Retry-After valide. Les 5xx/timeouts utilisent le backoff de tâche sans multiplier un retry caché du client.
- [ ] Retirer la boucle interne de cinq tentatives HTTP : un appel = une tentative. Désactiver les redirections automatiques pour que chaque requête soit budgétée ; une redirection inattendue produit une erreur fournisseur explicite. Conserver hôte fixé côté service, timeouts, client réutilisé par processus, `close()` en `finally`.
- [ ] Classer 429, 5xx et réseau comme temporaires ; 404/410 comme indisponibles ; 401/403 comme erreur d'accès globale nécessitant arrêt explicite plutôt qu'un million d'échecs individuels ; JSON/ID/format invalides comme payload invalide. Pour une recherche, une erreur ne signifie jamais « page vide ».
- [ ] Rejouer tests ciblés + tests client existants adaptés ; commit : `feat: enforce shared collection budgets and provider backoff`.

## Tâche 7 — Publication atomique et récupération après panne

**Fichiers :** créer `collection_publication.py`, `tests/test_collection_publication.py` ; modifier `archive_service.py`, `archive_catalog.py`.

**Interfaces produites :**

```python
prepare_replay(root: Path, replay_id: str, payload: dict[str, Any],
               source: ArchiveSource) -> PreparedReplay
publish_replay(store: CollectionStore, prepared: PreparedReplay,
               *, lease: TaskLease | None) -> PublicationResult
recover_publications(store: CollectionStore, *, limit: int = 100) -> int
```

`lease=None` est réservé à l'import explicite existant, qui respecte le verrou de maintenance et les mêmes contraintes de publication. Un import concurrent ne consomme pas un budget HTTP et ne double pas les nouveautés attribuées au collecteur.

- [ ] Écrire `test_same_id_hash_publishes_once_under_race`, `test_different_hash_keeps_both_versions`, `test_import_racing_with_download_has_one_first_publication`. Compter les replays distincts et versions séparément.
- [ ] Ajouter des points d'injection de panne : `test_recover_after_temp_fsync`, `test_recover_after_rename_before_sql_commit`, `test_disk_full_does_not_mark_archived`, `test_reassigned_worker_cannot_publish`, `test_missing_final_object_is_not_success`. Tests par sous-processus interrompu lorsque la panne doit simuler une perte de connexion SQL ; tous dans `tmp_path`.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_publication.py -q` avant implémentation.
- [ ] Conserver le calcul canonique de hash existant et l'enveloppe version 1. Valider ID/log/format, écrire un temporaire unique sur le même FS, flush/fsync, hors transaction. Enregistrer ensuite une intention durable avec références et token sous transaction validant la lease ; fsync n'est pas une preuve de commit SQL.
- [ ] Pour une collecte, cette préparation fichier n'est appelée **qu'après décision favorable** du filtre facultatif. Persister sélection de campagne, version retenue et compteur de sélection dans la transaction finale ; ils doivent survivre ou échouer ensemble. `lease=None` garde la sémantique d'import explicite et ne rattache rien arbitrairement à une collecte active. Tester `test_rejected_payload_never_reaches_publication_or_logs` : aucun objet/temporaire, aucune intention, aucune chaîne du log dans journaux ou DB après rejet.
- [ ] Transaction finale `BEGIN IMMEDIATE` : revalider token et réservation, vérifier fichier destination éventuel, publier sans écraser un objet valide, synchroniser les répertoires nécessaires, insérer replay/version, passer tâche à archived, attribuer les compteurs, libérer sa place, retirer l'intention, commit. Cette courte transaction peut contenir la finalisation fichier, jamais HTTP/hashing/sérialisation volumineuse.
- [ ] Sur crash avant commit, l'intention préalablement commitée permet de vérifier le fichier final ou temporaire et terminer avec une propriété récupérée, sans GET. La récupération et l'expiration de lease utilisent les mêmes transactions ; un ancien token ne retrouve aucun droit. Une intention sans fichier valide rend la tâche reprenable et libère correctement la place, avec erreur tracée.
- [ ] Un fichier publié mais non indexé n'est jamais visible à la liste/snapshot avant récupération. Un fichier temporaire sans intention peut être nettoyé après une durée supérieure à la réservation et vérification qu'aucun worker vivant ne le possède. Ne pas lancer un scan récursif des deux millions d'objets à chaque redémarrage : utiliser les intentions et un répertoire staging borné pour ces vérifications.
- [ ] Adapter `archive_replay()` pour appeler ce protocole et retourner son `ArchiveRef` actuel. Déduire les conflits de l'ensemble des versions, pas d'un flag calculé avant verrou. Rejouer `test_archive.py` + nouveaux tests ; commit : `feat: publish replay objects idempotently with crash recovery`.

## Tâche 8 — Découverte par fenêtres et couverture persistante

**Fichiers :** créer `collection_discovery.py`, `tests/test_collection_discovery.py` ; modifier `collection_store.py`, `collection_client.py`.

**Interfaces produites :**

```python
# collection_discovery.py
partition_range(source_format: str, start: int, end: int,
                *, partition_days: int) -> Iterator[WindowBounds]
seed_discovery(store: CollectionStore, campaign_id: str) -> int
process_search_page(store: CollectionStore, lease: TaskLease,
                    page: DiscoveryPage) -> None
# CollectionStore
commit_discovery_page(lease: TaskLease, page: DiscoveryPage) -> None
```

- [ ] Écrire `test_windows_are_disjoint_on_utc_grid` : grille ancrée à epoch 0, taille 7×86400, extrémités tronquées, aucun trou ; fin exclusive. `test_two_discovery_workers_do_not_scan_same_window` avec barrières interprocessus.
- [ ] Écrire `test_page_ids_and_cursor_commit_together` avec crash/rollback : soit tous les IDs et le curseur nouveau, soit ancien curseur. `test_same_timestamp_stall_is_reported`, `test_empty_middle_window_does_not_end_history`, `test_out_of_window_id_is_not_downloaded`, `test_partial_budget_does_not_complete_window`. Ajouter `test_ten_successful_pages_are_not_limited_to_five_attempts` : dix pages réussies avec `max_attempts=5` terminent la fenêtre ; cinq échecs du même curseur la bloquent avec erreur et couverture incomplète.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_discovery.py -q` avant implémentation.
- [ ] Adapter `search_once()` au contrat vérifié en tâche 1. Garder les couples ID/timestamp, l'ordre fournisseur et l'indication d'épuisement. Ne pas bricoler `before -= 1` pour résoudre un timestamp saturé. À défaut de continuation sûre, publier les IDs connus puis marquer la fenêtre bloquée avec son timestamp/signature.
- [ ] Créer les fenêtres à la demande : au plus `2 * discovery_workers` non terminées matérialisées en avance ; un générateur évite une liste de décennies de fenêtres. Pour `upload_after=None`, partir de la borne haute vers le passé. Une réponse valide vide à une recherche `before` sans borne inférieure serveur prouve l'absence de résultats plus anciens selon cette interface ; enregistrer cette preuve. Une réponse contenant seulement des IDs plus vieux que la fenêtre prouve uniquement que la fenêtre est vide, et permet d'avancer vers ces dates sans certifier des données inaccessibles.
- [ ] Affecter un seul propriétaire valide par fenêtre. `commit_discovery_page` vérifie le token et fait insertion dédupliquée, rattachement, compteurs, curseur et couverture dans la même transaction. Un ID sans timestamp valide ne doit pas faire déclarer la page complètement couverte : conserver l'erreur et marquer couverture incomplète.
- [ ] Réutiliser les segments complètement couverts des campagnes précédentes ; calculer les intervalles restants, y compris si leurs bornes étaient tronquées. Une génération `reinspect=True` force une redécouverte de la plage sans toucher à la déduplication globale des téléchargements.
- [ ] Ne pas assimiler « période couverte » à « tous les replays conservés ». L'insertion transactionnelle d'une page renseigne `replay_summaries` pour chaque ID/date découvert, même sans GET ; une campagne aux filtres différents réévalue cet index par lots (tâche 4). `reinspect=True` permet aussi de refaire une inspection si les métadonnées connues sont insuffisantes ou devenues obsolètes, avec budgets inchangés.
- [ ] Lorsque la file atteint 10 000 tâches non terminales, suspendre les recherches avant le GET suivant ; une page déjà reçue est conservée intégralement (dépassement autorisé limité aux pages en vol, mesuré). Ne pas perdre la moitié d'une page pour respecter exactement le seuil. Répartir équitablement les formats BO1/BO3 au lieu d'épuiser le premier format avant le second.
- [ ] Rejouer tests découverte + limites + leases ; commit : `feat: partition replay discovery and persist coverage`.

## Tâche 9 — Boucles des workers et mode incrémental

**Fichiers :** créer `collection_workers.py`, `tests/test_collection_workers.py`, `tests/test_collection_incremental.py` ; modifier `collection_discovery.py`, `collection_store.py` ; remplacer les tests basés sur `_save_progress` dans `test_collection_resume.py`.

**Interfaces produites :** `run_collection_worker(root: Path, campaign_id: str, worker_id: str, role: Literal['discovery','download']) -> int` dans `collection_workers.py` ; `incremental_bounds(store: CollectionStore, source_format: str, *, end: int, overlap_hours: int) -> WindowBounds` dans `collection_discovery.py`.

- [ ] Écrire `test_download_worker_skips_archived_without_http`, `test_download_worker_retries_without_losing_task`, `test_heartbeat_continues_during_slow_http`, `test_empty_queue_while_discovery_running_is_not_completed`, `test_stop_interrupts_rate_limit_wait`.
- [ ] Écrire `test_incremental_without_watermark_covers_last_48h`, `test_incremental_unchanged_downloads_zero`, `test_incremental_ten_new_ids_downloads_ten`, `test_old_pending_tasks_are_adopted_without_rescan` et `test_discovery_hole_survives_newer_watermark`.
- [ ] Exécuter les deux nouveaux fichiers et `test_collection_resume.py`, constater l'échec pertinent.
- [ ] Implémenter les boucles : lire commande → demander tâche → réévaluer métadonnées/objet existant → si contenu nécessaire, autorisation puis HTTP unique → filtrer → publier/rattacher si retenu, sinon abandonner le contenu et persister seulement la décision/résumé. Installer un heartbeat indépendant pendant HTTP, avec sa propre connexion ; perdre le token interdit publication et arrête cette exécution. Les renouvellements de leases et publications déjà engagées restent permis pendant `pausing/stopping`, alors que nouveaux claims et GET sont interdits. Fermer client/connexions/thread en sortie.
- [ ] Ajouter `test_basic_collection_archives_unknown_teams`, `test_filtered_collection_keeps_only_matches`, `test_unfiltered_campaign_downloads_previously_rejected_body`, `test_new_team_filter_uses_cached_summary_without_get_for_known_mismatch`. Une composition complète peut prouver un rejet sans GET ; une composition inconnue n'est pas un rejet universel. Si un ancien rejet devient admissible et que le corps est absent, retélécharger sous nouvelle lease. Un même replay ne doit pas reboucler dans la même campagne après sa décision terminale.
- [ ] Une boucle n'interprète jamais `ClaimResult(lease=None)` comme fin de campagne : seul le superviseur agrège découverte, retries différés, intentions et téléchargements. Chaque worker écrit son journal, pas le `state.json` du parent.
- [ ] Incrémental par format source : fenêtre `[dernier_repère - 48 h, fin_figée)` ; premier lancement `[fin - 48 h, fin)`. Rechercher à nouveau ces fenêtres même si déjà couvertes (nouvelle génération) pour voir les nouveaux IDs. Définir le repère sur la fin du segment récent dont la découverte a réussi, jamais sur le dernier replay téléchargé ; conserver séparément tous les trous connus.
- [ ] Lors d'une nouvelle campagne, rattacher par requête SQL les tâches inachevées compatibles avec ses formats et **filtres utilisateur actuels**. Elles restent globales et gardent attempts/retry_at ; ne pas charger toute la file en mémoire. En mode incremental, reprendre aussi les tâches de ces formats déjà découvertes avant la fenêtre de recouvrement technique, mais jamais en dehors des dates explicitement demandées ; réévaluer le filtre d'équipe courant. Un format non demandé n'est jamais adopté implicitement.
- [ ] Remplacer la reprise par fichiers `pending/cursors` par les tests équivalents sur SQLite ; conserver les tests de métier utiles, retirer uniquement les assertions de structure abandonnée. Aucune lecture automatique des anciennes progressions présentes dans le vrai stockage.
- [ ] Rejouer les tests ciblés ; commit : `feat: run cooperative collection workers and incremental updates`.

## Tâche 10 — Superviseur et coexistence des jobs

**Fichiers :** créer `collection_supervisor.py`, `tests/test_collection_supervisor.py` ; modifier `jobs.py`, `job_service.py`, `worker.py`, `main.py`, éventuellement `dataset_service.py`/`training_service.py` ; adapter `tests/test_worker_lifecycle.py` et `tests/test_jobs.py`.

**Interfaces produites :** `ensure_supervisor(root: Path, campaign_id: str) -> None`, `run_collection_supervisor(root: Path, campaign_id: str, *, lock_fd: int) -> int` dans `collection_supervisor.py` ; `acquire_resource_lock(resource: Literal['collection','compute'], job_id: str) -> int` et `release_resource_lock(fd: int) -> None` dans `jobs.py`. Adapter `job_service.run_job(job_id: str, *, lock_fd: int | None = None) -> None` : le dispatcher possède et libère le fd compute transmis ; les services prepare/training ne tentent plus de libérer un singleton global inexistant. Le superviseur possède et libère son fd collection.

- [ ] Écrire `test_two_launchers_spawn_one_supervisor`, `test_api_exit_does_not_release_child_lock`, `test_api_restart_keeps_live_workers`, `test_pid_reuse_is_not_live_worker`, `test_dead_supervisor_is_reconciled_without_duplicate_children`, `test_scale_changes_pool_without_new_windows`.
- [ ] Remplacer `test_only_one_long_job_across_types` par deux tests : `test_collection_and_one_compute_job_can_coexist`, `test_second_compute_job_conflicts`. Ajouter `test_import_prepare_keeps_compute_lock_after_api_restart` avec un vrai enfant minimal.
- [ ] Exécuter les tests ciblés et constater les échecs avant modification.
- [ ] Créer des fichiers de verrou stables `.collection.lock` et `.compute.lock`. Sous verrou court de lancement, rejoindre le superviseur vivant ou réserver le slot et lancer via `sys.executable`, liste d'arguments et `pass_fds=(fd,)`. **Le parent ferme sa copie du fd sans `LOCK_UN`** ; l'enfant propriétaire libère à sa sortie. Ne jamais unlink le fichier verrou. Supprimer le singleton global `_lock_fd` comme source de vérité multi-jobs ; les alias training restent orientés compute.
- [ ] Ajouter à `worker.py` des rôles explicites superviseur/worker collecte et l'argument fd hérité, tout en gardant le chemin prepare/training. Démarrer le superviseur avec `start_new_session=True` et sorties redirigées vers son journal pour qu'une fin de terminal/API ne l'arrête pas. Le superviseur lance les enfants via `subprocess.Popen`, sans leur transmettre le verrou de campagne. Tous utilisent la même racine absolue de stockage. Éviter l'import du module CLI et son environnement ML dans les enfants.
- [ ] Persister identité Linux `(boot_id, pid, /proc/<pid>/stat starttime)` et token de lancement ; ne jamais tuer un PID sans correspondance. Heartbeat seul n'autorise pas à démarrer un deuxième superviseur si le verrou est toujours détenu. Les workers contrôlent aussi l'identité/heartbeat du superviseur et s'arrêtent après son abandon ; le superviseur de remplacement attend/arrête les enfants identifiés, puis réconcilie les leases avant de relancer.
- [ ] Scale up crée seulement les travailleurs manquants ; scale down demande le drainage des workers sélectionnés. Pause bloque les nouveaux claims/GET, termine publications et rend les leases inutilisées ; stop fait de même puis termine les enfants. Après 60 s de grâce, terminer uniquement les enfants possédés, puis récupérer par intentions/leases. Heartbeat et limitation de débit doivent rester actifs durant le drainage.
- [ ] Le superviseur, seul, écrit la projection `state.json` du job parent. L'état canonique collecte reste SQLite. Mapping public : phases actives → `JobStatus.running/queued` ; paused/stopped → nouveaux états explicites à ajouter ; budget_reached/completed → `completed` avec raison distincte ; couverture trouée jamais présentée comme exhaustive. Persister erreurs et fin dans `finally`, avant de relancer une exception.
- [ ] Modifier `reconcile_jobs()` pour distinguer collectes cataloguées et jobs de calcul. Ne pas réécrire un worker vivant au startup/shutdown de FastAPI. Un ancien job de collecte présent seulement en JSON reste consultable historiquement, mais n'est jamais relancé ni importé dans la nouvelle file.
- [ ] Rejouer tous les tests de lifecycle et nouveaux tests superviseur ; commit : `feat: supervise collection pools independently from compute jobs`.

## Tâche 11 — Façade, API et CLI de contrôle

**Fichiers :** modifier `collection_service.py`, `api/collection.py`, `cli.py`, `job_service.py` ; créer `tests/test_collection_cli.py` ; compléter `tests/test_collection_api.py`, `tests/test_collection_campaigns.py`.

**Interfaces produites (dans `collection_service.py`) :**

```python
submit_collection(config: CollectionConfig) -> JobRecord
get_collection(job_id: str) -> CampaignView
pause_collection(job_id: str) -> CampaignView
stop_collection(job_id: str) -> CampaignView
resume_collection(job_id: str, request: ResumeCollectionRequest) -> CampaignView
scale_collection(job_id: str, request: ScaleCollectionRequest) -> CampaignView
retry_failed_collection(job_id: str, request: RetryFailedRequest) -> int
```

Les fonctions délèguent au store, puis démarrent/réveillent le superviseur si nécessaire. Pas d'algorithme SQL dupliqué dans les routes.

- [ ] Écrire `test_post_returns_202_without_waiting_for_network`, `test_double_post_returns_same_job`, `test_different_active_post_returns_409`, `test_controls_wrong_kind_return_404`, `test_resume_after_budget_requires_new_attempt`, `test_unknown_field_is_422`.
- [ ] Écrire `test_cli_start_loads_config_from_invocation_directory`, `test_cli_status_and_controls_share_service`, `test_old_collection_yaml_remains_valid`. Mock du lancement superviseur pour les tests unitaires API ; garder aussi un test d'intégration réel en tâche 14.
- [ ] Exécuter les tests API/CLI avant implémentation.
- [ ] Garder `POST /collection/jobs` (202), `GET /collection/jobs/{id}`, `GET /collection/jobs/{id}/logs`. Ajouter POST `/{id}/pause`, `/stop`, `/resume`, `/scale`, `/retry-failed` (202) et GET `/{id}/workers`, `/{id}/workers/{worker_id}/logs`. Les GET conservent les champs actuels `JobRecord` et ajoutent la progression structurée ; ne pas casser le client existant avec une enveloppe totalement différente.
- [ ] Ajouter CLI `pause`, `stop`, `resume`, `scale`, `retry-failed` avec `--job-id`. `resume --new-attempt --budgets chemin.yaml` correspond exactement à `ResumeCollectionRequest`; sans `--new-attempt`, tout `--budgets` est refusé. `start --config` accepte `reinspect: true` pour une réinspection explicite. État JSON déterministe en stdout, erreurs en stderr et code non nul.
- [ ] Ajouter GET `/collection/jobs/{id}/replays` paginé, limité à `decision=selected`, et POST `/collection/jobs/{id}/snapshot`, qui figent les références effectivement retenues. Conserver les routes `/archive` comme vues de l'ensemble de l'archive. Tester `test_campaign_listing_and_snapshot_exclude_other_campaign_objects`. Afficher `team_filter=null` et dates utilisateur nulles comme filtres inactifs, distincts des bornes techniques.
- [ ] Retirer les chemins de lancement `submit_job(JobKind.collection)` qui reprendraient le verrou compute ; les rediriger vers la façade sans récursion. Remplacer l'ancienne boucle `collect_replays()` et ses écritures JSONL/progression ; ne pas maintenir deux moteurs de collecte concurrents. Déplacer ses tests utiles vers workers/discovery.
- [ ] Afficher pour un job legacy consulté que sa source a pu être supprimée ; une tentative de resume renvoie une erreur explicite « ancien job non reprenable, lancer une nouvelle collecte », jamais une campagne vide portant le même ID.
- [ ] Rejouer API/CLI/jobs ; commit : `feat: expose parallel collection controls through API and CLI`.

## Tâche 12 — Lecteurs archive, snapshots et maintenance

**Fichiers :** modifier `archive_service.py`, `archive_catalog.py`, `cli.py`, `api/archive.py` si nécessaire ; compléter `tests/test_archive.py`, `test_archive_snapshots.py`, `test_archive_api.py`, `test_dataset_publication.py` ; créer `tests/test_archive_maintenance.py`.

**Interfaces conservées :** `archive_replay`, `is_replay_archived`, `list_replays`, `create_snapshot`, `read_snapshot`, `import_legacy_logs`, `read_archived_object` avec signatures existantes. `rebuild_index_from_objects() -> int` reconstruit désormais le catalogue des objets, pas les campagnes. Ajouter `export_catalog_jsonl(root: Path, destination: Path) -> int`, `backup_catalog(root: Path, destination: Path) -> None` dans `archive_catalog.py`.

**Interfaces supplémentaires :** `list_campaign_replays(campaign_id: str, *, offset: int=0, limit: int=100) -> tuple[list[ArchiveRef], int]` et `create_campaign_snapshot(campaign_id: str) -> ArchiveSnapshot` dans `archive_service.py`. Partager la construction du snapshot avec `create_snapshot` ; le sous-ensemble de campagne est une jointure sur ses décisions retenues et versions figées, pas un refiltrage approximatif de toute l'archive.

- [ ] Écrire `test_archive_hot_paths_do_not_read_jsonl` : rendre toute lecture du vieux fichier impossible et vérifier publication/présence/liste/snapshot. `test_list_filters_use_source_format_and_dates` : anciens filtres logiques et formats fournisseurs produisent le même résultat attendu.
- [ ] Écrire `test_snapshot_while_collecting_freezes_versions`, `test_snapshot_conflict_requires_explicit_hash`, `test_prepared_dataset_unchanged_while_archive_grows`, `test_rebuild_requires_exclusive_maintenance`, `test_backup_restores_committed_wal_content`.
- [ ] Exécuter ces tests avant modification.
- [ ] Remplacer les scans JSONL par requêtes indexées. `load_index_entries()` peut rester uniquement comme helper d'export/compatibilité hors chemin courant ; aucune utilisation dans `_existing_versions`, `is_replay_archived` ou le collecteur. `list_replays` applique SQL LIMIT/OFFSET avec ordre stable et ne charge pas tout le catalogue.
- [ ] Pour snapshot : lire dans une transaction courte la borne `max(publication_seq)` ; les versions étant immuables, paginer ensuite les références bornées par cette séquence et les filtres figés. Déduire les conflits uniquement des versions sous cette borne, puis calculer métadonnées hors transaction. Ne pas garder un lecteur WAL ouvert pendant des minutes de parsing. Le format `ArchiveSnapshot` reste celui attendu par le Dataset Builder ; aucun fichier référencé ne peut être supprimé en parallèle par la maintenance.
- [ ] Introduire `.maintenance.lock` : collecte, import et snapshot prennent un verrou partagé pendant leurs opérations ; reconstruction prend l'exclusif sans attendre indéfiniment. Ne pas mélanger cet ordre de verrouillage : maintenance d'abord, puis transaction SQL. La reconstruction parcourt les objets une fois par maintenance explicite, préserve les tables de coordination et signale les fichiers invalides ; elle ne prétend pas recréer les curseurs perdus.
- [ ] Export JSONL explicite par lots et renommage atomique, jamais double écriture à chaque replay. Backup avec l'API SQLite de sauvegarde, pas simple copie du `.sqlite3` en oubliant WAL. Ajouter `archive export-index --output ...`, `archive backup --output ...`, `archive rebuild-index` avec chemins bornés et erreurs de verrou explicites.
- [ ] Conserver l'import legacy demandé explicitement via le Dataset Builder ; il utilise `publish_replay(lease=None)` et la même base. Rien dans cette tâche ne réimporte les données supprimées le 4 novembre.
- [ ] Rejouer archive/API/dataset ciblés ; commit : `refactor: read archive and snapshots from transactional catalog`.

## Tâche 13 — Progression, logs et configurations opérateur

**Fichiers :** créer `collection_metrics.py`, `tests/test_collection_metrics.py` ; modifier `job_service.py`, `collection_supervisor.py`, `api/collection.py`, les deux configs collecte et `README.md` ; créer `configs/collection-mc-incremental.yaml`.

**Interfaces produites :** `collection_progress(store: CollectionStore, campaign_id: str) -> dict[str, Any]` dans `collection_metrics.py`. `read_job_logs(job_id: str, tail: int=100) -> list[str]` garde sa signature ; ajouter `read_collection_worker_logs(job_id: str, worker_id: str, tail: int=100) -> list[str]`.

- [ ] Écrire `test_metrics_count_unique_transitions`, `test_attempt_and_campaign_counters_are_distinct`, `test_idle_discovery_and_provider_pause_are_visible`, `test_eta_only_describes_known_queue`, `test_log_tail_does_not_read_whole_file`.
- [ ] Exécuter les tests métriques, constater les échecs.
- [ ] Calculer les compteurs dans les mêmes transactions que les transitions, ou depuis des tables agrégées/indexées ; ne pas scanner deux millions de tâches à chaque polling. Échantillonner la vue au maximum une fois/s ; buckets de minute pour débit récent, cumuls pour moyennes. Compacter les permis terminés après agrégation durable ; garder les permis incertains nécessaires à la reprise.
- [ ] Exposer les indicateurs de la spec, y compris couverture par format, débit effectif vs configuré, réservations, échecs définitifs, retries et temps par étape. Estimation uniquement sur file connue, `null` si débit insuffisant ; jamais pourcentage inventé de l'ensemble des replays Showdown.
- [ ] Exposer `selected_replays`, `new_replays`, `reused_replays`, `examined_replays`, `rejected_by_reason`, filtres actifs et hash de configuration. Le débit de téléchargement et celui de sélection sont différents ; tester dix GET dont deux matches → dix requêtes, deux retenus, huit rejets, aucun échec réseau. Une requête de statut ne doit pas afficher les corps des replays rejetés.
- [ ] Logs parent dans `worker.log`, enfants dans `workers/<worker-id>.log`, un seul écrivain par fichier, rotation 10 Mio × 3 sauvegardes. `tail` borné 1..1000, lecture depuis la fin avec budget de lecture borné ; longue ligne tronquée explicitement si nécessaire. Pas de relecture complète à chaque requête HTTP.
- [ ] Mettre à jour `collection-mc.yaml` avec les valeurs parallèles exactes de la spec. Exemple incrémental : mode incremental, recouvrement 48 h et budgets explicitement plus petits ; conserver un smoke à 1/1 et débit 1/s. Les configs futures ne doivent pas être utilisées par l'ancien schéma en cours de tâche avant raccordement.
- [ ] Garder le YAML basique **sans `upload_after`, `upload_before`, `teams_file` ni `team_filter`**. Ajouter un exemple séparé `configs/collection-mc-filtered.yaml` et `configs/teams.example.json` montrant les options facultatives. Documenter les quatre combinaisons, le chemin relatif au YAML, les formes distinctes, les compositions inconnues et le fait que filtrer après inspection économise le stockage, pas nécessairement les GET.
- [ ] README : démarrage backfill, pause, reprise simple vs nouvelle tentative, scale, mise à jour incrémentale, budget atteint, réinspection, limites de couverture et logs. Indiquer clairement que deux millions est une cible de stock, pas le résultat attendu d'un budget de 24 h à 2 req/s.
- [ ] Rejouer tests métriques/API/CLI et validation des YAML ; commit : `feat: report collection progress and document operating modes`.

## Tâche 14 — Validation intégrée, pannes et performance à deux millions

**Fichiers :** créer `tests/test_collection_integration.py`, `tests/collection_http_server.py`, `scripts/benchmark_collection.py`, `docs/audits/2026-11-04-collecte-parallele-resultats.md`. Compléter uniquement les fichiers dont un défaut est démontré.

**Interfaces produites :** script `python scripts/benchmark_collection.py --sizes 20000 100000 1000000 2000000 --workers 1 2 4 8 --output /tmp/collection-benchmark.json`, exécuté depuis le sous-projet ; serveur local démarré avec un port attribué par l'OS, injecté uniquement dans les tests/benchmarks. Aucun paramètre URL arbitraire ajouté à l'API publique.

- [ ] Écrire `test_parallel_pipeline_matches_serial_unique_set` : recherche paginée avec doublons/frontières, 1 000 IDs, huit workers, exactement 1 000 IDs publiés et un GET/ID en fonctionnement normal ; même ensemble que le run séquentiel sur autre racine.
- [ ] Écrire `test_restart_after_crash_preserves_pipeline`, `test_rate_limit_and_budget_hold_under_process_races`, `test_snapshot_and_compute_coexist_with_collection`. Injecter crash parent/enfant, timeout, 429 et erreur fichier. Tous les enfants sont attendus avec timeout et nettoyés en `finally`, sans `os._exit()` pour maquiller un test qui pend.
- [ ] Ajouter un parcours complet des quatre modes de sélection et deux campagnes successives aux équipes différentes ; une période couverte doit réutiliser les IDs, réévaluer les rejets, et retélécharger seulement les corps désormais requis. Vérifier absence de corps rejetés en DB, objets, temporaires et logs, isolation des snapshots de campagne, et plafond de sélection concurrent incluant la réutilisation d'objets existants.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_integration.py -q`, puis la suite complète `uv run --locked pytest` et `uv run --locked ruff check src tests`. Toute modification de code à cette étape doit rejouer ses tests ciblés ; rapporter les éventuels échecs préexistants séparément.
- [ ] Implémenter le benchmark synthétique catalogue avec insertions par lots de 1 000 et consommation par lots bornés. Générer deux millions de métadonnées réalistes sans fabriquer deux millions de gros logs pour le test d'index. Mesurer latence p50/p95 de présence, claim/publication, débit, taille DB/WAL, RSS maximale, contention et nombre de lectures complètes ; vérifier plans SQL et absence de croissance mémoire proportionnelle à N côté worker.
- [ ] Benchmark pipeline distinct : serveur local avec 200 ms de latence, 1 000 petits replays, limiteur non contraignant uniquement dans le harness. Trois répétitions à 1 et 8 workers, racines vierges ; médiane du débit 8 workers ≥3× médiane 1 worker. Si le seuil échoue, profiler l'étape limitante, corriger puis rejouer ; ne pas baisser le seuil silencieusement. Ne pas mettre une assertion de timing fragile dans la suite unitaire par défaut.
- [ ] Mesure réelle bornée : configurations 1/2/4/8 workers, même plafond global initial, maximum 1 000 requêtes ou 10 minutes par essai. Plages comparables, stockage de benchmark isolé. Conserver code/révision ancienne pour comparaison si possible sans le réintroduire dans le nouveau runtime ; à défaut déclarer la baseline nocturne « rapportée, non contrôlée ». Hausse éventuelle du débit dans un essai distinct, stop/ralentissement sur 429 ; ne pas confondre ajout de workers et augmentation de charge réseau.
- [ ] Publier le rapport : commandes, versions, configuration, matériel, résultats réels et synthétiques séparés, limites de couverture rencontrées, échecs/pannes testés, réglage retenu. Si réseau ou ressources empêchent une mesure, marquer exactement le test non réalisé ; ne pas annoncer le gain Showdown comme établi par le seul test local.
- [ ] Commit : `test: validate parallel collection recovery and scale`.

## Matrice de couverture et remise

| Exigence de la spec | Tâches | Preuve principale |
| --- | --- | --- |
| Répartition des recherches et déduplication avant GET | 5, 8, 9, 14 | Huit processus, mêmes IDs que séquentiel, un GET/ID neuf sans erreur. |
| Archive commune et coût indexé | 3, 7, 12, 14 | Contraintes SQL, aucun scan JSONL par ID, benchmark 2 M. |
| Reprise après panne et ancien token rejeté | 5, 7, 10, 14 | Crash aux frontières fichier/SQL, PID réutilisé, API redémarrée. |
| Budgets/débit communs | 4, 6, 14 | Courses multi-processus, retries, 429, zéro dépassement de plafond. |
| Backfill, incrémental, couverture honnête | 8, 9 | Fenêtres vides, timestamps bloqués, zéro GET sans nouveautés. |
| CLI/FastAPI et anciens YAML | 2, 11, 13 | Tests de validation, commandes, idempotence de start. |
| Filtres facultatifs, rejet avant publication, sélection par campagne | 2, 2 bis, 3–9, 11–14 | Quatre combinaisons, zéro corps rejeté conservé, changement de filtre réévalué, snapshots isolés. |
| Datasets figés pendant collecte | 10, 12, 14 | Snapshot concurrent, job compute indépendant, import coordonné. |
| Archive neuve sans migration/purge récurrente | 3 | Initialisation répétée, fichiers non concernés inchangés. |
| Supervision/visibilité | 10, 13 | Scale/pause/stop, compteurs durables, logs bornés. |
| Gains mesurés | 1, 14 | Rapport contrôlé local et mesure réelle distincte. |

Avant remise de l'implémentation : vérifier chaque ligne, exécuter `git diff --check`, relire les commandes README sur un stockage de démonstration, confirmer qu'aucun test n'a écrit dans le stockage personnel. Indiquer les commandes exécutées et les mesures manquantes. Ne pas lancer automatiquement une collecte de deux millions de replays, un entraînement ou un déploiement au terme de ce plan.
