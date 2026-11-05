# Préparation des datasets sur plusieurs cœurs — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Le travail demandé ici est la rédaction du plan : aucun prepare, entraînement ou provisionnement cloud n'a été lancé pendant sa rédaction.

**Goal:** Utiliser automatiquement les CPU accessibles à la machine d'exécution pour convertir les replays en trajectoires, avec résultats reproductibles, mémoire de conversion bornée et fermeture fiable des processus.

**Architecture:** Une configuration d'exécution de `prepare`, indépendante de la recette du dataset, pilote un unique `ProcessPoolExecutor`. Le parent prépare les identités/splits, soumet un nombre borné de conversions, valide et écrit progressivement en staging ; lui seul publie le dataset. Un diagnostic des ressources est exécuté dans le job qui effectue réellement la préparation, en local comme sur VM CPU.

**Tech Stack:** Python ≥3.13, `concurrent.futures`, `multiprocessing` en mode `spawn`, Pydantic 2, NumPy, convertisseur VGC-Bench existant, pytest, uv/Devbox, CLI et FastAPI actuelles. Aucune conversion GPU ni file distribuée.

**Spec:** Le contrat fonctionnel de la section 1 ci-dessous formalise la demande utilisateur ; le pipeline BC et les invariants dataset restent ceux de la [spec collecte/datasets/fine-tuning](../specs/2026-11-03-collecte-datasets-finetuning-mc.md). La collecte déjà développée suit sa [spec dédiée](../specs/2026-11-04-collecte-parallele-incrementale.md) ; elle n'est pas à réimplémenter.

## Contraintes globales

- Conserver la collecte locale 100k M-C, sa config `configs/collection-mc-100k.yaml` et ses données. Ne pas l'arrêter, la relancer, la purger ou changer ses paramètres pour développer ce chantier.
- `conversion_workers: auto` par défaut : tous les CPU logiques utilisables, sans soustraire systématiquement un cœur. Valeur explicite entière ≥1 pour contrôler la RAM ou laisser de la capacité à la collecte.
- Détecter les ressources **dans le processus prepare au moment de l'exécution**, pas à la création du dataset, au chargement du module ou sur la machine de l'API.
- Le nombre de workers est un choix d'exécution, pas une propriété sémantique des données : ne pas modifier `recipe_sha256`, la sélection, le regroupement BO3 ou les splits quand seule cette option change.
- Conserver les identités `(replay_id, raw_sha256, player_side)`, les filtres métier existants et les contrôles de masques/actions. Aucune erreur bloquante ne devient une simple perspective rejetée.
- Un seul pool par préparation, aucune création de pool dans une tâche enfant. Les workers de conversion ne publient pas le manifest et ne modifient pas l'archive.
- Borner le nombre de futurs/logs/trajectoires en vol ; ne pas garder toutes les trajectoires d'un dataset 100k en RAM avant écriture.
- Les tests utilisent des racines temporaires et aucune requête Showdown. Le smoke réel utilise un petit snapshot figé de l'archive existante, avec des IDs de datasets distincts.
- Un dataset `ready` reste immuable. Pas d'écrasement d'un dataset existant pour comparer 1/2/4 workers.
- Préserver les modifications de collecte déjà présentes dans le workspace. Ce plan ne donne pas l'autorisation de louer une VM, de transférer des données vers un service externe ou de lancer le fine-tuning 100k.

## Points de revue

1. VM/container voyant 64 CPU hôte mais autorisé à en utiliser 8 : détection bornée par affinité/quota, pas 64 workers — tâche 2.
2. Résultats terminant dans le désordre, deux versions d'un replay ou deux côtés : aucune permutation de provenance, aucun futur écrasé par une clé insuffisante — tâches 4 et 6.
3. Erreur bloquante après plusieurs succès déjà écrits en staging : aucun dataset partiellement `ready` — tâches 5 et 6.
4. Exception, interruption ou timeout durant le reader asyncio du vendor : fermeture du pool, pas d'enfant oublié ni de verrou compute indéfiniment détenu — tâches 3 et 6.
5. Même snapshot/recette sur deux machines et nombres de workers différents : mêmes contenus et splits, ressources d'exécution enregistrées séparément — tâches 1, 5 et 7.

---

## 1. Contrat fonctionnel et décisions techniques

### Parcours cible

1. Conserver la collecte locale jusqu'au stock souhaité de 100k replays M-C.
2. Développer/tester prepare multicœur localement ; smoke réel à 1, 2 et 4 workers, puis vérification du mode auto sur la même petite entrée.
3. Copier le trainer reproductible et l'archive cohérente sur une VM CPU adaptée ; exécuter snapshot → `dataset create` → `dataset prepare` avec auto ou une limite explicite après mesure RAM.
4. Récupérer le dataset complet `ready`, ses manifestes et contrôles d'intégrité.
5. Utiliser une VM GPU pour le fine-tuning BC à partir du dataset et du checkpoint parent M-B. Pas de dépendance GPU dans la conversion.

Les 100k replays collectés ne garantissent pas 100k replays exploitables après les filtres et la conversion. Le statut doit refléter les données effectivement produites, sans desserrer silencieusement les critères pour atteindre une cible.

### Configuration proposée

Créer `PrepareExecutionConfig`, distinct de `PrepareConfig` legacy et de `DatasetRecipe` :

```yaml
prepare:
  conversion_workers: auto
  conversion_timeout_s: 120
```

- `conversion_workers` : `Literal['auto'] | StrictInt`, défaut `auto`. Rejeter booléens, zéro, négatifs, nombres décimaux et chaînes numériques dans YAML/JSON ; la CLI convertit explicitement `"4"` vers l'entier 4.
- `conversion_timeout_s` : nombre fini strictement positif, défaut proposé 120 secondes par perspective **en cours d'exécution**, pas depuis sa mise en file. C'est une borne de diagnostic et de nettoyage, pas une limite de durée du replay.
- CLI future : `dataset prepare --dataset-id ID --conversion-workers auto|N`, option `--config configs/prepare.yaml` et `--conversion-timeout-s S`. Priorité : option CLI explicitement fournie > fichier prepare > défaut. API : body facultatif incluant ces mêmes champs.
- `DatasetJobConfig.execution` contient la configuration demandée ; anciens jobs sans ce champ → défaut auto. Un résumé `prepare_execution` dans le manifest et un fichier `prepare_execution.json` enregistrent valeurs demandées/effectives, Python, ressources, nombre de tâches, start method, dates et durée. Ces données n'entrent pas dans le hash de recette ou de contenu.
- `conversion_workers=1` utilise le même chemin de conversion que N>1, afin que la comparaison teste le parallélisme et non deux implémentations métier.
- Une conversion ponctuelle d'une seule perspective peut naturellement limiter le pool à un worker ; le problème à supprimer est le plafond imposé à une préparation contenant plusieurs perspectives.

### Résolution de `auto`

Prendre les CPU accessibles au processus, et non seulement le total hôte. Python fournit `os.process_cpu_count()` depuis 3.13 ; l'affinité peut restreindre ce nombre. [Documentation Python](https://docs.python.org/3.13/library/os.html#os.process_cpu_count).

Algorithme retenu, explicite pour éviter des interprétations différentes :

1. Lire `os.process_cpu_count()` si disponible, `len(os.sched_getaffinity(0))` si disponible et `os.cpu_count()` comme dernier repli. Utiliser le minimum des limites positives applicables ; un appel indisponible ou `None` n'est pas une limite zéro. Repli ultime : 1.
2. Sous Linux, résoudre le cgroup courant depuis `/proc/self/cgroup` et les montages de `/proc/self/mountinfo`. Pour cgroup v2, lire `cpu.max` dans le groupe et ses ancêtres visibles ; appliquer le quota fini le plus restrictif. Ne pas lire seulement `/sys/fs/cgroup/cpu.max`. Supporter v1 via `cpu.cfs_quota_us`/`cpu.cfs_period_us` si rencontré. Les restrictions CPU sont hiérarchiques ; `max` signifie absence de quota à ce niveau. [Documentation du noyau](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).
3. Pour un quota fractionnaire, `ceil(quota / period)` workers permet d'exploiter la fraction restante, sous le plafond d'affinité ; le scheduler conserve le quota effectif. Exemple : quota 2,5 CPU et affinité 8 → plafond de 3 processus, capacité CPU reportée 2,5. Ce choix d'arrondi est une décision de ce plan, pas une garantie de débit.
4. `auto` prend ce plafond ; une valeur explicite supérieure est ramenée au plafond, avec valeur demandée et limitation clairement affichées. Ne pas utiliser de constante liée à la machine actuelle.
5. `effective_workers = min(plafond choisi, nombre de perspectives soumises)` ; zéro tâche → aucun pool, traitement métier `insufficient_data` approprié.
6. Ne pas réduire silencieusement auto à cause d'une estimation RAM non mesurée. Afficher RAM accessible si détectable, mesurer RSS au smoke et permettre un override explicite. Auto signifie capacité CPU accessible, pas maximum instantanément inoccupé.

Constat de rédaction du 4 novembre : le Python **3.13.2 du trainer** voit 8 CPU logiques et une affinité de 8 ; les quotas cgroup visibles contrôlés n'exposent pas de limite numérique. Ce n'est pas une constante à copier : le job doit refaire la détection sur la VM.

### Gestion des processus

Utiliser `multiprocessing.get_context('spawn')` explicitement, avec fonctions de travail au niveau module et paramètres sérialisables. Un nouvel interpréteur ne doit pas hériter des verrous/boucles asyncio du parent ; auditer les imports et changements de CWD nécessaires au vendor. [Documentation Python multiprocessing](https://docs.python.org/3.13/library/multiprocessing.html#contexts-and-start-methods).

`shutdown(cancel_futures=True)` annule seulement le travail non démarré ; il ne stoppe pas une conversion en cours. Prévoir fin des tâches engagées, timeout du reader et nettoyage vérifié. Ne pas employer `terminate_workers()` comme API disponible en Python 3.13, ni des attributs privés `_processes` de l'executor. [Documentation Python concurrent.futures](https://docs.python.org/3.13/library/concurrent.futures.html#concurrent.futures.Executor.shutdown).

---

## 2. État réel du code et fichiers à toucher

Chemins relatifs à `pokecoach2-trainer/` ; les chemins `src/...`, `tests/...` et commandes uv qui suivent sont relatifs au sous-projet `pokecoach-trainer/`.

| Emplacement observé | Constat / changement attendu |
| --- | --- |
| `src/pokecoach_trainer/dataset_service.py::_prepare_snapshot_dataset` | Pool `max_workers=1`, tous les résultats collectés avant écriture ; remplacer par pool résolu + consommateur incrémental. |
| `dataset_service.py::_prepare_legacy_raw_logs` | Autre pool fixé à 1 ; faire passer la configuration explicite sans oublier ce parcours. |
| `src/pokecoach_trainer/conversion.py::convert_perspectives_parallel` | Tous les logs/futurs soumis ; clé `(replay_id, side)` pouvant écraser une autre version ; résultats conservés en liste. Introduire un itérateur borné. |
| `conversion.py::convert_perspective` | Peut créer son propre pool ; conserver le parcours ponctuel mais interdire un pool imbriqué dans les workers. |
| `src/pokecoach_trainer/schemas.py` | `PrepareConfig` legacy, `DatasetJobConfig`, `DatasetManifest` ; ajouter configuration runtime séparée et métadonnées d'exécution. |
| `src/pokecoach_trainer/api/datasets.py` | Body prepare actuellement ignoré pour snapshot ; propager l'option réellement jusqu'au worker. |
| `src/pokecoach_trainer/cli.py` | `dataset prepare` n'a pas de paramètre de workers ; transmettre config CLI/YAML/API sans modifier la recette. |
| `src/pokecoach_trainer/job_service.py`, `worker.py`, `jobs.py` | Réutiliser la supervision compute ; vérifier ownership du verrou et fermeture des enfants sans modifier la collecte. |
| `vendor/vgc-bench/vgc_bench/logs2trajs.py` | `_init_worker_loop` crée boucle + thread daemon ; `process_log` attend un future asyncio sans timeout. Toute adaptation vendor doit passer par un patch reproductible. |
| `tests/test_conversion_provenance.py` | Tests présents surtout avec faux executors, donc aucune preuve de parallélisme réel ou d'absence de fuite. |
| `tests/test_dataset_publication.py` | Conversion mockée ; compléter avec vrais processus et échecs après staging. |

Nouveaux modules : `prepare_resources.py` pour la détection/résolution, `conversion_runtime.py` pour le pool/initialisation/nettoyage, `conversion_worker.py` pour le travail sérialisable. Garder la sélection/splits/publication dans le service existant ; pas de service distribué.

Le workspace contient déjà de nombreuses modifications de collecte non committées. Faire un inventaire initial et ne pas les réinitialiser. Les fichiers partagés `schemas.py`, `cli.py`, `job_service.py` demandent des diffs particulièrement ciblés.

## 3. Méthode et commandes

Exécuter chaque tâche avec tests rouges pertinents puis verts et diff revu. Les tests doivent échouer sur le comportement absent, pas seulement sur un environnement indisponible. Ne pas lancer maintenant les étapes du plan dans la machine de l'utilisateur.

Commandes de référence depuis `pokecoach2-trainer/pokecoach-trainer/` :

```bash
uv run --locked pytest tests/test_conversion_provenance.py tests/test_dataset_publication.py -q
uv run --locked ruff check src tests
```

Le projet modifie parfois le CWD vers le vendor. Calculer les chemins depuis les settings/`__file__`, et résoudre les chemins utilisateur avant ce changement. Utiliser le même interpréteur uv pour parent, enfants, tests et smoke.

À chaque fin de tâche, commit seulement les fichiers concernés, si l'agent travaille dans une branche autorisée : identité `benacFlorian <florianbenac46@gmail.com>`, sans modifier la config Git globale. Aucun push ou commit des données locales. Ne pas mélanger le travail de collecte préexistant avec ce chantier.

---

## Tâche 1 — Contrat d'exécution, API et CLI

**Fichiers :** modifier `src/pokecoach_trainer/{schemas.py,cli.py,dataset_service.py,api/datasets.py}` ; créer `tests/test_prepare_config.py`, `tests/test_prepare_cli.py` ; compléter `tests/test_dataset_api.py`.

**Interfaces à produire :**

```python
class PrepareExecutionConfig(BaseModel):
    conversion_workers: Literal['auto'] | StrictInt = 'auto'
    conversion_timeout_s: float = 120.0

submit_preparation(dataset_id: str, execution: PrepareExecutionConfig | None = None) -> JobRecord
prepare_dataset(dataset_id: str, config: PrepareConfig | None = None, *,
                job_id: str | None = None,
                execution: PrepareExecutionConfig | None = None) -> DatasetManifest
```

`DatasetJobConfig.execution` utilise une `default_factory`. Pour préserver les appels legacy, ajouter les deux champs runtime à `PrepareConfig` par un modèle de base partagé ou une conversion explicite ; `DatasetRecipe` ne reçoit aucun de ces champs. Dans l'API snapshot, valider le body comme `PrepareExecutionConfig`, tout en conservant l'ancien body `PrepareConfig` uniquement pour les datasets importés. Aucune option non applicable ne doit être silencieusement ignorée.

- [ ] Écrire `test_prepare_defaults_to_auto`, `test_workers_reject_bool_zero_negative_float`, `test_old_job_without_execution_loads`, `test_snapshot_api_forwards_execution_to_job`, `test_cli_overrides_yaml_only_when_explicit`. Assertions centrales : `config.conversion_workers == 'auto'`, `job.config.execution.conversion_workers == 4`, et même `recipe_sha256` pour exécutions 1/4/auto.
- [ ] Lancer `uv run --locked pytest tests/test_prepare_config.py tests/test_prepare_cli.py tests/test_dataset_api.py -q` avant implémentation ; confirmer les échecs attendus.
- [ ] Implémenter le modèle avec `extra='forbid'`, validations, loader `prepare:` et flags CLI. Ajouter `DatasetManifest.prepare_execution: dict[str, Any] | None = None` pour consulter les résultats anciens sans modifier leur contenu.
- [ ] Propager `execution` de la façade au `DatasetJobConfig`, puis depuis `run_prepare_job` vers les deux branches de préparation. Ne pas résoudre auto lors de `dataset create` ni dans `submit_preparation`.
- [ ] Tester le body absent (auto), le snapshot avec body workers=4, legacy avec workers=2 et body incompatible rejeté. Vérifier que `ready` reste non modifiable.
- [ ] Rejouer les tests ciblés et Ruff ; commit `feat: expose dataset preparation execution settings`.

## Tâche 2 — Détection CPU et résolution portable

**Fichiers :** créer `src/pokecoach_trainer/prepare_resources.py`, `tests/test_prepare_resources.py` ; ajouter diagnostic CLI `dataset resources` dans `cli.py`.

**Interfaces :**

```python
@dataclass(frozen=True)
class PrepareResources:
    logical_cpus: int | None
    affinity_cpus: int | None
    process_cpus: int | None
    quota_cpus: float | None
    available_workers: int
    detection_notes: tuple[str, ...]

detect_prepare_resources() -> PrepareResources
resolve_conversion_workers(requested: Literal['auto'] | int,
                           resources: PrepareResources, *, task_count: int) -> int
```

- [ ] Écrire des tests sans dépendre de l'hôte : hôte64/affinité8 →8 ; affinité8/quota4 →4 ; quota2,5 →3 ; quota enfant illimité/parent2 →2 ; tout indisponible →1 ; `task_count=0` →0 ; machine8/tâches3 →3 ; explicite4 sur machine2 →2 avec diagnostic ; explicite1 reste1.
- [ ] Tester cgroup v2 imbriqué, namespace/montage non standard, cgroup v1, données manquantes/malformées et quota illimité. Injecter les lectures de fichiers dans les tests ; ne pas modifier le cgroup du test runner.
- [ ] Lancer `uv run --locked pytest tests/test_prepare_resources.py -q`, puis implémenter l'algorithme de section 1. Ne pas interpréter un fichier manquant comme zéro CPU ni une erreur de détection comme autorisation de dépasser une affinité connue.
- [ ] Ajouter `dataset resources` affichant les valeurs détectées et les limites visibles, sans créer de pool. Dans le prepare, résoudre une fois après constitution des tâches, journaliser demandé/effectif et enregistrer le diagnostic.
- [ ] Rejouer tests config/ressources/CLI ; commit `feat: detect available CPU capacity for dataset conversion`.

## Tâche 3 — Pool unique, initialisation et fermeture

**Fichiers :** créer `conversion_runtime.py`, `conversion_worker.py` sous `src/pokecoach_trainer/`, `tests/test_conversion_runtime.py`, `tests/prepare_process_helpers.py`. Si nécessaire modifier `patches/logs2trajs-provenance.patch`, `patches/README.md`, `scripts/sync_vgc_bench.sh`, `tests/test_vendor_sync.py` pour rendre timeout/nettoyage vendor reproductibles.

**Interfaces :**

```python
# conversion_runtime.py
conversion_pool(*, workers: int, storage_root: Path, vendor_root: Path,
                timeout_s: float) -> ContextManager[ProcessPoolExecutor]
# conversion_worker.py
initialize_conversion_worker(storage_root: str, vendor_root: str, timeout_s: float) -> None
convert_archive_task(ref: ArchiveRef, side: str,
                     contract: ModelContract, task_seed: int) -> ConversionResult
```

Le worker charge le log à partir de l'identité d'archive ; le parent n'envoie pas un dict de tous les logs. Le wrapper enfant appelle directement le convertisseur, jamais `convert_perspective()` si celui-ci crée un pool. `ConversionResult` reste attribué par l'identité de tâche fournie, pas par son ordre d'arrivée.

- [ ] Écrire `test_pool_uses_spawn_and_requested_workers`, `test_initializer_runs_once_per_child`, `test_worker_can_read_archive_from_other_cwd`, `test_pool_closes_on_success_and_exception` avec de vrais processus importables. Les fonctions enfants ne doivent être ni lambda ni closure de pytest.
- [ ] Exécuter `uv run --locked pytest tests/test_conversion_runtime.py -q` et constater les échecs avant implémentation.
- [ ] Utiliser un context spawn local au pool, pas `multiprocessing.set_start_method` global. Initialiser les settings et le CWD vendor avant ses imports dépendants de `data/*.json`. Auditer les imports du point d'entrée CLI/worker pour que les enfants n'initialisent pas GPU ou runtime serveur.
- [ ] Limiter les threads numériques par processus à 1 (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` avant imports ; `torch.set_num_threads(1)` / interop si Torch est chargé). Tester la limite effective : si des bibliothèques étaient déjà chargées avant initializer, déplacer leurs imports ou utiliser un contrôle runtime explicite, pas une variable d'environnement trop tardive. Aucun changement des paramètres d'entraînement GPU.
- [ ] Conserver une boucle asyncio vendor par enfant. Ajouter au vendor, via patch si nécessaire, un timeout optionnel rétrocompatible pour l'attente de `run_coroutine_threadsafe(...).result(timeout=...)`. À expiration, annuler le future coroutine et remonter `conversion_timeout` comme échec global de préparation, sans continuer à réutiliser un reader potentiellement occupé.
- [ ] Enregistrer un finalizer enfant qui arrête proprement la boucle/thread, ferme les ressources du reader disponibles et ne s'exécute pas deux fois. Le parent ferme son pool en `finally`, annule les tâches non commencées et attend les tâches engagées bornées par le timeout. Intercepter `KeyboardInterrupt`/SIGTERM au niveau du job pour passer par ce nettoyage ; ne pas avaler ces exceptions comme des rejets métier.
- [ ] Prévoir un chemin d'abandon borné pour enfant réellement bloqué/crashé : enregistrer PID + identité de démarrage des seuls enfants du pool via l'initializer et un canal fermé en sortie ; après délai de grâce, terminer seulement les enfants identifiés puis attendre leur fin avant de fermer l'executor. Sous Linux, protéger aussi le cas de disparition du parent (surveillance du parent ou signal de mort du parent avec vérification de la course au démarrage). Ne pas employer `pkill python`, tuer la collecte ou prétendre qu'annuler un future arrête son processus.
- [ ] Rejouer runtime et vendor-sync si patch ; commit `feat: manage isolated conversion process pools and cleanup`.

**Limite à documenter :** un arrêt brutal de machine ne garantit pas l'exécution des finalizers ; la publication atomique et l'état non-ready doivent protéger les données. Les tests de fuite portent sur enfants effectivement créés et les chemins de sortie provoqués, pas uniquement sur l'appel d'un mock `shutdown`.

## Tâche 4 — Conversion bornée, provenance et erreurs

**Fichiers :** modifier `src/pokecoach_trainer/conversion.py`, `conversion_worker.py` ; compléter `tests/test_conversion_provenance.py` ; créer `tests/test_conversion_bounded.py`.

**Interfaces :**

```python
iter_converted_perspectives(tasks: Iterable[tuple[ArchiveRef, str]],
                            executor: ProcessPoolExecutor, *,
                            contract: ModelContract, seed: int,
                            max_in_flight: int) -> Iterator[ConversionResult]
```

`convert_perspectives_parallel()` peut rester comme adaptateur de compatibilité retournant une liste dans l'ordre d'entrée pour les petits appelants/tests ; le prepare 100k doit utiliser l'itérateur. `max_in_flight = 2 * effective_workers` par défaut interne, pas une nouvelle option publique indispensable.

- [ ] Écrire `test_completion_order_preserves_full_identity`, `test_same_replay_two_hashes_do_not_overwrite_future`, `test_duplicate_full_identity_is_rejected`, `test_in_flight_is_bounded`, `test_results_do_not_retain_all_trajectories`. Avec tâches sentinelles, affirmer que `max_observed_pending <= 2 * workers` et que chaque résultat porte le hash et côté de sa tâche.
- [ ] Écrire `test_child_parse_error_keeps_provenance`, `test_blocking_mask_is_not_plain_rejection`, `test_broken_process_pool_aborts`, `test_timeout_aborts`. Un `process_log` retournant `None` reste un rejet ordinaire ; une erreur d'infrastructure ne doit pas être traduite en cent mille `conversion_exception` puis un dataset ready.
- [ ] Lancer les tests avant modification, puis remplacer soumission globale + liste de résultats par une fenêtre de futurs. Consommer avec `wait(..., FIRST_COMPLETED)`, retirer les références terminées, traiter/écrire via le consommateur avant de remplir à nouveau. Ne pas appeler `list(iterator)` dans le chemin prepare.
- [ ] Utiliser une table `future -> (index, ref, side)` et la clé complète `(replay_id, content_sha256, side)` ; vérifier les identités du résultat avant consommation. Deux versions d'un même replay dans un snapshot normal restent un conflit de source ; le convertisseur ne doit toutefois jamais confondre silencieusement ces tâches.
- [ ] Extraire un traitement partagé des trajectoires/erreurs utilisé par conversion unitaire et parallèle : même `trajectory_to_imitation`, mêmes contrôles, même contrat. Ne pas modifier la signification des codes `blocking_mask_first_target`, `blocking_mask_second_target`, `action_index_out_of_bounds`, `blocking_mask_error`.
- [ ] Si le convertisseur ou ses dépendances consomment de l'aléatoire, initialiser chaque tâche avec une seed stable dérivée de `(recipe.seed, replay_id, raw_sha256, side)` via hash stable ; jamais PID, numéro de worker, ordre de fin ou `hash()` Python. Appliquer la même règle à 1 worker.
- [ ] Sur erreur bloquante/infrastructure : arrêter de soumettre, propager l'échec au propriétaire du pool pour nettoyage ; ne pas attendre la conversion de tout le dataset pour découvrir un masque invalide. Rejouer les tests et commit `refactor: stream bounded conversion results with stable provenance`.

## Tâche 5 — Intégration prepare et publication progressive

**Fichiers :** modifier `src/pokecoach_trainer/dataset_service.py`, `schemas.py` ; compléter `tests/test_dataset_publication.py`, `tests/test_dataset.py` ; créer `tests/test_prepare_parallel_service.py`.

**Interfaces à adapter :** `_prepare_snapshot_dataset(dataset_id, *, job_id, execution)` et `_prepare_legacy_raw_logs(dataset_id, config, *, execution)` consomment `PrepareExecutionConfig`. `run_prepare_job` transmet la config persistée. Un helper parent écrit un `ConversionResult` validé dans le staging et retourne la ligne d'index et le nombre de transitions, sans garder la trajectoire.

- [ ] Écrire `test_snapshot_prepare_uses_resolved_worker_count`, `test_legacy_prepare_honors_workers`, `test_no_pool_when_no_tasks`, `test_runtime_does_not_change_recipe_or_splits`, `test_partial_success_then_blocking_error_never_publishes`.
- [ ] Exécuter les tests ciblés avant implémentation. Les anciens mocks de `convert_perspectives_parallel` devront être remplacés par mocks de l'itérateur, sans supprimer leurs assertions sur l'immuabilité/splits.
- [ ] Garder sélection des groupes et `assign_splits` avant le pool. Résoudre les ressources sur le job courant après calcul du nombre de perspectives ; créer un seul pool et consommer les résultats au fil de l'eau. Écrire `.pkl` et hash dans le staging uniquement, accumuler compteurs et petites lignes d'index, libérer immédiatement la trajectoire/future. Ne pas refaire `count_transitions` en rechargeant tous les pickles : les longueurs sont connues lors de l'écriture.
- [ ] Trier l'index final par identité complète avant publication ; les fichiers restent nommés par replay/côté avec une seule version autorisée par snapshot. Les lignes d'erreurs/décisions incluent ID/hash/côté. Les timestamps et métadonnées runtime peuvent varier ; la sélection, les splits et valeurs obs/acts/masks doivent être identiques.
- [ ] Compter les replays réellement convertis (au moins une perspective valide), et les perspectives/transitions par split. Ne pas déclarer `ready` uniquement parce que le nombre de candidats avant conversion atteignait la cible. Si les rejets ordinaires font tomber sous la cible ou rendent un split inexploitable, publier l'état `insufficient_data` et un diagnostic, sans `ready` ni changement silencieux de sélection/splits.
- [ ] Pour le chemin raw_logs legacy, appliquer également un pool unique et limiter les lots alimentant `process_logs_with_provenance`, qui accumule actuellement les trajectoires. Conserver ses filtres `min_rating/only_winner/strict` et provenance ; consommer/écrire un lot avant le suivant. Le chargement du JSON legacy entier reste une limite historique explicitement documentée, pas le parcours recommandé pour 100k.
- [ ] En erreur/interruption, fermer d'abord les workers, supprimer uniquement le staging appartenant à cette tentative, persister `failed`/`interrupted` et laisser intacts objets bruts, snapshots et datasets précédemment ready. Le chemin direct de service doit respecter le même invariant que le job API. Un timeout ou un pool cassé est un échec, pas `insufficient_data`.
- [ ] Reporter au moins une fois/s demandé/effectif, total/terminé/en-vol, succès/rejets/erreurs, perspectives/s et phase d'écriture. Le parent écrit seul manifest/progression ; aucune concurrence sur `state.json` entre enfants.
- [ ] Vérifier le verrou compute actuel avec un vrai enfant : son propriétaire doit survivre au retour de l'API et le libérer après nettoyage. Corriger seulement les défauts prouvés dans `job_service.py`/`jobs.py`/`worker.py` et les doubles libérations du service ; ne pas réouvrir le chantier de collecte.
- [ ] Rejouer tests dataset/API/conversion et Ruff ; commit `feat: prepare datasets with bounded multicore conversion`.

## Tâche 6 — Tests réels de parallélisme, déterminisme et fuite

**Fichiers :** créer `tests/test_prepare_parallel_processes.py`, compléter `tests/prepare_process_helpers.py`, `tests/test_conversion_runtime.py` et `tests/test_worker_lifecycle.py` ; ajouter de petites fixtures convertibles sous `tests/fixtures/prepare/`.

- [ ] Écrire un harness importable qui lance la préparation dans un processus dédié avec `TRAINER_STORAGE_ROOT` temporaire ; pas un faux executor. Capturer PID/identité des workers et signal de démarrage des tâches. Deux tâches se synchronisent par un mécanisme de test borné pour prouver qu'au moins deux processus travaillent simultanément, sans assertion fragile « deux fois plus rapide ».
- [ ] Sur mêmes fixtures, seed et recette, comparer 1/2/4 workers : mêmes identités triées, provenance/hash brut, décisions, `split_id`, appartenance BO3, nombre de transitions ; `numpy.testing.assert_array_equal` pour obs/actions/masques correspondants. Comparer les manifests après exclusion explicite des IDs de dataset/job, dates et runtime ; pas après suppression arbitraire des différences gênantes.
- [ ] Injecter des durées différentes pour finir dans un autre ordre. Ajouter un log rejeté normalement, un log en erreur, puis chaque erreur bloquante de masque ; vérifier l'état et les compteurs attendus. Ajouter crash d'enfant et timeout du reader. Les fixtures numériques seules ne prouvent pas la compatibilité du convertisseur réel : au moins un test utilise réellement `process_log` sur des logs valides.
- [ ] Après succès, échec bloquant et interruption, vérifier que chaque enfant de ce pool a disparu et a été attendu, que les canaux sont fermés et que le prochain job compute peut démarrer. Répéter trois préparations pour repérer une accumulation de processus/descripteurs. Ne pas se contenter de `multiprocessing.active_children()` du processus pytest si les workers sont petits-enfants.
- [ ] Ajouter une épreuve de milliers de tâches légères : maximum de futurs et trajectoires vivantes borné par le pool, aucune rétention de tous les tableaux. RSS réelle reportée dans le smoke, sans imposer un seuil universel à une CI différente.
- [ ] Lancer `uv run --locked pytest tests/test_prepare_parallel_processes.py tests/test_conversion_provenance.py tests/test_conversion_runtime.py tests/test_dataset_publication.py tests/test_worker_lifecycle.py -q`, puis la suite complète et Ruff. Chaque harness a un timeout et un nettoyage ciblé des processus possédés ; un test qui pend n'est pas un succès.
- [ ] Commit `test: verify multicore preparation provenance and process lifecycle`.

## Tâche 7 — Smoke local et mode opératoire CPU → GPU

**Fichiers :** créer `configs/prepare.yaml`, `configs/smoke/prepare.yaml`, `scripts/smoke_prepare_parallel.py`, `docs/audits/prepare-multicoeur-local.md` et `docs/runbooks/dataset-cpu-finetuning-gpu.md` ; compléter `pokecoach-trainer/README.md`. Ne pas modifier `collection-mc-100k.yaml`.

**Script smoke futur :**

```bash
uv run --locked python scripts/smoke_prepare_parallel.py --snapshot-id SNAPSHOT_ID --recipe configs/smoke/dataset-mc.yaml --workers 1 2 4 auto --report /tmp/prepare-parallel-report.json
```

- [ ] Écrire le script avec sélection fixe d'un petit snapshot **identique pour tous les runs**, environ 20–50 replays et assez de groupes pour les trois splits. Ne pas télécharger de replays supplémentaires ni modifier la recette entre runs. Utiliser une racine scratch séparée contenant le snapshot et les seuls objets nécessaires, avec chemins relatifs et nouveaux IDs de datasets. Si aucun petit snapshot exploitable n'est disponible, rapporter le prérequis manquant ; ne pas fabriquer un smoke réussi en remplaçant le convertisseur.
- [ ] Vérifier que le script lance des jobs prepare réels, attend leur état terminal avec timeout, puis lit manifest/index/fichiers. Sur machine ayant moins de 2 ou 4 CPU accessibles, afficher le plafonnement et distinguer demandé/effectif ; ne pas prétendre avoir testé quatre processus si un seul a tourné.
- [ ] Exécuter localement 1, 2, 4, puis auto. Pour le mode auto, une seule petite préparation suffit à valider la résolution ; ne pas saturer l'archive 100k pour ce contrôle. La collecte locale peut rester active ; noter cette charge concurrente dans les résultats.
- [ ] Reporter Python, cœurs/quota/affinité, workers effectifs, nombre de perspectives, durée conversion et totale, débit, pic RAM parent + enfants (préciser la méthode), erreurs, replays convertis et vérification d'absence de processus restants. Mesurer le gain sans exiger une accélération linéaire sur un petit jeu où spawn/imports peuvent dominer.
- [ ] Comparer effectivement sorties 1/2/4/auto avec les invariants de la tâche 6. Le rapport doit déclarer `ready` ou expliquer un véritable `insufficient_data` ; un échec métier commun aux quatre runs ne valide pas le parcours ready demandé.
- [ ] Documenter le transfert local → VM CPU : trainer et lockfiles, patches et provenance du vendor, configuration, archive cohérente et snapshot/index nécessaires. Si la collecte écrit encore, figer un sous-ensemble de références/objets immuables et utiliser une sauvegarde SQLite cohérente ; ne pas rsync une base WAL vivante comme un fichier isolé. Ne pas démarrer sur la VM les anciens jobs de collecte copiés.
- [ ] Documenter le parcours CPU avec commandes vérifiées contre `--help` : `archive snapshot --config …`, `dataset create --snapshot-id … --recipe … --dataset-id …`, puis `dataset prepare --dataset-id … --conversion-workers auto`. Pour la RAM, mesurer sur le smoke la mémoire d'un processus chaud et le pic parent avant de choisir une valeur explicite ; aucun calcul « vCPU moins un » ne remplace cette mesure. Garder marges pour OS, index, compression et fichiers temporaires.
- [ ] Documenter le retour du dataset : transférer l'intégralité du dossier ready (manifest, recipe/source, splits, index, décisions et trajectoires) avec une liste de SHA-256 ; copier d'abord vers un répertoire temporaire, vérifier complétude/hashes puis rendre disponible. Conserver snapshot/provenance pour traçabilité. Ne pas transférer seulement les `.pkl` ni déclarer valide un manifest sans fichiers correspondants.
- [ ] Documenter VM GPU : installer le trainer aux mêmes versions, vérifier le dataset et enregistrer le checkpoint M-B parent avec son hash ; configurer le fine-tuning BC existant vers cet ID de dataset et `device: cuda`. La conversion n'utilise ni checkpoint parent ni GPU. Le choix/location de VM et les transferts effectifs sont des opérations ultérieures, pas des actions à lancer automatiquement à la fin de l'implémentation.
- [ ] Vérifier documentation et commandes, ajouter les résultats mesurés (pas des estimations présentées comme mesures), commit `docs: validate local parallel prepare and CPU GPU handoff`.

## Matrice de validation finale

| Exigence | Tâches / preuve |
| --- | --- |
| Auto utilise la capacité CPU accessible sans valeur machine figée | 2, 7 — cas simulés affinité/quota + diagnostic réel. |
| Override utile sur VM contrainte en RAM | 1, 2, 7 — config/API/CLI et valeur effective visible. |
| Réel `ProcessPoolExecutor` multicœur | 3, 6 — workers différents observés simultanément. |
| Provenance et splits stables | 4–7 — comparaison 1/2/4/auto avec mêmes données. |
| Erreurs bloquantes empêchent ready | 4–6 — défaut injecté après écritures partielles. |
| Aucune fuite de processus dans les parcours testés | 3, 6, 7 — descendants attendus, interruption, timeout, répétition. |
| Mémoire des trajectoires bornée pour 100k | 4–6 — nombre de futurs borné et consommation progressive. |
| Collecte 100k conservée | Inventaire/diff final ; aucune action sur sa config ou ses données. |
| Déplacement CPU puis GPU documenté | 7 — archive cohérente, dataset complet avec checksums, checkpoint parent. |

Avant remise : exécuter `git diff --check`, présenter les tests exécutés et le rapport smoke, distinguer les échecs préexistants du chantier, et lister les vérifications non réalisées. Ne pas présenter la vitesse sur 100k comme mesurée par un smoke de 20 replays. Seule l'implémentation suivie de ces vérifications permet d'affirmer que prepare multicœur fonctionne ; ce document est le plan pour y parvenir.
