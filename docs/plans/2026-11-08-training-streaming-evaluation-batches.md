# Fine-tuning : lecture progressive et évaluation par batches — Plan d'implémentation

> **Pour Composer / agent exécutant :** implémenter ce plan tâche par tâche, avec les cases à cocher et les tests indiqués. Si disponible, appliquer `superpowers:executing-plans`. Lire les contrats des sections 1–5 avant de coder. La demande actuelle est la rédaction du plan ; ne pas lancer un entraînement complet, louer une machine ou modifier les données pendant sa rédaction.

**Goal:** Entraîner sur le dataset prêt de 80 900 replays sans le charger entièrement en RAM, et calculer la NLL par batches sans changer sa définition ni modifier le modèle entraîné pendant l'évaluation.

**Architecture:** Une source réitérable lit les pickles existants un par un et produit des batches de transitions. À l'entraînement, un tampon de taille bornée mélange les transitions ; à l'évaluation, la lecture est séquentielle. `CompleteBatchBC` consomme ces batches via la boucle `imitation` existante, tandis qu'une évaluation vectorisée utilise une copie isolée de la policy.

**Tech Stack:** Python ≥3.13, NumPy, PyTorch, `imitation`, Stable-Baselines3, VGC-Bench, Pydantic 2, pytest, environnement uv verrouillé existant. Aucune nouvelle dépendance de production.

**Spec:** Les sections 1–5 de ce document constituent le contrat technique des deux adaptations demandées le 8 novembre 2026. Contexte du dataset : [recette 80k](../recettes/prepare-dataset-mc-80k.md). La [préparation multicœur](2026-11-04-prepare-dataset-multicoeur.md) est déjà réalisée et n'est pas à réimplémenter.

## Contraintes globales

- Le dataset `ds-mb-mc-collectable-80k-v1` reste immuable : aucun changement de pickle, manifest, index, recette ou split. Aucun nouveau `prepare` requis.
- Conserver le parent `vgcbench-mb-reference`, l'architecture héritée, les masques conditionnels, les actions, la fonction de perte BC et l'optimiseur existants.
- Ne pas réduire le dataset, supprimer la validation ou évaluer seulement un échantillon pour annoncer une accélération.
- Conserver `batch_size: 32`, `num_epochs: 3`, `learning_rate: 0.0001` dans le profil de production. Le batch d'évaluation est un paramètre distinct.
- Une epoch réussie consomme exactement une fois chaque transition train, y compris le dernier batch incomplet ; ne pas compter l'observation terminale comme une action.
- Ne pas recopier la boucle d'optimisation `imitation`, modifier `.venv`, patcher le vendor, utiliser `PPO.learn()`, ni convertir les poids en FP16/BF16 dans ce chantier.
- Un seul processus de lecture pour cette V1 : pas de workers DataLoader, de queue de préchargement, de threads de lecture ou de pool imbriqué. Optimiser cela plus tard uniquement si les mesures le justifient.
- Pas de nouveau cache disque de dizaines de Go. Lire les pickles déjà disponibles, avec une liste de chemins en mémoire autorisée.
- Distinguer mémoire des tableaux utilisateur, mémoire du modèle/framework, mémoire GPU et cache disque du système. Ne pas promettre un plafond RSS total égal au seul tampon.
- Le budget cloud de 50 € motive ces changements ; les tests locaux ne prouvent ni la durée GPU ni le prix final d'un entraînement.
- Préserver les modifications utilisateur déjà présentes dans les YAML, ainsi que `storage/`. Si des commits sont réalisés, utiliser `benacFlorian <florianbenac46@gmail.com>` comme auteur et sélectionner explicitement les fichiers du chantier.

## Points de revue

1. Une ligne NumPy gardée comme vue retient toute sa trajectoire : recopier dans les buffers possédés, vérifier les durées de vie — tâches 2 et 7.
2. `imitation` calcule la taille via `len(batch["obs"])` : un simple dictionnaire à deux clés donne 2, pas la taille du batch ; utiliser `DictObs` — tâches 2 et 3.
3. 1, 31, 32, 33 transitions, répétition d'epoch et exception de lecture : pas de perte, répétition, précomptage fictif ou succès partiel — tâches 2, 3 et 5.
4. Embeddings `max_norm` et second masque dépendant de la première action : l'évaluation doit rester isolée et équivalente numériquement — tâche 4.
5. Tampon qui fuit, splits chargés ensemble, ou mémoire croissant à chaque epoch : vérifier structurellement et dans des processus séparés — tâches 5 et 7.

---

## 1. Constats vérifiés et limites de preuve

Chemins `src/`, `tests/`, `configs/` et `scripts/` relatifs à `pokecoach2-trainer/pokecoach-trainer/`. Les commandes `uv` se lancent depuis ce sous-projet.

Le manifeste local lu le 8 novembre indique :

| Élément | Valeur |
| --- | ---: |
| État | `ready` |
| Replays sélectionnés | 80 900 |
| Trajectoires train / validation / test | 111 923 / 14 018 / 13 986 |
| Transitions train / validation / test | 1 051 059 / 132 224 / 130 181 |
| Taille des pickles train / validation / test | environ 31,035 / 3,902 / 3,847 Gio |
| Paramètres du parent enregistré | 2 941 335 |

Ce sont des lectures de métadonnées et tailles de fichiers, pas une vérification intégrale des hashes de 139 927 trajectoires.

### 1.1 Chemin actuel

| Emplacement | Comportement à remplacer / préserver |
| --- | --- |
| `training.py::load_split_trajectories` | Retourne une liste de tous les pickles désérialisés. Garder éventuellement comme helper legacy, jamais dans le parcours de production. |
| `training.py::train_bc` | Charge train, validation et test ensemble ; appelle `set_demonstrations` avant puis dans la boucle d'epochs. |
| `bc_adapter.py::CompleteBatchBC.set_demonstrations` | Aplatit tout via `rollout.flatten_trajectories`, notamment `obs` et `next_obs`, alors que BC ne consomme que `obs`/`acts`. |
| `bc_adapter.py::remember_trajectories` | Conserve une liste de trajectoires pour déduire le compteur ; à éviter dans le nouveau parcours. |
| `bc_adapter.py::evaluate_nll` | Crée une copie de policy, puis un appel `evaluate_actions` et une synchronisation CPU par transition ; conserve une liste de toutes les pertes. |
| `trajectory_dataset.py::load_trajectory_pickle` | Point de lecture existant avec conversion legacy vers `DictObs` et vérification des actions. À réutiliser. |
| `training_service.py::preflight` | Vérifie dataset ready, parent et contrat. Garder les protections également pour le chemin CLI/direct. |
| `schemas.py::TrainingConfig` | Pas de paramètres de mémoire/évaluation ; `extra="forbid"`. |
| `cli.py::_load_experiment` | Valide déjà le bloc YAML `training` via `TrainingConfig`. |

La bibliothèque installée dans `.venv/lib/python3.13/site-packages/imitation/algorithms/bc.py` accepte un iterable de mappings de batches via `_demo_data_loader`. Sa boucle `train` consomme `obs` et `acts`, et gère le dernier batch incomplet. Inspecter ce fichier en lecture seule ; ne pas dépendre d'une modification manuelle de cet environnement.

Le modèle dans `vendor/vgc-bench/vgc_bench/src/policy.py` accepte les batches dans `evaluate_actions`, applique le masque de la deuxième action en fonction de la première et possède des embeddings `max_norm`. Ne pas réécrire ce calcul dans le trainer.

### 1.2 Approche retenue

Choix de ce plan : **lecture des fichiers + mélange par tampon borné**, et non index global de chaque transition dans un cache supplémentaire. Cela évite une seconde préparation coûteuse et une duplication du dataset sur le SSD loué.

Compromis explicite : le mélange borné n'est pas une permutation uniforme globale des 1 051 059 transitions. Il conserve la couverture exacte et offre un ordre déterministe pour une seed/epoch/config donnée, mais les poids entraînés ne seront pas identiques à ceux de l'ancien DataLoader. Ne pas promettre une équivalence bit à bit d'entraînement. La NLL à poids fixés, en revanche, doit rester équivalente à la version scalaire dans la tolérance définie plus bas.

## 2. Configuration et contrats de données

### 2.1 Configuration proposée

Ajouter à `TrainingConfig` deux champs :

```python
shuffle_buffer_mib: StrictInt = Field(default=256, ge=1, le=4096)
evaluation_batch_size: StrictInt = Field(default=256, ge=1)
```

Ces défauts s'appliquent aux anciens YAML/JSON. Rejeter `True`, `0`, les négatifs, les flottants et les chaînes numériques pour les nouveaux champs ; aucun correctif automatique silencieux. `shuffle_buffer_mib` est une allocation maximale pour les tableaux du tampon de mélange, pas un plafond de RAM du processus.

Le profil `configs/finetune-mb-mc-prod.yaml` reçoit seulement ces deux valeurs explicites dans `training`, sans changer ses autres paramètres. La CLI/API existante doit les propager jusqu'au job et au résultat. Pas de nouveau mode `legacy` public ni de nouvelle commande cloud.

### 2.2 Source des splits

Créer `src/pokecoach_trainer/training_data.py` avec :

```python
SplitName = Literal["train", "validation", "test"]

@dataclass(frozen=True, slots=True)
class TrajectorySplitSource:
    split: SplitName
    files: tuple[Path, ...]
    expected_transitions: int

def build_split_source(
    dataset_dir: Path, manifest: DatasetManifest, split: SplitName,
) -> TrajectorySplitSource: ...

def iter_source_trajectories(
    source: TrajectorySplitSource,
    *, file_order: Sequence[int] | None = None,
) -> Iterator[Trajectory]: ...

def iter_transition_batches(
    trajectories: Iterable[Trajectory], *, batch_size: int,
) -> Iterator[TransitionBatch]: ...

class TransitionBatch(TypedDict):
    obs: DictObs
    acts: np.ndarray
```

`build_split_source` ne désérialise aucun pickle. Il trie les chemins de manière stable et vérifie les nombres de fichiers du manifeste (`trajectories_train`, `trajectories_val`, `trajectories_test`). La convention du dossier est `validation`, celle du champ manifeste est `val`. Les anciens datasets ready sans `trajectory_index.jsonl` restent supportés.

Si `trajectory_index.jsonl` existe, le parcourir ligne à ligne pour vérifier que les chemins/splits/comptages décrivent exactement les fichiers retenus et les totaux du manifeste. Pas de `read_text().splitlines()` ni de liste des lignes complètes. Refuser doublon de chemin, chemin hors split, fichier absent, fichier supplémentaire ou total divergent. Les lignes n'ajoutent aucune transition au mélange ; elles servent uniquement à vérifier les métadonnées. Ne pas refaire systématiquement tous les hashes de fichiers à chaque epoch dans ce chantier.

Pour une trajectoire de T actions : prendre `observation[:T]`, `action_mask[:T]`, `acts[:T]` ; l'observation numéro T n'est pas un exemple supervisé. Conserver les dtypes observation/masque et produire des actions `np.int64` de forme `(B, 2)`. Refuser les formes incohérentes et les trajectoires vides ; ne pas les ignorer pour atteindre artificiellement un nombre attendu.

`iter_transition_batches` remplit des tableaux possédés, regroupe plusieurs trajectoires jusqu'à B lignes et émet un dernier batch partiel. Chaque batch est indépendant des sources et du suivant, y compris si le consommateur le conserve ou le modifie. Ne pas exposer une vue d'un buffer réutilisé. Ce helper sert à l'évaluation et aux tests legacy ; il n'effectue aucun mélange.

### 2.3 Source train mélangée et réitérable

Toujours dans `training_data.py` :

```python
SAMPLER_VERSION = "file_buffer_shuffle/1"

class ShuffledTransitionBatches:
    def __init__(
        self, source: TrajectorySplitSource, *, batch_size: int,
        shuffle_buffer_mib: int, seed: int, epoch: int,
    ) -> None: ...
    def __iter__(self) -> Iterator[TransitionBatch]: ...
```

À chaque appel de `__iter__`, créer un RNG local issu de `np.random.SeedSequence([seed, epoch, 1])`, puis mélanger une permutation des indices des fichiers. Ne pas utiliser `hash()` de Python, les RNG globaux, ni conserver un générateur épuisé entre deux itérations.

Algorithme obligatoire pour éviter qu'un exécutant invente un rééchantillonnage avec remise :

1. Déduire la structure des lignes à la première trajectoire et calculer `row_bytes = observation[0].nbytes + action_mask[0].nbytes + 2 * sizeof(int64)`.
2. `capacity = min(source.expected_transitions, budget_bytes // row_bytes)`, avec `budget_bytes = shuffle_buffer_mib * 1024**2`. Si une seule ligne ne rentre pas, erreur explicite. Allouer des tableaux contigus de cette capacité, pas une liste de vues NumPy.
3. Remplir le tampon avec les premières transitions en recopiant chaque ligne. Il peut contenir moins de `capacity` lignes si la source termine tôt ; le contrôle de compte total signalera ensuite l'incohérence éventuelle.
4. Pour chaque nouvelle ligne lorsque le tampon est plein : tirer uniformément un index, copier l'ancienne ligne dans le batch de sortie, puis la remplacer dans le tampon par la nouvelle ligne.
5. À la fin de la source : tirer un index parmi les lignes restantes, copier cette ligne vers la sortie, remplacer sa case par la dernière case active et réduire la taille active. Répéter jusqu'à vidage. Cela émet chaque ligne une fois.
6. Le batch de sortie traverse les frontières de fichiers et le vidage final. Seul le tout dernier batch peut être incomplet.

La mémoire de données vivantes doit dépendre du tampon, de la plus grosse trajectoire désérialisée et de quelques batches, pas de la somme des trajectoires. Une liste de chemins/permutation reste O(nombre de fichiers) et est acceptable. Supprimer les références aux observations/trajectoires précédentes avant de charger la suivante : attention aux variables locales de générateurs et aux slices.

Limite assumée : les pickles existants imposent de charger une trajectoire entière. Une trajectoire exceptionnellement volumineuse peut dépasser la taille du tampon ; ne pas qualifier le lecteur de plafonné strictement à 256 Mio. Aucune désérialisation globale des fichiers et aucun buffer ne doivent grossir automatiquement en cas de mémoire insuffisante.

## 3. Intégration avec `CompleteBatchBC`

Ajouter à `bc_adapter.py` :

```python
def set_transition_batches(
    self, batches: Iterable[TransitionBatch], *,
    expected_transitions: int, batch_size: int,
) -> None: ...
```

Cette méthode installe un iterable enveloppé dans `_demo_data_loader`, sans appeler `set_demonstrations`, `flatten_trajectories`, `list(batches)` ou `remember_trajectories`.

- `effective_batch_size = min(batch_size, expected_transitions)` ; les deux valeurs doivent être >0. Fixer `self.batch_size` et `self.minibatch_size` à cette valeur. Le paramètre `batch_size` de la source reçoit aussi cette valeur.
- L'enveloppe mesure le nombre **réel de lignes livrées** par `len(batch["acts"])`, vérifie l'accord avec `len(batch["obs"])`, puis incrémente `transitions_seen` quand elle livre chaque batch. Ne pas remplacer cette mesure par `expected_transitions` après `train()`.
- Conserver le wrapper de `optimizer.step` existant pour le compteur de pas et les gradients non finis.
- `train()` remet les compteurs à zéro au début. Après un `train(n_epochs=1)` réussi, exiger exactement le nombre attendu de lignes et `ceil(N/effective_batch_size)` pas. Un flux tronqué ou doublé doit échouer, pas publier une epoch réussie.
- L'appel de production reste un `train(n_epochs=1)` par epoch, comme aujourd'hui. Pour le nouvel interface, refuser explicitement `n_batches` ou `n_epochs != 1` jusqu'à l'ajout d'un contrat de comptage multi-epoch ; l'ancien interface legacy peut conserver sa compatibilité.
- Conserver la pondération existante du dernier batch (`tail_size / effective_batch_size`) : ce chantier ne change pas la définition de l'apprentissage.
- Ne pas confondre cette pondération d'apprentissage avec la moyenne NLL d'évaluation, qui utilise le nombre exact de transitions.

Garder `set_demonstrations` et `remember_trajectories` pour les petits tests/appels legacy, mais enlever toute référence legacy active lorsqu'on installe `set_transition_batches`. Un changement de mode sur le même objet ne doit pas retenir les anciens tableaux ni écraser le nouveau compteur avec une somme legacy.

## 4. Évaluation vectorisée, bornée et isolée

Créer `src/pokecoach_trainer/bc_evaluation.py` :

```python
def evaluate_nll_batches(
    policy, batches: Iterable[TransitionBatch], device: str,
) -> EvaluationResult: ...
```

Conserver dans `bc_adapter.py` la fonction publique existante, avec extension compatible :

```python
def evaluate_nll(
    policy, demonstrations: Iterable[Trajectory], device: str,
    *, batch_size: int = 256,
) -> EvaluationResult: ...
```

Ce wrapper transforme le flux avec `iter_transition_batches` puis délègue au nouvel évaluateur. Importer ce dernier localement dans le wrapper si nécessaire pour éviter le cycle avec `duplicate_policy`, qui peut rester dans `bc_adapter.py`.

Règles de calcul :

1. Créer une seule copie de policy par appel complet, via `duplicate_policy`, puis `eval()`, `actor_grad=False`, `requires_grad=False`.
2. Préserver les états RNG Torch CPU/CUDA concernés lors de la construction et utilisation de la copie (`torch.random.fork_rng` adapté au device). L'évaluation ne doit pas changer l'ordre de mélange ni le hasard du prochain apprentissage. Ne pas initialiser CUDA si l'évaluation est CPU.
3. Sous `torch.no_grad()`, transférer seulement les tableaux du batch courant. Déballer `DictObs` en dictionnaire de tenseurs pour `policy.evaluate_actions`.
4. Un seul appel `clone.evaluate_actions(obs, acts)` par batch. Vérifier que `log_prob` fournit une valeur par transition ; ne pas moyenner séparément les deux actions et ne pas réimplémenter leurs masques conditionnels.
5. Vérifier les valeurs finies. Accumuler `(-log_prob).sum(dtype=torch.float64)` dans un scalaire et ajouter B au compteur. Éviter une liste de pertes et les appels `.item()` par transition.
6. Retourner `total_negative_log_probability / total_transitions`, jamais une moyenne non pondérée des moyennes de batches. Un flux vide lève `ValueError("évaluation sans transitions")`, comme aujourd'hui.
7. Libérer copie et références aux tenseurs en `finally`, y compris après erreur. Ne pas appeler `torch.cuda.empty_cache()` à chaque batch comme substitut à la libération des références.

Le modèle source, son mode train/eval, ses gradients, ses poids et buffers doivent rester inchangés. C'est nécessaire même sous `no_grad()` : les embeddings `max_norm` peuvent modifier leurs poids lors d'un forward. Les tableaux d'observation/masque source restent inchangés également.

Référence de vérité : conserver **dans les tests seulement** l'ancienne boucle scalaire exécutée sur sa propre copie indépendante, initialisée avec les mêmes poids. Comparer sur la policy réelle et des observations valides, avec `abs_tol=1e-6, rel_tol=1e-5` sur CPU FP32 ; sur CUDA FP32 utiliser `abs_tol=1e-5, rel_tol=1e-4`. En cas d'échec, diagnostiquer masques, poids, ordre et modes avant d'élargir une tolérance.

Pour 1 051 059 transitions train et 132 224 validation, une évaluation à batch=256 doit appeler la policy respectivement 4 106 et 517 fois, au lieu d'une fois par transition. Le test final de 130 181 transitions demande 509 appels. Ce sont des nombres d'appels, pas des facteurs d'accélération mesurés.

## 5. Intégration, métriques et périmètre

Dans `training.py::train_bc` :

1. Vérifier dataset ready et compatibilité parent comme aujourd'hui, y compris pour un appel direct à `train_bc`. Ne pas ajouter un import cyclique vers `training_service`, qui importe déjà `train_bc` ; factoriser seulement si nécessaire.
2. Construire les trois sources de chemins et de compteurs. Ne désérialiser aucun split à ce stade. Train vide → erreur ; validation/test véritablement vides → comportement actuel conservé.
3. Pour chaque epoch, créer `ShuffledTransitionBatches` avec seed et numéro d'epoch (commencer à 1), l'installer, puis appeler BC une fois. Le générateur doit avoir libéré tampon et trajectoire avant l'évaluation ; le libérer explicitement si une exception interrompt la consommation.
4. Évaluer train puis validation via de **nouveaux** itérateurs séquentiels des fichiers et `iter_transition_batches(..., batch_size=evaluation_batch_size)`. Aucun pickle du test n'est lu avant la fin de toutes les epochs.
5. Conserver checkpoints, preuves de variation des poids, parent inchangé, `result.json`, et test final isolé. Avant la publication d'une métrique réussie, vérifier les totaux consommés contre le manifeste ; ne pas transformer une erreur de lecture en split vide.

Compléter chaque ligne `metrics.jsonl` sans supprimer les champs actuels :

- `training_batch_size`, `evaluation_batch_size`, `shuffle_buffer_mib`, `sampler_version` ;
- `train_duration_s`, `train_eval_duration_s`, `validation_duration_s` ;
- `train_transitions_per_s = observed_transitions / train_duration_s` ;
- `epoch_wall_duration_s`, incluant l'installation/lecture du flux, l'entraînement, les évaluations et la sauvegarde du checkpoint.

Mesurer avec `time.perf_counter()`. Synchroniser CUDA uniquement aux bornes des segments chronométrés lorsqu'il est utilisé ; aucune synchronisation supplémentaire par transition. Garder le champ historique `duration_s` avec son périmètre documenté, ou l'aligner explicitement sur `epoch_wall_duration_s` en le documentant.

Ajouter `TrainingResult.training_execution: dict[str, Any] = Field(default_factory=dict)` pour les options effectives et la version du sampler. Les anciens résultats restent lisibles. Reporter la mémoire réelle dans le benchmark de tâche 7 plutôt que multiplier les appels système à chaque batch.

**Hors périmètre :** location/upload cloud, entraînement complet local, changement de learning rate/batch train/epochs, entraînement distribué, AMP, gradient accumulation, early stopping, reprise complète de l'état d'optimiseur, modification des splits BO3. Le manifeste indique actuellement `series_leakage_guarantee=false` ; ces adaptations ne corrigent pas ce point et ne doivent pas le déclarer résolu.

---

## 6. Séquence d'implémentation pour Composer

Chaque tâche : écrire les tests cités → lancer pour observer le défaut attendu → implémenter → relancer → examiner le diff. Les commandes ci-dessous sont des instructions pour l'exécution future, pas des tests déjà exécutés lors de la rédaction. Faire des commits locaux par tâche si cela correspond au workflow choisi, sans `git add .`.

### Tâche 1 — Configuration et sources de fichiers validées

**Fichiers :** modifier `src/pokecoach_trainer/schemas.py` ; créer `src/pokecoach_trainer/training_data.py`, `tests/test_training_data.py`, `tests/test_training_execution_config.py` ; compléter `tests/test_training_api.py` et `tests/training_fixtures.py`.

**Produit :** les deux nouveaux champs `TrainingConfig`, `TrajectorySplitSource`, `build_split_source`, `iter_source_trajectories` des sections 2.1/2.2. Pas encore de mélange.

- [ ] Tester valeurs par défaut 256/256, rejet booléens/zéro/négatifs/float/chaîne, limites du tampon 1/4096 Mio, sérialisation JSON aller-retour et chargement via `cli._load_experiment`.
- [ ] Tester qu'une construction de source ne déclenche **aucun** `load_trajectory_pickle`. Vérifier chemins triés, mapping validation→`transitions_val`, manifeste sans index accepté, index incohérent/fichiers manquants refusés.
- [ ] Tester un itérateur dont le chargeur instrumenté ne charge que le premier pickle lors du premier `next()`, puis le suivant à la demande. Une corruption remonte avec le chemin concerné.
- [ ] Lancer `uv run --locked pytest tests/test_training_execution_config.py tests/test_training_data.py tests/test_training_api.py` : nouveaux tests rouges avant implémentation.
- [ ] Implémenter les interfaces sans accumuler de trajectoires. Enrichir les fixtures pour pouvoir produire des longueurs différentes et un index optionnel ; garder les anciennes fixtures utilisables.
- [ ] Relancer la commande : tests verts. Aucun fichier du dataset réel modifié.

### Tâche 2 — Batches séquentiels et mélange borné

**Fichiers :** compléter `src/pokecoach_trainer/training_data.py`, `tests/test_training_data.py` ; créer `tests/test_training_shuffle.py`.

**Produit :** `TransitionBatch`, `iter_transition_batches`, `ShuffledTransitionBatches`, `SAMPLER_VERSION` selon section 2. Utilise le lecteur de tâche 1.

- [ ] Créer des trajectoires dont chaque transition porte un identifiant unique et dont l'observation terminale porte une sentinelle distincte. Tester longueurs `[2, 3, 1]`, batch=4 : sorties de tailles `[4, 2]`, six identifiants uniques, aucune sentinelle.
- [ ] Tester types/alignement : `isinstance(batch["obs"], DictObs)`, `len(obs)==len(acts)`, dtype des actions int64 ; même observation, masque et action associés après mélange.
- [ ] Tester N=1/31/32/33 et des capacités effectives 1, inférieures/supérieures à N. Comparer `Counter(ids_sortis)` à `Counter(ids_entrants)` et le total : aucun tirage avec remise.
- [ ] Tester même seed+epoch → même ordre sur deux itérations et deux objets ; epoch différente → ordre différent sur une fixture assez grande ; état RNG NumPy/Python global inchangé.
- [ ] Conserver un batch, avancer le générateur et modifier le batch suivant : le premier ne change pas. Vérifier `np.shares_memory` faux avec les sources. Utiliser des références faibles pour vérifier que le tampon ne conserve pas toutes les trajectoires déjà traversées.
- [ ] Lancer `uv run --locked pytest tests/test_training_data.py tests/test_training_shuffle.py` : rouge.
- [ ] Implémenter le tampon contigu et l'algorithme remplacement/vidage décrit, sans `list(trajectories)`, `np.concatenate` global ni conservation de slices de trajectoires. Le générateur ferme ses ressources et références en sortie normale et en `close()`.
- [ ] Relancer : vert. Mesurer les allocations NumPy possédées du tampon et vérifier `sum(array.nbytes) <= budget_bytes` ; ce test ne prétend pas mesurer toute la RAM du processus.

### Tâche 3 — Brancher les batches dans la boucle BC existante

**Fichiers :** modifier `src/pokecoach_trainer/bc_adapter.py` ; compléter `tests/test_bc_batches.py` ; créer `tests/test_bc_streaming.py`.

**Produit :** `CompleteBatchBC.set_transition_batches`, comptage observé de la section 3. Aucun changement à la perte ou à la policy vendor.

- [ ] Réutiliser la petite vraie policy des tests existants. Pour N=1/31/32/33 et batch demandé=32, vérifier `observed_transitions=N` et les pas `[1,1,1,2]` ; exécuter deux appels d'une epoch et vérifier la remise à zéro des compteurs.
- [ ] Remplacer `rollout.flatten_trajectories` par une fonction qui lève : le nouveau chemin doit entraîner malgré cela. Vérifier absence de `_cached_trajectories` contenant le dataset dans ce mode.
- [ ] Ajouter un flux annoncé N=33 mais fournissant 32 lignes, et un flux fournissant 34 : erreurs explicites après consommation, pas succès avec un compteur recopié du manifeste. Rejeter `n_batches`/`n_epochs=2` dans le nouveau mode.
- [ ] Tester le dernier batch : sur les mêmes batches ordonnés, même parent et optimiseur, les nouveaux et anciens chemins donnent le même nombre de pas et des poids proches. Ce test fige les entrées ; il ne compare pas deux algorithmes de shuffle.
- [ ] Lancer `uv run --locked pytest tests/test_bc_batches.py tests/test_bc_streaming.py` : rouge.
- [ ] Installer le wrapper d'iterable et le comptage réel sans recopier `BC.train`. Garder les tests legacy verts ; supprimer la référence legacy lorsqu'on change de mode.
- [ ] Relancer : vert. Tester une exception de lecture au milieu d'epoch : elle se propage, le flux se ferme et aucun compteur de succès n'est fabriqué.

### Tâche 4 — NLL par batches avec parité et isolation

**Fichiers :** créer `src/pokecoach_trainer/bc_evaluation.py`, `tests/test_batched_evaluation.py` ; modifier `src/pokecoach_trainer/bc_adapter.py` ; renforcer `tests/test_evaluation_isolation.py`, `tests/test_training.py`.

**Produit :** `evaluate_nll_batches` et wrapper compatible `evaluate_nll(..., batch_size=256)`. Utilise `iter_transition_batches` de tâche 2 et `duplicate_policy` existant.

- [ ] Écrire un oracle scalaire dans le module de tests, copié conceptuellement du calcul existant, et comparer aux batches 1/2/32/256 avec 33 transitions de valeurs variées et plusieurs trajectoires.
- [ ] Tester avec une fausse policy contrôlée des pertes de `[1, 1, 1, 9, 9]`, batch=3 : résultat **4,2**, pas 5. Vérifier exactement `ceil(N/B)` appels au clone et aucune liste de N pertes.
- [ ] Tester sur vraie policy des actions où le second masque dépend de la première : switchs distincts, passe forcée et cas légal de mécanique supportée par les fixtures. Les tableaux doivent rester inchangés ; ne pas inventer des IDs d'actions sans consulter les conventions vendor.
- [ ] Forcer un embedding `max_norm` au-delà de sa norme puis comparer les `state_dict` de la policy source avant/après avec égalité exacte. Vérifier gradients, flags train/eval et `actor_grad`, ainsi que RNG CPU et CUDA disponible. Effectuer ce contrôle aussi après une erreur injectée.
- [ ] Tester entrée vide, NaN/Inf et forme de log_prob incohérente. Le test d'isolation normal doit **réussir** et retourner une NLL finie : enlever les `except ValueError: pass` permissifs des tests existants ; garder des tests d'erreur séparés.
- [ ] Lancer `uv run --locked pytest tests/test_batched_evaluation.py tests/test_evaluation_isolation.py tests/test_training.py` : rouge.
- [ ] Implémenter accumulation pondérée float64, copie isolée et fermeture selon section 4. Garder la compatibilité des appels à trois arguments d'`evaluate_nll` et d'`evaluate_validation`.
- [ ] Relancer sur CPU : vert aux tolérances prévues. Ajouter des tests CUDA conditionnels avec `skip` explicite si indisponible ; ne pas les déclarer validés en l'absence de GPU.

### Tâche 5 — Parcours d'entraînement complet et métriques

**Fichiers :** modifier `src/pokecoach_trainer/training.py`, `schemas.py` ; compléter `tests/test_training.py`, `test_finetuning.py`, `test_training_api.py` ; créer `tests/test_training_streaming_integration.py`.

**Produit :** `train_bc` utilisant exclusivement les sources/batches pour les données et `TrainingResult.training_execution`, métriques section 5. La CLI/API existante conserve ses routes/commandes.

- [ ] Construire un dataset temporaire avec 33 transitions train, 5 validation et 7 test, avec index cohérent ; deux epochs, batch train=32, batch évaluation=4. Attendre `[33,33]` transitions observées et `[2,2]` pas d'optimiseur.
- [ ] Interdire par monkeypatch `load_split_trajectories`, `flatten_trajectories` et `remember_trajectories` : le test d'intégration doit tout de même produire ses checkpoints et `result.json`.
- [ ] Instrumenter les lectures : aucun pickle test ouvert avant la fin des epochs ; les lectures train/validation d'évaluation sont séquentielles et ne réutilisent pas le générateur d'apprentissage épuisé.
- [ ] Vérifier parent inchangé, variation des poids acteur, NLL finies, provenance du dataset et champs de métriques nouveaux/existants. Un ancien `TrainingResult` sans `training_execution` doit rester lisible.
- [ ] Tester train vide, dataset non-ready, comptes incohérents, validation réellement vide et fichier validation manquant : vide autorisé seulement si déclaré vide, fichier manquant = erreur.
- [ ] Lancer `uv run --locked pytest tests/test_training_streaming_integration.py tests/test_training.py tests/test_finetuning.py tests/test_training_api.py` : rouge.
- [ ] Intégrer les flux, validations de comptes et timings de section 5. Ne pas reconstruire de DataLoader global « juste pour les métriques ». Fermer explicitement les itérateurs en erreur, préserver les checkpoints déjà terminés sans annoncer la réussite du run interrompu.
- [ ] Relancer : vert. Confirmer que ce parcours fonctionne à partir des pickles ready actuels et ne déclenche aucun `prepare`.

### Tâche 6 — Profil et documentation d'utilisation

**Fichiers :** modifier uniquement les champs concernés dans `configs/finetune-mb-mc-prod.yaml`, `README.md`, `../docs/recettes/prepare-dataset-mc-80k.md` ; compléter `tests/test_training_execution_config.py`.

**Produit :** configuration explicite et explication opérationnelle, sans lancement de location ou entraînement.

- [ ] Tester le chargement du vrai YAML : dataset/parent/learning_rate/batch train/epochs inchangés, nouveaux champs à 256/256.
- [ ] Ajouter `shuffle_buffer_mib: 256` et `evaluation_batch_size: 256`. Préserver les autres modifications utilisateur au lieu de régénérer tout le fichier.
- [ ] Documenter : différence batch train/batch évaluation ; RAM = tampon + trajectoire courante + batches + framework ; lecture disque répétée par epoch ; mélange borné déterministe mais différent de l'ancien shuffle global ; possibilité de diminuer le batch évaluation en cas d'OOM, sans retry automatique.
- [ ] Documenter la mesure préalable CPU/GPU et le fait qu'un dataset ready reste réutilisable sans conversion. Sur une VM GPU, demander explicitement `device: cuda:0` dans une copie de configuration pour éviter un repli CPU silencieux ; ne pas imposer CUDA au profil portable du dépôt.
- [ ] Lancer `uv run --locked pytest tests/test_training_execution_config.py` : vert. Relire les exemples pour vérifier qu'aucune commande de benchmark ne lance automatiquement trois epochs sur les 80k.

### Tâche 7 — Benchmark reproductible et preuve de mémoire bornée

**Fichiers :** créer `scripts/benchmark_training_pipeline.py`, `tests/test_training_memory.py`, `tests/test_training_benchmark.py` et, après exécution seulement, `../docs/audits/2026-11-08-training-streaming-resultats.md`.

**Produit :** un benchmark limité en lecture seule du dataset avec sorties dans un dossier de run distinct. Arguments exacts : `--dataset-id`, `--base-model-id`, `--device cpu|cuda:0`, `--max-files` (défaut 64), `--batch-size` (32), `--evaluation-batch-size` (256), `--shuffle-buffer-mib` (256), `--output` (chemin JSON obligatoire). Refuser `--max-files < 1` et les sorties à l'intérieur du dataset/parent.

- [ ] Tester les arguments et qu'une exécution ne lit pas au-delà du sous-ensemble demandé. Prendre les premiers fichiers train triés, compter leurs vraies transitions en lecture progressive et traiter ce sous-ensemble comme une source distincte pour les assertions ; ne pas comparer ses comptes à ceux du train complet.
- [ ] Construire une comparaison NLL scalaire/batchée sur le **même sous-ensemble et les mêmes poids** avec copies indépendantes, sans apprentissage préalable de l'une seulement. Les temps incluent la lecture pour le chiffre « bout en bout » ; afficher séparément si un temps de calcul pur est mesuré.
- [ ] Mesurer une epoch de BC streaming sur ce petit sous-ensemble avec un parent chargé en copie ; ne pas sauvegarder cette policy comme modèle de production. Montrer transitions/s, durée du chargement/mélange, évaluation et mémoire, pas seulement un temps de forward.
- [ ] Chaque mode de comparaison mémoire s'exécute dans un processus enfant frais : ne pas réutiliser un `ru_maxrss` qui conserve le maximum d'un autre mode. Pour Linux utiliser la bibliothèque standard (`resource` et, si utile, `/proc`) ; relever aussi `torch.cuda.max_memory_allocated()` lorsque CUDA est réellement utilisé, avec remise à zéro par mesure.
- [ ] Tester la propriété mémoire avec un chargeur synthétique instrumenté : même taille de trajectoire et de tampon, 100 puis 1000 trajectoires. Le nombre de trajectoires simultanément retenues reste constant ; les allocations du tampon restent bornées ; la hausse RSS, hors métadonnées, ne doit pas approcher la taille cumulée des trajectoires. Fixer les tailles de fixture et une marge dans le test, pas une assertion universelle sur tout l'environnement PyTorch.
- [ ] Ajouter une épreuve sur trois epochs pour détecter les références retenues dans `CompleteBatchBC`. Pas d'assertion « RSS revient exactement à zéro » : les allocateurs peuvent garder des blocs réutilisables.
- [ ] Lancer `uv run --locked pytest tests/test_training_memory.py tests/test_training_benchmark.py` : rouge, puis implémenter et obtenir vert.
- [ ] Lancer un benchmark local borné, en commençant par 8 fichiers, CPU si aucun GPU compatible :

```bash
uv run --locked python scripts/benchmark_training_pipeline.py \
  --dataset-id ds-mb-mc-collectable-80k-v1 \
  --base-model-id vgcbench-mb-reference \
  --device cpu --max-files 8 \
  --batch-size 32 --evaluation-batch-size 256 \
  --shuffle-buffer-mib 256 \
  --output /tmp/pokecoach-training-streaming-benchmark.json
```

- [ ] Enregistrer les résultats réels : logiciel/device, fichiers/transitions, options, NLL et écart, nombre d'appels, durées, RSS, mémoire GPU si applicable. Une mesure CUDA reste « non exécutée » si indisponible. Ne pas extrapoler une durée GPU à partir du CPU, ni promettre un speedup minimal non observé.
- [ ] Vérifier explicitement qu'aucun pickle/manifest/parent n'a été modifié. Le benchmark ne doit jamais lancer une location cloud ni appeler une API de facturation.

### Tâche 8 — Vérification finale et livraison

**Fichiers :** l'ensemble des changements précédents et le bilan d'audit.

- [ ] Exécuter la suite ciblée :

```bash
uv run --locked pytest tests/test_training_execution_config.py tests/test_training_data.py tests/test_training_shuffle.py tests/test_bc_batches.py tests/test_bc_streaming.py tests/test_batched_evaluation.py tests/test_evaluation_isolation.py tests/test_training_streaming_integration.py tests/test_training_memory.py tests/test_training_benchmark.py tests/test_training.py tests/test_finetuning.py tests/test_training_api.py
```

- [ ] Exécuter `uv run --locked pytest`, puis `uv run --locked ruff check src tests scripts`. Consigner les échecs préexistants séparément avec leur preuve ; ne pas les masquer ou les déclarer corrigés sans exécution.
- [ ] Exécuter `git diff --check` et relire la liste de fichiers modifiés : pas de `.venv`, vendor, données, checkpoint parent ni YAML sans rapport.
- [ ] Vérifier les critères de livraison ci-dessous et compléter le bilan réel. Si la parité numérique ou la couverture exacte échoue, corriger avant de proposer un entraînement GPU.

## 7. Critères de livraison

- [ ] Le chemin `train_bc` ne charge jamais un split entier de trajectoires et n'appelle pas `flatten_trajectories`.
- [ ] Le tampon est borné en octets, ses lignes sont des copies possédées et les batches n'immobilisent pas toutes les trajectoires passées.
- [ ] Une epoch couvre chaque transition une fois ; dernier batch conservé ; compteurs réellement observés.
- [ ] Les options de mélange et sa version sont enregistrées ; la différence avec le shuffle global est documentée.
- [ ] Évaluation complète à batches, moyenne pondérée par transition, équivalence testée contre un oracle scalaire.
- [ ] Aucun changement aux poids, gradients, modes ou RNG de l'entraînement lors de l'évaluation.
- [ ] Le test final n'est lu qu'après les epochs ; les erreurs de lecture/validation ne sont pas ignorées.
- [ ] Les anciens YAML/appels publics restent compatibles ; le profil 80k garde ses hyperparamètres d'apprentissage.
- [ ] Dataset ready et parent inchangés ; aucun nouveau prepare ou cache volumineux requis.
- [ ] Tests et benchmark local borné documentés, mesures CPU/CUDA clairement distinguées, aucun coût cloud engagé.

**Texte à transmettre à Composer :** « Implémente `docs/plans/2026-11-08-training-streaming-evaluation-batches.md` dans l'ordre, en respectant les contrats de flux, de comptage et de parité NLL. Préserve le dataset ready et mes modifications existantes. Ne lance ni entraînement complet ni location GPU. Termine avec les tests et le benchmark local borné, puis rapporte les mesures et les limites restantes. »
