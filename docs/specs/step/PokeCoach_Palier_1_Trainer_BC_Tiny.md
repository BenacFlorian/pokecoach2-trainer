# PokeCoach Trainer — Palier 1 : entraîner un BC Tiny local et l'exécuter dans PokeCoach

**Nature du document :** cahier des charges technique / plan d'implémentation pour Cursor ou Codex  
**État :** spécification à implémenter, pas un programme déjà développé  
**Date :** 2 novembre 2026  
**Périmètre :** reprise de VGC-Bench, FastAPI local, petit dataset, BC Tiny configurable, export, preuve de compatibilité PokeCoach

---

## 1. Objectif du palier

Construire un **petit serveur Python indépendant de PokeCoach**, capable de :

1. embarquer une version **identifiable et figée** du code de VGC-Bench, ainsi que ses dépendances compatibles ;
2. préparer un **mini-dataset réel de replays humains** à partir des logs publiés par VGC-Bench ;
3. rendre **paramétrables** la largeur du Transformer, le nombre de couches et le nombre de têtes d'attention ;
4. lancer depuis **FastAPI** un entraînement **Behavior Cloning (BC)** en arrière-plan dans un **processus worker distinct** ;
5. consulter les états du job, ses logs, sa loss et ses checkpoints ;
6. télécharger un checkpoint dans un format **réutilisable par PokeCoach** ;
7. **charger le checkpoint dans PokeCoach** et effectuer au moins une **partie complète** avec lui, depuis le Team Preview jusqu'à la fin.

**Le succès n'est pas d'obtenir une bonne IA.** Un Tiny entraîné sur très peu de données peut jouer très mal. Le succès est de démontrer une chaîne reproductible **données → entraînement → checkpoint → chargement → match PokeCoach**.

### Hors périmètre

- Nouvelle régulation, nouveau scraper, ingestion quotidienne, dataset massif.
- Reinforcement learning, self-play d'entraînement, optimisation du niveau de jeu.
- Dashboard Angular complet, multi-utilisateur, SaaS, authentification distante.
- Comparaison de 4 tailles de modèles et recherche d'hyperparamètres.
- Refonte générale des mécanismes de masquage des actions.
- Déploiement sur un serveur GPU distant.

**Priorité :** rester simple, local et compatible. La première démonstration doit se faire avec une régulation **déjà gérée à la fois par le code de VGC-Bench récupéré et par PokeCoach**.

---

## 2. Sources réelles à reprendre

**Dépôt de référence :** https://github.com/cameronangliss/vgc-bench

Fichiers importants :

| Fichier d'origine | Pourquoi le conserver / l'adapter |
| --- | --- |
| `vgc_bench/src/policy.py` | `MaskedActorCriticPolicy`, `AttentionExtractor`, masquage et Transformer. |
| `vgc_bench/src/policy_player.py` | Encodage d'un combat en observation, exécution de la policy, conversion des actions en ordres Showdown. |
| `vgc_bench/pretrain.py` | Entraînement BC à partir des trajectoires et sauvegarde PPO/SB3. |
| `vgc_bench/logs2trajs.py` | Transformation des logs humains en trajectoires d'entraînement. |
| `vgc_bench/src/env.py` | Environnement Showdown servant à instancier la policy. |
| `vgc_bench/src/utils.py` | Identifiants, dimensions d'observations, tables de correspondance et longueurs d'actions. |
| `pyproject.toml` | Versions et forks spécifiques des dépendances. |
| `pokemon-showdown/` | Sous-module Git du moteur Showdown, fixé par le dépôt. |
| `pretrain.sh` | Script de référence pour l'orchestration du serveur Showdown et de l'entraînement. |

Données :

- Logs courants : https://huggingface.co/datasets/cameronangliss/vgc-battle-logs
- Archives Scarlet/Violet : https://huggingface.co/datasets/cameronangliss/vgc-battle-logs-sv
- Checkpoints de référence : https://huggingface.co/cameronangliss/vgc-bench-models

**Attention :** les datasets Hugging Face ci-dessus contiennent des **logs JSON de combats**, pas directement des checkpoints ni nécessairement des trajectoires `.pkl` prêtes pour `pretrain.py`. Il faut les transformer avec `logs2trajs.py`.

**Vérifications connues du code d'origine :**

- `AttentionExtractor` définit actuellement `embed_layers = 3`, `num_heads = 4`, `embed_len = 32` comme attributs de classe.
- La largeur `d_model` est déjà transmise en argument. Le `pretrain.py` de référence utilise `d_model = 256`.
- Le BC utilise la classe `imitation.algorithms.bc.BC`, et la sauvegarde passe par `ppo.save(...)` : les checkpoints obtenus sont des **archives `.zip` Stable-Baselines3**.
- `TrajectoryDataset` lit les trajectoires dans `trajs/*.pkl` et fabrique aujourd'hui un **masque synthétique**. Ne pas considérer ce masque comme une représentation exhaustive des actions légales.
- Le BC de référence réalise également des évaluations contre une heuristique Showdown, coûteuses pour un simple smoke test : les rendre optionnelles.
- Le projet référence des forks spécifiques de `poke-env` et `imitation` et fixe d'autres versions (SB3, Torch, etc.). **Ne pas remplacer aveuglément ces dépendances par des versions PyPI plus récentes.**

Sources code :

- https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/src/policy.py
- https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/src/policy_player.py
- https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/pretrain.py
- https://github.com/cameronangliss/vgc-bench/blob/main/vgc_bench/logs2trajs.py
- https://github.com/cameronangliss/vgc-bench/blob/main/pyproject.toml
- https://github.com/cameronangliss/vgc-bench/blob/main/README.md

---

## 3. Récupération du code et installation locale

### 3.1. Principe de reprise

Créer un dépôt autonome `pokecoach-trainer` et y placer une copie contrôlée de VGC-Bench. **Ne pas réécrire immédiatement l'encodeur, la policy ou la conversion des actions** : ce serait augmenter le risque d'incompatibilité avec PokeCoach.

Exemple de démarrage :

```bash
mkdir pokecoach-trainer
cd pokecoach-trainer

git clone --recurse-submodules https://github.com/cameronangliss/vgc-bench.git vendor/vgc-bench

# Enregistrer le commit exact retenu ; reporter sa valeur dans la documentation.
git -C vendor/vgc-bench rev-parse HEAD
# Conserver également le commit du sous-module Pokémon Showdown.
git -C vendor/vgc-bench/pokemon-showdown rev-parse HEAD
```

Le choix entre **copie versionnée**, **fork Git**, ou **sous-module Git** doit permettre les modifications de `policy.py` et `pretrain.py`. Pour ce palier, un **fork du dépôt VGC-Bench** ou une copie versionnée avec l'historique des modifications est approprié. La commande `git clone` ci-dessus est un point de départ, pas une architecture de dépendances imposée.

### 3.2. Dépendances et Showdown

Le README officiel indique Python 3.13 comme environnement utilisé par l'auteur, même si `pyproject.toml` déclare une compatibilité Python plus large. Pour réduire les surprises, essayer Python 3.13 avec les dépendances effectivement figées par le dépôt retenu.

Dans l'environnement virtuel adapté :

```bash
# Exemple une fois Python 3.13 disponible
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './vendor/vgc-bench[dev]'
python -m pip install fastapi 'uvicorn[standard]'
```

Le cas d'une installation impossible à cause de `open-spiel` est signalé dans le README officiel ; vérifier d'abord si cette dépendance est utile pour ce palier et documenter toute exclusion volontaire. Ne pas masquer une incompatibilité par des changements arbitraires.

Pour les tests qui demandent une partie réelle :

```bash
cd vendor/vgc-bench/pokemon-showdown
npm install
# Sur un terminal LOCAL séparé, jamais exposé sur Internet :
node pokemon-showdown start 8000 --no-security
```

**Sécurité :** `--no-security` est réservé à un environnement de développement isolé. Restreindre réellement l'accès par bind/firewall/container réseau ; le serveur Showdown peut écouter sur `0.0.0.0`. Ne pas ouvrir ce port sur Internet.

### 3.3. Critère de sortie de l'installation

- Les imports `torch`, `imitation`, `stable_baselines3`, `poke_env`, `vgc_bench` réussissent.
- Une policy existante se charge sur CPU avec les dépendances retenues.
- La version et le commit de VGC-Bench, de `poke-env` et de Showdown sont enregistrés.
- Un serveur Showdown local démarre lorsque nécessaire.
- Les tests de référence les plus pertinents du dépôt sont exécutés et leur résultat est conservé.

---

## 4. Structure proposée du mini-serveur

```text
pokecoach-trainer/
├── README.md
├── pyproject.toml                  # Dépendances du wrapper FastAPI
├── vendor/
│   └── vgc-bench/                 # Fork / copie versionnée du code de référence
│       ├── vgc_bench/
│       ├── pokemon-showdown/
│       └── ...
├── app/
│   ├── main.py                    # Application FastAPI
│   ├── api/
│   │   ├── health.py
│   │   ├── datasets.py
│   │   ├── training.py
│   │   └── models.py
│   ├── schemas.py                 # Requêtes / réponses Pydantic
│   ├── jobs.py                    # Gestion état, PID, statut, logs
│   ├── dataset_service.py         # Import/sous-échantillonnage
│   ├── training_service.py        # Lancement du worker
│   ├── artifact_service.py        # Checkpoints + manifests
│   └── worker.py                  # Processus d'entraînement distinct
├── configs/
│   └── tiny.yaml                  # Configuration du premier essai
├── storage/                       # Ignoré par Git
│   ├── raw_logs/
│   ├── datasets/
│   ├── jobs/
│   └── models/
├── tests/
│   ├── test_policy_config.py
│   ├── test_dataset.py
│   ├── test_training_api.py
│   ├── test_model_reload.py
│   └── test_pokecoach_compatibility.py
└── .gitignore
```

Cette structure est une proposition. L'important est de séparer les responsabilités **API / préparation des données / entraînement / artefacts**.

**Architecture minimale :** 1 processus FastAPI + 1 processus worker Python lancé via `subprocess` (ou un gestionnaire de jobs simple), disque local, fichiers JSON/JSONL pour les états et logs. **Pas de Redis, Celery, PostgreSQL ni interface Angular pour ce palier.** Ne pas lancer le travail GPU directement dans le thread d'une requête FastAPI, ni compter sur `BackgroundTasks` pour survivre aux redémarrages.

Définir une politique claire au redémarrage : par exemple, un job `running` dont le PID n'existe plus devient `interrupted`. Pour ce palier, **un seul entraînement simultané** suffit.

---

## 5. Rendre le Transformer configurable

### 5.1. Modification souhaitée

Dans `vgc_bench/src/policy.py` :

- Transformer les valeurs fixes `embed_layers` et `num_heads` en arguments explicites de `AttentionExtractor.__init__()`.
- Exposer ces paramètres dans `MaskedActorCriticPolicy.__init__()`.
- Transmettre les arguments dans `features_extractor_kwargs`.
- Garder **les valeurs par défaut historiques** afin de pouvoir encore charger les anciens checkpoints : `num_layers=3`, `num_heads=4`, `embed_len=32`, avec `d_model` conservé comme auparavant.
- Prévoir que **les paramètres sont enregistrés dans la configuration de la policy à la sauvegarde**, puis recréés automatiquement à l'ouverture du `.zip` SB3.
- Ajouter une validation : `d_model > 0`, `num_layers >= 1`, `num_heads >= 1`, `d_model % num_heads == 0`.

Convention recommandée pour la nouvelle API : `num_layers` et `num_heads` ; à l'intérieur de `AttentionExtractor`, mapper proprement vers les attributs existants plutôt que modifier la logique métier.

**Exemple conceptuel** de transmission (le code exact doit respecter les signatures de VGC-Bench/SB3) :

```python
policy_kwargs = {
    "d_model": 128,
    "num_layers": 2,
    "num_heads": 4,
    "embed_len": 32,
    "choose_on_teampreview": True,
}

ppo = PPO(
    MaskedActorCriticPolicy,
    env,
    policy_kwargs=policy_kwargs,
    device=device,
)
```

### 5.2. Tiny initial

| Paramètre | Valeur de départ |
| --- | --- |
| `d_model` | `128` |
| `num_layers` | `2` |
| `num_heads` | `4` |
| `embed_len` | `32` |
| `choose_on_teampreview` | `true` |
| `device` | `auto` (CPU si GPU incompatible / indisponible) |
| `batch_size` | `32` (modifiable) |
| `num_epochs` | `3` pour le premier smoke test |
| `seed` | `42` |

Ce sont des **valeurs de validation de pipeline**, pas des hyperparamètres optimisés. Le Tiny conserve les mêmes **dimensions d'observations, espace d'actions et conventions de masquage** que VGC-Bench ; seuls les paramètres internes de capacité changent.

### 5.3. Tests obligatoires

1. Instancier les configurations historique (`256 / 3 / 4`) et Tiny (`128 / 2 / 4`).
2. Calculer le nombre de paramètres de chacune et vérifier que Tiny est effectivement plus petit.
3. Vérifier les dimensions des sorties et de l'espace d'actions.
4. Sauvegarder puis recharger un modèle Tiny neuf et vérifier que `num_layers` et `num_heads` n'ont pas changé.
5. Vérifier le comportement du masquage et les contraintes entre actions des deux Pokémon.
6. Vérifier qu'un checkpoint original compatible demeure chargeable après les modifications, avec les imports/classes appropriés.

**Ne pas** affirmer qu'une archive Tiny est compatible avec un ancien binaire PokeCoach sans mise à jour : **PokeCoach doit disposer de la version compatible de la classe policy**, désormais capable de recréer l'architecture choisie.

---

## 6. Préparer un mini-dataset réel

### 6.1. Pour ce palier, pas de scraping

Partir des **replays humains publiés** par VGC-Bench ; télécharger **un fichier de logs d'un format compatible** avec les versions figées de PokeCoach, du moteur Showdown et du parseur. Éviter un mélange de plusieurs régulations lors du premier essai.

Objectif indicatif : **200 à 1 000 replays**, selon disponibilité et taux de conversion. Ce volume est volontairement insuffisant pour viser un excellent BC : il sert à vérifier que le modèle apprend et que la chaîne technique fonctionne.

Chaîne :

```text
JSON brut de replays Showdown
    ↓ sélection d'un format + échantillonnage déterministe
storage/raw_logs/<dataset-id>/logs_<format>.json
    ↓ logs2trajs.py (adapté pour chemins explicites)
Trajectoires (une ou deux perspectives par partie)
    ↓ contrôle de validité + split PAR PARTIE
storage/datasets/<dataset-id>/train/*.pkl
storage/datasets/<dataset-id>/validation/*.pkl
storage/datasets/<dataset-id>/manifest.json
```

### 6.2. Adaptations nécessaires

Les scripts d'origine sont conçus pour des répertoires **fixes** (`battle_logs/`, `trajs/`) et non pour une bibliothèque de datasets. Les adapter afin de recevoir des **chemins explicites** et de ne jamais écraser un autre dataset.

Règles :

- Sélectionner les replays **avant** la conversion ; conserver les IDs des combats.
- Utiliser un `seed` constant pour pouvoir régénérer exactement l'échantillon.
- Pour la séparation `train/validation`, grouper les deux perspectives d'une même partie dans **le même split** ; ne pas mélanger les tours d'un même replay entre les splits.
- Les trajectoires peuvent avoir des longueurs différentes : adapter correctement le `DataLoader` et le nombre de transitions.
- Contrôler la taille des observations et les valeurs/indices des actions.
- Détecter les conversions échouées et produire un compte rendu plutôt que stopper sans explication.
- Ne jamais charger des fichiers `.pkl` de provenance non fiable : `pickle` peut exécuter du code à la désérialisation.
- Calculer les statistiques : replays retenus, replays rejetés, trajectoires, décisions, splits, format, origine.

**Point de vigilance particulier :** le `pretrain.py` d'origine calcule des groupes de chargement via `div_frac = 0.01`. Avec un dataset minuscule, il peut en résulter un batch de fichiers vide (`len(dataset) // 100 == 0`). Implémenter un chargement adapté aux petits jeux de données ; **ne pas reprendre tel quel ce découpage**.

### 6.3. Masque d'actions — dette technique maîtrisée

Le lecteur BC de référence reconstruit un masque **synthétique**, pas la liste complète des actions légales de chaque tour. Il faut :

- documenter précisément l'implémentation utilisée pour ce premier POC ;
- vérifier, pour chaque exemple, que les actions humaines cibles ne sont pas interdites par le masque utilisé dans la loss ;
- garantir l'absence de valeurs invalides et de `NaN` à l'entraînement ;
- garder **pendant l'inférence** les masques légaux calculés à partir de l'état Showdown et les contraintes de double action ;
- **ne pas inventer** de règles stratégiques qui interdiraient des actions légales.

Reconstituer les masques légaux exacts dans les trajectoires serait une amélioration intéressante **après ce palier**, si elle nécessite une refonte importante du convertisseur. En revanche, une contradiction entre les actions humaines et le masque de training est **bloquante**.

---

## 7. Lancement du BC Tiny

### 7.1. Configuration d'expérience

Fichier `configs/tiny.yaml` (spécification du format attendu) :

```yaml
experiment:
  name: "bc-tiny-smoke"
  seed: 42

dataset:
  id: "sample-compatible-format-001"
  validation_ratio: 0.10

model:
  policy_class: "MaskedActorCriticPolicy"
  d_model: 128
  num_layers: 2
  num_heads: 4
  embed_len: 32
  choose_on_teampreview: true

training:
  method: "bc"
  device: "auto"
  batch_size: 32
  num_epochs: 3
  evaluation_battles_per_epoch: 0
  save_each_epoch: true
```

Au premier palier, l'évaluation contre `SimpleHeuristicsPlayer` **après chaque époque** est désactivable. Elle requiert Showdown et alourdit le test. La **preuve de combat complet**, à la fin, reste obligatoire.

### 7.2. Traitement par le worker

1. Valider la configuration et l'existence du dataset.
2. Vérifier les prérequis : dépendances, espace disque, device, ports utilisés si Showdown est requis.
3. Initialiser `PPO(MaskedActorCriticPolicy, ...)` et entraîner **sa policy** au moyen de `imitation.algorithms.bc.BC` (c'est la méthode VGC-Bench, même si le conteneur utilisé s'appelle PPO).
4. Pour chaque époque, enregistrer `train_loss` et `validation_loss` avec un passage de validation **sans gradients, sans mise à jour des poids**.
5. Émettre les logs, le numéro d'époque, le temps écoulé, le nombre de transitions, la mémoire et un checkpoint.
6. Sauvegarder un modèle final **en `.zip` Stable-Baselines3** avec `ppo.save(...)` et son manifeste JSON.
7. Faire un round-trip de chargement `PPO.load(..., device="cpu")` **dans un nouveau processus**.
8. Marquer le job `completed` seulement après validation des artefacts.

Prévoir les états : `queued`, `running`, `completed`, `failed`, `interrupted` et idéalement `cancelled`.

### 7.3. Observabilité minimale

- Logs textuels horodatés dans un fichier propre au job.
- Métriques par époque dans `metrics.jsonl` ou JSON.
- TensorBoard si disponible, sans l'imposer pour consommer les métriques de l'API.
- Détail des erreurs Python et statut d'échec récupérables via l'API.
- Nombre de paramètres et device effectif visibles.
- Contrôles explicites sur les pertes non finies (`NaN` / infini), métriques de validation absentes et fichiers de poids incomplets.

**Une loss qui baisse est souhaitable mais non obligatoire pour déclarer la compatibilité.** En revanche, le processus doit réellement effectuer des mises à jour des paramètres : calcul des gradients, appel à l'optimiseur, différence mesurable entre certains poids initiaux et finaux.

---

## 8. API FastAPI minimale

Ce palier n'a pas besoin d'un scraper. L'API expose seulement l'import d'un jeu de données local, sa préparation, l'entraînement, les résultats et le téléchargement d'un modèle.

| Méthode | Route | Fonction |
| --- | --- | --- |
| `GET` | `/health` | API opérationnelle ; informations versions essentielles. |
| `POST` | `/datasets/import` | Enregistrer un fichier local de logs déjà disponible sur le serveur, ou téléverser un JSON (limite de taille). |
| `POST` | `/datasets/{id}/prepare` | Échantillonnage, conversion en trajectoires, contrôles, split. |
| `GET` | `/datasets/{id}` | Statut et statistiques du dataset. |
| `POST` | `/training/jobs` | Lancer un job BC Tiny (réponse immédiate avec ID). |
| `GET` | `/training/jobs/{id}` | État, paramètres, progression, métriques, erreurs. |
| `GET` | `/training/jobs/{id}/logs` | Dernières lignes de logs ; pagination ou `tail`. |
| `GET` | `/models` | Liste des artefacts disponibles. |
| `GET` | `/models/{id}` | Manifeste, taille, configuration, validation. |
| `GET` | `/models/{id}/download` | Télécharger le package contenant `model.zip` et `manifest.json`. |

**Évolution éventuelle :** `GET /training/jobs/{id}/events` en Server-Sent Events, mais seulement si utile. Le polling HTTP suffit pour le POC.

### Exemple de lancement

`POST /training/jobs`

```json
{
  "dataset_id": "sample-compatible-format-001",
  "experiment_name": "bc-tiny-smoke",
  "seed": 42,
  "model": {
    "d_model": 128,
    "num_layers": 2,
    "num_heads": 4,
    "embed_len": 32,
    "choose_on_teampreview": true
  },
  "training": {
    "device": "auto",
    "batch_size": 32,
    "num_epochs": 3
  }
}
```

Réponse attendue (`202 Accepted`) :

```json
{
  "job_id": "train-001",
  "status": "queued"
}
```

### Exigences de sûreté (même en local)

- `127.0.0.1` par défaut pour FastAPI ; aucune exposition publique de l'API de lancement de processus.
- Un seul entraînement à la fois ; contrôle des ressources.
- Pas d'argument shell construit à partir de valeurs libres de l'utilisateur ; liste d'arguments validés.
- Pas d'accès arbitraire à des chemins via les endpoints ; chemins bornés à `storage/` et IDs contrôlés.
- Fichiers entrants limités et validés ; checkpoints / pickles traités comme **artefacts de confiance** uniquement.
- Ne pas exposer `--no-security` de Showdown à un réseau non maîtrisé.

---

## 9. Format du modèle exporté et compatibilité PokeCoach

### 9.1. Artefact à produire

Téléchargement sous forme d'un package simple :

```text
pokecoach-bc-tiny-train-001/
├── model.zip            # Archive SB3 produite par PPO.save(...)
├── manifest.json        # Architecture + version des protocoles et dépendances
├── metrics.json         # Résultats du test et courbe d'entraînement
└── README.md            # Comment charger le modèle
```

**Ne pas renommer ou convertir arbitrairement le checkpoint interne en `.safetensors` ou en modèle Hugging Face** : PokeCoach utilise déjà une policy dérivée de VGC-Bench. Réutiliser le mécanisme de chargement SB3 existant est le chemin le moins risqué.

### 9.2. Manifest minimal

Exemple de contenu (valeurs `TO_FILL` à remplacer par les valeurs réellement enregistrées, ne pas inventer de hash/version) :

```json
{
  "model_id": "bc-tiny-train-001",
  "format": "stable-baselines3-ppo-zip",
  "algorithm": "behavior_cloning",
  "policy_class": "vgc_bench.src.policy.MaskedActorCriticPolicy",
  "architecture": {
    "d_model": 128,
    "num_layers": 2,
    "num_heads": 4,
    "embed_len": 32,
    "choose_on_teampreview": true
  },
  "dataset_id": "sample-compatible-format-001",
  "battle_format": "TO_FILL",
  "observation_schema_version": "TO_FILL",
  "action_schema_version": "TO_FILL",
  "vgc_bench_commit": "TO_FILL",
  "showdown_commit": "TO_FILL",
  "poke_env_commit": "TO_FILL",
  "torch_version": "TO_FILL",
  "stable_baselines3_version": "TO_FILL",
  "imitation_revision": "TO_FILL",
  "checkpoint_sha256": "TO_FILL"
}
```

Pour `observation_schema_version` et `action_schema_version`, **ne pas se contenter d'un nom libre** : documenter / calculer une empreinte stable des conventions d'encodage (ordre des champs, longueurs, tables d'identifiants, espace d'actions, sémantique des masques). Les commits du code restent indispensables même si des empreintes sont introduites.

### 9.3. Condition de compatibilité réelle

Il ne suffit pas que `PPO.load()` réussisse dans le Trainer. **PokeCoach doit reproduire les mêmes observations et le même sens des actions.** Les points suivants sont indispensables :

1. Même classe compatible `MaskedActorCriticPolicy`, avec ses nouveaux arguments configurables.
2. Même ordre/dimension des features issus de `PolicyPlayer.embed_battle(...)`.
3. Mêmes mappings Pokémon / talents / objets / attaques / météo et mêmes conventions de version.
4. Même espace d'actions et mêmes indices.
5. Même construction du masque légale à l'inférence : `DoublesEnv.get_action_mask(battle)` ; mêmes contraintes de double action.
6. Mêmes règles de Team Preview : le modèle doit sélectionner un quartet **et** fonctionner aux tours suivants.
7. Convergence des versions `poke-env` / Showdown requises par le modèle et supportées par PokeCoach.

Si PokeCoach utilise une copie **modifiée** de VGC-Bench, effectuer un **diff d'interface** entre Trainer et PokeCoach ; ne pas supposer la compatibilité uniquement parce que les classes ont le même nom.

### 9.4. Import dans PokeCoach

Prévoir dans PokeCoach, pour ce palier, un **chemin local configurable** vers l'archive exportée. Pas besoin d'API distante de modèles.

Par exemple, au niveau de son adaptateur de policy existant :

```python
from stable_baselines3 import PPO

model = PPO.load("/path/to/model.zip", device="cpu")
policy = model.policy
```

C'est seulement le **chargement de principe**. La véritable intégration doit utiliser le joueur / adaptateur VGC-Bench actuellement en place dans PokeCoach, pour transformer les objets `DoubleBattle` en observations masquées puis en ordres jouables. Elle doit aussi enregistrer quel modèle a été utilisé pour chaque simulation.

**Important :** ne pas tenter de charger des poids Tiny dans une instance de policy déjà créée avec l'architecture Small historique. Soit recréer la policy selon les données du checkpoint, soit vérifier explicitement les dimensions avant tout chargement de `state_dict`.

---

## 10. Scénario de validation de bout en bout

### Étape A — Préparation

- Récupérer / figer VGC-Bench.
- Démarrer le Trainer FastAPI.
- Préparer un dataset de quelques centaines de replays compatibles.
- Vérifier que les trajectoires sont valides et que les comptes sont non nuls.

### Étape B — Entraînement

- Envoyer `POST /training/jobs` avec `d_model=128`, `num_layers=2`, `num_heads=4`.
- Observer la progression et les logs par HTTP.
- Terminer les 3 époques sans crash, `NaN` ou lot vide.
- Vérifier que les poids ont effectivement évolué.
- Sauvegarder `model.zip` et son manifeste, puis recharger le checkpoint dans un **nouveau processus CPU**.

### Étape C — Passage à PokeCoach

- Télécharger le package.
- Installer / désigner ce modèle dans PokeCoach.
- Vérifier les métadonnées de compatibilité, ou refuser proprement l'import en cas d'incompatibilité.
- Construire dans PokeCoach un **matchup d'essai** utilisant une régulation et deux équipes supportées par le moteur de référence.
- Forcer un des agents à utiliser le **Tiny fraîchement entraîné** ; donner un adversaire heuristique ou une policy connue.
- Exécuter **au moins un match complet** avec Team Preview, tours normaux, switches et fin de partie.
- Contrôler que tous les ordres générés sont **acceptés par Showdown**, sans erreur d'index, exception de chargement ou blocage du worker.
- Vérifier que la partie et ses traces sont sauvegardées et affichables par PokeCoach.
- Relancer le **même checkpoint** dans une nouvelle exécution de PokeCoach pour démontrer que le résultat n'était pas lié à un état mémoire résiduel.

### Étape D — Rapport de test

Émettre un rapport lisible comprenant :

- Versions et commits du Trainer / VGC-Bench / Showdown / `poke-env`.
- Dataset, taille et taux de conversion.
- Configuration Tiny, nombre de paramètres, device.
- Loss d'entraînement / validation et statut des sauvegardes.
- Chemin et empreinte SHA-256 du checkpoint.
- Résultat des tests du modèle rechargé, côté Trainer puis côté PokeCoach.
- `PASSED` / `FAILED` explicite et traces en cas d'échec.

Le **niveau de jeu ou le win rate** est hors critère d'acceptation ; une défaite complète et valide est un succès de compatibilité.

---

## 11. Critères d'acceptation (« Definition of Done »)

- [ ] Le dépôt VGC-Bench et son sous-module Showdown sont disponibles, et leurs commits sont identifiés.
- [ ] Le serveur FastAPI local démarre et expose `/docs` et `/health`.
- [ ] Un petit dataset de replays humains réels est téléchargé/importé, converti et vérifié.
- [ ] Le nombre de couches (`num_layers`) et de têtes (`num_heads`) est configurable ; la largeur (`d_model`) l'est aussi.
- [ ] Le Tiny `128 / 2 / 4` est plus petit que le modèle `256 / 3 / 4` et son architecture reste intacte après sauvegarde/rechargement.
- [ ] Un entraînement BC de trois époques peut être lancé par HTTP sans bloquer FastAPI.
- [ ] Les statuts de job, logs et loss sont consultables via HTTP.
- [ ] Des mises à jour effectives des paramètres ont été constatées ; aucun `NaN` et aucune action cible impossible du fait du masque.
- [ ] Le modèle entraîné est exporté en archive `.zip` SB3 avec manifeste de version et hash.
- [ ] Le checkpoint se recharge sur CPU dans un **nouveau processus**.
- [ ] PokeCoach charge ce checkpoint avec le bon encodeur, la bonne policy et les bonnes versions d'espace d'actions.
- [ ] Le Tiny joue une **partie complète** depuis la preview jusqu'au résultat final, avec ordres valides.
- [ ] Le test se reproduit après redémarrage de PokeCoach.
- [ ] Un README explique installation, reproduction du dataset, entraînement, téléchargement et intégration.

**Le palier n'est validé qu'après les deux derniers tests PokeCoach : un fichier exportable seul n'est pas une preuve de compatibilité.**

---

## 12. Ordre d'implémentation recommandé (pour l'agent de code)

1. **Audit de l'existant** : examiner le dépôt VGC-Bench figé et l'adaptateur de modèle actuellement utilisé dans PokeCoach ; inventorier les différences réelles. Ne pas modifier PokeCoach inutilement.
2. **Installer et reproduire** : garantir qu'une policy de référence peut être instanciée et chargée dans l'environnement retenu.
3. **Paramétrer la policy** : `num_layers`, `num_heads`, `d_model`, sauvegarde/rechargement ; tests unitaires immédiats.
4. **Préparer le dataset réduit** : importer des logs, convertir, vérifier la structure et splitter par replay. Créer un manifeste de dataset.
5. **Entraîner par CLI d'abord** : faire fonctionner le Tiny de bout en bout sans FastAPI ; corriger les pièges des petits datasets et désactiver les évaluations répétées.
6. **Envelopper dans FastAPI** : faire lancer exactement le même entraînement par un worker distinct ; endpoints de statut, logs, fichiers.
7. **Exporter/charger** : archive SB3 + manifeste, round-trip dans un nouveau processus.
8. **Prouver avec PokeCoach** : import local, combat complet, validation des actions, rapport final.

**Règle de développement :** réutiliser autant que possible les classes, wrappers et mappings VGC-Bench ; faire les plus petites modifications nécessaires. Éviter une seconde implémentation divergent de l'encodage des combats dans le Trainer.

---

## 13. Risques identifiés et garde-fous

| Risque | Garde-fou |
| --- | --- |
| Versions différentes de `poke-env` ou Showdown entre Trainer et PokeCoach | Pin exact des commits, test de contrat observation/action, échec explicite si incompatible. |
| Jeu de données trop petit pour le `div_frac` de référence | Batch de trajectoires non nul, logique de mini-dataset dédiée. |
| Replays non convertibles | Comptes d'échecs et échantillons rejetés, sans prétendre que toutes les parties sont valides. |
| Masque de BC synthétique incompatible avec une action humaine | Contrôler la cible avant la loss, échouer si le masque interdit l'action de vérité terrain. |
| Entrée dans un entraînement coûteux sans retour de contrôle | Processus worker, logs persistants, un job à la fois, arrêt/reprise d'état. |
| Évaluation de 100 combats à chaque époque trop lente | Désactiver l'évaluation de combat pendant le smoke test et garder le match complet final. |
| Le `.zip` se charge mais joue une action illégale | Test de bout en bout passant obligatoirement par `DoubleBattle`, `get_action_mask` et `action_to_order`. |
| Fichiers `.pkl` ou checkpoints malveillants | Seulement des artefacts locaux / de source connue ; ne pas les traiter comme données arbitraires non fiables. |
| Fork VGC-Bench qui diverge de PokeCoach | Documenter les patches et tester la même interface de policy des deux côtés. |

---

## 14. Livrables attendus

1. Un **dépôt Python exécutable** `pokecoach-trainer` avec FastAPI, worker, scripts d'import, configuration et tests.
2. La **copie/fork documentée** du VGC-Bench utilisé, avec les modifications pour un Transformer configurable.
3. Un **dataset de démonstration** réduit, reproductible et accompagné de son manifeste.
4. Un **checkpoint Tiny BC entraîné**, exportable et rechargé (`model.zip` + manifeste).
5. Un **exemple de lancement FastAPI** (commande ou script), de démarrage de job et de téléchargement.
6. Un **test d'intégration PokeCoach** établissant la compatibilité sur une partie complète, ainsi que ses traces.
7. Un compte rendu final indiquant clairement **ce qui fonctionne**, **ce qui a dû être adapté** et **les limites restantes**.

### Résultat attendu en une phrase

> « J'ai lancé un entraînement BC Tiny depuis une API FastAPI locale sur quelques centaines de replays humains, téléchargé son checkpoint Stable-Baselines3, puis utilisé ce modèle dans PokeCoach pour terminer une partie Pokémon VGC. »

---

## Notes pour le développeur chargé de l'implémentation

- Ce document est une **spécification fonctionnelle et technique** : adapter les exemples de code à l'état exact du dépôt et ne pas présumer de chemins, signatures ou versions non vérifiés.
- **Ne pas remplacer l'apprentissage BC par PPO** : le checkpoint est dans un conteneur PPO/SB3, mais les poids de la policy sont ajustés par l'algorithme `BC` de la bibliothèque `imitation`.
- L'entraînement n'a **pas besoin de générer de nouvelles parties en permanence**, contrairement au RL ; le serveur Showdown est notamment utile pour l'instanciation/validation de l'environnement et la preuve de combat finale.
- Le fait de sélectionner une attaque légale ne prouve pas sa qualité stratégique ; ce premier Tiny est une **preuve technique**, non une IA de production.
- Faire un **rapport d'écart précis** si une version récente de VGC-Bench, les nouvelles mécaniques ou la copie actuelle de PokeCoach ont modifié le contrat observations/actions.
