# PokeCoach Trainer

## Collecte parallèle Showdown

La collecte repose sur une **campagne** SQLite (`archive/catalog.sqlite3`), un **superviseur** détaché et deux pools (découverte / téléchargement). Les budgets réseau et de sélection sont **globaux** : ajouter des workers n’augmente pas le plafond de requêtes.

### Démarrage backfill (M-C)

Depuis ce répertoire :

```bash
uv run --locked pokecoach-trainer collection start --config configs/collection-mc.yaml
uv run --locked pokecoach-trainer collection status --job-id job-…
```

`configs/collection-mc.yaml` cible ~2 M replays retenus, 2 workers découverte, 8 téléchargement, 2 req/s, fenêtres 7 jours. Ce chiffre est une **cible de stockage**, pas le résultat d’un budget 24 h à 2 req/s.

### Mise à jour incrémentale

```bash
uv run --locked pokecoach-trainer collection start --config configs/collection-mc-incremental.yaml
```

Mode `incremental` explicite, recouvrement 48 h par défaut.

### Contrôles

```bash
uv run --locked pokecoach-trainer collection pause --job-id job-…
uv run --locked pokecoach-trainer collection resume --job-id job-…
uv run --locked pokecoach-trainer collection stop --job-id job-…
uv run --locked pokecoach-trainer collection scale --job-id job-… --discovery-workers 2 --download-workers 8
uv run --locked pokecoach-trainer collection retry-failed --job-id job-…
```

Après un budget atteint, une **nouvelle tentative** exige des budgets explicites :

```bash
uv run --locked pokecoach-trainer collection resume --job-id job-… --new-attempt --budgets configs/budgets.yaml
```

### Filtres facultatifs

Sans `upload_after`, `upload_before`, `teams_file` ni `team_filter`, tous les replays valides du format sont éligibles. Exemple filtré : `configs/collection-mc-filtered.yaml` et `configs/teams.example.json` (chemins relatifs au YAML). Les dates et équipes se combinent par ET ; plusieurs compositions dans le fichier d’équipes s’alternent par OU. Un replay inspecté mais rejeté n’est pas archivé.

### Collecte filtrée par profil dataset

`configs/collection-mc-100k-valid.yaml` charge `configs/dataset-mc-collectable.yaml` (`dataset_recipe_file`) et fige le profil dans la campagne. Seuls les replays **admissibles** (OTS, Elo individuel, inactivité, etc.) sont **publiés** dans l’archive. Un candidat peut être téléchargé pour examen puis rejeté : ça consomme des requêtes, pas le quota de retenus.

```bash
# Après validation locale, pas pendant les tests unitaires.
uv run --locked pokecoach-trainer collection start --config configs/collection-mc-100k-valid.yaml
uv run --locked pokecoach-trainer collection status --job-id JOB_ID
```

Les budgets (`max_requests`, `max_duration_s`) ne sont **pas** augmentés automatiquement. À un taux d’acceptation partiel, la campagne peut s’arrêter avant 100k retenus ; reprendre avec `--new-attempt --budgets`. Un **nouveau profil** exige une **nouvelle campagne**. Les ~104k objets historiques ne sont pas supprimés.

Snapshot de campagne (pas le snapshot global historique) :

`POST /collection/jobs/{job_id}/snapshot` (préfixe `/collection`).

```bash
uv run --locked pokecoach-trainer dataset create \
  --snapshot-id SNAPSHOT_ID \
  --recipe configs/dataset-mc-collectable.yaml \
  --dataset-id DATASET_ID
```

La cible dataset se mesure sur ce snapshot. Splits et conversion restent des contrôles ultérieurs : **admissible ≠ convertible**.

### Archive (export / sauvegarde / reconstruction)

```bash
uv run --locked pokecoach-trainer archive export-index --output /tmp/replays.jsonl
uv run --locked pokecoach-trainer archive backup --output /tmp/catalog-backup.sqlite3
uv run --locked pokecoach-trainer archive rebuild-index
```

### Reprise et anciens jobs

Les jobs de collecte séquentielle (progression JSON) ne sont **pas** repris automatiquement : relancer une nouvelle campagne. Les snapshots et datasets restent figés pendant une collecte active.

### Tests

```bash
uv run --locked pytest tests/test_collection_config.py tests/test_collection_integration.py -q
uv run --locked ruff check src tests
```

SQLite embarquée : vérifier `sqlite3.sqlite_version` (correctif WAL recommandé pour la prod multi-processus).

## Fine-tuning BC en flux (dataset ready)

Le parcours `train_bc` lit les pickles **un par un** : pas de chargement intégral du split train en RAM. L’entraînement utilise un **tampon de mélange borné** (`shuffle_buffer_mib`, défaut 256 Mio pour les tableaux de transitions) ; l’**évaluation NLL** utilise un batch distinct (`evaluation_batch_size`, défaut 256).

- **batch train** (`batch_size`) et **batch évaluation** sont indépendants ; en cas d’OOM à l’évaluation, diminuer `evaluation_batch_size` dans une copie de config (pas de retry automatique).
- **RAM utile** ≈ tampon de shuffle + trajectoire pickle courante + batch courants + framework PyTorch/SB3 (pas un plafond RSS garanti).
- Chaque epoch **relit tous les fichiers** train et applique un ordre de mélange **déterministe** (seed + numéro d’epoch) ; ce n’est pas l’ancien shuffle global du `DataLoader`, les poids peuvent différer légèrement à hyperparamètres égaux.
- Mesurer CPU/GPU avant une location : sur VM GPU, mettre `device: cuda:0` explicitement dans la config d’exécution pour éviter un repli CPU silencieux avec `device: auto`.
- Un dataset déjà `ready` se réutilise **sans** conversion ni nouveau cache disque.

Profil production : `configs/finetune-mb-mc-prod.yaml`. Benchmark lecture seule (sous-ensemble de fichiers) :

```bash
uv run --locked python scripts/benchmark_training_pipeline.py \
  --dataset-id ds-mb-mc-collectable-80k-v1 \
  --base-model-id vgcbench-mb-reference \
  --device cpu --max-files 8 \
  --output /tmp/pokecoach-training-streaming-benchmark.json
```
