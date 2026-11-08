# Préparer le dataset M-B + M-C ~80k (archive collectable)

Profil figé : `pokecoach-trainer/configs/profiles/mc-collectable-80k.yaml`.

Prérequis :

- `TRAINER_STORAGE_ROOT` = répertoire `storage/` du trainer (archive ~80 948 versions, ~75k M-C + ~5,6k M-B).
- Espace disque : prévoir **~80–200 Go** pour le dataset converti.
- RAM : éviter le swap pendant `dataset prepare` (fermer le reste si besoin).

## 1. Snapshot archive (M-B + M-C)

```bash
cd pokecoach2-trainer/pokecoach-trainer
uv run --locked pokecoach-trainer archive snapshot \
  --config configs/snapshot-mc-full-archive.yaml
```

Copier `snapshot_id` (ex. `snap-xxxxxxxx`). **Ne pas** réutiliser `snap-ce52587f5821cbad` (smoke 300 replays).

## 2. Créer le dataset (sélection + splits, sans conversion)

```bash
uv run --locked pokecoach-trainer dataset create \
  --snapshot-id <SNAPSHOT_ID> \
  --recipe configs/dataset-mb-mc-collectable.yaml \
  --dataset-id ds-mb-mc-collectable-80k-v1
```

## 3. Prepare (conversion perspectives)

```bash
uv run --locked pokecoach-trainer dataset prepare \
  --dataset-id ds-mb-mc-collectable-80k-v1 \
  --config configs/prepare-mc-collectable.yaml
```

Workers : **3** par défaut (`prepare-mc-collectable.yaml`). Ajustement :

```bash
uv run --locked pokecoach-trainer dataset resources
uv run --locked pokecoach-trainer dataset prepare \
  --dataset-id ds-mb-mc-collectable-80k-v1 \
  --conversion-workers auto
```

**Sans** `--sync-legacy` (legacy = raw_logs uniquement).

## 4. Vérifier

```bash
uv run --locked pokecoach-trainer dataset show --dataset-id ds-mb-mc-collectable-80k-v1
```

Attendre `state: ready` et les dossiers `storage/datasets/ds-mb-mc-collectable-80k-v1/{train,validation,test}/`.

## Suite (hors prepare)

Entraînement : `configs/finetune-mb-mc-prod.yaml` (`dataset_id` déjà aligné). Puis export + profil PokeCoach (`VGC_MODEL_PROFILE`).
