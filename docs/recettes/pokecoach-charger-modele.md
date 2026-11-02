# Charger un modèle Trainer dans PokeCoach

## 1. Entraîner et (optionnel) exporter

```bash
cd pokecoach2-trainer
devbox run cli -- train --config configs/tiny.yaml
devbox run cli -- export --config configs/tiny.yaml \
  --result pokecoach-trainer/storage/runs/bc-tiny-smoke/result.json
```

## 2. Un seul fichier profil

Copier `configs/pokecoach-vgc_model.example.yaml` vers un emplacement stable, par exemple :

`~/pokecoach-models/bc-tiny-smoke/vgc_model.yaml`

| Changer de modèle | Éditer |
|-------------------|--------|
| Nouveau run Trainer | `checkpoint.path` + `architecture.*` (ou `trainer_package_dir`) |
| Export package | `trainer_package_dir` seul |
| Identifiant traces | `model_id` |

Les dimensions `architecture` doivent correspondre à l’entraînement (`configs/tiny.yaml` ou `manifest.json` export).

## 3. PokeCoach

Dans `pokecoach2-python/.env` :

```bash
POKECOACH_ENGINE=vgc_bench
POKECOACH_VGC_REAL=1
VGC_MODEL_PROFILE=/chemin/vers/vgc_model.yaml
# SHOWDOWN + engine-runtime setup inchangés
```

Relancer une analyse comme d’habitude : le bridge charge le zip et instancie la policy avec l’architecture du profil (même modèle p1 et p2).

## 4. Vérifications

```bash
cd pokecoach2-python/engine-runtime
PYTHONPATH=src pytest tests/test_model_profile.py -q
```

Chargement réel (optionnel) :

```bash
export POKECOACH_TEST_CHECKPOINT=/chemin/final.zip
PYTHONPATH=src pytest tests/test_policy_factory.py -q
```
