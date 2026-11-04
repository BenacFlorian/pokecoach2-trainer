# Revue lot A — socle M-C

**Résultat technique :** PASSED (baseline + schémas lot B)  
**Validation humaine :** en attente

## Fichiers à lire

1. `docs/audits/mc-compatibilite.md`
2. `docs/audits/mc-runtime.json`
3. `docs/audits/PokeCoach_Trainer_Corrections_Dataset_Service.md` (acquis)

## Commandes exécutées

```bash
cd pokecoach-trainer && uv run --locked pytest tests/ -q
# code de sortie 0 (~100+ tests)
```

## Preuves

- Corrections dataset conservées (`test_dataset.py`).
- Patch provenance vendor versionné (`patches/logs2trajs-provenance.patch`).

## Limites

- Checkpoint M-B non chargé en audit réel dans ce rapport.
- Formats Showdown BO1/BO3 : registre `configs/formats.json` ; preuves replay réelles à collecter manuellement.

## Suite autorisée après validation

Lots B–E implémentés en code ; pauses B–E à valider séparément avant campagne de démonstration 300 replays / combats PokeCoach.
