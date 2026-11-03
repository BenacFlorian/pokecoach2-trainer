# Patches VGC-Bench (vendor)

| Patch | Fichier | Motif |
| --- | --- | --- |
| `policy-configurable.patch` | `vgc_bench/src/policy.py` | `num_layers`, `num_heads`, `embed_len` configurables (étape 2) |
| Provenance replays | `vgc_bench/logs2trajs.py` | `TaggedTrajectory`, `process_logs_with_provenance` (split train/val par replay) |
| Dépendances | `vendor/vgc-bench/pyproject.toml` | Commits Git figés pour `imitation` / `poke-env` ; retrait de `open-spiel` (BC uniquement) |

Regénérer le vendor :

```bash
devbox run setup-vendor
```
