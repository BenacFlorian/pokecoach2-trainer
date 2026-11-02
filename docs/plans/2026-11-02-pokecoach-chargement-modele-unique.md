# Plan d’implémentation — Charger un modèle Trainer dans PokeCoach (palier 1, étapes 8–9 réorientées)

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Remplacer le checkpoint VGC-Bench historique par **ton** `.zip` SB3 dans PokeCoach, avec **un seul fichier de profil** à modifier pour changer de modèle (chemin + dimensions + options).

**Architecture:** Un fichier YAML `vgc_model.yaml` (ou référence vers un package Trainer exporté) est la **source de vérité**. Le bridge `pokecoach2-python` le charge au démarrage, vérifie le SHA-256, instancie la policy avec les **`policy_kwargs` du profil**, charge les poids, puis utilise **le même checkpoint pour p1 et p2** (comportement actuel de `battle_session.py`). Pas d’adversaire historique obligatoire.

**Tech stack:** PokeCoach (`pokecoach2-python`), bridge `engine-runtime`, VGC-Bench vendor épinglé, checkpoints SB3 `.zip`, manifeste Trainer (`ModelManifest`), pytest.

**Périmètre dépôts:**

| Dépôt | Rôle |
|--------|------|
| `pokecoach2-python/` | Implémentation principale (profil, bridge, tests, recette) |
| `pokecoach2-trainer/` | Doc + optionnel : exemple de profil / lien export → profil |

**Hors scope (palier 1):** duel Tiny vs `100.zip`, file d’attente de modèles, UI de sélection, entraînement dans PokeCoach.

## Global Constraints

- Même révision VGC-Bench que Trainer (`d79f953…`) et policy patchée (`num_layers`, `num_heads`, `embed_len` configurables).
- Ne pas charger un `state_dict` Tiny dans une instance policy créée avec l’architecture historique : **recréer** la policy via `PPO(..., policy_kwargs=…)` puis `load_state_dict` ou `PPO.load` cohérent avec le zip.
- Aucun repli silencieux vers stub/fixture si `POKECOACH_VGC_REAL=1` et profil explicite.
- Succès palier 1 : **partie complète** (preview → fin) avec **ton** modèle ; défaite acceptable ; traces persistées ; **redémarrage** + second combat avec le **même** profil.
- Un seul endroit utilisateur pour changer de modèle : **`VGC_MODEL_PROFILE`** pointant vers **un** YAML (voir ci-dessous).

---

## Fichier unique : `vgc_model.yaml`

Convention proposée (à copier depuis `engine-runtime/config/vgc_model.example.yaml`) :

```yaml
# Seul fichier à éditer pour changer de modèle entraîné.
model_id: bc-tiny-mb-002

checkpoint:
  # Chemin absolu ou relatif au répertoire de ce fichier YAML
  path: ../../pokecoach2-trainer/pokecoach-trainer/storage/runs/bc-tiny-smoke/final.zip
  sha256: null  # null = calculer au chargement ; recommandé : SHA du zip après export/train

architecture:
  d_model: 128
  num_layers: 2
  num_heads: 4
  embed_len: 32
  choose_on_teampreview: true

# Optionnel : vérifier alignement Trainer (export)
trainer_manifest: null
# trainer_manifest: ../../pokecoach2-trainer/.../storage/models/bc-tiny-smoke/package/manifest.json

inference:
  device: auto  # auto | cpu | cuda:0 — même sémantique que Trainer (CC GPU)
  deterministic: true
```

**Mode package Trainer (recommandé après `export`):** une seule clé à changer :

```yaml
model_id: bc-tiny-smoke
trainer_package_dir: /chemin/vers/storage/models/bc-tiny-smoke/package
# Le loader lit package/manifest.json (architecture + sha256) et package/model.zip
```

Règle produit : **`trainer_package_dir` XOR (`checkpoint` + `architecture`)** — pas les deux contradictoires ; si `trainer_package_dir` est set, il **remplit** checkpoint + architecture depuis `manifest.json`.

**Variable d’environnement unique côté ops :**

```bash
VGC_MODEL_PROFILE=/chemin/vers/vgc_model.yaml
```

Les anciennes variables `VGC_CHECKPOINT_PATH` / `VGC_CHECKPOINT_SHA256` deviennent **dérivées** du profil (ou deprecated avec message clair si le profil est absent).

---

## Cartographie fichiers

| Fichier | Responsabilité |
|---------|----------------|
| `pokecoach2-python/engine-runtime/config/vgc_model.example.yaml` | Modèle documenté + exemple Tiny mb-002 |
| `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/model_profile.py` | Parse YAML, validation Pydantic, résolution chemins, lecture manifeste Trainer |
| `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/policy_factory.py` | Construire/charger `MaskedActorCriticPolicy` + poids depuis profil |
| `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/checkpoint.py` | Étendre si besoin (hash, erreurs explicites) |
| `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/vgc_real.py` | Charger via profil, provenance enrichie (model_id, arch, sha256) |
| `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/battle_session.py` | `set_policy` via factory (même zip p1/p2) |
| `pokecoach2-python/src/pokecoach_python/core/config.py` | `vgc_model_profile: Path \| None` |
| `pokecoach2-python/.env.example` | Documenter `VGC_MODEL_PROFILE` |
| `pokecoach2-python/engine-runtime/tests/test_model_profile.py` | Tests unitaires profil |
| `pokecoach2-python/engine-runtime/tests/test_model_load_integration.py` | Chargement zip Tiny réel (skip si absent) |
| `pokecoach2-python/tests/integration/test_full_battle_trained_model.py` | Combat réel optionnel (`POKECOACH_VGC_REAL=1`) |
| `pokecoach2-trainer/docs/reviews/palier-1-etapes-8-9.md` | Rapport PASSED/FAILED |
| `pokecoach2-trainer/docs/recettes/pokecoach-charger-modele.md` | Recette utilisateur (profil → combat) |

---

### Task 1 : Schéma `VgcModelProfile` (Pydantic)

**Files:**
- Create: `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/model_profile.py`
- Create: `pokecoach2-python/engine-runtime/tests/test_model_profile.py`
- Modify: `pokecoach2-python/engine-runtime/pyproject.toml` (si dépendance `pyyaml` manquante)

**Interfaces:**
- Produces: `load_vgc_model_profile(path: Path) -> VgcModelProfile`
- Produces: `VgcModelProfile.resolved_checkpoint_path: Path`, `.architecture_dict`, `.checkpoint_sha256_expected: str | None`

- [ ] **Step 1:** Test — YAML minimal valide, chemins relatifs résolus par rapport au dossier du YAML.
- [ ] **Step 2:** Test — `d_model % num_heads != 0` → erreur.
- [ ] **Step 3:** Test — `trainer_package_dir` avec manifeste Trainer mock → remplit architecture + path `model.zip`.
- [ ] **Step 4:** Implémenter `VgcModelProfile` (champs alignés sur Trainer `ModelConfig` : mêmes noms).
- [ ] **Step 5:** `pytest engine-runtime/tests/test_model_profile.py -q`

**Commit:** `feat(bridge): schéma vgc_model.yaml et loader de profil`

---

### Task 2 : Factory de chargement policy (architecture + zip)

**Files:**
- Create: `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/policy_factory.py`
- Modify: `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/model_profile.py` (types)
- Test: `pokecoach2-python/engine-runtime/tests/test_policy_factory.py`

**Interfaces:**
- Consumes: `VgcModelProfile`, `torch.device`
- Produces: `load_policy_from_profile(profile: VgcModelProfile, device: torch.device) -> MaskedActorCriticPolicy`

Stratégie de chargement (ordre) :

1. Vérifier `checkpoint.path` existe ; `verify_checkpoint(path, profile.sha256)`.
2. Créer env minimal `ShowdownEnv` + `PPO(MaskedActorCriticPolicy, env, policy_kwargs=profile.policy_kwargs(), …)` **sans** `learn`.
3. Charger poids : préférer `PPO.load(zip, device=…)` si les métadonnées SB3 du zip correspondent ; sinon `load_state_dict` depuis `policy.pth` dans le zip (comme `PolicyPlayer.set_policy`).
4. Vérifier `choose_on_teampreview` sur l’instance policy == profil.
5. Lever `ModelLoadError` avec message explicite (hash, shape mismatch, missing file).

- [ ] **Step 1:** Test avec zip Tiny local (fixture path env `POKECOACH_TEST_CHECKPOINT`) — skip si fichier absent.
- [ ] **Step 2:** Test — mauvais sha256 → `ValueError` clair.
- [ ] **Step 3:** Test — profil historique (256/3/4/32) + zip référence si disponible (skip sinon).
- [ ] **Step 4:** Implémenter factory.
- [ ] **Step 5:** `pytest engine-runtime/tests/test_policy_factory.py -q`

**Commit:** `feat(bridge): chargement policy depuis profil modèle`

---

### Task 3 : Brancher le profil dans `vgc_real` et la config PokeCoach

**Files:**
- Modify: `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/vgc_real.py`
- Modify: `pokecoach2-python/src/pokecoach_python/core/config.py`
- Modify: `pokecoach2-python/.env.example`
- Test: `pokecoach2-python/tests/unit/test_vgc_model_profile_config.py`

**Interfaces:**
- Consumes: `load_vgc_model_profile`, `Settings.vgc_model_profile`
- Produces: `RealRuntime` stocke `model_id`, `checkpoint_sha256`, `architecture` pour traces

- [ ] **Step 1:** Ajouter `VGC_MODEL_PROFILE` → `Settings.vgc_model_profile: Path | None`.
- [ ] **Step 2:** `RealRuntime.__init__` : si profil présent, charger profil + factory ; sinon comportement actuel (`VGC_CHECKPOINT_PATH`) avec **warning** deprecation.
- [ ] **Step 3:** `hello` / métadonnées runtime : exposer `modelId`, `checkpointSha256`, `architecture` (dict).
- [ ] **Step 4:** Tests unitaires config + mock runtime.

**Commit:** `feat(pokecoach): VGC_MODEL_PROFILE et runtime enrichi`

---

### Task 4 : `battle_session` — même modèle p1/p2 via factory

**Files:**
- Modify: `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/battle_session.py`
- Modify: `pokecoach2-python/engine-runtime/src/pokecoach_vgc_bridge/battle_runner.py` (propager profil ou policy pré-chargée)
- Test: `pokecoach2-python/engine-runtime/tests/test_battle_session_model_profile.py`

**Interfaces:**
- Consumes: `load_policy_from_profile` ou policy déjà chargée passée par `RealRuntime`
- Produces: `create_policy_pair(..., model_profile: VgcModelProfile | None)` charge **deux fois la même policy** (deux instances, mêmes poids) — pas de `VGC_OPPONENT_*`.

- [ ] **Step 1:** Test — mock factory appelée 2 fois avec même profil.
- [ ] **Step 2:** Remplacer `p1.set_policy(checkpoint_path)` direct par factory quand profil actif.
- [ ] **Step 3:** Conserver chemin legacy `checkpoint_path` seul si pas de profil (compatibilité `100.zip` + `PPO.load` seul tant que zip auto-descriptif).

**Commit:** `feat(bridge): session combat avec profil modèle unique`

---

### Task 5 : Compatibilité encodeur / masques (inspiré étape 8)

**Files:**
- Create: `pokecoach2-python/engine-runtime/tests/test_encoder_parity.py`
- Optional: `pokecoach2-trainer/tests/test_pokecoach_encoder_parity.py` (appelle même vecteur de test)

**Interfaces:**
- Produces: test comparant `PolicyPlayer.embed_battle` + `DoublesEnv.get_action_mask` sur états fixture (ou replay pickle) entre Trainer vendor CWD et bridge CWD.

- [ ] **Step 1:** Fixture JSON minimal (observation shape, mask shape) enregistrée depuis `dataset inspect` mb-002.
- [ ] **Step 2:** Test parité dimensions `(6936,)`, mask `(214,)` ou équivalent act_len.
- [ ] **Step 3:** Documenter dans recette : si échec → contrat / commit VGC-Bench divergent.

**Commit:** `test(bridge): parité encodeur et masque BC`

---

### Task 6 : Export Trainer → profil (doc + exemple)

**Files:**
- Create: `pokecoach2-trainer/docs/recettes/pokecoach-charger-modele.md`
- Create: `pokecoach2-trainer/configs/pokecoach-vgc_model.example.yaml`
- Modify: `pokecoach2-trainer/README.md` (lien recette)

Contenu recette :

1. `devbox run cli -- export --config configs/tiny.yaml --result storage/runs/bc-tiny-smoke/result.json`
2. Copier ou référencer `storage/models/<id>/package/`
3. Remplir **un** `vgc_model.yaml` (`trainer_package_dir` **ou** path `final.zip` + architecture depuis `manifest.json`)
4. `VGC_MODEL_PROFILE=... POKECOACH_VGC_REAL=1` + commande combat documentée

- [ ] **Step 1:** Exemple YAML pointant vers package exporté.
- [ ] **Step 2:** Table « changer de modèle » : une colonne = fichier à éditer, une ligne par essai.

**Commit:** `docs(trainer): recette chargement modèle PokeCoach`

---

### Task 7 : Preuve combat complet + redémarrage (étape 9 réorientée)

**Files:**
- Create: `pokecoach2-python/tests/integration/test_full_battle_trained_model.py`
- Create: `pokecoach2-trainer/docs/reviews/palier-1-etapes-8-9.md`

**Critères PASSED:**

- [ ] Profil `vgc_model.yaml` charge le zip Tiny (ou export) sans erreur.
- [ ] `POKECOACH_VGC_REAL=1`, pas de fixture bridge.
- [ ] Combat complet : preview (policy `choose_on_teampreview: true`), au moins un tour, fin terminal ou timeout documenté.
- [ ] Traces persistées avec `model_id` + sha256 + architecture.
- [ ] Redémarrage processus → **même** profil → second combat OK.
- [ ] Zéro ordre rejeté Showdown (pas de repli action secrète).

**Critères NON requis:** battre le modèle historique ; deux checkpoints différents.

- [ ] **Step 1:** Script/recette manuelle documentée (Showdown local, devbox engine).
- [ ] **Step 2:** Test d’intégration marqué `@pytest.mark.real` skip par défaut.
- [ ] **Step 3:** Remplir `palier-1-etapes-8-9.md` avec commandes, hashes, verdict.

**Commit:** `test(pokecoach): preuve combat modèle entraîné (profil unique)`

---

### Task 8 : Nettoyage plan / checklist palier 1

**Files:**
- Modify: `pokecoach2-trainer/docs/plans/2026-11-02-palier-1-trainer-bc-tiny.md` (addendum : étapes 8–9 remplacées par ce plan)
- Modify: checklist section 6 (retirer adversaire historique obligatoire)

- [ ] **Step 1:** Addendum en tête du plan historique ou lien vers ce document.
- [ ] **Step 2:** Checklist : « profil unique + combat + redémarrage ».

**Commit:** `docs: aligner palier 1 étapes 8–9 sur profil modèle unique`

---

## Ordre d’exécution recommandé

```text
Task 1 → 2 → 3 → 4 → 5 (parallèle possible avec 4) → 6 → 7 → 8
```

## Vérification globale

```bash
# pokecoach2-python (engine-runtime)
cd pokecoach2-python/engine-runtime && pytest tests/test_model_profile.py tests/test_policy_factory.py -q

# Config
grep VGC_MODEL_PROFILE pokecoach2-python/.env.example

# Manuel (après export Trainer + vgc_model.yaml rempli)
export VGC_MODEL_PROFILE=/chemin/vgc_model.yaml
export POKECOACH_VGC_REAL=1
# … recette docs/recettes/pokecoach-charger-modele.md
```

## Self-review (spec vs plan)

| Exigence utilisateur | Task |
|---------------------|------|
| Remplacer modèle VGC-Bench, pas duel | 4, 7 |
| Un seul fichier path + dimensions | Fichier `vgc_model.yaml`, Task 1, 6 |
| Changement de modèle facile | `trainer_package_dir` ou éditer path + arch, Task 6 |
| Inspiré encodeur/contrat étape 8 | Task 2, 5 |
| Preuve combat étape 9 | Task 7 |
| Option adversaire historique | Hors scope (note addendum Task 8) |

---

## Exécution

Plan enregistré dans `docs/plans/2026-11-02-pokecoach-chargement-modele-unique.md`.

**Deux options d’exécution :**

1. **Subagent-Driven (recommandé)** — une tâche = un sous-agent + revue entre tâches.
2. **Inline** — implémenter Tasks 1→8 dans cette session avec pauses aux Tasks 4 et 7.

Quelle approche préfères-tu ?
