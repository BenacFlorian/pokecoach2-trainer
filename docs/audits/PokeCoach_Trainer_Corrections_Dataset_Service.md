# PokeCoach Trainer — Corrections de `dataset_service.py`

**Objet :** corriger deux problèmes identifiés lors de la lecture du service de préparation des datasets.  
**Priorité :** avant de constituer un dataset volumineux pour une nouvelle régulation.  
**Périmètre :** préparation des trajectoires, traçabilité des replays et découpage entraînement/validation.

> **Note de portée :** cette spécification décrit les corrections à implémenter. Les extraits sont des exemples de conception à adapter aux signatures et structures exactes du dépôt. Les problèmes rapportés proviennent de la lecture précédente du code, et doivent être confirmés par les tests ci-dessous.

---

## Problème 1 — Attribution erronée des trajectoires aux replays

### Code problématique

Dans `dataset_service.py`, l'association entre une trajectoire et un replay repose sur une déduction par indice :

```python
replay_guess = replay_ids[i % len(replay_ids)]
```

Cette formule **n'établit aucun lien réel** entre la trajectoire `i` et son replay d'origine.

Si `process_logs()` traite les replays en parallèle et collecte les résultats dans leur ordre d'achèvement, l'ordre des trajectoires peut différer de l'ordre des `replay_ids`. Un replay peut en outre générer plusieurs trajectoires (une par perspective de joueur), voire aucune si sa conversion échoue.

### Conséquences

- Mauvais `replay_id` enregistré pour une trajectoire.
- Risque de placer les perspectives `p1` et `p2` d'un même combat dans des ensembles différents.
- **Fuite de données** entre entraînement et validation : l'évaluation peut paraître meilleure qu'elle ne l'est réellement.
- Difficile de retrouver le replay responsable d'une erreur de conversion ou d'une observation aberrante.

### Correction demandée

**Ne jamais reconstruire la provenance à partir de la position de sortie.** La conserver dès le début de la conversion et la transmettre jusqu'au dataset final.

Pour chaque trajectoire, stocker au minimum :

```python
{
    "replay_id": "identifiant-replay",
    "player_side": "p1",  # ou "p2"
    "trajectory": trajectory,
}
```

Approche recommandée :

1. Faire en sorte que la fonction qui traite **un replay** retourne les trajectoires accompagnées de leur `replay_id` d'origine et du côté (`p1`/`p2`).
2. Si une exécution parallèle est utilisée, transporter ces métadonnées **dans le résultat du worker** ; ne pas dépendre de l'ordre de complétion.
3. Supprimer complètement la ligne `replay_ids[i % len(replay_ids)]` et toute logique équivalente.
4. Effectuer le **split au niveau du replay** : affecter chaque `replay_id` une seule fois à `train` ou `validation` (et éventuellement `test`), puis y placer **toutes** ses trajectoires.
5. Enregistrer le lien `trajectory_id → replay_id → player_side → split` dans le manifeste du dataset.

Exemple de structure (pseudocode Python, **pas un patch direct**) :

```python
from dataclasses import dataclass

@dataclass
class TaggedTrajectory:
    replay_id: str
    player_side: str  # "p1" ou "p2"
    trajectory: object

# Le worker de conversion retourne des TaggedTrajectory.
converted: list[TaggedTrajectory] = convert_replays_with_provenance(replays)

# Important : un même replay ne doit appartenir qu'à un seul split.
unique_replay_ids = sorted({item.replay_id for item in converted})
train_ids, validation_ids = split_replay_ids(
    unique_replay_ids, seed=42, validation_ratio=0.1
)
train = [item for item in converted if item.replay_id in train_ids]
validation = [item for item in converted if item.replay_id in validation_ids]

assert {x.replay_id for x in train}.isdisjoint(
    {x.replay_id for x in validation}
)
```

**Attention :** si la fonction `process_logs()` de VGC-Bench ne renvoie que les trajectoires, adapter son wrapper ou la fonction interne qui transforme un replay, plutôt que d'essayer de deviner les identifiants après coup. Maintenir autant que possible l'interface amont compatible.

### Tests de non-régression

- [ ] Un replay connu avec deux perspectives donne deux trajectoires portant **le même** `replay_id`, avec `player_side` distinct.
- [ ] En inversant artificiellement l'ordre d'achèvement des workers, les associations restent correctes.
- [ ] Une conversion qui échoue ou ne produit aucune trajectoire ne décale pas l'identité des autres.
- [ ] Les ensembles de `replay_id` dans `train` et `validation` sont **disjoints**.
- [ ] Une préparation répétée avec la même seed et les mêmes données donne la même répartition.

---

## Problème 2 — Conversion des logs exécutée uniquement si `force=True`

### Comportement problématique

Dans la version examinée de `dataset_service.py`, l'appel à `process_logs()` semble se trouver sous :

```python
if config.force:
    # ...
    trajs = process_logs(...)
```

Le code suivant utilise ensuite `trajs`, même dans le cas où `config.force` est faux. Si aucun autre chemin ne définit `trajs`, cela peut provoquer une exception de type `UnboundLocalError`, ou empêcher la préparation normale d'un dataset.

### Comportement attendu

- **Première préparation :** convertir les logs même si `force=False`.
- **Dataset déjà préparé et `force=False` :** réutiliser le résultat existant ou renvoyer une réponse explicite du type « déjà préparé », selon le contrat actuel de l'API.
- **Dataset déjà préparé et `force=True` :** relancer la conversion et remplacer les artefacts de façon contrôlée.
- **Erreur de conversion :** conserver un état d'échec explicite, sans publier un dataset incomplet comme valide.

### Correction demandée

La condition `force` doit contrôler **la réutilisation / régénération des artefacts**, pas l'existence de l'étape de conversion.

Pseudocode de principe :

```python
def prepare_dataset(dataset_id: str, config):
    existing = find_prepared_dataset(dataset_id)

    if existing is not None and not config.force:
        return existing  # ou lever une erreur métier « déjà préparé »

    # Chemin normal : atteint à la première préparation ET lors d'un force=True.
    logs = load_source_logs(dataset_id)
    tagged_trajs = convert_logs_with_provenance(logs)
    train, validation = split_by_replay(tagged_trajs, seed=config.seed)

    # Publier de façon atomique : fichiers temporaires -> validation -> remplacement.
    return save_prepared_dataset(dataset_id, train, validation)
```

Si le code actuel nettoie des fichiers uniquement quand `force=True`, conserver cette logique **à l'endroit approprié**, en évitant de supprimer une version valide avant qu'une nouvelle préparation ait réussi.

### Tests de non-régression

- [ ] Dataset brut jamais préparé + `force=False` → préparation réussie.
- [ ] Dataset déjà préparé + `force=False` → aucun calcul coûteux inutile, réponse cohérente.
- [ ] Dataset déjà préparé + `force=True` → régénération réelle.
- [ ] Erreur pendant la régénération → l'ancien dataset valide n'est pas détruit.
- [ ] Aucune branche de code ne lit une variable locale (`trajs` ou équivalent) avant son initialisation.

---

## Ordre d'implémentation conseillé

1. **Corriger la provenance** au niveau de la fonction de conversion d'un replay.
2. Ajouter les tests prouvant que `p1` et `p2` restent ensemble, même en conversion parallèle.
3. Corriger les branches `force=False` / `force=True` dans la préparation.
4. Produire un petit dataset local et inspecter son manifeste (`replay_id`, côté, split, nombre de transitions).
5. Exécuter un entraînement Tiny sur ce dataset préparé pour vérifier que le format reste compatible avec le pipeline BC actuel.

## Critères d'acceptation globaux

Le correctif est terminé lorsque :

- Une préparation initiale fonctionne **sans `force=True`**.
- Chaque trajectoire conserve sa provenance réelle, sans déduction par indice.
- Aucun replay n'apparaît à la fois dans `train` et `validation`.
- Une préparation forcée est reproductible et ne détruit pas un dataset valide en cas d'échec.
- Le dataset obtenu permet de lancer l'entraînement Tiny existant **sans changer son contrat d'entrée**.

## Références du projet

- Dépôt Trainer : <https://github.com/BenacFlorian/pokecoach2-trainer>
- VGC-Bench (référence amont) : <https://github.com/cameronangliss/vgc-bench>
- Fichier ciblé : `dataset_service.py` (vérifier son chemin exact dans le dépôt local)
