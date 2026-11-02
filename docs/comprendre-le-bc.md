# Comprendre le behavior cloning (BC) — Palier 1

Guide pédagogique du Trainer PokeCoach. Sera enrichi avec un exemple réel anonymisé après la première préparation de dataset (étape 3).

## Glossaire

| Terme | Sens dans ce projet |
| --- | --- |
| **Replay** | Enregistrement JSON d’une partie Showdown (humain). |
| **Trajectoire** | Suite d’observations et d’actions pour un joueur sur une partie. |
| **Transition** | Une décision : observation (dict avec masque) + paire d’actions humaines. |
| **Observation** | Vecteur encodé par VGC-Bench + masque (entraînement : masque synthétique). |
| **Action** | Deux indices entiers (un par Pokémon actif), espace de taille `act_len=107` chacun. |
| **Policy** | Réseau `MaskedActorCriticPolicy` (actor + critic ; BC n’utilise que l’actor via la loss). |
| **Batch** | Lot de transitions consommées par `imitation.BC` / PyTorch. |
| **Époque** | Un passage complet sur le split train. |
| **Loss** | Écart entre log-probabilités de la policy et actions humaines (masquées). |
| **Gradient** | Dérivées pour ajuster les poids ; **absents** pendant la validation (copie isolée). |
| **Optimiseur** | Met à jour les poids après `backward()` ; géré par BC, pas par `ppo.learn()`. |
| **Checkpoint** | Archive `.zip` Stable-Baselines3 (policy + métadonnées). |
| **Inférence** | Choix d’actions en combat avec masques légaux Showdown. |

## Chaîne Tiny (résumé)

1. Logs JSON → trajectoires `.pkl` (`logs2trajs`, chemins explicites).
2. Split par **identité de partie** (deux perspectives ensemble).
3. `PPO(..., MaskedActorCriticPolicy)` conteneur ; **`BC.train()`** seul ajuste les poids.
4. Export zip + manifeste contrat ; reload CPU dans un processus neuf.

## Masque synthétique BC (dette documentée)

Le fork VGC-Bench met `action_mask` à 1 partout sauf indices `47:87` pour chaque moitié du vecteur `2 * act_len`, mimiquant l’absence de certaines mega/z/dmax/tera sans recalculer les légalités tour par tour. Toute action humaine tombant sur un bit à 0 est **bloquante** pour la préparation.

## Qui fait quoi pendant `train_bc`

| Étape | Responsable |
| --- | --- |
| Forward + logits | `MaskedActorCriticPolicy.get_logits` / `evaluate_actions` |
| Loss imitation | `imitation.algorithms.bc.BC` |
| `backward` / pas optimiseur | BC (pas `ppo.learn()`) |
| Validation loss | Notre `evaluate_validation` sur **`copy.deepcopy(policy)`** |
| Sauvegarde | `ppo.save(...)` |

*(Exemple chiffré et schéma observation → Transformer ajoutés après étape 2–4.)*
