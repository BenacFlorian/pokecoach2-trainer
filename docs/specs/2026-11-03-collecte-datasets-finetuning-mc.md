# PokeCoach Trainer — Collecte, datasets et fine-tuning BC M-B → M-C

**Date :** 3 novembre 2026.  
**Nature :** spécification fonctionnelle et technique à implémenter.  
**Statut :** décisions fonctionnelles discutées avec l’utilisateur ; propositions techniques précisées ci-dessous ; aucun développement exécuté pour rédiger ce document.

## 1. Résultat attendu et décisions validées

> Depuis FastAPI local, je lance une collecte de replays Showdown M-C. Je construis un dataset `natural` d’environ 300 replays exploitables, puis je poursuis l’apprentissage BC du checkpoint VGC-Bench M-B en conservant son architecture. J’exporte le modèle et l’utilise pour terminer un combat M-C dans PokeCoach, puis je reproduis cette preuve après redémarrage.

Le mot « Tiny » ne désigne **pas** une nouvelle architecture dans cette version. Il s’agit d’un **petit entraînement de démonstration**, avec peu de replays. Il ne faut pas appliquer les valeurs `128/2/4` du précédent palier au checkpoint M-B.

| Sujet | Décision utilisateur |
| --- | --- |
| Chaîne | Collecte brute → Dataset Builder → entraînement BC → PokeCoach. |
| Initialisation du modèle | Charger le checkpoint VGC-Bench M-B et conserver son architecture et ses poids comme point de départ. |
| Données d’adaptation | Replays de la régulation M-C. |
| Déclenchement de collecte | Manuel, depuis FastAPI ; collecte incrémentale, pas de planification automatique. |
| Archivage | Garder les replays bruts, y compris ceux qu’un dataset rejettera. |
| Filtre Elo | Appliquer le seuil au joueur dont on imite les décisions, perspective par perspective. |
| Défauts du dataset | Minimum 1 100, sans maximum ; Elo inconnu exclu ; gagnants et perdants acceptés. |
| Reconstructibilité | OTS et équipes complètes requis ; rejet expliqué si reconstruction impossible. |
| Types de partie | BO1 et BO3 acceptés. |
| Abandons | Acceptés après au moins trois tours terminés ; défaites par inactivité exclues. |
| Variantes | `natural` uniquement ; `diverse` plus tard. Pas de plafond de replays par équipe en V1. |
| Volume | Cible de 300 replays exploitables après filtrage et conversion. |
| Répartition | Environ 80 % train, 10 % validation, 10 % test, reproductible. |
| BO3 incomplet/inconnu | Chaque partie reste admissible individuellement. Ne pas rejeter une partie valide parce que la série est incomplète ou non identifiée. |
| Traçabilité BO3 | Conserver les liens de série connus. Grouper les parties d’une série identifiée dans le même split ; sinon grouper par replay et signaler la limite. |
| Socle | Devbox, uv, FastAPI ; application locale. |

Les valeurs de seed, learning rate, budgets réseau et noms de nouvelles routes ci-dessous sont des **choix techniques proposés pour rendre la V1 exécutable**. Elles ne sont pas présentées comme des préférences exprimées par l’utilisateur.

## 2. Portée et articulation avec les documents précédents

Ce travail prolonge le [palier 1](step/PokeCoach_Palier_1_Trainer_BC_Tiny.md). Pour cette version, les exigences suivantes remplacent celles du premier palier :

- un checkpoint initial M-B obligatoire, au lieu d’un réseau Tiny initialisé aléatoirement ;
- la collecte Showdown et M-C entrent dans le périmètre ;
- un split `test` s’ajoute à `train` et `validation` ;
- l’archive de replays est commune à plusieurs constructions de datasets ;
- les datasets publiés sont immuables et associés à leur recette complète.

L’intégration PokeCoach suit le [plan de chargement par profil unique](../plans/2026-11-02-pokecoach-chargement-modele-unique.md) et le code existant `VGC_MODEL_PROFILE`. Ne pas réintroduire l’ancienne obligation de deux checkpoints adverses distincts. Pour la preuve de compatibilité, le profil peut être utilisé par les deux joueurs, dans deux instances distinctes ; ce combat n’est pas un entraînement par self-play.

Les deux prérequis identifiés dans l’[audit du service dataset](../audits/PokeCoach_Trainer_Corrections_Dataset_Service.md) ont été corrigés avant la réalisation de cette version : préparation sans `force=True` et provenance explicite des trajectoires. Ces corrections constituent le socle existant à conserver, pas des travaux à refaire. L’audit décrit les problèmes d’origine ; l’état actualisé figure en section 3.

L’immuabilité reste une évolution à réaliser : une régénération publiée reçoit un nouvel ID, sans écraser un dataset utilisé par un modèle. Corriger la préparation sans `force` ne signifie pas que ce nouveau contrat de publication est déjà implémenté.

Hors périmètre : collecte planifiée, nouvelle interface Angular, entraînement RL, distillation, changement de capacité du réseau, recherche d’hyperparamètres, variantes `diverse`, classement stratégique des équipes, apprentissage de stratégies entre manches BO3, cloud GPU et accès public à l’API.

Le travail reste progressif et commenté en français. À chaque lot de réalisation, fournir une démonstration, les tests exécutés et les limites, puis attendre la revue humaine. L’approbation du code par l’utilisateur porte sur sa compréhension et le comportement montré ; les preuves techniques sont à la charge de l’implémenteur.

## 3. État du code, prérequis corrigés et évolutions nécessaires

Constats actualisés par lecture du code local après les deux corrections préalables, sans attester d’une exécution complète du parcours M-C :

| Emplacement | État observé | Exigence pour cette version |
| --- | --- | --- |
| VGC-Bench `scrape_logs.py` | Filtres métier avant sauvegarde ; recherche incrémentale ; JSON global réécrit ; formats M-A/M-B dans la liste. | Séparer téléchargement et admissibilité au dataset ; reprise et erreurs explicites ; formats configurables vérifiés. |
| `dataset_service.py` — corrigé | La conversion est appelée indépendamment de `config.force` ; une première préparation sans `force=True` est prise en charge. | Conserver ce comportement et son test de non-régression ; ajouter les états persistants et l’immuabilité exigés par cette version. |
| Convertisseur et `dataset_service.py` — corrigé | `process_logs_with_provenance()` associe chaque future à son replay/côté et retourne des `TaggedTrajectory(replay_id, player_side, trajectory)`. `_write_prepared_trajectories()` utilise ces identités pour les fichiers et les splits. | Réutiliser cette interface, sans attribution par position ; enrichir la provenance avec hashes, liens BO3 et décisions détaillées de conversion. |
| `dataset_service.py` | Premier fichier JSON du répertoire choisi ; limite appliquée dans l’ordre du fichier ; split avant résultats de conversion. | Source figée explicite, sélection déterministe et comptes établis sur les trajectoires réellement retenues. |
| Route `/datasets/{id}/prepare` | Conversion exécutée pendant la requête. | Job en processus séparé, réponse `202`, polling et logs. |
| `trajectory_dataset.py` | Vérification des deux demi-masques séparément ; pas de contrôle conditionnel complet ni de bornes explicites. | Contrôler indices, valeurs, cibles et contrainte de la seconde action après la première. |
| `training.py` | Nouvelle instance PPO ; aucun checkpoint initial chargé ; environnement `format_map["mb"]`. | Charger M-B, conserver ses poids et son architecture ; format M-C issu du contrat vérifié. |
| `training.py` | `learning_rate` présent dans le schéma mais non transmis à BC ; pertes non finies transformées en NaN sans arrêt immédiat. | Appliquer le learning rate effectif et arrêter sur une perte non finie avant publication. |
| Validation | Copie reconstruite de la policy ; risque de device différent ; métriques actuellement de type négative log-vraisemblance. | Copie sur le bon device, immuabilité de l’original et définition explicite de chaque métrique. |
| `training_service.py` / `jobs.py` | Sortie du processus en `PIPE` sans lecteur apparent ; sauvegarde de l’échec après un `raise` ; réconciliation interrompant tous les jobs actifs. | Logs drainés/persistés, échec écrit avant propagation, identification des workers vivants avant libération de verrou. |
| Bridge PokeCoach | Profil de modèle existant ; factory utilisant encore M-B pour créer l’environnement ; mapping explicite M-C absent. | Réutiliser le profil, rendre le format cohérent de bout en bout, vérifier réellement M-C. |

Les tests existants `test_prepare_dataset_runs_without_force`, `test_process_logs_with_provenance_keeps_metadata` et `test_write_prepared_trajectories_split_by_replay_id`, dans `tests/test_dataset.py`, couvrent ces corrections avec des fixtures. Les conserver et les compléter pour les nouveaux cas de la section 11 ; ils ne prouvent pas encore une conversion réelle M-C ni le regroupement BO3.

Chemins Trainer ci-dessous relatifs à `pokecoach-trainer/`. Chemins PokeCoach relatifs au dépôt voisin `pokecoach2-python/`. Les sources vendor et leurs patches doivent rester reproductibles : une synchronisation ne doit pas effacer silencieusement les adaptations du convertisseur.

## 4. Architecture retenue

Trois modules dans le **même Trainer FastAPI**, avec des services utilisables en CLI et des workers séparés pour les opérations longues :

1. **Collecteur** : découvrir les IDs, télécharger, archiver, reprendre une collecte.
2. **Dataset Builder** : figer une source, filtrer, convertir, sélectionner, splitter et publier un dataset.
3. **BC Trainer** : charger le checkpoint initial, entraîner sur le dataset prêt, évaluer et exporter.

Leur séparation permet de reconstruire un dataset avec d’autres filtres sans télécharger de nouveau les données. Des microservices distincts ajouteraient des déploiements et des échanges réseau inutiles pour cette application locale ; une chaîne monolithique mêlant filtres et téléchargement empêcherait de réexploiter les rejets. La séparation en trois responsabilités est celle discutée et retenue avec l’utilisateur.

**Choix de stockage V1 : fichiers locaux**, un objet JSON brut par replay/version de contenu, index JSONL reconstruisible et manifestes JSON. Pas de serveur de base de données ni de nouveau gestionnaire de tâches externe. Le volume visé permet ce choix ; une migration future de l’index n’affectera pas le contrat des objets bruts.

```text
storage/
  archive/
    objects/<replay-id>/<sha256>.json
    index.jsonl
    cursors/<format-id>.json
    snapshots/<snapshot-id>.json
  datasets/<dataset-id>/
    recipe.json
    source.json
    replay_decisions.jsonl
    trajectory_index.jsonl
    split_assignments.json
    train/*.pkl
    validation/*.pkl
    test/*.pkl
    manifest.json
  base_models/<base-model-id>/
    model.zip
    manifest.json
  jobs/<job-id>/
    config.json
    state.json
    worker.log
    metrics.jsonl
  runs/<job-id>/
  models/<model-id>/package/
```

Conserver la structure Devbox/uv déjà installée. Tous les chemins sont bornés à des racines configurées, les IDs sont validés, et les écritures de publication sont atomiques. L’API, la CLI, les workers et les vérifications de reload utilisent le même environnement uv ; les enfants sont lancés avec `sys.executable` et une liste d’arguments.

**Concurrence proposée : un seul job long à la fois**, collecte, préparation ou entraînement. Les consultations restent disponibles. Aucun job long ne s’exécute dans le thread HTTP ou via un simple `BackgroundTasks` non supervisé.

## 5. Collecte manuelle et archive brute

### 5.1 Contrat de collecte

La requête contient la régulation/les formats validés, des bornes de date facultatives et des limites de travail. Le collecteur n’applique aucun filtre Elo, résultat, équipe, OTS, espèce ou longueur de partie. La découverte reste naturellement bornée aux formats et dates demandés.

L’objectif est de récupérer les replays publics **accessibles par les interfaces de recherche**, sans promettre une archive exhaustive de tous les combats joués. Les limites de pagination et les plages non vérifiées sont visibles dans le résultat.

Configuration technique initiale proposée :

- un seul téléchargement simultané, débit maximal d’une requête par seconde ;
- timeout réseau de 30 secondes, cinq tentatives maximum par requête ;
- budget par lancement : au plus 5 000 nouveaux replays archivés, 10 000 requêtes ou deux heures, arrêt au premier plafond atteint ;
- budgets modifiables dans des bornes validées ; enregistrer les valeurs effectives ;
- respecter `Retry-After`, limiter les tentatives et appliquer un délai croissant sur erreurs temporaires ;
- l’atteinte d’un budget est une fin normale avec `stop_reason=budget_reached`, pas une déclaration de complétude historique.

Reprendre les appels utiles de VGC-Bench, sans conserver ses filtres destructifs ni son écriture d’un JSON géant. Hébergeur de téléchargement fixé par le service ; ne pas accepter d’URL arbitraire fournie à l’API.

### 5.2 Objet archivé et métadonnées

Conserver la réponse JSON complète disponible, le log intégral, son hash et au minimum :

- `replay_id`, format exact du fournisseur et régulation résolue ;
- date de téléchargement et `uploadtime` avec leur signification explicite ; ne pas présenter une date d’upload comme la date certaine du combat ;
- URL d’origine reconstruite depuis l’ID, révision du collecteur et empreinte du contenu ;
- informations de joueurs, ratings et format BO1/BO3 lorsqu’elles sont fournies ;
- informations de série/numéro de manche lorsqu’elles sont explicitement identifiables.

Les métadonnées dérivées portent la source de leur extraction. Une valeur absente reste `null` ou `unknown`, jamais inventée. Conserver l’objet brut permet de réextraire les métadonnées avec une nouvelle version du parseur.

Un replay déjà archivé avec le même ID/contenu n’est pas téléchargé à nouveau lors de la reprise normale. Les réponses dont l’ID diffère de l’ID demandé, JSON invalides et contenus incomplets produisent une erreur de collecte, pas un replay utilisable. Si un même ID est explicitement réimporté avec un contenu différent, conserver les versions et signaler le conflit ; un snapshot doit désigner une version précise, sans écrasement silencieux.

### 5.3 Pagination, reprise et erreurs

- Distinguer la recherche de nouveaux uploads de l’exploration de l’historique ; persister les deux progressions par format.
- Enregistrer les IDs découverts et leur état : à télécharger, archivé, temporairement indisponible, définitivement indisponible selon réponse observée.
- N’avancer un curseur qu’après avoir persisté les découvertes de la page. Une erreur sur un replay ne doit pas le faire disparaître des reprises futures.
- Traiter les pages vides, réponses HTTP 429/5xx, erreurs réseau, JSON invalides, doublons et plusieurs replays au même timestamp. Détecter les pages répétées sans progression ; ne pas boucler indéfiniment.
- Si l’API de recherche empêche de parcourir tous les IDs à une frontière de timestamp, enregistrer cette limite et `coverage_incomplete`. Ne pas inventer un mécanisme de pagination non pris en charge.
- À l’arrêt ou au crash, les objets déjà publiés restent utilisables ; les fichiers temporaires ne figurent pas dans l’index validé. L’index est reconstruisible depuis les objets et journaux persistés.
- Un nouveau lancement manuel reprend les téléchargements en attente et la recherche de nouveautés ; aucun cron n’est créé.

### 5.4 Snapshot d’archive

La création d’un dataset utilise un **snapshot immuable** : liste d’IDs et versions/hashes bruts, formats, bornes éventuelles, version de métadonnées et informations BO3 connues à cette date. Une collecte ultérieure ne modifie ni ce snapshot ni les datasets déjà publiés.

Le snapshot est produit à la demande depuis l’archive. La fin d’un job de collecte renvoie les compteurs et les formats concernés ; elle ne déclenche automatiquement ni préparation ni entraînement.

## 6. Dataset Builder : filtrer, convertir, publier

### 6.1 Recette du premier dataset

```yaml
dataset:
  variant: natural
  regulation: M-C
  target_replays: 300
  seed: 42
  min_elo: 1100
  max_elo: null
  unknown_elo: exclude_perspective
  only_winners: false
  require_ots: true
  require_complete_teams: true
  include_bo1: true
  include_bo3: true
  include_forfeits: true
  min_completed_turns_forfeit: 3
  exclude_inactivity_losses: true
  deduplicate: true
  max_replays_per_team: null
  splits:
    train: 0.80
    validation: 0.10
    test: 0.10
```

Les paramètres sont typés, validés et enregistrés. Les clés inconnues sont refusées, pour éviter qu’un filtre écrit dans le YAML soit silencieusement ignoré. En V1, `variant` n’accepte que `natural` et un plafond par équipe non nul est refusé avec une explication.

L’API existante utilise `min_rating` et `only_winner` : migrer vers les noms ci-dessus ou fournir des alias documentés. Si ancien et nouveau nom sont simultanément fournis avec des valeurs différentes, refuser la requête. Ne pas entretenir deux recettes indépendantes selon CLI/API.

### 6.2 Deux niveaux de décision

**Niveau partie :** bon format M-C, type BO1/BO3 accepté, absence de duplication, log exploitable, présence de la preview et de décisions, fin identifiée, absence de perte par inactivité et respect du seuil d’abandon.

**Niveau perspective :** Elo du joueur dans les bornes inclusives, Elo connu lorsque requis, victoire si `only_winners=true`, informations suffisantes pour reconstruire cette perspective, conversion et contrôles de trajectoire valides.

Avec les défauts retenus, une partie entre 1 300 et 1 000 fournit uniquement la perspective 1 300. `min_elo=1100` porte sur le rating individuel fourni dans le log, pas sur une moyenne de la partie. Si cette valeur manque, ne pas lui substituer un rating moyen provenant d’un autre champ.

Pour `require_ots=true` et `require_complete_teams=true`, vérifier les informations nécessaires des **deux équipes** : six membres identifiables et feuilles ouvertes utilisables par le convertisseur. La seule présence de la chaîne `showteam` n’est pas suffisante. Un cas non supporté, notamment une ambiguïté liée à Illusion, est archivé mais rejeté pour cette construction avec un motif technique documenté ; ne pas inventer des identités de Pokémon.

### 6.3 Résultats, abandons et données incomplètes

Extraire un résultat typé : victoire normale, abandon, inactivité, égalité, fin inconnue. Conserver les messages source justifiant la classification. Ne pas déduire « abandon » du seul fait qu’une partie est courte, ni « inactivité » de sa durée.

Le seuil d’abandon compte les **tours terminés**, pas le dernier numéro de tour apparu. Par exemple, une entrée dans le tour 3 suivie d’un abandon avant sa résolution ne prouve pas trois tours terminés. Documenter les marqueurs du protocole utilisés et couvrir les cas limites avec des logs réels/fixtures. Si le nombre nécessaire n’est pas déterminable, rejeter ce replay avec `forfeit_turn_count_unknown`.

Choix technique V1 : conserver dans l’archive les égalités et fins inconnues mais les exclure du dataset (`draw_unsupported`, `termination_unknown`), le convertisseur actuel recherchant explicitement un gagnant. Une future prise en charge ne nécessite pas de nouvelle collecte. Un replay manifestement tronqué ou malformé reste non exploitable même si sa série BO3 est connue.

### 6.4 BO3 : admissibilité individuelle, liens conservés

**Ne pas utiliser `bo3_series_unknown` comme motif de rejet.** Une première manche valide peut être retenue même si la seconde est abandonnée, rejetée par les filtres ou absente de l’archive. Ne pas exiger une série complète ni une victoire dans la série.

Conserver par replay : `match_type`, `series_id`, `game_number`, `linked_replay_ids`, `series_link_source` et `series_link_status` (`confirmed`, `unknown` ou `conflict`). Conserver les liens vers des replays non collectés sans les traiter comme des fichiers disponibles.

Un lien est confirmé à partir d’un identifiant ou d’une référence explicite du protocole/fournisseur. Le même duo de joueurs, les mêmes équipes ou des horaires proches ne suffisent pas à inventer un identifiant de série. En cas de conflit, garder les éléments source, marquer la relation incertaine et ne pas en déduire un regroupement fiable.

Les parties BO3 sans lien fiable sont acceptées selon les mêmes règles individuelles. Les limites de séparation sont déclarées dans le manifeste : nombre de BO3 non groupables et `series_leakage_guarantee=false` s’il en existe. Cela n’est pas un échec de la V1 ; cette limite a été acceptée par l’utilisateur.

### 6.5 Conversion et provenance

Réutiliser `LogReader`, `process_log` et `PolicyPlayer.embed_battle` du VGC-Bench figé, ainsi que l’interface locale `process_logs_with_provenance()` et ses `TaggedTrajectory` déjà ajoutées. Enrichir cette enveloppe avec les métadonnées et décisions ci-dessous ; ne pas réimplémenter l’encodeur dans le Trainer.

Une unité de conversion identifie **un replay et une perspective**. Chaque résultat contient :

```text
replay_id, raw_sha256, player_side (p1|p2), series_id éventuel,
status (converted|rejected|error), reason_code, details,
trajectory si succès, converter_revision, encoding_contract
```

Chaque tâche de conversion doit produire un résultat avec ces champs, même en exécution parallèle et même en cas d’erreur. L’association actuelle entre future et `(replay_id, player_side)` assure déjà la provenance des succès, indépendamment de l’ordre d’achèvement ; la préserver. Étendre le suivi aux résultats vides et aux exceptions, actuellement seulement comptés globalement, pour fournir une décision détaillée par perspective. Le Builder utilise l’interface avec provenance ; `process_logs()` reste une interface historique renvoyant les seules trajectoires, insuffisante pour ce parcours.

Le Builder effectue les filtres métier de façon explicite ; désactiver les filtres redondants du convertisseur après sélection pour éviter des rejets silencieux. Les erreurs techniques du convertisseur restent rapportées par perspective.

Contrôles avant publication : observations de bonnes dimensions et finies, indices d’actions entiers et bornés, correspondance observations/actions attendue par `imitation`, transitions non vides, état terminal cohérent et cible autorisée par le masque BC. Appliquer à la seconde cible le masque conditionnel après la première action, avec la logique de la policy.

Une cible humaine interdite par le masque BC est **bloquante pour la construction** : ne pas supprimer silencieusement la trajectoire, ouvrir artificiellement le bit cible ou la compter parmi les simples rejets de qualité. Enregistrer replay, perspective, transition et paire d’actions. Les échecs de reconstruction ordinaires sont comptés et permettent de poursuivre sur d’autres replays.

Documenter exactement le masque synthétique et son empreinte ; ne pas le présenter comme un masque légal exhaustif. À l’inférence, conserver les masques légaux issus de l’état Showdown.

### 6.6 Sélection, cible de 300 et splits

- Une partie est « exploitable » si au moins une perspective passe les filtres, la conversion et les contrôles. Deux perspectives ne comptent jamais pour deux replays.
- Dédupliquer les objets identiques par hash de log ; conserver les IDs alias dans la provenance. Ne pas confondre deux parties différentes jouées avec les mêmes équipes avec un doublon.
- Former les groupes de séparation à partir des replays, de leurs aliases et des séries confirmées dans le snapshot. Une partie rejetée ne provoque pas le rejet des autres parties de sa série.
- Parcourir les groupes dans un ordre pseudo-aléatoire déterministe dépendant du seed et d’IDs triés, jamais dans l’ordre d’arrivée des workers. Retenir les perspectives admissibles des groupes jusqu’à atteindre 300 replays exploitables. Finir le groupe en cours pour ne pas tronquer une série connue : un léger dépassement est permis et compté.
- Une fois la cible atteinte, les candidats non traités sont `not_selected_target_reached`, pas « rejetés pour mauvaise qualité ». Aucune pondération de diversité ni quota par équipe en V1.
- Affecter les groupes retenus à train/validation/test avec un algorithme déterministe visant 80/10/10 **en nombre de replays**, sans couper de groupe. Documenter l’algorithme et les écarts effectifs ; trois splits non vides sont nécessaires.
- Un même replay, ses deux perspectives, ses aliases et les membres connus d’une série ne doivent pas traverser les splits. Tester ces propriétés sur les fichiers effectivement publiés, pas seulement sur une fonction de mélange d’IDs.
- Publier `split_assignments.json` avec un `split_id`. Les futures variantes dérivées du même dataset devront réutiliser ce split de référence ; `diverse` ne doit pas être implémenté maintenant.

Si le snapshot ne fournit pas 300 replays exploitables, produire le rapport et un état `insufficient_data`. Ne pas collecter davantage automatiquement, abaisser les seuils ou publier silencieusement un dataset plus petit. L’utilisateur peut collecter à nouveau et construire depuis un nouveau snapshot, ou lancer explicitement une nouvelle construction avec une cible réduite. Signaler dans le rapport final l’écart à l’objectif de démonstration.

### 6.7 Manifestes, versionnement et rejets

Chaque construction possède un nouvel ID. Le manifeste contient : hash du snapshot, recette complète/hash, versions de parseur/conversion/mappings, seed, formats exacts, régulation, split/version d’algorithme, tableaux de provenance et nombres par split de replays, perspectives, trajectoires et transitions.

Les décisions sont enregistrées par construction avec `stage`, `reason_code`, replay, perspective facultative et détails. Distinguer rejet métier, erreur de parsing, téléchargement manquant, non-sélection et erreur bloquante. Fournir les totaux et exemples de chaque motif.

Écrire les sorties dans une zone temporaire puis publier le manifeste `ready` seulement après tous les contrôles. Un échec ne laisse pas un dataset partiellement prêt. Un dataset déjà `ready` ne peut pas être modifié ; `force=true` doit être remplacé par une nouvelle construction, avec `parent_dataset_id` facultatif. Les modèles existants conservent leurs références exactes.

L’import historique de JSON VGC-Bench reste disponible : convertir explicitement `id → [timestamp, log]` en objets d’archive, figer la source et signaler les métadonnées manquantes. Ne plus sélectionner arbitrairement le premier JSON d’un répertoire. Refuser les clés JSON dupliquées contradictoires avant leur écrasement par le décodeur.

## 7. Prérequis M-C et contrat d’encodage

**Un nom de format ajouté au scraper ne suffit pas à rendre le modèle compatible M-C.** Avant fine-tuning, établir un rapport de compatibilité sur :

1. identifiants réels des formats M-C BO1/BO3 exposés par Showdown ;
2. moteur Showdown local capable de valider les équipes et de jouer la régulation cible ;
3. forks `poke-env`/VGC-Bench capables de reconstruire et encoder ces replays ;
4. mappings de Pokémon, capacités, objets, attaques et mécaniques utilisés par les données M-C ;
5. mêmes dimensions **et même sens** des observations/actions que le checkpoint M-B ;
6. correspondance du libellé PokeCoach `M-C` avec le format moteur approprié.

Les formats BO1 et BO3 peuvent être réunis dans le dataset seulement si leur encodage par partie partage le même contrat. Conserver leur format source exact, même si l’entraînement utilise un environnement de conteneur commun à M-C.

Choix V1 : **architecture et contrat d’encodage du checkpoint initial conservés**. Un nouvel index absent de ses tables ou un changement de dimensions est un blocage explicite, pas une invitation à étendre aléatoirement des embeddings, renuméroter les tables ou utiliser `strict=False` pour charger partiellement les poids. Documenter l’écart et demander une décision de périmètre si une migration de contrat devient nécessaire.

La collecte brute peut fonctionner pour un format encore non convertible. Le Builder et l’entraînement doivent alors refuser de présenter des données incompatibles comme prêtes. Ne jamais remplacer M-C par M-B silencieusement pour faire passer un test.

Conserver commits exacts, hashes des patches et empreintes canoniques des tables et conventions. Utiliser les mêmes révisions compatibles dans Trainer et PokeCoach. Un test avec des dimensions identiques mais des tables permutées doit échouer.

## 8. Fine-tuning du checkpoint M-B

### 8.1 Modèle source et configuration

Enregistrer localement un modèle de base fiable sous `base_models/<base-model-id>/`, avec source précise, hash SB3, architecture réellement constatée, classe/imports, versions et contrat. Ce registre est distinct des modèles produits par le Trainer. L’enregistrement est une opération locale explicite ; l’API d’entraînement accepte un `base_model_id`, pas un chemin arbitraire ou une URL de poids.

Le terme « checkpoint M-B » identifie ici le modèle source choisi par l’utilisateur. Documenter la provenance réelle ; ne pas affirmer qu’il a été entraîné exclusivement sur M-B sans preuve de la source.

Configuration d’exemple, avec IDs issus d’opérations préalables :

```yaml
experiment:
  name: bc-mb-to-mc-smoke
  seed: 42
  dataset_id: mc-natural-001
initialization:
  mode: finetune
  base_model_id: vgcbench-mb-reference
  architecture: inherit
training:
  method: bc
  device: auto
  batch_size: 32
  num_epochs: 3
  learning_rate: 0.0001
  evaluation_battles_per_epoch: 0
```

Les trois époques, batch 32, seed 42 et learning rate `1e-4` constituent des défauts techniques de démonstration, configurables et enregistrés. Pas de promesse d’amélioration du niveau de jeu. Une architecture libre fournie en contradiction avec `inherit` est refusée ; aucun défaut `ModelConfig(128/2/4)` ne doit remplacer l’architecture héritée.

### 8.2 Chargement et preuve du point de départ

- Vérifier le hash source avant désérialisation et charger le conteneur SB3 avec la classe de policy compatible. Si les anciens métadonnées sont incomplètes, utiliser les valeurs historiques auditées de cette révision ; ne pas deviner le nombre de têtes depuis les seules formes de matrices.
- Avant toute passe avant ou mise à jour, vérifier l’égalité des paramètres/buffers chargés avec le checkpoint source et enregistrer l’architecture constatée. Une nouvelle instance créée comme conteneur est admissible seulement si **tous** les poids source y sont chargés strictement avant usage.
- Créer un optimiseur BC neuf pour ce fine-tuning et le documenter : réutiliser les poids ne signifie pas reprendre l’état d’un ancien optimiseur PPO ou BC.
- Transmettre `learning_rate` à l’optimiseur de `imitation.BC` et enregistrer la valeur effectivement utilisée. Garder le même optimiseur entre les époques de ce job.
- L’environnement servant aux espaces PPO utilise le format M-C vérifié du dataset, ou une construction d’espaces explicitement prouvée équivalente ; supprimer la sélection M-B codée en dur dans ce parcours.
- L’apprentissage est réalisé par `imitation.algorithms.bc.BC`, jamais `PPO.learn()`. Aucun reset des poids entre époques et aucun entraînement sur validation/test.

### 8.3 Métriques et validation sans mutation

Consommer toutes les transitions train, y compris un dernier batch incomplet ; traiter les trajectoires de longueurs différentes sans batch vide. Vérifier loss/gradients finis avant mise à jour et export.

Publier par époque : nombre réel de transitions, pas d’optimiseur, loss BC d’optimisation avec coefficients, `train_nll` et `validation_nll` comparables, durée, device, mémoire disponible et checkpoint. La NLL est la moyenne de la négative log-vraisemblance des actions humaines, pondérée par transitions. Si les anciens champs `train_loss`/`validation_loss` sont conservés, documenter leur équivalence à la NLL ; ne pas les confondre avec un objectif BC incluant entropie ou régularisation.

L’évaluation utilise une copie indépendante de la policy, reconstruite avec son architecture réelle puis chargée strictement ; cela réutilise la solution actuelle si `deepcopy` n’est pas possible. Déplacer la copie sur le device choisi, la passer en `eval()`, fixer `actor_grad=False`, désactiver les gradients et utiliser `evaluate_actions()` sous `no_grad()`.

Les embeddings `max_norm` peuvent renormaliser la copie pendant une passe avant ; ils ne doivent pas modifier l’original. Ne pas retirer globalement `max_norm`. Vérifier l’égalité exacte du `state_dict`, des gradients existants, des modes et de l’état d’optimiseur de l’original avant/après évaluation, y compris en cas d’erreur. Ne jamais sauvegarder la copie d’évaluation.

Mesurer le changement des poids après **de vrais pas BC**, notamment sur un paramètre de l’acteur qui n’est pas un embedding renormalisé ; une différence due uniquement à `max_norm` n’est pas une preuve d’apprentissage. Le fichier source reste inchangé et ses paramètres finaux ne sont pas écrasés.

### 8.4 Test final réservé

Le checkpoint final retenu pour cette démonstration est celui de la troisième époque, sans sélection à partir du test. Évaluer la NLL sur le split `test` seulement à la fin du job, une fois le checkpoint choisi. Utiliser le même mécanisme isolé que pour la validation et publier un rapport lié au hash exact du modèle/dataset/split.

La baisse de loss et la victoire en combat ne sont pas des critères obligatoires ; les mises à jour effectives, métriques finies et compatibilité le sont. Ne pas lancer une recherche de paramètres guidée par le test sous couvert de ce smoke test.

## 9. API, jobs et erreurs

Maintenir FastAPI sur `127.0.0.1`, Uvicorn avec un seul worker HTTP, `/health` et `/docs`. Les routes suivantes décrivent le contrat cible ; conserver les routes existantes compatibles lorsqu’elles ne contredisent pas l’immuabilité.

| Méthode et route | Fonction |
| --- | --- |
| `POST /collection/jobs` | Lancer manuellement une collecte, répondre `202` avec ID et configuration retenue. |
| `GET /collection/jobs/{id}` | État, compteurs, formats, reprise, erreurs et raison d’arrêt. |
| `GET /collection/jobs/{id}/logs` | Logs bornés/paginés. |
| `GET /archive/replays` | Inventaire paginé et filtrable par format/date ; métadonnées et versions. |
| `POST /archive/snapshots` | Figer une sélection de l’archive, renvoyer son ID/hash. |
| `GET /archive/snapshots/{id}` | Décrire la source figée et ses limites. |
| `POST /datasets` | Créer un ID de dataset avec `snapshot_id` et recette validée. |
| `POST /datasets/{id}/prepare` | Lancer la préparation en worker, répondre `202` avec job et dataset IDs. |
| `GET /datasets/{id}` | État, config, statistiques, splits, limites BO3 et référence de job. |
| `GET /datasets/{id}/rejections` | Décisions/rejets paginés avec filtres par étape et motif. |
| `GET /datasets/{id}/logs` | Logs du job de préparation associé. |
| `POST /datasets/import` | Importer un JSON local historique via l’adaptateur d’archive, renvoyer snapshot et dataset IDs. |
| `POST /training/jobs` | Lancer le fine-tuning avec `dataset_id` et `base_model_id`, réponse `202`. |
| `GET /training/jobs/{id}` et `/logs` | État, parent M-B, métriques, erreurs, checkpoints. |
| `GET /models`, `/models/{id}`, `/models/{id}/download` | Lister, inspecter et télécharger les modèles publiés. |

Préparation : le manifeste dataset expose l’ID/état du job ; inutile de créer un autre système de suivi. L’API de création des jobs accepte des schémas Pydantic stricts ; `404` pour ID absent, `409` pour ressource occupée/dataset déjà publié, `422` pour configuration ou contrat incompatible.

États de jobs : `queued`, `running`, `completed`, `failed`, `interrupted`. États dataset : `created`, `preparing`, `ready`, `insufficient_data`, `failed`, `interrupted`. Un job de préparation peut finir `completed` avec un dataset `insufficient_data` : le traitement a terminé, mais ce résultat n’est pas entraînable. Exposer cette distinction, ne pas afficher un succès de préparation de données prêtes.

Le verrou doit être interprocessus, acquis atomiquement et lié à l’identité réelle du worker. Au redémarrage de l’API, vérifier PID et identité/heure de démarrage : un worker vivant n’est pas déclaré interrompu et son verrou n’est pas libéré. Écrire les états de succès/échec avant de propager une exception, conserver traceback et journaux, et traiter une erreur de création du sous-processus.

Rediriger stdout/stderr vers des fichiers ou les drainer ; un `PIPE` non lu peut bloquer le calcul. Une requête de consultation ne charge pas l’ensemble d’un énorme log en mémoire. Les jobs avec IDs inconnus ou chemins sortant du stockage sont refusés.

Conserver les points d’entrée CLI, pilotés par `devbox run cli -- ...`, pour collecte, snapshot, préparation, enregistrement du modèle de base, entraînement et export. CLI et HTTP appellent les mêmes services ; documenter les commandes exactes une fois implémentées. Le lancement via HTTP reste obligatoire dans la démonstration finale.

## 10. Export et utilisation dans PokeCoach

Publier un package contenant `model.zip` SB3, `manifest.json`, métriques par époque, rapport de test final et README de chargement. Publication seulement après un rechargement CPU réussi dans un nouveau processus, depuis les fichiers exportés et les imports réellement disponibles.

Le manifeste versionné conserve les champs déjà consommés par PokeCoach (`architecture`, `checkpoint_sha256`, `contract_fingerprint`) et ajoute :

- `training_mode=bc_finetune`, identité/source/hash du checkpoint parent ;
- architecture effective héritée et nombre de paramètres ;
- `target_regulation=M-C`, formats source BO1/BO3 et format exact du combat de validation ;
- dataset ID/hash, snapshot ID/hash, recette et `split_id` ;
- contrat d’encodage, commits/révisions des forks/Showdown/Trainer et hashes des patches ;
- hyperparamètres effectifs, seed, device, métriques et preuve de mise à jour des poids ;
- statuts séparés de reload Trainer, test réservé et preuve PokeCoach ;
- limites déclarées, notamment les séries BO3 non identifiées.

Utiliser le profil existant `vgc_model.yaml` via `VGC_MODEL_PROFILE`, de préférence avec `trainer_package_dir`. Lire l’architecture depuis le manifeste, pas depuis une taille Tiny codée en dur. Propager et vérifier la régulation/les formats du modèle au runtime PokeCoach ; un profil dont le format contredit le matchup est refusé explicitement.

La preuve finale comporte :

1. deux équipes acceptées en M-C par le moteur figé ;
2. chargement effectif du modèle fine-tuné, identité/hash tracés ;
3. sélection réelle du quartet par la policy à la preview, puis tours normaux, switches et fin de partie ;
4. zéro ordre rejeté et aucun repli silencieux vers une fixture, une autre policy ou une action réparée après erreur ;
5. traces persistées par le parcours PokeCoach et consultables dans son affichage existant ;
6. redémarrage des processus PokeCoach/bridge puis nouveau combat complet avec le même checkpoint exporté ;
7. rapport lisible et exploitable par machine, avec IDs de parties, hashes et verdict.

Si un premier combat ne contient pas de switch, ajouter un scénario compatible pour couvrir ce cas en conservant tous les essais et erreurs. Un timeout n’est jamais une preuve de partie terminée. Le modèle peut perdre et rester compatible ; aucun objectif de win rate n’est imposé.

## 11. Tests d’acceptation et preuves

| Domaine | Cas obligatoires |
| --- | --- |
| Collecte brute | Replay sans OTS, à faible Elo, avec abandon ou espèce non convertible conservé dans l’archive ; aucun filtrage métier dans le collecteur. |
| Reprise | Relance sans retéléchargement des objets connus ; crash après découverte d’une page ; téléchargement échoué remis en attente ; page vide/répétée, même timestamp, 429 et timeout. |
| Sources | Snapshot inchangé après une nouvelle collecte ; conflit de contenu sous le même ID signalé ; import JSON historique avec clés contradictoires rejeté. |
| Filtres | 1 300/1 000 → une seule perspective ; Elo inconnu ; borne 1 100 incluse ; gagnant/perdant ; OTS manquant d’un seul côté ; équipe incomplète/ambiguë. |
| Fins de partie | Abandon après deux tours terminés rejeté, après trois accepté ; entrée dans le tour 3 insuffisante ; inactivité rejetée ; victoire normale ; égalité/fin inconnue avec motif. |
| BO3 | Première manche acceptée malgré abandon/rejet de la deuxième ; série incomplète acceptée ; série inconnue acceptée et comptée ; liens connus groupés ; métadonnées conservées. |
| Provenance | Conserver les tests existants d’identité et de split ; compléter avec workers terminant dans l’ordre inverse, replay produisant deux, une ou zéro perspective, erreur intermédiaire sans décaler les IDs des suivants et décisions détaillées de rejet/erreur. |
| Splits | Aucune intersection réelle des IDs/aliases/séries confirmées entre les trois dossiers ; répétition identique à snapshot/recette/seed identiques ; ratios effectifs et dépassement de cible explicités. |
| Préparation | Conserver le test acquis du premier appel sans `force` ; ajouter dataset publié non écrasable, moins de 300 exploitables → `insufficient_data` et absence de publication partielle après erreur. |
| Masques | Indice négatif/hors bornes refusé ; cible interdite détectée ; seconde cible interdite après la première détectée ; erreur bloquante traçable. |
| Fine-tuning | Poids initiaux égaux au checkpoint source ; architecture héritée ; config contradictoire refusée ; learning rate effectif ; changement réel de l’acteur par optimiseur ; source intacte. |
| Petits datasets | Trajectoires de longueurs variables, batch incomplet, dernière mise à jour avant validation/sauvegarde ; aucun exemple perdu silencieusement. |
| Évaluation | Aucun poids/gradient/mode/état d’optimiseur original modifié par validation/test, même avec `max_norm` ou erreur ; copie sur le bon device ; pas d’accès au test pendant les époques. |
| Contrat M-C | Mauvais format, tables permutées ou architecture différente refusés ; aucun fallback M-B ; parité d’encodage/mapping entre Trainer et bridge. |
| Jobs/API | `202` rapide, `/health` disponible pendant le job ; second job refusé ; logs volumineux sans blocage ; erreur sauvegardée ; worker vivant conservé après restart API ; PID réutilisé détecté. |
| Artefacts | ZIP tronqué, hash faux ou reload en échec non publiés ; architecture/poids conservés en nouveau processus CPU. |
| PokeCoach | Deux combats réels M-C séparés par redémarrage, même checkpoint, preview effective, résultat terminal, ordres acceptés et traces persistées. |

Les fixtures unitaires accélèrent les contrôles ; elles ne remplacent pas les preuves sur replays M-C, fine-tuning réel et combats. Un test ignoré faute de dépendances n’est pas un succès d’acceptation.

## 12. Livrables et ordre de réalisation

Ordre recommandé, chaque lot se terminant par une revue humaine :

1. **Audit M-C et modèle source** : établir formats, moteur, mappings, architecture et provenance ; signaler tout blocage de contrat avant un entraînement.
2. **Collecte et archive** : worker manuel FastAPI, reprises, stockage brut, inventaire et snapshots.
3. **Builder fiable** : s’appuyer sur la préparation sans `force` et la provenance déjà corrigées ; enrichir les décisions de conversion, appliquer les filtres, conserver BO3 et publier un dataset `natural` immuable avec trois splits et rapport.
4. **Fine-tuning réel** : charger M-B, appliquer BC sur M-C, prouver les mises à jour, validation isolée et test final réservé.
5. **Export et PokeCoach** : manifeste complet, reload, profil unique et deux exécutions M-C réelles.

Livrables : code commenté, tests, archive locale reproductible avec provenance, dataset/version/recette, checkpoint exporté, recette CLI/HTTP, profil PokeCoach d’exemple et rapport final.

Le rapport final distingue `PASSED`, `FAILED` et `BLOCKED`, avec les preuves de chaque critère. Il indique les limites de couverture de collecte, le nombre réel de replays/perspectives/transitions, les rejets, les BO3 sans lien fiable, les ratios effectifs, les versions, les hashes parent/enfant, le device, les métriques et les traces des combats. Si la cible a été réduite explicitement, le résultat technique et l’écart à 300 sont présentés séparément.

**Fin de V1 :** archive brute réutilisable, dataset `natural` fiable selon les limites déclarées, petit fine-tuning BC hérité du checkpoint M-B et modèle effectivement utilisable en M-C dans PokeCoach. La variante `diverse` est reportée.
