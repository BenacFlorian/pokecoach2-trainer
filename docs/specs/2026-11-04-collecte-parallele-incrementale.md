# PokeCoach Trainer — Collecte parallèle et incrémentale des replays

**Date :** 4 novembre 2026.  
**Statut :** spécification validée par l'utilisateur le 4 novembre 2026 ; implémentation à réaliser.  
**Décisions utilisateur :** plusieurs jobs sur **une seule machine**, cible d'environ **deux millions de replays**, démarrage sur une archive vide et suppression des anciens replays. Aucune migration de l'ancien index ni reprise des anciens curseurs n'est demandée. Les valeurs de réglage restent des points de départ à mesurer.

## 1. Besoin et résultat attendu

> Je lance une collecte avec plusieurs workers. Ils se répartissent automatiquement les recherches et les téléchargements, alimentent la même archive, et ne refont pas chacun le même parcours. Je peux arrêter, reprendre et augmenter le parallélisme sans perdre le travail accompli. Après le rattrapage initial, je collecte seulement les nouveautés pour préparer les prochains fine-tunings.

L'objectif est de constituer une archive importante pour le modèle BC PokeCoach, puis de la maintenir suffisamment fraîche pour suivre l'évolution de la méta. L'utilisateur rapporte environ 20 000 replays collectés entre 18 h et 7 h, soit environ 1 540 replays/h. À débit constant, 700 000 replays représenteraient environ **19 jours**, et un million environ **27 jours**, depuis une archive vide. Ce sont des extrapolations du volume et de la durée rapportés, pas un benchmark du trainer.

Le chiffre de 700 000 attribué à VGC-Bench est ici un repère fourni par l'utilisateur, non une comparaison vérifiée à périmètre égal. La cible technique est une archive d'environ **deux millions de replays**, **si suffisamment de replays pertinents sont accessibles**. Le système ne promet ni un volume disponible en M-C, ni un modèle meilleur par le seul nombre de parties. Les volumes bruts, les replays exploitables et les décisions d'entraînement doivent rester distingués.

La fraîcheur et la couverture par format/date comptent autant que le cumul historique. Le chantier accélère la collecte ; les gains du modèle seront évalués séparément sur les datasets et métriques BC existants.

**Complément utilisateur : sélection facultative.** Une collecte basique n'applique **aucun filtre de date ni d'équipe**. L'utilisateur peut ajouter une période, une liste d'équipes fournie par fichier JSON, ou les deux. Un replay peut être téléchargé pour vérifier sa composition, mais un replay non retenu ne doit pas être enregistré dans la collecte lancée.

## 2. Constats sur le trainer actuel

Chemins ci-dessous relatifs à `pokecoach-trainer/` ; observations issues de la lecture du code, sans profilage d'une collecte réelle.

| Élément | État observé | Conséquence |
| --- | --- | --- |
| `collection_service.py::collect_replays` | Recherche une page, puis télécharge tous ses replays séquentiellement avant la suivante. | Découverte et téléchargement se bloquent mutuellement. |
| `schemas.py::CollectionConfig` et `configs/collection-mc.yaml` | Intervalle minimal d'une seconde, appliqué avant recherche et téléchargement. | La latence réseau et les écritures s'ajoutent à cette seconde ; plafond inférieur à 3 600 replays/h même sans échec. |
| `archive_service.py::_existing_versions` | Recharge entièrement `archive/index/replays.jsonl` pour chaque vérification ; plusieurs vérifications lors d'un nouvel archivage. | Coût croissant avec l'archive, pouvant produire un coût cumulé quadratique. À mesurer, mais à retirer du chemin courant. |
| `collection_service.py::_load_progress` | Curseurs et IDs en attente attachés au `job_id`. | Une nouvelle soumission ne reprend pas automatiquement le parcours d'une précédente collecte. |
| `job_service.py` / `jobs.py` | Slot global `.long_job.lock` commun aux jobs longs. | Le mécanisme de soumission ne fournit pas le pool de collecte demandé. |
| `archive_service.py` | Fichiers par replay/hash et verrou de publication existent ; les vérifications préalables ne constituent pas une réservation de téléchargement. | Partager un dossier ne suffit pas à empêcher deux workers de télécharger le même ID. |
| `collection_client.py` | Pagination par `before`, avec détection de répétition dans le service ; bornes de dates filtrées après téléchargement. | Le partitionnement doit réellement borner la découverte, sans inventer une API de recherche par intervalle. |
| Cycle de vie des jobs | Réconciliation marquant les jobs actifs interrompus sans vérification de leur activité ; erreur de collecte levée avant la sauvegarde finale de l'état. | La supervision et la persistance des échecs doivent être corrigées pour gérer plusieurs processus. |

Le YAML ouvert limite aussi un lancement à 30 000 nouveautés, 60 000 requêtes ou 28 800 secondes (8 h). Vérifier la configuration effective et la raison d'arrêt dans les jobs de la collecte nocturne avant d'attribuer les 13 heures à ce fichier précis.

Cette spec remplace, pour la collecte, les décisions « un seul téléchargement », « index JSONL opérationnel » et « un seul job long global » de la [spec précédente](2026-11-03-collecte-datasets-finetuning-mc.md). Elle conserve l'archive brute, la séparation collecte/dataset/entraînement et les snapshots immuables.

## 3. Approches et choix retenus

| Approche | Intérêt | Limite |
| --- | --- | --- |
| Lancer plusieurs collecteurs actuels avec des dates différentes | Peu de modifications apparentes. | Répartition manuelle, reprise fragile, index coûteux et concurrence non coordonnée ; insuffisant. |
| Campagne commune, tâches persistantes, pool local et index SQLite | Partage atomique du travail, reprise et déduplication sans serveur supplémentaire. | Demande un nouvel index et une supervision explicite. **Approche retenue, sur archive vide.** |
| File distribuée et base serveur | Permet plusieurs machines. | Infrastructure supplémentaire sans besoin utilisateur dans cette version. |

Une **campagne** représente la collecte demandée ; elle possède plusieurs **workers**, qui sont les jobs de travail parallèles. L'utilisateur ne découpe pas les replays à la main. Une commande démarre et supervise l'ensemble. Tous les workers utilisent le même `TRAINER_STORAGE_ROOT`.

V1 : une campagne active par archive, avec plusieurs workers. Un deuxième `start` ayant la même configuration normalisée retourne la campagne active sans la dupliquer ; une configuration différente retourne un conflit explicite. Le nombre de workers se modifie par une opération dédiée. Cela satisfait le besoin de jobs coopérants sans introduire plusieurs campagnes concurrentes aux budgets ambigus.

Périmètre : collecte publique Showdown, formats déjà supportés, une machine et disque local, CLI/FastAPI existantes. Pas de cluster, Redis, nouvelle interface Angular, planificateur automatique, nouvelle architecture BC ni changement des filtres métier du Dataset Builder. La collecte continue pendant une préparation ou un entraînement sur données figées.

## 4. Organisation du travail

```text
CLI / FastAPI → campagne persistante et superviseur local
                         |
             plages de recherche exclusives
                         |
                 workers de découverte
                         |
          file partagée, une tâche par replay_id
                         |
          workers de téléchargement et sélection
                         |
        objets bruts communs + index transactionnel
                         |
             snapshots → datasets → BC
```

### 4.0 Filtres facultatifs et contenu de la collecte

| Configuration | Replays retenus |
| --- | --- |
| `format_ids: [mc]` seulement | Tous les replays accessibles de ces formats, sans sélection par période ni composition. |
| Dates seulement | Replays dans la période demandée, toutes équipes acceptées. |
| Fichier d'équipes seulement | Replays où l'une des deux équipes correspond, sans restriction temporelle utilisateur. |
| Dates et fichier d'équipes | Replays satisfaisant **la période ET une des équipes** configurées. |

- `upload_after` et `upload_before` sont optionnels, indépendants, et portent sur la date d'upload. Une borne absente ne crée pas une fenêtre implicite de 7/30 jours. Les fenêtres de sept jours servent seulement à répartir le parcours. La borne haute technique figée au lancement donne un parcours fini, pas un filtre utilisateur sur les dates.
- Mode par défaut : `backfill`. Le mode `incremental`, limité aux nouveautés et à son recouvrement, doit être demandé explicitement ; une collecte basique ne devient pas automatiquement une collecte des dernières 48 h.
- `teams_file` est une option du chargement YAML/CLI, résolue relativement au fichier de configuration. Le JSON contient `{"teams": [["id1", "id2", "id3", "id4", "id5", "id6"]]}` avec une ou plusieurs compositions. Il est lu/validé une fois au démarrage, puis son contenu normalisé et son hash sont persistés. Modifier ce fichier ne change pas une campagne en cours.
- Sans fichier ni `team_filter` explicite, le filtre d'équipe est désactivé : aucune extraction réussie des six Pokémon n'est alors requise pour archiver. Une liste fournie vide ou mal formée est une erreur, jamais un filtre silencieusement désactivé. L'API reçoit le contenu dans `team_filter: {teams: [...]}` ; elle n'ouvre pas de chemin arbitraire transmis par HTTP.
- Règle proposée pour les compositions : six identités d'espèces/formes initiales, comparées sans ordre, sans sets, surnoms, sexe, niveau ni shiny. Utiliser les IDs Showdown normalisés ; distinguer les formes initiales ayant des IDs différents. Ne pas réunir les quatre Pokémon joués ou des Pokémon des deux côtés pour inventer une équipe de six. Une seule équipe connue correspondante suffit ; sinon une composition inconnue donne `team_unknown`, distinct de `team_mismatch`.
- Après téléchargement/validation, appliquer les filtres **avant toute publication**. Un rejet ne laisse ni objet brut, ni copie du log dans les journaux, ni référence dans la sélection de la campagne. Conserver au plus ID, date, compositions connues, hash/version d'extraction et décision technique. Sans filtre d'équipe, un replay valide est conservé même si sa composition reste inconnue.
- Séparer le stockage commun et l'appartenance à une collecte. Un objet déjà archivé par une autre campagne reste intact ; il ne rejoint la collecte courante que s'il correspond à ses filtres. Un replay retenu déjà présent est réutilisé sans GET. Les listes et snapshots d'une collecte utilisent ses seules références retenues.
- Les rejets dépendent des filtres et ne sont jamais définitifs pour toutes les campagnes. Une nouvelle liste d'équipes ou une collecte sans filtre réévalue les métadonnées ; si le replay devient admissible mais son contenu n'a pas été conservé, il est téléchargé à nouveau. Les compositions inconnues ne sont pas mises en cache comme une preuve de non-correspondance universelle.
- La couverture de découverte signifie « IDs parcourus », pas « tous les contenus archivés ». Persister les IDs découverts et leurs dates permet de les réévaluer par lots lors d'un changement de filtre, sans reparcourir inutilement les pages. Une nouvelle campagne ne doit pas sauter les IDs rejetés d'une ancienne uniquement parce que la période est marquée couverte.

### 4.1 Découverte répartie

- Découper la plage demandée en tâches `(format_source, début inclus, fin exclue)`, par exemple des fenêtres de sept jours UTC. Les bornes logiques sont converties vers les formats source existants.
- Normaliser les fenêtres sur une grille UTC stable ; tronquer seulement les extrémités de la plage demandée. Enregistrer leur couverture dans l'archive, indépendamment du job qui les a parcourues.
- Figer `upload_before` à l'instant du lancement si la borne n'est pas fournie. Sans `upload_after`, traiter les fenêtres depuis cette borne vers le passé, avec un nombre borné de fenêtres créées à l'avance. Une fenêtre vide ne prouve pas que les périodes précédentes sont vides : poursuivre jusqu'à épuisement historique vérifiable par une recherche non bornée vers le passé, ou jusqu'au budget. En l'absence de preuve, la couverture reste partielle.
- Un worker réserve une fenêtre, initialise la recherche à sa borne supérieure, parcourt les pages vers le passé et s'arrête à sa borne inférieure. Il ne télécharge pas les replays hors fenêtre lorsque leurs métadonnées de recherche suffisent à les exclure.
- Le contrat de recherche connu utilise `before`, fondé sur `uploadtime`. La documentation décrit des pages avec chevauchement ; ne pas supposer un paramètre `after`, des offsets fiables, ni des IDs numériques continus. Vérifier les bornes et l'ordre effectifs lors de l'implémentation avec des tests de contrat et une vérification réseau limitée. [API officielle Showdown](https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md#replay-search).
- Enregistrer **dans une même transaction** les IDs de la page, leur rattachement à la campagne et le curseur suivant. Un crash ne doit jamais avancer le curseur sans conserver les IDs découverts.
- Une contrainte unique sur `replay_id` empêche les doublons de file, même aux frontières des pages/fenêtres. Les IDs déjà archivés sont comptabilisés sans nouveau téléchargement.
- Une page répétée ou une pagination bloquée sur plusieurs replays de même timestamp ne doit pas provoquer un saut silencieux. Utiliser uniquement une continuation vérifiée ; sinon conserver les IDs obtenus, déclarer la plage `coverage_incomplete` et exposer la borne bloquée.
- Une fenêtre n'est terminée que lorsque sa découverte est terminée, pas lorsque son budget est épuisé. Le téléchargement des IDs découverts peut se poursuivre indépendamment.
- Borner le stock de tâches en attente : la découverte ralentit lorsque la file atteint son seuil configuré et conserve son curseur. Ne pas charger tous les IDs en mémoire.

Un chevauchement limité des réponses fournisseur est normal. L'exigence est l'absence de parcours complets identiques attribués à plusieurs workers et l'absence de téléchargements redondants en fonctionnement normal.

### 4.2 Téléchargements exclusifs et reprise

Chaque tâche replay suit les états `pending → leased → archived`, avec sorties `retry_wait`, `unavailable`, `failed` et `examined_only` (contenu examiné mais non conservé). La décision `selected`/`team_mismatch`/`team_unknown`/`outside_period` appartient à la campagne ; `examined_only` n'est pas une interdiction globale de retéléchargement. La réservation est transactionnelle et contient `worker_id`, un jeton de réservation, une expiration, le nombre de tentatives et la date de prochaine tentative.

Un seul worker possède une réservation valide pour un ID. Il la renouvelle pendant son activité, y compris pendant un délai réseau ; il ne garde aucune transaction SQL ouverte pendant HTTP. À expiration, la tâche redevient réservable. Tout acquittement/publication vérifie le jeton courant pour empêcher un ancien worker de valider le travail après réattribution. La supervision utilise identité de processus et heartbeat, pas le PID seul.

La garantie est **un archivage logique unique et idempotent**, avec traitement pouvant être répété après panne. Un timeout ou un crash après réception peut imposer un second GET : une garantie absolue de « jamais deux requêtes réseau » serait incompatible avec la reprise. Ces répétitions doivent être mesurées, non confondues avec une mauvaise répartition.

Les erreurs temporaires passent en `retry_wait` avec délai croissant et jitter. Les erreurs définitives (par exemple replay supprimé) sont tracées et terminent la tâche. Une erreur de payload/format est distincte d'une indisponibilité réseau. Le plafond de tentatives est durable et ne se réinitialise pas à chaque réservation ; une relance explicite des échecs reste possible et auditée.

### 4.3 Stockage commun et publication

Conserver `archive/objects/<replay-id>/<content-sha256>.json` et les enveloppes existantes. Introduire `archive/catalog.sqlite3` pour :

- les campagnes, tentatives d'exécution et workers ;
- les fenêtres, curseurs et couvertures persistantes ;
- les tâches de téléchargement, réservations et rattachements aux campagnes ;
- les replays et leurs versions, avec unicité `(replay_id, content_sha256)` ;
- les budgets, paramètres de débit et compteurs nécessaires à la coordination.

SQLite devient l'index opérationnel. `index/replays.jsonl` devient un export de compatibilité reconstructible, jamais une deuxième vérité mise à jour en parallèle. Adapter les lecteurs actuels (liste, snapshots, présence, import, reconstruction) dans le même chantier.

Choix retenu : WAL, connexions propres aux processus, transactions courtes, délai d'attente borné sur contention et index SQL adaptés aux réservations/recherches. WAL autorise la concurrence lecteurs/écrivain mais conserve un seul écrivain à la fois et exige le même hôte avec stockage local. Utiliser une version SQLite intégrant le correctif WAL-reset documenté (3.51.3 ou ultérieure, ou backport officiel corrigé). [Documentation SQLite](https://www.sqlite.org/wal.html).

Protocole de publication :

1. Télécharger et valider, écrire un temporaire unique à proximité de la destination, calculer les hashes et synchroniser le contenu.
2. Vérifier la réservation et publier le fichier complet sans écraser une version existante, dans une courte section de publication protégée ; rendre le renommage durable, y compris le répertoire si nécessaire.
3. Insérer la version et passer la tâche à `archived` dans une transaction. Incrémenter les compteurs uniquement si la première publication logique du replay a effectivement été créée.

Le fichier et SQLite ne forment pas une transaction unique. Prévoir une intention durable de publication et une réconciliation ciblée : après un crash entre les étapes 2 et 3, retrouver le fichier valide sans le retélécharger ; ne jamais exposer une ligne `archived` pointant vers un fichier incomplet ou absent. Les temporaires isolés sont nettoyables. Le contrôle du jeton et la publication sont sérialisés avec une réattribution concurrente.

Même ID et même hash : opération sans effet. Même ID avec contenu différent lors d'un import ou d'une vérification explicite : garder les versions et signaler le conflit, sans écraser ni compter deux replays uniques. Conserver la sélection explicite de version exigée par les snapshots. La collecte ordinaire ne retélécharge pas les IDs déjà archivés.

## 5. Débit, budgets et performance

Le nombre de workers masque la latence ; le débit global détermine le nombre maximal de requêtes. **Huit workers avec un plafond global d'une requête/s ne permettent pas huit replays/s.** Le plafond doit pouvoir être configuré au-delà de la limite actuelle, puis ajusté d'après les mesures et réponses du fournisseur.

- Un limiteur partagé par hôte couvre recherche, téléchargement et chaque nouvelle tentative, quel que soit le worker. Pas de `sleep(1)` indépendant multiplié par le nombre de processus.
- Un `429` déclenche une pause commune respectant `Retry-After` (secondes ou date HTTP), avec délai de repli si invalide ; reprendre progressivement sous le plafond configuré. Persister la pause pour qu'un redémarrage ne la contourne pas.
- Les budgets appartiennent à la campagne/tentative de reprise, jamais à chaque worker. Le plafond `max_selected_replays` compte les IDs nouvellement **retenus par la campagne**, y compris s'ils existent déjà dans l'archive ; les rejets ne le consomment pas. `max_new_replays` reste accepté comme ancien nom de configuration, normalisé vers ce plafond ; documenter cette clarification. Réserver atomiquement les autorisations de requête et les places de sélection avant de les consommer, rendre la place en cas de rejet. Aucun dépassement de `max_requests` ni de `max_selected_replays` par multiplication du parallélisme. Le budget réseau inclut tous les téléchargements examinés et rejetés. Afficher séparément retenus, nouvellement archivés et réutilisés.
- À la durée limite, ne plus lancer de requête ; permettre uniquement la finalisation des opérations déjà engagées dans un délai de grâce borné. Exposer ce dépassement de durée de finalisation.
- Une reprise conserve tâches, curseurs, historique des tentatives et compteurs cumulés. Elle ouvre explicitement une nouvelle tentative d'exécution avec ses budgets ; afficher à la fois les compteurs de tentative et de campagne. Une pause/reprise simple dans la même tentative ne remet aucun budget à zéro.
- Une fin de budget signifie `budget_reached`, jamais « tout l'historique est collecté ».

Instrumenter séparément attente du limiteur, HTTP, index, publication, découverte et file vide. Fixer un objectif de débit réseau absolu seulement après une mesure contrôlée ; aucune promesse « un million en une nuit ».

## 6. Historique et mises à jour de la méta

Deux modes exposés :

- **`backfill`** : parcourir l'historique demandé et reprendre les fenêtres non terminées. Un nouveau lancement réutilise les couvertures persistantes compatibles ; il ne recommence pas depuis le replay le plus récent pour rejoindre son ancien curseur.
- **`incremental`** : rechercher les nouveautés jusqu'à une borne haute figée, avec recouvrement temporel configurable (proposition : 48 h) par rapport à la dernière couverture récente réussie. Le recouvrement est volontaire ; les objets déjà archivés ne sont pas téléchargés à nouveau ; les anciens rejets sont réévalués selon les filtres courants.

Maintenir deux informations distinctes : couverture de **découverte** et complétude des **téléchargements**. Une découverte terminée peut avancer son repère uniquement si tous ses IDs ont été persistés ; les téléchargements en attente restent récupérables. Une plage bloquée ne doit pas être cachée par un repère plus récent. Sans repère incrémental, démarrer sur les dernières 48 h et annoncer cette couverture limitée ; un backfill traite l'historique séparément.

Le recouvrement absorbe une partie des retards d'indexation sans garantir de détecter un replay devenu visible bien plus tard avec une ancienne date. Une commande de réinspection d'une plage historique reste possible, explicitement déclenchée et dédupliquée.

Une collecte conserve les données brutes **retenues par ses filtres facultatifs**, sans filtre Elo/OTS/résultat. Le Dataset Builder garde ces dernières responsabilités. Les snapshots figent une sélection cohérente de versions publiées dans une transaction de lecture courte, puis calculent les métadonnées hors transaction sur ces objets immuables. Un snapshot demandé pour une campagne se limite à ses replays retenus ; les replays d'autres campagnes présents dans l'archive n'y entrent pas implicitement. Un entraînement utilise son dataset figé pendant que l'archive continue de grandir.

## 7. Interface et configuration proposées

Conserver `collection start --config`, `collection status --job-id` et `POST /collection/jobs`. Le `job_id` public représente la campagne parente ; les workers et leurs journaux sont consultables séparément. Les routes de consultation existantes continuent de fonctionner.

Ajouter les opérations CLI et équivalents FastAPI :

| Opération | Comportement |
| --- | --- |
| `collection pause --job-id …` | Suspend les nouvelles réservations ; termine les publications engagées et préserve la file. |
| `collection resume --job-id …` | Reprend une pause sans réinitialiser les budgets ; après fin de budget, exige l'ouverture explicite d'une nouvelle tentative avec budgets enregistrés. |
| `collection scale --job-id … --download-workers 8 --discovery-workers 2` | Augmente/réduit le pool sans recréer les fenêtres ; les workers retirés rendent proprement leurs tâches. |
| `collection stop --job-id …` | Arrête proprement les workers ; conserve le travail inachevé pour reprise explicite. |
| `collection retry-failed --job-id …` | Réactive explicitement des tâches échouées, avec journalisation et nouveau plafond de tentatives. |

Les noms exacts des nouvelles routes et options secondaires sont laissés à l'implémenteur ; leurs sémantiques ci-dessus sont obligatoires. Les opérations sont idempotentes lorsque répétées et renvoient rapidement l'état persistant.

Exemple **futur**, non utilisable tel quel avec le schéma actuel ; chiffres proposés pour démarrer les mesures, pas quotas garantis par Showdown :

```yaml
collection:
  format_ids: [mc]
  mode: backfill
  discovery_workers: 2
  download_workers: 8
  partition_days: 7
  pending_high_watermark: 10000
  requests_per_second: 2.0
  max_selected_replays: 2000000
  max_requests: 3000000
  max_duration_s: 86400
  timeout_s: 30
  max_attempts: 5
  lease_duration_s: 120
  heartbeat_interval_s: 15
```

La limite de deux millions est un plafond, pas la quantité attendue en 24 h. À deux requêtes/s, le budget temps est susceptible d'être atteint bien avant, même sans erreur.

Ce YAML n'active ni filtre de date ni filtre d'équipe. Pour les activer, ajouter uniquement les options souhaitées :

```yaml
  upload_after: "2026-11-01T00:00:00Z"  # facultatif, inclus
  upload_before: "2026-11-04T00:00:00Z" # facultatif, exclu
  teams_file: "teams.json"              # facultatif, relatif au YAML
```

Les trois lignes sont des ajouts au bloc `collection`, pas un second bloc requis. Un JSON d'équipes vide n'est pas l'équivalent de l'absence de `teams_file`.

Les anciens YAML restent acceptés : absence des nouveaux champs = un worker de découverte, un de téléchargement, et conversion de `request_interval_s` en limite globale équivalente. Si `request_interval_s` et `requests_per_second` sont fournis ensemble, retourner une erreur de configuration explicite. Valider valeurs positives, limites de workers, cohérence heartbeat/réservation et dates UTC ; une date naïve est interprétée en UTC et cette valeur effective est affichée. Le passage aux fenêtres semi-ouvertes est documenté.

## 8. Supervision et indicateurs

Le superviseur détient un verrou de campagne transmis/possédé correctement par le processus qui en assure la durée de vie. Une API redémarrée retrouve le superviseur vivant ; elle ne marque pas aveuglément ses workers interrompus. Une identité de démarrage et des heartbeats permettent de distinguer PID réutilisé, processus vivant et tâche abandonnée. Les erreurs sont persistées avant propagation.

Le slot de collecte est indépendant des slots préparation/entraînement. La V1 conserve au plus un job de calcul long (préparation ou entraînement), parallèlement à la campagne de collecte. Les réparations structurelles restent exclusives aux écritures d'archive.

Afficher au minimum :

- workers configurés, actifs, en attente réseau et interrompus ;
- fenêtres en attente/en cours/terminées/bloquées et bornes de couverture par format ;
- IDs uniques découverts, déjà archivés, en attente, réservés, nouveaux archivés, indisponibles et échoués ;
- filtres actifs/inactifs, IDs examinés, retenus, réutilisés et écartés par motif ; un rejet de sélection n'est pas une erreur réseau ;
- requêtes de recherche/téléchargement/retry, réponses 429/5xx et pause globale ;
- débit de nouveaux replays sur fenêtre glissante, débit moyen et temps passé par étape ;
- budgets restants, raison d'arrêt, état de couverture et nombre de tâches restant à reprendre ;
- estimation du temps pour vider la file connue, distincte d'une estimation pour une cible arbitraire. Pas de « 90 % de tout Showdown » quand le volume total est inconnu.

Les compteurs proviennent des transitions persistées, pas de sommes de fichiers `state.json` écrits concurremment. Limiter la fréquence d'affichage et de flush des journaux ; rotation et lecture en fin de fichier pour éviter de relire un journal complet à chaque polling.

## 9. Critères d'acceptation

Tests déterministes avec serveur HTTP simulé et stockage temporaire ; campagnes réseau réelles uniquement pour mesurer le débit et vérifier le contrat fournisseur.

1. **Répartition** : huit workers et des fenêtres distinctes couvrent le même ensemble d'IDs qu'une exécution séquentielle ; aucune fenêtre n'a deux propriétaires valides. Rejouer un `start` identique ne crée pas un deuxième parcours.
2. **Déduplication** : injecter des IDs répétés dans plusieurs pages et aux frontières ; en l'absence d'erreur, un seul téléchargement et une seule publication par ID neuf. Les IDs déjà archivés ne déclenchent aucun GET replay.
3. **Dates et pagination** : vérifier bornes UTC, timestamp partagé, page répétée, fenêtre vide entre deux fenêtres non vides, et absence de continuation fiable. Aucun trou silencieux ni boucle infinie ; les limites de couverture sont exposées.
4. **Pannes** : interrompre un worker après découverte, après réservation, après publication du fichier et avant commit. La reprise ne perd aucun ID ; un worker au jeton périmé ne peut pas acquitter une nouvelle réservation.
5. **Budgets** : avec huit workers, tous les appels, retries compris, respectent le plafond global ; les budgets de replays/requêtes ne sont pas dépassés. Pause, reprise et nouvelle tentative conservent les compteurs attendus.
6. **429 et erreurs** : `Retry-After` en secondes/date et erreurs répétées déclenchent une régulation commune, sans tempête de retries ni blocage SQL pendant l'attente.
7. **Démarrage à neuf** : initialisation idempotente du catalogue vide, aucune importation automatique de l'ancien index/des anciens jobs, aucun effacement automatique au lancement. Les modèles et datasets préparés ne sont pas supprimés par l'initialisation.
8. **Incrémental** : sans filtre d'équipe, deuxième collecte sans nouveautés = zéro téléchargement ; apparition de dix IDs = dix nouveaux archivages, y compris dans le recouvrement. Les anciennes tâches en attente restent traitées.
9. **Coexistence** : un snapshot créé pendant la collecte est cohérent et immuable ; préparation/entraînement restent utilisables ; redémarrer FastAPI ne détruit pas une campagne vivante.
10. **Montée en volume** : sur un catalogue synthétique de 20 000, 100 000, un million puis deux millions d'IDs, aucune relecture intégrale de l'index par tâche. Mesurer débit, mémoire, contention SQL et coût disque ; la mémoire des workers reste bornée par les lots, pas par la taille totale de l'archive.
11. **Gain de parallélisme isolé** : sur un serveur local à latence fixe de 200 ms, 1 000 petits replays neufs et un limiteur non contraignant, huit workers atteignent au moins trois fois le débit d'un worker sur la même machine. Exécuter des essais séparés avec archive vierge ; publier la configuration et les résultats. Ce seuil teste le pipeline, pas les capacités de Showdown.
12. **Mesure réelle** : comparer l'ancien collecteur et la nouvelle version à un worker, puis 2/4/8 workers, sur des plages de volume comparable avec budget réseau borné. Rapporter replays uniques/h, erreurs, coût local, débit global et couverture. Retenir une configuration soutenable ; distinguer gain d'indexation, gain de concurrence et hausse éventuelle du débit autorisé.
13. **Options indépendantes** : tester les quatre combinaisons sans filtre/date/équipes/date+équipes ; absence de dates = aucun seuil historique implicite, absence d'équipes = aucune exclusion pour composition inconnue. JSON vide/invalide refusé ; reprise conservant la configuration figée malgré édition du fichier.
14. **Sélection avant publication** : une équipe correspondante côté p1 ou p2 suffit, ordre/sets ignorés, quatre Pokémon révélés insuffisants. Les rejets ne laissent aucun log brut ni objet publié et ne consomment pas le plafond de sélection ; tous les GET consomment le budget réseau. Une autre campagne peut sélectionner un ancien rejet ; une campagne sans filtre le télécharge si le contenu a été abandonné. Snapshot/listing par campagne exclut les objets non sélectionnés d'autres collectes.

## 10. Ordre de réalisation proposé

1. Initialisation du catalogue transactionnel vide et mesures du coût local.
2. File durable, réservations et publication idempotente.
3. Découverte partitionnée, reprise et limites de couverture.
4. Pool supervisé, limiteur/budgets globaux et commandes de contrôle.
5. Mode incrémental, coexistence avec les datasets et vérifications de charge.

L'archive repart de zéro par décision utilisateur. Chaque lot doit ensuite préserver les données produites par le nouveau système ; aucune purge automatique ne fait partie du fonctionnement normal. Le livrable de ce chantier sera une collecte dont les workers partagent réellement le travail, avec un débit mesurable et une reprise fiable ; l'entraînement continue à consommer des snapshots de cette archive.
