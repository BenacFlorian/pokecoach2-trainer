# Collecte filtrée par le profil dataset — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Lire le document entier avant de modifier le code. Ce document est un plan : il n'autorise pas à purger l'archive existante, modifier une campagne active ou lancer une collecte de production pendant le développement.

**Goal:** Ne retenir et ne publier, dans une nouvelle campagne filtrée, que des replays M-C qui satisfont le profil métier du dataset et possèdent au moins une perspective admissible.

**Architecture:** Partager le modèle de filtres, le parseur de métadonnées et les décisions métier entre collecte et dataset. Évaluer chaque candidat avant publication, conserver les raisons de rejet et les faits nécessaires à la reprise dans SQLite, puis compter une sélection uniquement lorsque sa version archivée est disponible. Construire le dataset à partir du snapshot de cette campagne.

**Tech Stack:** Python ≥3.13, Pydantic 2, SQLite, workers de collecte existants, httpx, YAML, FastAPI, pytest et uv. Aucune dépendance ML supplémentaire dans la collecte.

**Spec:** Le contrat fonctionnel des sections 1 à 5 formalise la demande du 5 novembre 2026. Les anciennes specs [collecte/datasets](../specs/2026-11-03-collecte-datasets-finetuning-mc.md) et [collecte parallèle](../specs/2026-11-04-collecte-parallele-incrementale.md) apportent le contexte ; la demande actuelle remplace leur principe de collecte sans filtres métier pour les campagnes qui activent ce nouveau profil.

## Contraintes globales

- Livrer le filtrage à la collecte, pas un nettoyage a posteriori présenté comme du filtrage.
- Conserver les anciens fichiers, objets, campagnes, snapshots et datasets. Ne pas transformer rétroactivement une campagne sans profil en campagne filtrée.
- Réutiliser les fonctions métier du dataset ; ne pas recopier leurs conditions dans un second moteur.
- Appliquer `exclude_inactivity_losses: true`, conformément à la demande, bien que le fichier actif `configs/smoke/dataset-mc.yaml` contienne actuellement `false`.
- `min_completed_turns_forfeit` vaut **2** dans le nouveau profil. Le défaut actuel de `DatasetRecipe`, **3**, ne doit pas être modifié implicitement pour tous les autres usages.
- Elo individuel : une perspective inconnue est exclue ; la moyenne du replay ne remplace jamais l'Elo individuel.
- Un replay accepté peut fournir une seule perspective. Ne pas exiger que les deux joueurs dépassent 1100.
- Aucun téléchargement, parsing de log ou accès fichier long sous une transaction SQLite d'écriture.
- Respecter les budgets réseau, les leases, la reprise et le débit global déjà présents. Les rejets consomment des requêtes, pas le quota de replays retenus.
- Tous les tests écrivent dans `tmp_path` avec HTTP simulé. Aucun test ne doit utiliser le stockage de production.
- Ne pas réimplémenter le prepare multicœur. Ne pas convertir les replays en trajectoires dans les workers de collecte.
- Préserver les changements préexistants. Si des commits sont réalisés, utiliser l'auteur `benacFlorian <florianbenac46@gmail.com>` ; ne jamais inclure `storage/` ni des changements étrangers au chantier.

## Points de revue

1. Elo pris dans le champ avatar, timer pris pour une défaite ou OTS mal décodée : les tests doivent reproduire le protocole Showdown réel — tâche 2.
2. Archive déjà remplie ou cache d'une ancienne version : aucun chemin de reprise ne sélectionne sans évaluation des bons faits et du bon profil — tâches 4 et 7.
3. Deux workers, dernier emplacement disponible, doublons ou crash après renommage : pas de dépassement, de double comptage ou de sélection sans objet — tâches 5 et 6.
4. Profil assoupli après un rejet : le rejet précédent ne bloque pas définitivement le replay ; aucun téléchargement en boucle — tâche 7.
5. Snapshot dépassant 1000 replays et replay comportant plusieurs versions : le dataset reçoit toutes les versions précisément validées — tâche 8.

---

## 1. Définition de « valide » et résultat attendu

### 1.1 Profil à appliquer

```yaml
min_elo: 1100
max_elo: null
unknown_elo: exclude_perspective
only_winners: false
require_ots: true
require_complete_teams: true
include_bo1: true
include_bo3: true
include_forfeits: true
min_completed_turns_forfeit: 2
exclude_inactivity_losses: true
deduplicate: true
```

Le contrat est : **100 % des replays sélectionnés satisfont ces filtres et contiennent au moins une perspective admissible**, pour la version de parseur/filtres enregistrée avec la campagne. Il ne signifie pas que 100 % des candidats trouvés sur Showdown seront acceptés.

Certaines informations nécessitent le log complet. Le chemin réseau actuel fournit essentiellement les identifiants, dates et formats lors de la recherche. Il faut donc encore télécharger des candidats invalides pour les examiner ; leurs corps ne sont pas publiés dans l'archive. Seuls les faits compacts et décisions nécessaires à la reprise sont conservés.

Il ne s'agit pas non plus d'une garantie que 100 % des replays seront convertibles en trajectoires : le convertisseur, les actions/masques et les contraintes de splits restent des contrôles ultérieurs. Une garantie de conversion nécessiterait un chantier supplémentaire et un coût CPU différent. Les rapports distingueront admissibilité et conversion.

### 1.2 Conditions d'acceptation

```text
format/payload autorisés
ET filtres collection existants (date et éventuel team_filter)
ET decide_game(metadata, filters).accepted
ET au moins un decide_perspective(metadata, side, filters).accepted
ET absence de doublon retenu dans cette campagne, si deduplicate=true
```

Exemples, en supposant les autres critères satisfaits :

| Situation | Replay retenu | Perspectives admissibles |
| --- | --- | --- |
| p1=1100, p2=1099 | oui | p1 |
| p1=1099, p2=1400 | oui | p2 |
| p1 inconnu, p2=1400 | oui | p2 |
| deux Elo inconnus, moyenne=1500 | non | aucune |
| deux Elo <1100 | non | aucune |
| p1 perd mais vaut 1300, p2=1000 | oui | p1, car only_winners=false |
| OTS présente pour un seul joueur | non | aucune |
| seulement cinq membres dans une équipe initiale | non | aucune |
| abandon après deux tours terminés | oui | selon Elo |
| abandon après un seul tour terminé | non | aucune |
| minuterie activée, puis victoire normale | oui | selon Elo |
| défaite effective par inactivité | non | aucune |
| égalité ou fin indéterminée | non | conserver les raisons du dataset |

`max_elo: null` signifie aucune borne supérieure. Une borne renseignée reste inclusive. Les options BO1/BO3 portent sur les parties : accepter les BO3 n'exige pas de collecter trois manches et ne permet pas de dédupliquer les trois manches comme si elles étaient une seule partie.

### 1.3 Archive existante

Les « 104k replays / 25 % valides » sont le constat utilisateur, pas une mesure refaite pendant la rédaction. Le catalogue `pokecoach2-trainer/storage/archive/catalog.sqlite3` accessible lors de cette lecture contient zéro version ; il ne permet pas de reproduire cette statistique. Ne pas choisir un autre stockage au hasard ni conclure que les données utilisateur ont disparu.

La nouvelle campagne peut réutiliser les objets déjà présents dans le **stockage effectivement configuré**, après validation locale, sans HTTP. Les anciens replays invalides restent dans l'archive historique mais sont absents de la sélection et du snapshot filtrés. Il faut utiliser le snapshot de campagne, pas un snapshot global mélangeant ancien et nouveau stock.

## 2. État réel du code et pièges constatés

Chemins ci-dessous relatifs à `pokecoach2-trainer/pokecoach-trainer/`, sauf mention contraire.

| Fichier / symbole | Constat et conséquence |
| --- | --- |
| `schemas.py::CollectionConfig` | Aucun profil dataset ; seulement dates, équipes et paramètres d'exécution. |
| `schemas.py::DatasetRecipe` | Contient les filtres demandés, avec un défaut de trois tours pour les abandons et deux anciens alias. |
| `dataset_filters.py::decide_game`, `decide_perspective` | Fonctions pures à partager. L'admissibilité d'un replay correspond déjà à au moins une perspective acceptée. |
| `replay_metadata.py` | Extraction attachée à un `ArchiveRef`, mais accepte une enveloppe `cached`. Plusieurs hypothèses de parsing sont incorrectes pour le protocole produit par Showdown local. |
| `collection_filters.py::evaluate_selection` | Ne gère que date/équipe ; `filter_fingerprint` ignore les critères dataset. |
| `collection_workers.py::_run_download_step` | Archive après sélection date/équipe. Le raccourci « déjà archivé » évalue avec `uploadtime=None, summary=None`. |
| `collection_store.py::_reevaluate_known_metadata` | Peut écrire directement une sélection sur la seule base du résumé date/équipe, avant le passage d'un worker. |
| `collection_store.py::record_selection` | Incrémente `selected_in_attempt` avant publication, sans garde suffisante contre les répétitions. |
| `collection_store.py::claim_download` | Pioche dans une file globale : il faut vérifier le rattachement/format et la décision de campagne avant de traiter une ancienne tâche. |
| `collection_publication.py::prepare_replay` | Construit identité et enveloppe puis écrit déjà en staging. Extraire la construction pure pour évaluer avant toute écriture du corps. |
| `collection_publication.py::publish_replay`, `recover_publications` | L'intention de publication existe ; la reprise doit aussi finaliser sélection, réservation et compteur de façon idempotente. |
| `archive_catalog.py` | Schéma version 1, migration additive à prévoir. `campaign_replays` possède déjà `version_content_sha256` et `first_selected_attempt`, actuellement sous-exploités. |
| `dataset_split.py::build_replay_groups` | Regroupe les logs identiques pour les splits, mais ne supprime pas les replays doublons. |
| `dataset_service.py::_prepare_snapshot_dataset` | Le drapeau `deduplicate` n'y pilote pas une élimination explicite des alias ; ce n'est pas une déduplication complète actuellement. |
| `archive_service.py::create_campaign_snapshot` | Demande jusqu'à 1 000 000 refs en un appel, mais `catalog_list_campaign_replays` borne chaque page à 1000 : pagination à corriger. |
| `collection_service.py::collect_replays` | Ancien collecteur encore appelable, explicitement sans filtres métier. Ne pas laisser un appel avec nouveau profil le contourner silencieusement. |

Préfixer les chemins `schemas.py`, etc. de cette table par `src/pokecoach_trainer/`.

### Sources locales du protocole à consulter avant le parseur

- `../../pokemon-showdown/sim/battle.ts` : `add('player', side.id, side.name, side.avatar, options.rating || '')`, `add('turn', ...)`, `add('showteam', side.id, Teams.pack(team))`.
- `../../pokemon-showdown/data/rulesets.ts` : Team Preview émet `poke` avec le côté `p1`/`p2`, pas nécessairement `p1a`…`p1f`.
- `../../pokemon-showdown/sim/teams.ts` : encodage packed team, membres séparés par `]`, champs séparés par `|`.
- `../../pokemon-showdown/server/room-battle.ts` : `|inactive|` sert aussi aux avertissements ; la perte effective est signalée par un message « lost due to inactivity. ».

Les fixtures actuelles dans `tests/replay_log_fixtures.py` ont notamment un Elo placé à la place de l'avatar, des slots `p1a`… et des tours terminés par `|`. Ne pas les utiliser comme unique preuve du protocole réel.

## 3. Architecture et configuration retenues

### 3.1 Choix d'architecture

Trois approches ont été considérées : recopier les conditions dans la collecte (rapide mais divergence garantie), appeler la conversion complète pendant la collecte (garantie plus forte mais hors besoin et coûteuse), ou partager extraction et filtres avant publication. **Ce plan recommande la troisième**, choix technique proposé ici à partir du code existant.

Créer `ReplayFilterConfig` dans `schemas.py`, contenant exactement les 12 champs de la section 1.1, avec les défauts actuels de `DatasetRecipe`. Faire hériter `DatasetRecipe` de cette classe, en préservant son format JSON plat et ses autres champs (`target_replays`, `seed`, `splits`, etc.). Garder la normalisation des anciens alias dans `DatasetRecipe`.

Ajouter `CollectionConfig.dataset_filters: ReplayFilterConfig | None = None`. `None` conserve le comportement historique sans filtrage métier. Une campagne activant ce champ applique tous ses paramètres ; aucune activation implicite lors de la lecture d'un ancien job.

Créer `replay_policy.py` pour les fonctions pures communes :

```python
FILTER_POLICY_VERSION = "replay_policy/1"

def filters_from_recipe(recipe: DatasetRecipe) -> ReplayFilterConfig: ...
def replay_filter_fingerprint(filters: ReplayFilterConfig) -> str: ...
def evaluate_replay_policy(
    metadata: ReplayMetadata,
    filters: ReplayFilterConfig,
) -> ReplayEligibility: ...

@dataclass(frozen=True, slots=True)
class ReplayEligibility:
    accepted: bool
    reason_code: str
    accepted_sides: tuple[Literal["p1", "p2"], ...]
    decisions: tuple[FilterDecision, ...]
```

Les fonctions `decide_game` et `decide_perspective` restent dans `dataset_filters.py`, mais acceptent `ReplayFilterConfig` ; `DatasetRecipe` reste compatible par héritage. `evaluate_replay_policy` les compose : échec partie → raison de cette partie ; partie acceptée mais zéro côté → `no_eligible_perspective`, avec les deux raisons détaillées ; sinon `replay_accepted`. Aucun doublon n'est décidé par cette fonction pure, car cela dépend de la campagne.

### 3.2 Une recette source, des valeurs figées dans la campagne

Étendre `_load_collection_config(path)` avec une clé YAML de chargement `dataset_recipe_file`, sur le modèle de `teams_file`. Résoudre le chemin **par rapport au YAML collection**, charger et valider `DatasetRecipe`, puis injecter `dataset_filters=filters_from_recipe(recipe)`.

- `dataset_recipe_file` et `dataset_filters` inline sont exclusifs ; erreur explicite s'ils coexistent.
- Le chemin n'est pas un champ runtime de `CollectionConfig` et n'est jamais rouvert par un worker/reprise.
- L'API reçoit les valeurs structurées `dataset_filters`, pas un chemin local fourni par le client.
- Les valeurs effectives sont sauvegardées dans la config de campagne. Modifier ensuite le fichier YAML ne modifie pas une campagne existante.
- Les empreintes des filtres incluent les 12 valeurs normalisées, la version de politique, la version du parseur, et les filtres date/équipe existants. Les limites réseau/workers n'entrent pas dans l'empreinte des décisions métier.
- La normalisation des alias a lieu avant extraction ; ne pas hacher `min_rating`, `only_winner`, seed, splits ou target comme filtres.

L'empreinte des filtres seuls produite par `replay_filter_fingerprint` inclut les valeurs et versions métier ; `collection_filters.filter_fingerprint` compose cette empreinte avec date/équipe. Figer également les versions parseur/policy dans la campagne à sa création, dans deux nouvelles colonnes nullable `campaigns.validation_parser_version` et `campaigns.validation_policy_version`. `None` signifie campagne historique sans garantie. Pour une campagne filtrée existante, un démarrage/reprise avec versions incompatibles doit échouer explicitement avant de lancer les workers ; créer une nouvelle campagne permet de revalider le stock avec le nouveau code.

Créer deux profils explicites, sans écraser le profil utilisé par une collecte en cours :

```yaml
# configs/dataset-mc-collectable.yaml
dataset:
  variant: natural
  regulation: M-C
  target_replays: 100000
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
  min_completed_turns_forfeit: 2
  exclude_inactivity_losses: true
  deduplicate: true
  max_replays_per_team: null
  splits:
    train: 0.80
    validation: 0.10
    test: 0.10
```

Créer `configs/collection-mc-100k-valid.yaml` en copiant les paramètres d'exécution de `collection-mc-100k.yaml` et en ajoutant :

```yaml
collection:
  # Autres paramètres de collection-mc-100k.yaml à conserver.
  dataset_recipe_file: dataset-mc-collectable.yaml
  max_selected_replays: 100000
```

Le fichier final doit être un YAML complet, pas ce fragment. À 25 % d'acceptation, atteindre 100k retenus demanderait approximativement 400k candidats, plus les requêtes de recherche/retry. **Ne pas augmenter automatiquement** `max_requests`/`max_duration_s` ni promettre la cible avec les budgets existants ; documenter le risque d'arrêt avant cible et la reprise avec nouveaux budgets. Le taux de 25 % peut changer après correction du parseur.

## 4. Extraction et cache : décisions détaillées

### 4.1 Identité commune avant publication

Extraire depuis `prepare_replay` une fonction pure dans le même module :

```python
def build_replay_envelope(
    replay_id: str, payload: dict[str, Any], source: ArchiveSource,
) -> tuple[ArchiveRef, dict[str, Any]]: ...
```

Elle conserve exactement la validation d'identifiant, le périmètre de format, `_PROVIDER_KEYS`, le hash de contenu canonique, le hash du log et le format d'enveloppe actuels. Aucun accès disque. `prepare_replay` la réutilise pour écrire en staging, sans changer les identités historiques.

La collecte peut alors appeler `extract_replay_metadata(ref, cached=envelope)` **avant publication**. Pas de faux hash, pas de `ArchiveRef` calculé différemment et pas de nouvelle dépendance à Torch/VGC-Bench pour cette extraction.

### 4.2 Parseur partagé corrigé

Passer la version publique à `REPLAY_METADATA_VERSION = "replay_metadata/2"` et utiliser cette constante dans l'extraction et les empreintes. Les snapshots historiques conservent leurs métadonnées ; ne pas les réécrire en place.

Règles à implémenter et tester :

1. `|player|p1|Alice|avatar|1450` : lire le rating après l'avatar. Un avatar numérique sans rating ne devient jamais un Elo. Une mise à jour de nom/avatar sans rating ne doit pas effacer un Elo précédemment observé. Ne pas deviner un Elo depuis la valeur d'avatar des anciennes fixtures.
2. `|poke|p1|Species, L50|` : lire les six membres du preview initial par côté, jusqu'au début du combat. Respecter un éventuel `clearpoke` qui réinitialise ce preview. Les changements/switches ultérieurs ne complètent pas une équipe initiale incomplète. Adapter aussi `extract_team_summary` à la même lecture pour éviter deux interprétations.
3. `showteam` : séparer uniquement le préfixe et le côté, conserver les `|` internes, décoder les six entrées packed séparées par `]`. Une entrée doit fournir une identité d'espèce exploitable (champ espèce ou nom selon le format packed). Six virgules dans une chaîne ne prouvent pas l'existence de six membres. OTS signifie feuille exploitable pour chacun des deux côtés ; ne pas imposer six espèces distinctes comme raccourci de parsing.
4. Accepter `|turn|2` sans séparateur final. Un tour est terminé s'il possède une preuve de clôture (`upkeep`, passage au tour suivant, ou fin normale après résolution). Un `move` isolé avant abandon ne suffit pas à compter le tour interrompu. Pour un abandon au début du tour 3 après deux tours clos : `completed_turns=2`. Si la preuve manque, ne pas inventer un compte suffisant.
5. Distinguer les messages de timer des pertes effectives. Ne jamais classer une partie en inactivité sur la seule présence de `|inactive|`.
6. Lire les événements de fin autorisés (`win`, `tie`, messages système de défaite/abandon), sans recherche libre de « forfeit » dans le chat. Conserver le gagnant indiqué par `|win|` même lorsqu'un message d'abandon le précède. Priorité : inactivité effective > abandon effectif > victoire normale, lorsque ces messages appartiennent à la fin de la même partie ; égalité explicite sans défaite reste `tie`.
7. `only_winners=true`, même hors profil cible : gagnant ou identité du côté inconnu → perspective rejetée avec `winner_unknown`, jamais acceptée par défaut. Conserver le comportement demandé lorsque `only_winners=false`.
8. Conserver la résolution BO existante sauf défaut démontré par fixture réelle ; ne pas inventer de liens BO3 à partir des seuls noms/dates.

Extraire les petits helpers de lecture preview/packed team dans un nouveau `replay_log.py` si nécessaire pour les partager entre `replay_metadata.py` et `collection_filters.py`. N'utiliser que la bibliothèque standard. Mettre les nouvelles fixtures dans `tests/fixtures/replays/` et leur description de provenance dans un README ; privilégier des fragments anonymisés issus du protocole local, plus un échantillon de vrais replays si le stockage configuré le permet.

### 4.3 Persistance et invalidation

Migration catalogue 1 → 2, additive, idempotente, sous verrou d'initialisation existant :

- Table `replay_metadata_cache` : `replay_id`, `content_sha256`, `parser_version`, `log_sha256`, `source_format`, `uploadtime`, `metadata_json`, clé primaire `(replay_id, content_sha256, parser_version)`.
- `campaign_replays` : ajouter `decision_json TEXT` (nullable pour l'historique) et `log_sha256 TEXT` ; exploiter `version_content_sha256` et `first_selected_attempt` existants.
- `campaigns` : ajouter les deux colonnes de versions de validation définies en section 3.2 ; les anciennes lignes restent nulles.
- Table `campaign_log_claims` : `campaign_id`, `log_sha256`, `replay_id`, `content_sha256`, `lease_token`, `state` (`pending`/`selected`), clé primaire `(campaign_id, log_sha256)`. Sert à réserver un contenu admissible sans course entre workers.
- Ne pas créer d'index unique sur les hashes de logs des anciens objets : ils peuvent légitimement contenir des doublons historiques.
- Réutiliser `publication_intents.metadata_json` pour sauvegarder le contexte de décision/finalisation avec l'intention, sans modifier le contenu canonique du provider.

Une décision n'est réutilisable que pour le même contenu, le même parseur et la même empreinte de filtres. `replay_summaries` reste utile au filtre équipe ; son hash actuel ne remplace pas le hash canonique du provider. `source_format` et `uploadtime` doivent être persistés avec leurs vraies valeurs, pas `""`/`None` lors de l'évaluation d'un payload disponible.

Un cache de faits ne contenant aucun objet archivé peut suffire à rejeter sans HTTP. S'il conduit à accepter, il faut récupérer le corps et **réévaluer le contenu obtenu** avant publication. Si le parseur change et que le corps n'existe pas localement, il faut refetch pour produire les faits du nouveau parseur.

## 5. Cycle transactionnel, déduplication et reprise

### 5.1 Déduplication exacte

- Même `replay_id` : un seul replay retenu par campagne ; la version retenue est épinglée par son hash.
- Deux IDs différents avec même `log_sha256` : un seul retenu si `deduplicate=true` ; raison `duplicate_log`, avec identifiant canonique dans les détails.
- Hash de log = algorithme `sha256_text` existant, sans normalisation approximative des équipes/joueurs/texte.
- Une même équipe ou deux manches BO3 différentes ne constituent pas des doublons.
- Appliquer les filtres **avant** la déduplication : un replay rejeté ne réserve jamais son hash et ne bloque pas un alias admissible.
- Périmètre de déduplication : sélection de la campagne. Un objet déjà archivé et admissible peut servir dans une nouvelle campagne ; la présence globale d'un hash ne doit pas empêcher la réutilisation du stock.
- En collecte parallèle, le premier contenu admissible finalisé gagne. Ce choix ne garantit pas le même ID représentant entre deux nouvelles campagnes ; les décisions enregistrées et le snapshot assurent la reproductibilité d'une campagne donnée.
- `deduplicate=false` conserve les différents IDs, mais ne désactive ni l'idempotence d'un même ID ni la gestion des versions.

### 5.2 État d'une sélection

Conserver la réservation de capacité prise par `claim_download` jusqu'au rejet ou à la finalisation. Ne pas incrémenter le quota à la simple décision métier.

```text
claim + réserve un emplacement
    → charge objet local OU fetch payload
    → construit enveloppe pure / extrait faits
    → décide date/équipe + métier
        → rejet : persiste décision/faits, libère emplacement
        → admissible : réserve éventuellement hash de log
            → publication (ou réutilisation de version locale)
            → finalise sélection + épingle version + compte une seule fois
```

Ajouter les interfaces suivantes à `CollectionStore` (types dans `collection_types.py`) :

```python
@dataclass(frozen=True, slots=True)
class CollectionEvaluation:
    selection: SelectionDecision
    metadata: ReplayMetadata | None
    eligibility: ReplayEligibility | None
    summary: TeamSummary | None

def record_evaluation(
    self, lease: TaskLease, ref: ArchiveRef, evaluation: CollectionEvaluation,
) -> Literal["rejected", "ready", "duplicate", "wait"]: ...

def finalize_selection(
    self, lease: TaskLease, ref: ArchiveRef, *, new_version: bool,
) -> bool: ...  # True seulement au premier passage vers selected.
```

`record_evaluation` valide le lease courant, persiste les faits et détails, et procède atomiquement à la réservation du hash si nécessaire. Une décision admissible en attente de publication porte `decision='pending_publication'`, jamais `selected`. Un doublon déjà finalisé devient `rejected/duplicate_log`. Un hash seulement réservé par un autre worker retourne `wait` : libérer le slot/tâche avec reprise différée, sans rejet définitif tant que la publication concurrente n'est pas résolue.

`finalize_selection` exige une version présente au catalogue et la réservation/identité correspondante ; elle épingle `version_content_sha256`, marque la sélection, fixe `first_selected_attempt` et libère la réservation. Mettre à jour sélection, hash réservé et compteurs dans **une seule transaction**. Un second appel renvoie False et ne modifie aucun compteur.

`selected_in_attempt` compte les premières sélections réussies de cet attempt ; `selected_replays` compte les replays retenus de la campagne ; `new_published` compte les nouvelles versions physiquement publiées. Une réutilisation locale peut accroître les deux premiers sans accroître le dernier. Une reprise avec nouvel attempt ne recompte pas une sélection déjà finalisée dans la même campagne.

En cas de crash entre publication et finalisation, `recover_publications` termine les deux opérations. Pour réconcilier un ancien lease expiré, utiliser une fonction interne dédiée à la récupération de l'intention persistée ; ne pas désactiver la validation des leases sur le chemin worker normal. Une intention dont l'objet final existe reste propriétaire de sa réservation de hash jusqu'à récupération. Libérer une réservation abandonnée uniquement après avoir exclu une publication à récupérer.

### 5.3 Cas déjà connus

- `_reevaluate_known_metadata` doit planifier du travail borné, pas créer directement des sélections. Un fait mis en cache incomplet n'est pas une décision finale négative.
- Parcourir par lots avec pagination par clé, pas en chargeant 104k logs. Ne pas parser les archives dans `start_campaign` sous `_tx`.
- Pour un replay archivé sans cache à jour, planifier une lecture locale par un worker ; le raccourci `catalog_has_replay → selected` disparaît.
- Pour plusieurs versions, choisir la plus récente au moment de l'évaluation, enregistrer son hash puis conserver cette version. Si une sélection de la campagne est déjà épinglée, la respecter.
- Pour un rejet même version/profil/parseur : pas de nouveau GET, pas de compteur/rejet ajouté à chaque polling. Un nouveau profil, une nouvelle version ou `reinspect` peut déclencher une réévaluation ; `reinspect` n'autorise pas une boucle de refetch illimitée.
- `examined_only` décrit un corps examiné, pas une interdiction globale définitive. Une autre campagne peut le reconsidérer.
- Une tâche d'un format hors campagne ne doit pas être claimée. Les sélections/rejets de l'ancienne campagne ne doivent pas être repris comme décisions de la nouvelle.
- Reprendre une campagne fige son profil. Modifier le profil exige une nouvelle campagne ; modifier seulement les budgets/workers utilise les mécanismes existants.

---

## 6. Tâches d'implémentation

Les commandes suivantes sont exécutées depuis `pokecoach2-trainer/pokecoach-trainer/`. Utiliser l'environnement uv/Devbox déjà prévu par le dépôt. Chaque tâche suit : test rouge ciblé → implémentation → test vert → revue du diff. Les commandes de tests de ce plan sont à exécuter pendant l'implémentation, pas des résultats déjà obtenus.

### Tâche 1 — Contrat de filtres et chargement du profil

**Fichiers :** modifier `src/pokecoach_trainer/schemas.py`, `cli.py`, `dataset_filters.py`, `collection_filters.py`, `collection_service.py` ; créer `src/pokecoach_trainer/replay_policy.py`, `tests/test_replay_policy.py` ; compléter `tests/test_collection_config.py`, `test_collection_cli.py`, `test_collection_api.py`.

**Interfaces produites :** `ReplayFilterConfig`, `CollectionConfig.dataset_filters`, `filters_from_recipe`, `replay_filter_fingerprint`, `ReplayEligibility`, `evaluate_replay_policy`, chargement de `dataset_recipe_file` décrit en section 3. Les règles de parsing restent celles d'avant jusqu'à la tâche 2.

- [ ] Écrire les tests : ancien YAML → `dataset_filters is None` ; référence relative → les 12 valeurs exactes ; config inline équivalente → même empreinte ; modifier le YAML après chargement → config déjà chargée inchangée ; inline + fichier → erreur.
- [ ] Tester les valeurs invalides : `min_elo > max_elo`, tours négatifs, `unknown_elo` inconnu, clé mal orthographiée ; garder les anciennes recettes et alias compatibles. Le nombre minimal de tours est un entier ≥0 ; les Elo, lorsqu'ils sont fournis, sont des entiers ≥0.
- [ ] Tester que changer chacun des 12 champs possibles modifie l'empreinte métier, tandis que seed/splits/target/workers ne la modifient pas. Pour `unknown_elo`, une seule valeur est actuellement autorisée : tester la validation et sa présence dans la projection plutôt qu'inventer une seconde valeur.
- [ ] Exécuter `uv run --locked pytest tests/test_replay_policy.py tests/test_collection_config.py tests/test_collection_cli.py tests/test_collection_api.py` et constater les échecs attendus sur les nouvelles interfaces.
- [ ] Implémenter l'héritage sans changer la sérialisation plate de `DatasetRecipe`, ni la représentation de ses alias dans les anciens manifestes. Pour les configurations legacy sans profil, préserver l'empreinte de requête historique en omettant le nouveau champ nul ; une campagne active identique doit rester reconnue comme identique.
- [ ] Implémenter le chargement et la projection explicite des 12 champs. L'API conserve `CollectionJobConfig` comme enveloppe et accepte le nouveau champ dans `config` ; aucun accès à un chemin client côté API.
- [ ] Rejouer la commande : succès. Vérifier aussi `tests/test_collection_campaigns.py` et `tests/test_dataset_filters.py`.

### Tâche 2 — Métadonnées fiables avant écriture

**Fichiers :** modifier `src/pokecoach_trainer/replay_metadata.py`, `collection_publication.py`, `collection_filters.py`, `dataset_filters.py` ; créer `src/pokecoach_trainer/replay_log.py`, `tests/fixtures/replays/`, `tests/test_replay_log.py` ; compléter `tests/test_replay_metadata.py`, `test_collection_publication.py`, `test_dataset_filters.py`, `test_collection_filters.py` ; corriger `tests/replay_log_fixtures.py`.

**Interfaces produites :** `build_replay_envelope`, `REPLAY_METADATA_VERSION`, et extraction d'une enveloppe en mémoire avec l'API existante `extract_replay_metadata(ref, cached=envelope)`. Helpers internes de protocole partagés par les deux extracteurs.

- [ ] Écrire une fixture correcte de chaque cas de la section 4.2 ; nommer notamment `test_numeric_avatar_is_not_elo`, `test_timer_warning_is_not_inactivity_loss`, `test_packed_showteam_has_six_members`, `test_forfeit_after_two_closed_turns`, `test_partial_second_turn_does_not_reach_two`, `test_chat_forfeit_does_not_end_game`.
- [ ] Ajouter les assertions : avatar=1500 sans rating → `None` ; avatar=1/rating=1100 → 1100 ; six `poke|p1` → équipe complète ; cinq → incomplète ; timer ON + win → `normal_win` ; message système de perte par inactivité + win → `inactivity` ; abandon + win → `forfeit` avec gagnant conservé.
- [ ] Ajouter un test où `build_replay_envelope` ne crée aucun fichier et donne les mêmes hashes/enveloppe que `prepare_replay`. Comparer les métadonnées avant publication et après lecture de l'archive : égalité complète.
- [ ] Exécuter `uv run --locked pytest tests/test_replay_log.py tests/test_replay_metadata.py tests/test_collection_publication.py tests/test_dataset_filters.py tests/test_collection_filters.py` : les nouveaux tests doivent exposer les défauts avant correction.
- [ ] Implémenter les règles de la section 4.2 et versionner le parseur. Corriger les fixtures incohérentes, sans mode heuristique qui traite les anciens avatars comme ratings pour maintenir artificiellement leurs résultats.
- [ ] Rejouer les tests ciblés puis `uv run --locked pytest tests/test_dataset.py tests/test_dataset_split.py` ; expliquer tout changement de résultat lié à une correction de protocole. Ne pas prétendre avoir recalculé le rendement des 104k replays.

### Tâche 3 — Décisions métier partagées et déduplication dataset

**Fichiers :** modifier `src/pokecoach_trainer/replay_policy.py`, `dataset_service.py`, `collection_filters.py`, `collection_types.py` ; créer `src/pokecoach_trainer/replay_deduplication.py`, `tests/test_replay_deduplication.py` ; compléter `tests/test_replay_policy.py`, `test_dataset_filters.py`, `test_dataset_split.py`.

**Interfaces produites :** `CollectionEvaluation` et `evaluate_collection_candidate(config: CollectionConfig, ref: ArchiveRef, envelope: dict[str, Any]) -> CollectionEvaluation` dans `collection_filters.py`. Étendre `SelectionDecision.reason` pour les raisons métier en utilisant `str`, garder `selected` et `filter_sha256` ; les détails restent structurés dans `CollectionEvaluation`.

Pour la déduplication dataset, définir :

```python
def duplicate_replay_ids(
    refs: Sequence[ArchiveRef], admissible_ids: set[str],
) -> dict[str, str]: ...  # alias_id -> canonical_id, plus petit ID admissible par hash.
```

- [ ] Transformer les exemples de la section 1.2 en tests paramétrés de `evaluate_replay_policy`, avec assertions sur `accepted`, `accepted_sides` et `reason_code`.
- [ ] Couvrir aussi `max_elo=1400` (1400 accepté/1401 rejeté), `only_winners=true`, BO1/BO3 exclus séparément, `require_ots=false`, `require_complete_teams=false`, `include_forfeits=false`, inactivité autorisée et seuils d'abandon 0/1/2/3.
- [ ] Tester la composition date/équipe/métier : un rejet date n'est pas annulé par un succès métier. Les sources de rejet et l'ordre date → équipe → partie → perspectives sont stables.
- [ ] Tester `duplicate_replay_ids` : deux IDs même log → un alias ; premier ID inadmissible → l'autre reste canonique ; logs distincts même équipe → aucun alias ; plusieurs manches BO3 distinctes → aucun alias.
- [ ] Exécuter `uv run --locked pytest tests/test_replay_policy.py tests/test_replay_deduplication.py tests/test_dataset_filters.py tests/test_dataset_split.py` avant implémentation.
- [ ] Faire utiliser `evaluate_replay_policy` par `_prepare_snapshot_dataset` et enregistrer les mêmes `FilterDecision` utiles qu'avant. Utiliser seulement `accepted_sides` pour préparer les tâches de conversion.
- [ ] Si `recipe.deduplicate`, retirer les alias des IDs admissibles et des tâches **avant** `select_groups`. Enregistrer `duplicate_log` au stage `selection`. Garder les groupes construits sur le snapshot complet pour préserver les liens BO3/anti-fuite ; ne pas éliminer toutes les manches d'une série sous prétexte qu'elles partagent un groupe.
- [ ] Implémenter `evaluate_collection_candidate` sans accès réseau/disque, puis rejouer les tests ciblés. À ce stade, la décision de doublon de campagne reste à la tâche 5.

### Tâche 4 — Migration et stockage des preuves

**Fichiers :** modifier `src/pokecoach_trainer/archive_catalog.py`, `collection_store.py`, `collection_types.py` ; créer `tests/test_collection_metadata_cache.py`, `tests/test_collection_catalog_migration.py`.

**Interfaces produites :** schéma section 4.3 ; méthodes `CollectionStore.get_cached_metadata(ref: ArchiveRef) -> ReplayMetadata | None` et `cache_metadata(metadata: ReplayMetadata, *, source_format: str, uploadtime: int | None) -> None`. La première valide la version de parseur courante et l'identité complète.

- [ ] Créer un catalogue v1 temporaire avec objets, anciennes campagnes, décisions et publication en cours. Tester que migration conserve toutes les lignes et références, qu'un second démarrage ne modifie rien et qu'une version inconnue est refusée clairement.
- [ ] Tester cache pour même ID/deux contenus, même contenu/deux versions de parseur, données JSON invalides et cache absent. Une entrée inutilisable entraîne recalcul, pas acceptation implicite.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_catalog_migration.py tests/test_collection_metadata_cache.py` : rouge.
- [ ] Ajouter une migration 1→2 sous le verrou bootstrap existant. Créer les tables/colonnes dans une transaction, puis fixer `user_version=2` seulement après succès. Le chemin catalogue neuf crée directement le schéma v2.
- [ ] Implémenter la persistance des métadonnées, du détail des décisions, des hashes et des champs date/format. Prévoir un accès par lots indexé pour la réévaluation, sans scanner des logs en transaction.
- [ ] Rejouer les nouveaux tests et `uv run --locked pytest tests/test_collection_campaigns.py tests/test_collection_resume.py tests/test_collection_publication.py`.

### Tâche 5 — Sélection atomique, quota et doublons concurrents

**Fichiers :** modifier `src/pokecoach_trainer/collection_store.py`, `collection_publication.py`, `collection_types.py`, `collection_supervisor.py` ; créer `tests/test_collection_admission.py` ; compléter `tests/test_collection_limits.py`, `test_collection_leases.py`, `test_collection_publication.py`, `test_collection_supervisor.py`.

**Interfaces produites :** `record_evaluation`, `finalize_selection`, réservations `campaign_log_claims`, récupération des sélections en attente, suivant section 5. Adapter les appelants legacy de `record_selection` ; ne pas laisser deux chemins concurrents incrémenter les compteurs.

- [ ] Écrire `test_rejected_replay_releases_slot`, `test_selection_counts_only_after_publication`, `test_finalize_selection_is_idempotent` : quota inchangé sur rejet/évaluation ; exactement +1 après objet catalogué ; deuxième finalisation sans effet.
- [ ] Écrire un test à deux workers, quota=1 : au plus un sélectionné, réservation finale=0. Ne pas simuler la concurrence par deux appels séquentiels si le test vise une course ; utiliser barrières/deux connexions SQLite.
- [ ] Écrire `test_concurrent_equal_logs_select_one`, `test_rejected_alias_does_not_reserve_hash`, `test_pending_duplicate_retries_if_owner_fails`, et tester `deduplicate=false`.
- [ ] Avec les hooks existants, injecter les crashes avant rename, après rename/avant catalogue et après catalogue/avant finalisation. Après récupération : un objet, une sélection, un compteur ; zéro emplacement ou hash réservé à l'infini.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_admission.py tests/test_collection_limits.py tests/test_collection_leases.py tests/test_collection_publication.py` : rouge sur les nouvelles garanties.
- [ ] Implémenter les transitions et l'idempotence. La réservation de capacité reste active pendant publication ; le superviseur ne conclut pas « cible atteinte » sur des seules évaluations non publiées. Réconcilier les réservations expirées avec les intentions avant de les relâcher.
- [ ] Rejouer les tests ciblés et superviseur. Le cas `publish_replay` trouvant une version déjà présente doit aussi finaliser correctement au lieu de retourner avant le traitement de la sélection.

### Tâche 6 — Brancher les workers avant publication

**Fichiers :** modifier `src/pokecoach_trainer/collection_workers.py`, `collection_service.py`, `collection_filters.py` ; compléter `tests/test_collection_workers.py`, `test_collection_integration.py`, `test_collection.py`.

**Interfaces consommées :** `build_replay_envelope`, `evaluate_collection_candidate`, `record_evaluation`, `finalize_selection`.

- [ ] Écrire un test avec 4 candidats distincts : OTS absente, Elo trop faible, inactivité, valide. Attendre un seul objet publié, trois décisions détaillées, `selected_in_attempt=1`, puis poursuite de la collecte jusqu'à cible/budget.
- [ ] Tester qu'un rejet métier ne crée ni objet ni fichier staging ni intention de publication ; son cache et sa raison existent.
- [ ] Tester erreur HTTP temporaire (retry), payload malformé (rejet permanent `invalid_replay_payload`), et exception interne inattendue (erreur visible, pas conversion silencieuse en rejet métier). Aucun de ces cas n'incrémente les sélections.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_workers.py tests/test_collection_integration.py tests/test_collection.py` : vérifier les nouveaux échecs.
- [ ] Remplacer le flux de `_run_download_step` par la section 5.2. Les payloads invalides sont validés avant parsing ; capture ciblée des erreurs de validation, pas `except Exception: rejected` autour de toute l'admission.
- [ ] Pour un rejet, ne pas rappeler `release_lease` après qu'une transaction a déjà invalidé le lease. Pour `wait`, suivre la reprise différée ; pour `ready`, publier puis finaliser. Le heartbeat reste actif pendant tout ce travail.
- [ ] Dans l'ancien `collect_replays`, refuser explicitement `dataset_filters is not None` avec `InvalidConfiguration("les filtres dataset nécessitent la collecte supervisée")`. Les entrées CLI/API normales passent déjà par `submit_collection` ; conserver les tests legacy sans profil. Cette décision évite de développer un second moteur de reprise obsolète.
- [ ] Rejouer les tests : vert, y compris la collecte historique qui accepte les équipes inconnues lorsque `dataset_filters=None` et aucun filtre équipe n'est activé.

### Tâche 7 — Stock existant, nouvelle campagne et invalidation

**Fichiers :** modifier `src/pokecoach_trainer/collection_store.py`, `collection_workers.py`, `collection_discovery.py` ; compléter `tests/test_collection_resume.py`, `test_collection_incremental.py`, `test_collection_workers.py`, `test_collection_metadata_cache.py`.

**Interfaces consommées :** cache de la tâche 4 ; admission/finalisation de la tâche 5. `_reevaluate_known_metadata` devient un ordonnanceur borné de candidats connus, sans passage direct à `selected`.

- [ ] Tester objet déjà archivé valide et invalide, avec transport HTTP qui lève sur tout appel : seul le valide est sélectionné et aucune nouvelle version n'est publiée.
- [ ] Tester les mêmes objets sans cache, puis avec cache d'un ancien parseur : extraction locale obligatoire, mêmes résultats que le chemin réseau.
- [ ] Tester un rejet OTS, puis même profil : zéro GET additionnel. Nouveau profil `require_ots=false` avec corps absent : un GET, réévaluation du payload et publication si admissible. Si le payload a changé et échoue, rejet de cette nouvelle version, pas acceptation héritée du cache.
- [ ] Tester pause/reprise et nouvel attempt : filtres identiques malgré modification du fichier YAML ; sélections déjà finalisées non recomptées ; quotas et historique conservés.
- [ ] Tester backlog d'un autre format et anciens replays rejetés/archivés : seules les tâches candidates de la campagne actuelle sont traitées ; aucun rejet d'une autre campagne n'agit comme blacklist globale.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_resume.py tests/test_collection_incremental.py tests/test_collection_workers.py tests/test_collection_metadata_cache.py` : rouge.
- [ ] Planifier par lots les IDs connus des formats demandés, incluant les versions archivées qui n'ont pas encore de résumé métier. Les workers relisent le corps local au besoin. Conserver les décisions finales déjà connues pour le même profil/contenu et borner le réexamen à un passage par identité de validation.
- [ ] Implémenter l'invalidation avec les empreintes de filtre/parseur, la nouvelle version et `reinspect`. Adapter la clause de claim aux formats et aux décisions de la campagne ; ne pas se fier seulement à `download_tasks.state` global.
- [ ] Rejouer la commande ciblée et les tests de découverte ; vérifier qu'aucun traitement long ne reste dans une transaction d'écriture de `start_campaign`.

### Tâche 8 — Snapshot complet, versions épinglées et preuve de parité

**Fichiers :** modifier `src/pokecoach_trainer/archive_service.py`, `archive_catalog.py`, `schemas.py` ; créer `tests/test_collection_dataset_parity.py` ; compléter `tests/test_collection_api.py`, `test_dataset.py` `test_archive_snapshots.py` et `test_archive_catalog.py`.

**Interfaces produites :** `create_campaign_snapshot(campaign_id)` inchangée côté appelants ; nouveaux champs optionnels de `ArchiveSnapshot` : `collection_filter_sha256`, `collection_dataset_filters`, `collection_parser_version`, `collection_policy_version`. Anciens snapshots lisibles avec valeurs par défaut nulles.

- [ ] Créer une campagne de 1001 sélections dans un stockage temporaire : le snapshot contient 1001 refs, pas 1000. Vérifier aussi pagination vide et dernier lot partiel.
- [ ] Ajouter une nouvelle version invalide à un ID après sélection d'une version valide : le snapshot utilise la version épinglée, jamais la dernière automatiquement.
- [ ] Tester que `pending_publication`, rejets et anciens objets non sélectionnés n'apparaissent pas. Un snapshot concurrent ne doit pas inclure une finalisation postérieure au point de capture.
- [ ] Construire une campagne synthétique couvrant les filtres, puis son snapshot ; pour chaque ref, recharger l'objet et vérifier `evaluate_replay_policy(...).accepted is True`, au moins une perspective et unicité de `log_sha256` si activée.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_dataset_parity.py tests/test_collection_api.py tests/test_dataset.py` : rouge.
- [ ] Fixer un point de capture cohérent des lignes sélectionnées et du plafond de publication dans une transaction SQLite de lecture ; parcourir les pages de 1000 jusqu'au total, avec les mêmes bornes. Ne pas garder cette transaction ouverte pendant le parsing de tous les objets.
- [ ] Construire les métadonnées depuis les versions épinglées/cache correspondant. Une version sélectionnée manquante ou corrompue produit une erreur explicite ; ne pas substituer une autre version ni omettre silencieusement l'échec d'extraction dans une campagne filtrée.
- [ ] Enregistrer les filtres et versions de validation dans le snapshot et son hash. Les anciens snapshots restent immuables. Avec un parseur/policy incompatible avec la campagne, refuser la création d'un nouveau snapshot filtré en demandant une nouvelle campagne de validation, plutôt que réinterpréter silencieusement les anciennes sélections.
- [ ] Ajouter un test d'intégration `dataset create/prepare` sur ce snapshot : remplacer uniquement la conversion lourde par un résultat contrôlé, laisser les vrais filtres/sélection actifs ; zéro rejet métier nouveau avec la même recette, et aucune conversion pour les côtés inadmissibles. Prévoir suffisamment de groupes pour les trois splits et une cible adaptée à la fixture.
- [ ] Rejouer les tests. Le mock de conversion démontre la parité des filtres, pas la convertibilité réelle ; ne pas annoncer cette dernière.

### Tâche 9 — Profils utilisables, métriques et documentation

**Fichiers :** créer `configs/dataset-mc-collectable.yaml`, `configs/collection-mc-100k-valid.yaml` ; modifier `src/pokecoach_trainer/collection_metrics.py`, `README.md`, `../docs/dataset-reason-codes.md` ; compléter `tests/test_collection_metrics.py`, `test_collection_config.py`, `test_collection_api.py`.

**Interfaces produites :** progression distinguant `selected_replays`, `selected_in_attempt`, `new_published`, `rejected_by_reason`, plus `accepted_perspectives`, `duplicate_replays`, `validation_pending` et l'empreinte du profil. `accepted_perspectives` compte uniquement les côtés des sélections finalisées, pas ceux des doublons/rejets ni des évaluations en attente.

- [ ] Écrire un test chargeant le nouveau YAML depuis un CWD différent et vérifier exactement les 12 paramètres de la demande, notamment inactivité=true et tours=2.
- [ ] Écrire un test de métriques : replay retenu avec un seul côté → +1 replay/+1 perspective ; rejet → +0 ; doublon → +0 avec compteur doublon et raison ; répétition idempotente → aucune augmentation.
- [ ] Exécuter `uv run --locked pytest tests/test_collection_metrics.py tests/test_collection_config.py tests/test_collection_api.py`.
- [ ] Créer les YAML complets, implémenter les agrégats bornés/indexés et documenter les nouvelles raisons (`no_eligible_perspective`, `duplicate_log`, `invalid_replay_payload`, `winner_unknown`) avec les détails individuels d'Elo. Garder les raisons existantes du dataset.
- [ ] Documenter les commandes futures :

```bash
# À exécuter volontairement après validation, pas pendant les tests.
uv run --locked pokecoach-trainer collection start --config configs/collection-mc-100k-valid.yaml
uv run --locked pokecoach-trainer collection status --job-id JOB_ID
```

- [ ] Documenter l'endpoint déjà existant `POST /collection/jobs/{job_id}/snapshot` (préfixe `/collection` confirmé dans `src/pokecoach_trainer/main.py`). Utiliser le snapshot retourné pour `dataset create --snapshot-id SNAPSHOT_ID --recipe configs/dataset-mc-collectable.yaml --dataset-id DATASET_ID` ; la commande commence par `uv run --locked pokecoach-trainer` comme les autres.
- [ ] Expliquer que la cible dataset doit être atteinte par le snapshot et que les contraintes de splits/conversion restent à vérifier. Documenter reprise avec budgets supplémentaires, nouveau profil → nouvelle campagne, et absence de suppression des 104k objets historiques.
- [ ] Rejouer les tests puis relire la documentation : aucun « 100 % convertibles », aucune promesse d'atteindre 100k avec 200k requêtes, aucune recommandation d'utiliser le snapshot global historique.

### Tâche 10 — Validation finale et bilan vérifiable

**Fichiers :** tests précédents ; créer `../docs/audits/2026-11-05-collecte-filtree-resultats.md` pendant l'implémentation seulement, avec les résultats réellement obtenus.

- [ ] Exécuter la suite ciblée complète :

```bash
uv run --locked pytest tests/test_replay_policy.py tests/test_replay_log.py tests/test_replay_metadata.py tests/test_replay_deduplication.py tests/test_collection_metadata_cache.py tests/test_collection_catalog_migration.py tests/test_collection_admission.py tests/test_collection_workers.py tests/test_collection_resume.py tests/test_collection_incremental.py tests/test_collection_limits.py tests/test_collection_leases.py tests/test_collection_publication.py tests/test_collection_dataset_parity.py tests/test_collection_metrics.py tests/test_collection_cli.py tests/test_collection_api.py tests/test_collection_config.py tests/test_dataset_filters.py tests/test_dataset_split.py
```

- [ ] Exécuter `uv run --locked pytest`, puis `uv run --locked ruff check src tests`. Critère : aucune régression attribuable aux changements ; consigner les échecs préexistants séparément avec une preuve, sans les masquer ni les déclarer corrigés.
- [ ] Exécuter un scénario de collecte via le serveur HTTP local de `tests/collection_http_server.py` : mélange de candidats, 1 puis plusieurs workers, interruption/reprise, quota=3. Attendre exactement trois replays admissibles distincts, aucune publication rejetée et un snapshot de trois refs. Adapter la fixture au protocole réel de la tâche 2.
- [ ] Vérifier sur une copie temporaire d'un catalogue v1 la migration et la lecture des anciens snapshots. Jamais de migration de production comme simple test.
- [ ] Si le stockage réel contenant les 104k replays est disponible, faire un audit local borné en lecture seule : maximum 200 objets, provenance enregistrée, extraction/raisons avant et après correction, durée moyenne et estimation de coût. Ne pas conclure à un rendement global à partir de cet échantillon. Si le stockage manque, le dire et conserver les tests synthétiques comme preuve effectivement disponible.
- [ ] Écrire le bilan : commandes et résultats, profil exact, nombres découverts/examinés/rejetés/sélectionnés/perspectives, raisons principales, reprise/crash testés, limites de preuve. Vérifier `git diff --check` et la liste des fichiers modifiés.

## 7. Critères de livraison

- [ ] Le nouveau profil reprend exactement les 12 paramètres demandés.
- [ ] Collecte et dataset appellent les mêmes filtres, sur des métadonnées issues du même parseur versionné.
- [ ] Les tests de protocole empêchent notamment la confusion avatar/Elo et timer/inactivité.
- [ ] Chaque replay sélectionné possède au moins une perspective admissible ; les côtés rejetés ne sont pas convertis.
- [ ] Aucun nouveau corps rejeté n'est publié ; les raisons et faits compacts restent disponibles.
- [ ] Déduplication exacte, par campagne, résistante à la concurrence et partagée dans sa définition avec le dataset.
- [ ] Le quota compte les sélections finalisées ; les rejets, retries et crashes ne le gonflent pas.
- [ ] Les archives connues passent les filtres sans téléchargement inutile ; un profil changé invalide les décisions appropriées.
- [ ] Le snapshot filtré est complet au-delà de 1000 refs, épingle les versions et exclut les objets historiques non sélectionnés.
- [ ] Les campagnes/configurations sans profil et les anciens snapshots restent lisibles ; aucune donnée existante n'est supprimée.
- [ ] La documentation distingue clairement téléchargement d'un candidat, sélection métier et conversion en trajectoire.

Le chantier est terminé lorsque ces critères sont démontrés par les tests et le bilan, sans nécessiter de lancer une collecte de 100k pour prouver le comportement.
