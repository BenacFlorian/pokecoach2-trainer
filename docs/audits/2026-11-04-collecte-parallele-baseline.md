# Baseline collecte parallèle — 2026-11-04

## Révision Git

- `HEAD`: `ad605969efc11e39ade61959cc21ce462b444eb6`
- État working tree (hors docs plan/spec non suivis) : propre au moment de l'audit.

## Environnement Python / SQLite

Commande : `uv run --locked python -c "import sys, sqlite3; print(sys.version); print(sqlite3.sqlite_version)"`

- Python : **3.13.2**
- `sqlite3.sqlite_version` chargée par Python : **3.47.1**

La spec exige une bibliothèque SQLite avec correctif WAL-reset (≥ 3.51.3 ou backports officiels 3.44.6 / 3.50.7). **3.47.1 n'est pas dans la liste autorisée** : les tests de concurrence catalogue injectent une version ; pour la prod locale, prévoir une build Python liée à SQLite corrigée avant collecte multi-processus réelle.

## Tests préexistants (avant implémentation parallèle)

```bash
cd pokecoach-trainer && uv run --locked pytest \
  tests/test_collection.py tests/test_collection_resume.py \
  tests/test_archive.py tests/test_archive_snapshots.py \
  tests/test_collection_api.py tests/test_worker_lifecycle.py -q
```

Résultat : **21 passed**, 0 failed (2026-11-04).

## Contrat API recherche Showdown

Source documentaire : [WEB-API.md — replay search](https://github.com/smogon/pokemon-showdown-client/blob/master/WEB-API.md#replay-search).

### Requête

`GET https://replay.pokemonshowdown.com/search.json?format=<format_id>&before=<unix_seconds>`

- `before` : borne haute **exclusive** (replays strictement plus anciens que ce timestamp).
- Réponse : tableau JSON d'objets `{ id, uploadtime, format, players, rating, ... }`.
- Ordre observé (2026-11-04, `gen9championsvgc2026regmc`, curl espacé ≥ 1 s) : **décroissant par `uploadtime`** (plus récent en tête).
- Taille de page observée : **51** entrées tant que l'historique n'est pas épuisé ; `[]` si aucun replay plus ancien que `before`.

### Pagination — règle de continuation retenue

1. Première page : `before = borne_haute_effective + 1` (compatible avec sémantique exclusive du paramètre).
2. Page suivante : `before = uploadtime` du **dernier** élément de la page courante (le plus ancien de la page), **sans** incrément arbitraire `+1` après la première requête.
3. Si la page est vide : preuve d'épuisement pour la plage accessible via cette interface (enregistrer couverture).
4. Si deux pages consécutives partagent le même `uploadtime` minimal sans avancer le curseur : **stall** → marquer `coverage_incomplete` (ne pas inventer de saut de timestamp).

### Limite de l'audit réseau

- Vérification partielle (≤ 10 GET, format MC 2026).
- L'ancien code (`before + 1` à chaque page) doit être réexaminé ; la fixture `tests/fixtures/collection/search_contract.json` fige les observations pour les tests simulés.

## Décision legacy `before + 1`

Le client historique utilisait `before + 1` systématiquement dans l'URL. L'audit confirme que `before` est exclusif ; la continuation doit reprendre le timestamp du dernier item, pas `oldest + 1` ni `oldest - 1` sans preuve. Les tests simulés codent le contrat retenu ci-dessus.
