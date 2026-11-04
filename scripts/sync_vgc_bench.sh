#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${POKECOACH_VGC_BENCH_ROOT:-$ROOT/../pokecoach2-python/engine-runtime/vendor/vgc-bench}"
DEST="$ROOT/pokecoach-trainer/vendor/vgc-bench"
if [[ ! -d "$SRC" ]]; then
  echo "Source VGC-Bench introuvable: $SRC" >&2
  exit 1
fi
mkdir -p "$DEST"
rsync -a --delete --exclude '.git' "$SRC/" "$DEST/"
PATCH="$ROOT/patches/logs2trajs-provenance.patch"
if [[ -f "$PATCH" ]]; then
  patch -p1 -d "$DEST" --forward --batch < "$PATCH" || {
    echo "Échec application patch provenance: $PATCH" >&2
    exit 1
  }
  echo "Applied patch: logs2trajs-provenance.patch"
fi
# Re-appliquer les épingles Trainer (voir patches/README.md)
sed -i 's|imitation @ git+https://github.com/cameronangliss/imitation.git@fix-build|imitation @ git+https://github.com/cameronangliss/imitation.git@56eb3b295da5d52b8f7277c6ba206913b19ab110|' "$DEST/pyproject.toml"
sed -i '/open-spiel==/d' "$DEST/pyproject.toml"
sed -i 's|poke-env @ git+https://github.com/cameronangliss/poke-env.git@vgc-bench|poke-env @ git+https://github.com/cameronangliss/poke-env.git@379a04628e42b2a2c94790c0788204b652f7ae1b|' "$DEST/pyproject.toml"
echo "Synced VGC-Bench from $SRC to $DEST"
