#!/usr/bin/env bash
# Push the weights to a target host, once.
#
#   ./docker/sync-models.sh tbtlr@orin.local
#   ./docker/sync-models.sh user@spark.local
#
# Kept out of the image on purpose: 5.6 GB that changes rarely, versus a
# binary that changes often. Bundled, every code change ships 6+ GB to
# every machine.
set -euo pipefail

TARGET="${1:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${MODELS_SRC:-$REPO/models}"
DST="${MODELS_DST:-/var/lib/gemma-live/models}"

[[ -n "$TARGET" ]] || { echo "usage: $0 <user@host>" >&2; exit 1; }
[[ -d "$SRC" ]] || { echo "error: no models at $SRC" >&2; exit 1; }

# tokenizer.bin is a hard dependency with no reference anywhere in the
# repo: moonshine resolves it as dirname(model)/tokenizer.bin, and
# without it startup fails complaining about the MODEL, not the
# tokenizer. Worth failing here instead.
[[ -f "$SRC/tokenizer.bin" ]] || {
    echo "error: $SRC/tokenizer.bin missing — run tools/fetch-models.sh" >&2
    exit 1; }

echo "==> $(du -sh "$SRC" | cut -f1) -> $TARGET:$DST"
ssh "$TARGET" "sudo install -d -m 0755 -o \$(id -u) -g \$(id -g) '$DST'"
# No compression: gguf is already compressed, so -z burns CPU for nothing
# on both ends. --partial so an interrupted push resumes.
rsync -a --info=progress2 --partial "$SRC"/ "$TARGET:$DST/"
echo "==> on the target:"
ssh "$TARGET" "ls -la '$DST' | tail -n +2 | awk '{printf \"    %8.2f GB  %s\n\", \$5/1073741824, \$9}'"
