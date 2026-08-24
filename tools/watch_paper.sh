#!/usr/bin/env bash
# Watch the manuscript .docx and re-render the .md sidecar + .pdf on every save.
#
# Runs in the foreground on purpose: one line per render, and you can see it die.
# Ctrl-C to stop.
#
#   ./tools/watch_paper.sh                          # default manuscript
#   ./tools/watch_paper.sh path/to/other.docx
#
# The render writes .md and .pdf into the same directory, which fswatch also
# sees. The hash guard below is what stops that from looping forever: an event
# only causes work when the .docx contents actually changed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="${1:-$REPO/paper/ksas_2026_fall/manuscript.docx}"
DIR="$(cd "$(dirname "$DOC")" && pwd)"
DOC="$DIR/$(basename "$DOC")"

command -v fswatch >/dev/null || { echo "fswatch not installed: brew install fswatch" >&2; exit 1; }

hash_of() { [ -f "$DOC" ] && shasum -a 256 "$DOC" | cut -d' ' -f1 || echo "absent"; }

render() {
  if [ ! -f "$DOC" ]; then
    echo "$(date +%H:%M:%S)  waiting — $(basename "$DOC") does not exist yet"
    return
  fi
  python3 "$REPO/tools/render_paper.py" "$DOC" || echo "$(date +%H:%M:%S)  render FAILED (see above)"
}

echo "watching  $DOC"
echo "renders   $(basename "${DOC%.docx}").md + $(basename "${DOC%.docx}").pdf"
echo "stop      Ctrl-C"
echo

last="$(hash_of)"
render

# --latency batches the burst of events LibreOffice emits when it saves.
fswatch -o --latency 1 --exclude '\.(md|pdf)$' --exclude '\.~lock' "$DIR" | while read -r _; do
  cur="$(hash_of)"
  [ "$cur" = "$last" ] && continue
  last="$cur"
  render
done
