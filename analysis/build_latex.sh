#!/usr/bin/env bash
# Build the TikZ figures. Needs pdflatex; ghostscript only for the PNG preview.
#
#   bash analysis/build_latex.sh            # every .tex in analysis/
#   bash analysis/build_latex.sh fig04_pipeline
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/figures"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

for tex in "${@:-$(ls "$HERE"/*.tex | xargs -n1 basename | sed 's/\.tex$//')}"; do
    name="${tex%.tex}"
    src="$HERE/$name.tex"
    [ -f "$src" ] || { echo "no such figure: $src" >&2; exit 1; }
    echo "== $name"
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$TMP" "$src" \
        >"$TMP/$name.out" 2>&1 || { grep -E '^!|l\.[0-9]+' "$TMP/$name.out" | head -20; exit 1; }
    cp "$TMP/$name.pdf" "$OUT/$name.pdf"
    if command -v gs >/dev/null 2>&1; then
        gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r220 -dTextAlphaBits=4 \
           -dGraphicsAlphaBits=4 -sOutputFile="$OUT/$name.png" "$OUT/$name.pdf" >/dev/null
    fi
    ls -l "$OUT/$name".{pdf,png} 2>/dev/null | awk '{printf "   %s  %s bytes\n", $NF, $5}'
done
