#!/usr/bin/env bash
# Verify a payload against MANIFEST.sha256. Works on macOS and Linux.
#
#   bash verify.sh [dir]        default: current directory
#
# Run this at EVERY hop. A 37 GB payload crossing cluster -> Mac -> USB -> Ubuntu
# -> Jetson has four chances to corrupt a byte, and a damaged .data file surfaces
# much later as an incomprehensible TensorRT build error.
set -euo pipefail
cd "${1:-.}"

[ -f MANIFEST.sha256 ] || { echo "no MANIFEST.sha256 here" >&2; exit 1; }

if command -v sha256sum >/dev/null; then
    CHECK=(sha256sum -c --quiet MANIFEST.sha256)
elif command -v shasum >/dev/null; then
    CHECK=(shasum -a 256 -c MANIFEST.sha256)     # macOS
else
    echo "no sha256 tool found" >&2; exit 1
fi

echo "checking $(wc -l < MANIFEST.sha256) files in $(pwd) ..."
n_missing=0
while read -r _ path; do
    [ -f "$path" ] || { echo "MISSING  $path"; n_missing=$((n_missing+1)); }
done < MANIFEST.sha256
[ "$n_missing" -gt 0 ] && { echo "$n_missing file(s) missing" >&2; exit 1; }

# NOTE: capture rather than pipe. Under `set -o pipefail` the pipeline inherits
# sha256sum's exit status 1 on mismatch, so an `if <pipeline>` reads as FALSE in
# exactly the case we need to catch -- the check would silently pass.
out="$("${CHECK[@]}" 2>&1 || true)"
bad="$(printf '%s\n' "$out" | grep -v ': OK$' | grep . || true)"
if [ -n "$bad" ]; then
    echo
    printf '%s\n' "$bad"
    echo
    echo "VERIFY FAILED -- re-copy the files listed above" >&2
    exit 1
fi
echo "all files match. payload is intact."

echo
echo "sizes:"
du -sh onnx fixtures golden 2>/dev/null || true
