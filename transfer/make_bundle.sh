#!/usr/bin/env bash
# Pack a payload into one file for the USB hop. Run on the Mac, from the payload dir.
#
#   bash make_bundle.sh vision     ~1.6 GB  repo + vision.onnx + golden/
#   bash make_bundle.sh all       ~37 GB   everything (see the warning below)
#
# NOT gzipped on purpose: ONNX weights are fp16 and compress by only a few percent,
# so gzip costs minutes of CPU at both ends for nothing. A plain tar also lets you
# stream-extract without a temporary copy.
set -euo pipefail

MODE="${1:-vision}"
PAYLOAD="${PAYLOAD:-$HOME/alpamayo-payload}"
cd "$PAYLOAD"

case "$MODE" in
  vision) OUT="alpamayo-vision.tar"
          ITEMS=(alpamayo-repo.tgz golden)
          for f in onnx/vision.onnx*; do [ -e "$f" ] && ITEMS+=("$f"); done ;;
  all)    OUT="alpamayo-all.tar"
          ITEMS=(alpamayo-repo.tgz golden fixtures onnx)
          [ -f MANIFEST.sha256 ] && ITEMS+=(MANIFEST.sha256)
          cat <<'WARN'
NOTE on `all`: a 37 GB tar needs 37 GB for the archive PLUS 37 GB extracted at
every hop -- 74 GB on the Ubuntu host. You also lose rsync's per-file resume: one
bad byte means re-copying the whole archive. For the full payload, copying the
directories straight across is usually the better trade.
WARN
          ;;
  *) echo "usage: make_bundle.sh [vision|all]" >&2; exit 2 ;;
esac

for i in "${ITEMS[@]}"; do
    [ -e "$i" ] || { echo "ERROR: missing $i -- pull it before bundling" >&2; exit 1; }
done

echo "packing: ${ITEMS[*]}"
rm -f "$OUT" "$OUT.sha256"
tar -cf "$OUT" "${ITEMS[@]}"

if command -v sha256sum >/dev/null; then sha256sum "$OUT" > "$OUT.sha256"
else shasum -a 256 "$OUT" > "$OUT.sha256"; fi

echo
ls -lh "$OUT" "$OUT.sha256"
cat "$OUT.sha256"
cat <<MSG

Copy BOTH files to the exFAT stick, then on the Ubuntu host:

  cd ~/alpamayo-payload
  cp /media/\$USER/ALPAMAYO/$OUT* .
  shasum -a 256 -c $OUT.sha256 || sha256sum -c $OUT.sha256
  tar -xvf $OUT

Then push to the Jetson:

  bash ~/alpamayo-xavier/transfer/to_xavier.sh <xavier-ip> ${MODE}
MSG
