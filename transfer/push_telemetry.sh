#!/usr/bin/env bash
# Send the completed telemetry update and its vocabulary. No model/engine rebuild.
set -euo pipefail
HOST="${1:?usage: bash transfer/push_telemetry.sh <xavier-ip>}"
XAVIER_LOGIN="${XAVIER_USER:-bimal}"
XAVIER_DATA="${XAVIER_WORK:-/mnt/ssdhome/models/alpamayo}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOCAB="${ALPAMAYO_VOCAB:-$REPO/artifacts/telemetry/vocab.json}"
[ -f "$VOCAB" ] || { echo "missing vocabulary: $VOCAB" >&2; exit 1; }
bash "$REPO/transfer/push_code.sh" "$HOST"
rsync -avh --checksum "$VOCAB" "$XAVIER_LOGIN@$HOST:$XAVIER_DATA/fixtures/vocab.json"
echo "Telemetry code and vocabulary transferred. Use your existing inference command."
