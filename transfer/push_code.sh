#!/usr/bin/env bash
# Push code-only changes to the Jetson. Seconds, and safe to re-run.
#
#   bash transfer/push_code.sh <xavier-ip>
#   XAVIER_USER=bimal bash transfer/push_code.sh 192.168.1.50
#
# Touches only source. Never touches onnx/, engines/, results/ or your CLIP setup.
set -euo pipefail

HOST="${1:?usage: push_code.sh <xavier-ip>}"
USER_="${XAVIER_USER:-bimal}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="alpamayo-xavier"          # relative to the remote home directory

rsync -avh --checksum \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '*.pdf' \
    "$SRC"/bench "$SRC"/xavier "$SRC"/transfer "$SRC"/env.sh "$SRC"/README.md \
    "$USER_@$HOST:$DEST/"

echo
echo "verify the files that matter:"
ssh "$USER_@$HOST" "cd $DEST && md5sum bench/alpamayo_stage_bench.py xavier/run_alpamayo.py \
    xavier/alpamayo_xavier/trt_runner.py"
echo "local:"
( cd "$SRC" && md5sum bench/alpamayo_stage_bench.py xavier/run_alpamayo.py \
    xavier/alpamayo_xavier/trt_runner.py )
