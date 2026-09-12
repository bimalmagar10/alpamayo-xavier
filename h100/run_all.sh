#!/usr/bin/env bash
# Clean rebuild of the whole H100 side, in order, with no stale artefacts.
#
#   bash h100/run_all.sh              full rebuild, A1 onwards
#   bash h100/run_all.sh --keep-golden  reuse A1's output (saves ~30 min)
#
# Ends by writing MANIFEST.sha256 over everything the Jetson needs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
alpamayo_check_paths || exit 1

# The prefill export only fits on one 80 GB H100 without allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Print as it happens: through `| tee`, Python otherwise holds output back in a
# buffer and a working job looks frozen.
export PYTHONUNBUFFERED=1

KEEP_GOLDEN=0
[ "${1:-}" = "--keep-golden" ] && KEEP_GOLDEN=1

step() { echo; echo "=============== $* ==============="; }

step "preflight  network"
# a1, a3b and a5 read the driving clip through physical_ai_av, which STREAMS the
# camera video from huggingface.co on every run -- only small metadata is cached.
# Without a route out, those steps sit in network timeouts and look hung.
if ! python -c "import socket; socket.create_connection(('huggingface.co', 443), timeout=10).close()"; then
  echo "No route to huggingface.co from $(hostname)." >&2
  echo "Allocate a compute node with internet access (rpg-93-5 worked before) and rerun." >&2
  exit 1
fi
echo "huggingface.co reachable from $(hostname)"

step "0  clearing derived artefacts"
# The checkpoint at $ALPAMAYO_MODEL is never touched.
rm -rf "$ALPAMAYO_ROOT/onnx" "$ALPAMAYO_ROOT/fixtures" "$ALPAMAYO_ROOT/frames" \
       "$ALPAMAYO_ROOT/ckpt/alpamayo-fp16"
[ "$KEEP_GOLDEN" = 0 ] && rm -rf "$ALPAMAYO_ROOT/golden"
mkdir -p "$ALPAMAYO_ROOT"/{onnx,fixtures,golden,frames,logs}
df -h "$ALPAMAYO_ROOT" | tail -1

if [ "$KEEP_GOLDEN" = 0 ]; then
  step "1  golden reference + CoC trace statistics  (~30 min)"
  python "$ALPAMAYO_REPO/h100/a1_golden.py" --clips 64 --max-new-tokens 256 \
      2>&1 | tee "$ALPAMAYO_ROOT/logs/a1.log"
else
  step "1  SKIPPED (--keep-golden)"
fi

step "1b per-stage H100 timing  (~5 min)"
python "$ALPAMAYO_REPO/h100/a1b_h100_stages.py" 2>&1 | tee "$ALPAMAYO_ROOT/logs/a1b.log"

step "2  fp16 cast audit  (audit only -- the 22 GB checkpoint is not needed)"
# a3 casts from the original bf16 itself; this only certifies that it is safe.
python "$ALPAMAYO_REPO/h100/a2_cast_fp16.py" --audit-only \
    2>&1 | tee "$ALPAMAYO_ROOT/logs/a2_audit.log"

step "3  export four ONNX graphs  (~40 min)"
python "$ALPAMAYO_REPO/h100/a3_export_onnx.py" 2>&1 | tee "$ALPAMAYO_ROOT/logs/a3.log"

step "3a make the graphs readable by TensorRT 8.5 (LayerNormalization -> primitives)"
python "$ALPAMAYO_REPO/h100/a3c_decompose_layernorm.py" 2>&1 | tee "$ALPAMAYO_ROOT/logs/a3c.log"

step "3b fixtures for the Jetson"
python "$ALPAMAYO_REPO/h100/a3b_fixtures.py" 2>&1 | tee "$ALPAMAYO_ROOT/logs/a3b.log"

step "3c the golden clip's camera frames, for run_alpamayo.py on the Jetson"
python "$ALPAMAYO_REPO/h100/a5_export_frames.py" 2>&1 | tee "$ALPAMAYO_ROOT/logs/a5.log"

step "4  manifest"
cd "$ALPAMAYO_ROOT"
rm -f MANIFEST.sha256
# Only what the Jetson actually needs. Trace scratch dirs are excluded by name.
find onnx fixtures golden frames -type f ! -path '*__trace*' ! -name '*.tmp.onnx*' -print0 \
  | sort -z | xargs -0 sha256sum > MANIFEST.sha256
echo "$(wc -l < MANIFEST.sha256) files"
du -sh onnx fixtures golden frames
echo "payload: $(du -cb onnx fixtures golden frames 2>/dev/null | tail -1 | cut -f1 | awk '{printf "%.2f GB", $1/1e9}')"

step "done"
cat <<MSG
Everything the Jetson needs is under $ALPAMAYO_ROOT:
  onnx/  fixtures/  golden/  frames/  MANIFEST.sha256

Pull it from your Mac (the cluster cannot reach the Jetson):
  rsync -avh --partial --progress \\
      <user>@<cluster>:$ALPAMAYO_ROOT/{onnx,fixtures,golden,frames,MANIFEST.sha256} \\
      ~/alpamayo-payload/
MSG
