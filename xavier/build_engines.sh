#!/usr/bin/env bash
# Build the four TensorRT engines ON the Xavier.
#
# Engines are not portable. A TensorRT plan file is locked to the compute
# capability, the TensorRT version and the exact GPU it was built for, so an
# engine built on the H100 (sm_90, TRT 10.x) cannot load here (sm_72, TRT 8.5.2).
# The H100 produces ONNX; only this machine can produce .plan files.
#
# Expect this to take 30-90 minutes. The prefill and decode graphs are large and
# TensorRT's tactic search on a Xavier CPU is slow. Run it once, keep the plans.
set -euo pipefail

WORK="${ALPAMAYO_WORK:-/mnt/ssdhome/models/alpamayo}"
ONNX="$WORK/onnx"
ENG="$WORK/engines"
LOG="$WORK/logs"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
PRECISION="${PRECISION:-int8}"      # int8 | fp16
WORKSPACE_MB="${WORKSPACE_MB:-4096}"

mkdir -p "$ENG" "$LOG"
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC" >&2; exit 1; }

# Lock clocks so tactic timing is not measured against a throttling board --
# a wandering clock makes TensorRT pick genuinely worse kernels.
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

build() {
  local name="$1"; shift
  local src="$ONNX/$name.onnx"
  [ "$PRECISION" = "int8" ] && [ -f "$ONNX/$name.int8.onnx" ] && src="$ONNX/$name.int8.onnx"
  local plan="$ENG/$name.$PRECISION.plan"

  if [ ! -f "$src" ]; then echo "[skip] $src missing"; return; fi
  if [ -f "$plan" ]; then echo "[have] $plan"; return; fi

  echo "== building $name from $(basename "$src") =="
  local flags=(--onnx="$src" --saveEngine="$plan"
               --memPoolSize=workspace:${WORKSPACE_MB}M
               --builderOptimizationLevel=3 --verbose)
  # fp16 stays on even for int8 builds: it is the fallback precision for any
  # layer TensorRT refuses to run in int8, and without it those fall back to fp32.
  flags+=(--fp16)
  [ "$PRECISION" = "int8" ] && flags+=(--int8)

  /usr/bin/time -v "$TRTEXEC" "${flags[@]}" "$@" 2>&1 | tee "$LOG/build_$name.$PRECISION.log"
  echo "[done] $plan  ($(du -h "$plan" | cut -f1))"
}

build vision
build prefill
build decode
build expert

echo
echo "== per-engine profiles =="
for name in vision prefill decode expert; do
  plan="$ENG/$name.$PRECISION.plan"
  [ -f "$plan" ] || continue
  echo "-- $name --"
  "$TRTEXEC" --loadEngine="$plan" --iterations=100 --avgRuns=50 --noDataTransfers \
      --dumpProfile --separateProfileRun 2>&1 | tee "$LOG/profile_$name.$PRECISION.log" \
      | grep -E "mean:|median:|GPU Compute Time" || true
done

echo
echo "Engines in $ENG. Read $LOG/profile_*.log before trusting any speedup:"
echo "a decode engine whose profile is dominated by Reformat nodes is losing to"
echo "quantization overhead, not winning from it."
