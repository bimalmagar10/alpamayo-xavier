#!/usr/bin/env bash
# Runbook step B2 -- build the TensorRT engines ON the Xavier.
#
#   PRECISION=fp16 bash xavier/build_engines.sh                 every graph in onnx/
#   PRECISION=fp16 bash xavier/build_engines.sh vision expert   just these
#
# Engines are not portable. A TensorRT plan file is locked to the compute
# capability, the TensorRT version and the exact GPU it was built for, so an
# engine built on the H100 (sm_90, TRT 10.x) cannot load here (sm_72, TRT 8.5.2).
# The H100 produces ONNX; only this machine can produce .plan files.
#
# Flags are limited to what trtexec 8.5 accepts. --builderOptimizationLevel only
# arrived in 8.6, and 8.5 exits on it as an unknown option.
#
# Expect 30-90 minutes in total; prefill and decode (15 GB each) are the slow ones.
set -euo pipefail

WORK="${ALPAMAYO_WORK:-/mnt/ssdhome/models/alpamayo}"
ONNX="$WORK/onnx"
ENG="$WORK/engines"
LOG="$WORK/logs"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
PRECISION="${PRECISION:-fp16}"      # fp16 | int8
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(vision expert prefill decode)   # smallest first

mkdir -p "$ENG" "$LOG"
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC" >&2; exit 1; }
case "$PRECISION" in fp16|int8) ;; *) echo "PRECISION must be fp16 or int8" >&2; exit 2 ;; esac

echo "TensorRT : $(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null || echo unknown)"
echo "disk     : $(df -h "$WORK" | awk 'NR==2 {print $4 " free on " $6}')"
free -g | sed 's/^/  /'

# Lock clocks so tactic timing is not measured against a throttling board --
# a wandering clock makes TensorRT pick genuinely worse kernels.
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

build() {
  local name="$1"
  local src="$ONNX/$name.onnx"
  local plan="$ENG/$name.$PRECISION.plan"

  if [ "$PRECISION" = "int8" ]; then
    # int8 without a4's calibrated Q/DQ graph makes trtexec invent scales: the
    # engine builds and runs, and everything it outputs is wrong.
    if [ ! -f "$ONNX/$name.int8.onnx" ]; then
      echo "[skip] $name: no $name.int8.onnx from h100/a4_quantize_int8.py -- refusing uncalibrated int8"
      return 0
    fi
    src="$ONNX/$name.int8.onnx"
  fi

  if [ -f "$plan" ]; then echo "[have] $plan"; return 0; fi
  if [ ! -f "$src" ]; then echo "[skip] $(basename "$src") not in $ONNX"; return 0; fi
  if [ "$PRECISION" = "fp16" ] && [ ! -f "$src.data" ]; then
    echo "[skip] $name: $(basename "$src").data missing -- the weights must sit next to the .onnx"
    return 0
  fi
  if [ "$name" = "prefill" ] || [ "$name" = "decode" ]; then
    local swap_gb; swap_gb=$(free -g | awk '/^Swap/ {print $2}')
    if [ "${swap_gb:-0}" -lt 16 ]; then
      echo "[warn] $name holds 15 GB of weights and only ${swap_gb} GB swap is configured;"
      echo "       the build can run out of memory. See the swap step in the runbook."
    fi
  fi

  echo; echo "== building $name from $(basename "$src")  ($(date +%H:%M)) =="
  echo "   progress: tail -f $LOG/build_$name.$PRECISION.log"
  # --precisionConstraints=prefer: with plain --fp16, TensorRT may run layers the
  # graph explicitly casts to fp32 in fp16 anyway. "prefer" makes it honour those
  # casts where a kernel exists -- the LayerNorm statistics from a3c, the RMSNorm
  # math, and the expert's fp32 action_in_proj (sin/cos of arguments up to 2*pi*100).
  local flags=(--onnx="$src" --saveEngine="$plan.tmp"
               --memPoolSize=workspace:"$WORKSPACE_MB"
               --timingCacheFile="$ENG/timing.cache"
               --precisionConstraints=prefer
               --verbose)
  # fp16 stays on even for int8 builds: it is the fallback precision for any
  # layer TensorRT refuses to run in int8, and without it those fall back to fp32.
  flags+=(--fp16)
  [ "$PRECISION" = "int8" ] && flags+=(--int8)

  # GNU time adds peak memory to the log when installed (apt install time); the
  # build does not depend on it.
  local timer=()
  [ -x /usr/bin/time ] && timer=(/usr/bin/time -v -o "$LOG/build_$name.$PRECISION.time")
  local t0=$SECONDS
  if ! ${timer[@]+"${timer[@]}"} "$TRTEXEC" "${flags[@]}" > "$LOG/build_$name.$PRECISION.log" 2>&1; then
    echo "[FAIL] $name -- TensorRT errors from $LOG/build_$name.$PRECISION.log:" >&2
    grep -E "\[E\]|ERROR|Assertion|Unsupported|not supported" \
        "$LOG/build_$name.$PRECISION.log" | head -n 20 >&2 || true
    echo "   ... last lines:" >&2
    tail -n 12 "$LOG/build_$name.$PRECISION.log" >&2
    rm -f "$plan.tmp"
    return 1
  fi
  mv "$plan.tmp" "$plan"
  echo "   built in $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s"
  grep -E "Elapsed \(wall clock\)|Maximum resident set size" \
      "$LOG/build_$name.$PRECISION.time" 2>/dev/null | sed 's/^[[:space:]]*/   /' || true
  echo "[done] $plan  ($(du -h "$plan" | cut -f1))"
}

for n in "${NAMES[@]}"; do build "$n"; done

echo
echo "== GPU compute time per engine call (random inputs, trtexec) =="
for n in "${NAMES[@]}"; do
  plan="$ENG/$n.$PRECISION.plan"
  [ -f "$plan" ] || continue
  if "$TRTEXEC" --loadEngine="$plan" --iterations=10 --avgRuns=10 --noDataTransfers \
        --dumpProfile --separateProfileRun > "$LOG/profile_$n.$PRECISION.log" 2>&1; then
    printf "  %-8s %s\n" "$n" "$(grep -E "GPU Compute Time" "$LOG/profile_$n.$PRECISION.log" | tail -1 | sed 's/.*GPU Compute Time: //')"
  else
    echo "  $n: profiling failed, see $LOG/profile_$n.$PRECISION.log"
  fi
done

echo
echo "Engines in $ENG. Next: python xavier/verify.py --precision $PRECISION"
