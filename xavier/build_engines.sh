#!/usr/bin/env bash
# Runbook step B2 -- build the TensorRT engines ON the Xavier.
#
#   PRECISION=fp16 bash xavier/build_engines.sh                  everything in onnx/
#   PRECISION=fp16 bash xavier/build_engines.sh vision expert    just these graphs
#   PRECISION=fp16 bash xavier/build_engines.sh decode.p03       one piece
#
# A graph that h100/a3d_split_graphs.py cut into pieces (onnx/<graph>.pNN.onnx)
# is built piece by piece: one engine per piece, all reading <graph>.onnx.data.
# A whole 4.6-15 GB graph cannot be built here -- TensorRT asks for one GPU block
# larger than all of its weights -- while a ~1.5 GB piece can.
#
# Engines are not portable. A plan is locked to the compute capability, the
# TensorRT version and the GPU it was built on, so only this machine can make them.
# Flags are limited to what trtexec 8.5 accepts (no --builderOptimizationLevel).
set -euo pipefail

WORK="${ALPAMAYO_WORK:-/mnt/ssdhome/models/alpamayo}"
ONNX="$WORK/onnx"
ENG="$WORK/engines"
LOG="$WORK/logs"
TRTEXEC=/usr/src/tensorrt/bin/trtexec
PRECISION="${PRECISION:-fp16}"      # fp16 | int8
WORKSPACE_MB="${WORKSPACE_MB:-2048}"
# Each candidate kernel is timed AVG_TIMING times (trtexec 8.5 default 8). Prefill
# pieces run 3,006-token attention per timing run, so 2 cuts their build time
# several-fold at the cost of slightly noisier kernel choices.
AVG_TIMING="${AVG_TIMING:-8}"
NAMES=("$@")
[ ${#NAMES[@]} -eq 0 ] && NAMES=(vision expert decode prefill)   # smallest first

mkdir -p "$ENG" "$LOG"
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC" >&2; exit 1; }
case "$PRECISION" in fp16|int8) ;; *) echo "PRECISION must be fp16 or int8" >&2; exit 2 ;; esac
# the runner reads the piece layout from engines/, so it survives deleting onnx/
[ -f "$ONNX/pieces.json" ] && cp "$ONNX/pieces.json" "$ENG/pieces.json"

# graph name -> its pieces if it was split, else itself
TARGETS=()
for n in "${NAMES[@]}"; do
  found=0
  for f in "$ONNX/$n".p[0-9][0-9].onnx; do
    [ -e "$f" ] || continue
    TARGETS+=("$(basename "$f" .onnx)"); found=1
  done
  [ "$found" = 1 ] || TARGETS+=("$n")
done

echo "TensorRT : $(dpkg-query -W -f='${Version}' tensorrt 2>/dev/null || echo unknown)"
echo "disk     : $(df -h "$WORK" | awk 'NR==2 {print $4 " free on " $6}')"
echo "targets  : ${TARGETS[*]}"
free -g | sed 's/^/  /'

# Lock clocks so tactic timing is not measured against a throttling board --
# a wandering clock makes TensorRT pick genuinely worse kernels.
sudo nvpmodel -m 0 || true
sudo jetson_clocks || true

build() {
  local name="$1"
  local src="$ONNX/$name.onnx"
  local plan="$ENG/$name.$PRECISION.plan"
  local data="$ONNX/${name%.p[0-9][0-9]}.onnx.data"      # pieces read their graph's weights

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
  if [ "$PRECISION" = "fp16" ] && [ ! -f "$data" ]; then
    echo "[skip] $name: $(basename "$data") missing -- the weights must sit in $ONNX"
    return 0
  fi

  # CPU and GPU share one memory pool on the Xavier. Page cache left over from
  # reading the previous .data file is not always handed back fast enough when
  # TensorRT asks for a multi-GB GPU block, so drop it before every build.
  sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null || true
  echo; echo "== building $name  ($(date +%H:%M)) =="
  echo "   memory before build: $(free -g | awk '/^Mem/ {print $7 " GB available"}'), $(free -g | awk '/^Swap/ {print $4 " GB swap free"}')"
  echo "   progress: tail -f $LOG/build_$name.$PRECISION.log"
  # --precisionConstraints=prefer: honour the graph's explicit fp32 casts where a
  # kernel exists (LayerNorm statistics, RMSNorm, the expert's action_in_proj).
  local flags=(--onnx="$src" --saveEngine="$plan.tmp"
               --memPoolSize=workspace:"$WORKSPACE_MB"
               --timingCacheFile="$ENG/timing.cache"
               --precisionConstraints=prefer
               --avgTiming="$AVG_TIMING"
               --verbose)
  # fp16 stays on even for int8 builds: it is the fallback precision for any
  # layer TensorRT refuses to run in int8, and without it those fall back to fp32.
  flags+=(--fp16)
  [ "$PRECISION" = "int8" ] && flags+=(--int8)

  # GNU time adds peak memory when installed (apt install time); nothing depends on it.
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
  grep -E "Maximum resident set size" \
      "$LOG/build_$name.$PRECISION.time" 2>/dev/null | sed 's/^[[:space:]]*/   /' || true
  echo "[done] $plan  ($(du -h "$plan" | cut -f1))"
}

for n in "${TARGETS[@]}"; do build "$n"; done

echo
echo "== GPU compute time per engine call (random inputs, trtexec) =="
declare -A SUM
for n in "${TARGETS[@]}"; do
  plan="$ENG/$n.$PRECISION.plan"
  [ -f "$plan" ] || continue
  if "$TRTEXEC" --loadEngine="$plan" --iterations=10 --avgRuns=10 --noDataTransfers \
        --dumpProfile --separateProfileRun > "$LOG/profile_$n.$PRECISION.log" 2>&1; then
    # "GPU Compute Time: min = .., mean = .., median = .." is per call;
    # "Total GPU Compute Time" is the sum over every timing run -- not the number to quote.
    line=$(grep -E "GPU Compute Time: min" "$LOG/profile_$n.$PRECISION.log" | tail -1)
    med=$(echo "$line" | sed -E 's/.*median = ([0-9.]+) ms.*/\1/')
    printf "  %-12s median %9s ms   mean %9s ms\n" "$n" "$med" \
        "$(echo "$line" | sed -E 's/.*mean = ([0-9.]+) ms.*/\1/')"
    g="${n%.p[0-9][0-9]}"
    SUM[$g]=$(awk -v a="${SUM[$g]:-0}" -v b="$med" 'BEGIN {printf "%.3f", a + b}')
  else
    echo "  $n: profiling failed, see $LOG/profile_$n.$PRECISION.log"
  fi
done
for g in "${!SUM[@]}"; do
  printf "  %-12s %9s ms per call, all pieces together\n" "$g total" "${SUM[$g]}"
done

echo
echo "Engines in $ENG. Next: python xavier/verify.py --precision $PRECISION"
