#!/usr/bin/env bash
# Run the Alpamayo-R1 dissection studies on an H100 interactive session.
#
#   bash study/run_study.sh              # both studies: capture on the GPU, then plot
#   bash study/run_study.sh plot         # re-plot both from the saved files, no GPU
#   bash study/run_study.sh act          # the activation study alone
#   bash study/run_study.sh vision plot  # one study, one stage
#   STUDY_LOAD_CUDA=0 bash study/run_study.sh    # skip the CUDA module (see below)
#
# Uses the SAME alpamayo environment as the rest of the pipeline
# ($ALPAMAYO_ROOT/alpamayo/ar1_venv) -- nothing new is installed except
# matplotlib, and only if the venv does not already have it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
# First argument is a stage when it names one, otherwise it names a study.
case "${1:-}" in
    all|capture|plot) WHICH=all;       STAGE="$1" ;;
    "")               WHICH=all;       STAGE=all ;;
    *)                WHICH="$1";      STAGE="${2:-all}" ;;
esac
case "$WHICH" in
    all|vision|act) ;;
    *) echo "usage: run_study.sh [vision|act|all] [all|capture|plot]" >&2; exit 2 ;;
esac

# --- environment modules -------------------------------------------------
# ORDER MATTERS: modules first, venv last, so the venv's python wins on PATH.
#
# h100/a1_golden.sbatch records why the CUDA module is probably unnecessary:
# torch brings its own CUDA runtime, cuBLAS and cuDNN as nvidia-*-cu12 wheels
# and needs only the driver at runtime. Loading system CUDA puts a different
# minor version of those libraries ahead of the ones torch was built against.
# It is loaded here because this project's sessions have been run that way; set
# STUDY_LOAD_CUDA=0 if anything looks subtly wrong.
if [ "${STUDY_LOAD_CUDA:-1}" = "1" ] && command -v module >/dev/null 2>&1; then
    module use /opt/apps/nfs/modules/h100/rocky9.6/Core
    module load cuda/12.9.1
    echo "modules: $(module list 2>&1 | tr '\n' ' ' | sed 's/  */ /g')"
else
    echo "modules: not loaded (STUDY_LOAD_CUDA=${STUDY_LOAD_CUDA:-1}, module $(command -v module >/dev/null 2>&1 && echo present || echo absent))"
fi

# --- paths and the alpamayo venv -----------------------------------------
# shellcheck disable=SC1091
source "$REPO/env.sh"
VENV="$ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate"
if [ -f "$VENV" ]; then
    # shellcheck disable=SC1091
    source "$VENV"
else
    echo "no venv at $VENV -- run h100/setup_h100.sh first" >&2
    exit 1
fi

# This study reads the SAVED golden pixels, so it never streams the driving
# clip -- unlike a1/a3b, which is why env.sh leaves HF_HUB_OFFLINE=0. The two
# Qwen configs the reference code resolves at load time are already in HF_HOME,
# so going offline here turns a possible network hang on a compute node with no
# route out into a cache hit. Export HF_HUB_OFFLINE=0 to override.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE_OVERRIDE:-1}"

echo "model   : $ALPAMAYO_MODEL"
echo "golden  : $ALPAMAYO_ROOT/golden"
echo "hf cache: $HF_HOME  (offline=$HF_HUB_OFFLINE)"
[ -d "$ALPAMAYO_MODEL" ] || { echo "checkpoint missing at ALPAMAYO_MODEL" >&2; exit 1; }
[ -f "$ALPAMAYO_ROOT/golden/inputs.npz" ] || {
    echo "no golden/inputs.npz under ALPAMAYO_ROOT -- run h100/a1_golden.py first" >&2
    exit 1; }

echo "python  : $(which python)  $(python -V 2>&1)"
python - <<'PY'
import torch
print("torch   : %s, cuda %s, device %s" % (
    torch.__version__, torch.version.cuda,
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))
PY

# matplotlib is the only thing the reference venv does not already carry.
python -c "import matplotlib" 2>/dev/null || {
    echo "installing matplotlib into the alpamayo venv"
    python -m pip install --quiet matplotlib
}

# --- the study -----------------------------------------------------------
mkdir -p "$HERE/out"
if [ "$WHICH" = all ] || [ "$WHICH" = vision ]; then
    echo
    echo "=============== vision tower ==============="
    python "$HERE/vision_study.py" --stage "$STAGE" --out "$HERE/out"
fi
if [ "$WHICH" = all ] || [ "$WHICH" = act ]; then
    echo
    echo "=========== activations vs the fp16 wall ==========="
    python "$HERE/activation_study.py" --stage "$STAGE" --out "$HERE/out"
fi

echo
echo "figures in $HERE/out:"
ls -1 "$HERE/out" | sed 's/^/  /'
