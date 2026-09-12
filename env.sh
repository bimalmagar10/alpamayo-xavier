# Site configuration for the Alpamayo -> Xavier pipeline.
#   source env.sh
# Every path below can be overridden by exporting it beforehand.

# --- cluster (H100 side) --------------------------------------------------
# Code. Persistent, backed up, small.
export ALPAMAYO_REPO="${ALPAMAYO_REPO:-/mnt/DISCL/work/bthapama/alpamayo-xavier}"

# The released bfloat16 checkpoint, already on disk. Read-only as far as this
# pipeline is concerned -- nothing here ever writes into it.
export ALPAMAYO_MODEL="${ALPAMAYO_MODEL:-/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B}"

# Everything derived: golden tensors, the fp16 checkpoint, ONNX graphs and their
# external-data files. This runs to roughly 60-90 GB, which is why it defaults to
# scratch next to the model rather than into the repo on /work.
export ALPAMAYO_ROOT="${ALPAMAYO_ROOT:-/mnt/SHARED-SCRATCH/bthapama/alpamayo-work}"

# Hugging Face cache. The reference code resolves Qwen/Qwen3-VL-8B-Instruct and
# Qwen/Qwen3-VL-2B-Instruct at load time even when the Alpamayo weights are local,
# so this must point somewhere shared and pre-populated -- compute nodes usually
# have no route to huggingface.co.
export HF_HOME="${HF_HOME:-/mnt/SHARED-SCRATCH/bthapama/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # keep 0: the dataset streams camera video

# --- Jetson AGX Xavier side ----------------------------------------------
# Used by xavier/*.sh and xavier/run_alpamayo.py on the device itself.
export MODELS_ROOT="${MODELS_ROOT:-/mnt/ssdhome/models}"
export ALPAMAYO_WORK="${ALPAMAYO_WORK:-$MODELS_ROOT/alpamayo}"

# --- derived --------------------------------------------------------------
export PYTHONPATH="$ALPAMAYO_REPO/h100:${PYTHONPATH:-}"

alpamayo_check_paths() {
    local bad=0 v val n
    for v in ALPAMAYO_REPO ALPAMAYO_MODEL ALPAMAYO_ROOT HF_HOME; do
        eval "val=\${$v:-}"
        if [ -z "$val" ]; then
            echo "UNSET    $v -- did you 'source env.sh' rather than 'VAR=x source env.sh'?" >&2
            bad=1
            continue
        fi
        case "$v" in
            ALPAMAYO_REPO|ALPAMAYO_MODEL)
                if [ -d "$val" ]; then echo "ok       $v = $val"
                else echo "MISSING  $v = $val" >&2; bad=1; fi ;;
            *)
                if mkdir -p "$val" 2>/dev/null; then echo "ok       $v = $val"
                else echo "CANNOT CREATE  $v = $val" >&2; bad=1; fi ;;
        esac
    done
    if [ -n "${ALPAMAYO_MODEL:-}" ] && [ -d "${ALPAMAYO_MODEL:-}" ]; then
        n=$(ls "$ALPAMAYO_MODEL"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n" = "5" ]; then
            echo "ok       checkpoint has 5 safetensors shards"
        else
            echo "WARNING  expected 5 safetensors shards in ALPAMAYO_MODEL, found $n" >&2
            bad=1
        fi
    fi
    return $bad
}
