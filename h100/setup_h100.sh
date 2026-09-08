#!/usr/bin/env bash
# Environment for the H100 side of the Alpamayo -> Xavier pipeline.
# Everything needing Python 3.12 / torch 2.8 / flash-attn / bfloat16 happens here.
#
# Run this on a LOGIN or DATA-TRANSFER node the first time: it fetches two Qwen
# config repos from huggingface.co, and compute nodes usually have no route out.
# Once HF_HOME is populated you can export HF_HUB_OFFLINE=1 and run the rest of
# the pipeline on a compute node.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

echo "== paths =="
alpamayo_check_paths || { echo; echo "Fix the paths above (or export them) and re-run." >&2; exit 1; }
echo

mkdir -p "$ALPAMAYO_ROOT"/{golden,fixtures,onnx,calib,logs,ckpt}

# --- 1. reference implementation, exactly as NVIDIA pins it ---------------
cd "$ALPAMAYO_ROOT"
[ -d alpamayo ] || git clone https://github.com/NVlabs/alpamayo.git
cd alpamayo
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ar1_venv --python 3.12
# shellcheck disable=SC1091
source ar1_venv/bin/activate
uv sync --active

# --- 2. export-side extras the reference repo does not need ---------------
# Deliberately not version-pinned. ModelOpt already pins onnx, onnxruntime,
# onnxscript and polygraphy per platform -- including choosing CPU onnxruntime on
# aarch64, where no onnxruntime-gpu wheel exists. Pinning them by hand fights that
# and breaks the resolver.
PY_BIN="$(which python)"

# Tier 1: needed by a3_export_onnx.py. Small, always resolvable.
uv pip install --python "$PY_BIN" onnx

# Tier 2: ONLY a4_quantize_int8.py needs this. A failure here must not block
# stages a1-a3, which are most of the pipeline and produce the FP16 path.
MODELOPT_OK=1
if [ "$(uname -m)" = "aarch64" ]; then
    cat <<'NOTE'
NOTE: aarch64 detected (Grace Hopper?). ModelOpt has no onnxruntime-gpu wheel for
      this architecture and will install CPU onnxruntime, so INT8 calibration in a4
      will be slow. Stages a1-a3 and the whole FP16 path are unaffected.
NOTE
fi
if ! uv pip install --python "$PY_BIN" "nvidia-modelopt[onnx]>=0.46"; then
    MODELOPT_OK=0
    cat <<'WARN'

WARNING: could not install nvidia-modelopt[onnx].
         This blocks ONLY a4 (INT8 quantization). Stages a1 (golden), a2 (fp16
         cast) and a3 (ONNX export) do not import it, and the FP16 engines are
         what you want working first anyway. Continuing.
WARN
fi

# ModelOpt requires torch>=2.8 while the reference repo pins torch==2.8.0. That is
# compatible, but `uv pip install` does not respect the project lock, so confirm
# nothing was silently upgraded out from under flash-attn.
python - <<'TORCHCHK'
import sys
import torch
v = torch.__version__.split("+")[0]
print("torch after extras:", torch.__version__)
if v != "2.8.0":
    sys.stderr.write(
        "\nWARNING: the reference repo pins torch==2.8.0 but %s is installed.\n"
        "         flash-attn is built against 2.8.0 -- re-run `uv sync --active`.\n" % v)
TORCHCHK

# onnxruntime-gpu falls back to CPU *silently* when it cannot load CUDA/cuDNN.
# On a 14 GB prefill graph that turns a4 calibration from minutes into hours, and
# nothing in the output says why -- so check the provider list now, not then.
if [ "$MODELOPT_OK" = 1 ]; then
python - <<'ORTCHK'
import sys
try:
    import onnxruntime as ort
except ImportError:
    sys.exit(0)
try:
    ort.preload_dlls()          # >=1.21: resolves CUDA/cuDNN from the nvidia-* wheels
except Exception:
    pass
provs = ort.get_available_providers()
print("onnxruntime:", ort.__version__)
print("  providers:", ", ".join(provs))
if "CUDAExecutionProvider" not in provs:
    sys.stderr.write(
        "\nWARNING: onnxruntime has no CUDAExecutionProvider -- a4 calibration would\n"
        "         run on CPU and take hours. Usually a cuDNN/CUDA load failure. Try:\n"
        "           export LD_LIBRARY_PATH=$(python -c \"import nvidia,os;"
        "print(':'.join(os.path.join(p,'lib') for p in __import__('glob').glob("
        "os.path.dirname(nvidia.__file__)+'/*')))\"):$LD_LIBRARY_PATH\n"
        "         Stages a1-a3 are unaffected.\n")
ORTCHK
fi

# --- 3. the two Qwen repos the reference code resolves at load time -------
# Alpamayo's weights are already on SHARED-SCRATCH, but base_model.py still calls
# Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct") for the architecture
# and AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct") for the image
# pipeline. Config and tokenizer only -- no weights, a few MB each.
hf download Qwen/Qwen3-VL-8B-Instruct --exclude "*.safetensors" "*.bin" "*.pth" >/dev/null
hf download Qwen/Qwen3-VL-2B-Instruct --exclude "*.safetensors" "*.bin" "*.pth" >/dev/null
echo "cached Qwen configs into $HF_HOME"

# --- 4. verify the checkpoint and the stack ------------------------------
python - <<'PY'
import os, json, glob, torch
model_dir = os.environ["ALPAMAYO_MODEL"]
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
n_params = idx["metadata"]["total_parameters"]
n_bytes = idx["metadata"]["total_size"]
shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
on_disk = sum(os.path.getsize(s) for s in shards)

print("checkpoint :", model_dir)
print("  shards   :", len(shards))
print("  params   : {:,}".format(n_params))
print("  expected : {:.2f} GB".format(n_bytes / 1e9))
print("  on disk  : {:.2f} GB".format(on_disk / 1e9))
assert len(shards) == 5, "expected 5 shards"
assert abs(on_disk - n_bytes) / n_bytes < 0.01, "shard sizes do not match the index -- incomplete download?"
assert n_params == 11_078_526_194, "unexpected parameter count"

print("torch      :", torch.__version__, "cuda", torch.version.cuda)
print("device     :", torch.cuda.get_device_name(0),
      "sm_%d%d" % torch.cuda.get_device_capability())
print("bf16       :", torch.cuda.is_bf16_supported())
print("\nOK")
PY

cat <<MSG

H100 environment ready.
  repo        : $ALPAMAYO_REPO
  checkpoint  : $ALPAMAYO_MODEL   (read-only)
  work / out  : $ALPAMAYO_ROOT
  reference   : $ALPAMAYO_ROOT/alpamayo (venv: ar1_venv)
  hf cache    : $HF_HOME
  modelopt    : $([ "$MODELOPT_OK" = 1 ] && echo "installed (a4 ready)" || echo "NOT installed -- a4 blocked, a1-a3 fine")

Next:
  source $ALPAMAYO_REPO/env.sh
  source $ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate
  python $ALPAMAYO_REPO/h100/a1_golden.py --clips 64
MSG
