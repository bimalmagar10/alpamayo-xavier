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
uv pip install --python "$(which python)" \
    onnx==1.17.0 onnxruntime-gpu==1.20.1 onnxscript==0.1.0 \
    nvidia-modelopt[torch]==0.23.0 polygraphy==0.49.9

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

Next:
  source $ALPAMAYO_REPO/env.sh
  source $ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate
  python $ALPAMAYO_REPO/h100/a1_golden.py --clips 64
MSG
