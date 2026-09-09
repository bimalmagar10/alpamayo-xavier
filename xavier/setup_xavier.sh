#!/usr/bin/env bash
# Environment for running Alpamayo TensorRT engines on a Jetson AGX Xavier.
# JetPack 5.1.7 / L4T r35.6.5 / Python 3.8.10 / CUDA 11.4 / TensorRT 8.5.2 / sm_72.
set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/mnt/ssdhome/models}"
VENV="$MODELS_ROOT/envs/alpamayo-jp5"
WORK="${ALPAMAYO_WORK:-$MODELS_ROOT/alpamayo}"

# --- 0. refuse to run if the NVMe is not mounted -------------------------
SRC="$(findmnt -n -o SOURCE "$(dirname "$MODELS_ROOT")" 2>/dev/null || true)"
case "$SRC" in
  /dev/nvme*) echo "NVMe OK: $SRC" ;;
  *) echo "ERROR: $(dirname "$MODELS_ROOT") is not on NVMe (got '${SRC:-nothing}')." >&2
     echo "Refusing to write ~15 GB of engines onto the 28 GB eMMC." >&2; exit 1 ;;
esac

# --- 0b. inventory: show what exists, and what this script will and will not touch
echo
echo "== already on this device (LEFT UNTOUCHED) =="
for d in "$MODELS_ROOT"/envs/*/; do
    [ -d "$d" ] || continue
    case "$(basename "$d")" in
      alpamayo-jp5) : ;;   # ours, reported below
      *) printf "  venv        %-24s %s\n" "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)" ;;
    esac
done
for d in "$MODELS_ROOT"/checkpoints/*/; do
    [ -d "$d" ] || continue
    printf "  checkpoints %-24s %s\n" "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
done
[ -d "$MODELS_ROOT/src" ] && printf "  src         %-24s %s\n" "" "$(du -sh "$MODELS_ROOT/src" 2>/dev/null | cut -f1)"
echo
echo "== this script will CREATE (and nothing else) =="
echo "  $VENV"
echo "  $WORK/{onnx,engines,golden,results,logs}"
echo "  apt packages (additive; no removals, no autoremove)"
echo
echo "It contains no rm, no apt purge and no autoremove. Your CLIP venv,"
echo "checkpoints and sources are in different directories and are not read,"
echo "modified or deleted."
echo

mkdir -p "$WORK"/{onnx,engines,golden,results,logs}

# --- 1. build deps -------------------------------------------------------
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libopenblas-dev libjpeg-dev zlib1g-dev
# TensorRT's Python bindings ship as an apt dist-package, not a wheel:
sudo apt-get install -y python3-libnvinfer python3-libnvinfer-dev
sudo apt-get clean

# --- 2. venv, WITH system site-packages ----------------------------------
# This is the step everyone gets wrong. `import tensorrt` resolves to
# /usr/lib/python3.8/dist-packages/tensorrt, which a plain venv hides. There is
# no pip-installable TensorRT 8.5 wheel for aarch64/cp38, so the venv must
# inherit system packages or nothing will import.
if [ -d "$VENV" ]; then
    echo "reusing the existing venv at $VENV (nothing removed)"
else
    python3 -m venv --system-site-packages "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export PIP_CACHE_DIR="$MODELS_ROOT/cache/pip"
export TMPDIR="$MODELS_ROOT/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

python -m pip install --no-cache-dir pip==23.3.2 setuptools==68.2.2 wheel==0.41.3
python -m pip install --no-cache-dir numpy==1.23.5 Pillow==9.5.0 tokenizers==0.13.3

# torch is used only for GPU buffer management and the embedding gather --
# the model itself runs entirely inside TensorRT.
python -m pip install --no-cache-dir \
  'https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl'

# --- 3. verify -----------------------------------------------------------
python - <<'PY'
import sys, torch, tensorrt as trt
cap = torch.cuda.get_device_capability()
print("python     :", sys.version.split()[0])
print("torch      :", torch.__version__, "cuda", torch.version.cuda)
print("device     :", torch.cuda.get_device_name(0), f"sm_{cap[0]}{cap[1]}")
print("tensorrt   :", trt.__version__)
free, total = torch.cuda.mem_get_info()
print("gpu memory : %.1f GiB free / %.1f GiB" % (free / 2**30, total / 2**30))
assert cap == (7, 2), "expected sm_72 (AGX Xavier)"
assert trt.__version__.startswith("8.5"), "expected TensorRT 8.5.x from JetPack 5.1.x"
print("\nOK")
PY

echo
echo "Xavier environment ready."
echo "  venv      : $VENV"
echo "  workspace : $WORK"
echo "Next: copy the ONNX graphs from the H100 into $WORK/onnx, then run build_engines.sh"
