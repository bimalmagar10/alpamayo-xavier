#!/usr/bin/env bash
#SBATCH --job-name=alpamayo-a1
#SBATCH --gres=gpu:1                 # one H100 is enough: 22 GB bf16 model
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
##SBATCH --partition=<your-partition>   # uncomment and set for your cluster
##SBATCH --account=<your-account>
set -euo pipefail

# --- environment modules -------------------------------------------------
# ORDER MATTERS: load modules FIRST, activate the venv LAST, so the venv's python
# and libraries win on PATH.
#
# You very likely do NOT need a CUDA module. torch 2.8.0 from PyPI bundles its own
# CUDA 12.8 runtime, cuBLAS and cuDNN 9.10 as nvidia-*-cu12 wheels inside the venv;
# at runtime it needs only the driver (libcuda.so.1), which the kernel provides.
# Loading system CUDA 12.9 puts a different minor version of cuBLAS/cuDNN on
# LD_LIBRARY_PATH ahead of the ones torch was built against, which is a source of
# subtle breakage rather than a fix.
#
# Run preflight.py once with and once without these two lines. If the "which CUDA
# libraries actually got loaded" section is unchanged, leave them commented out.
#
# module use /opt/apps/nfs/modules/h100/rocky9.6/Core
# module load cuda/12.9.1

source /mnt/DISCL/work/bthapama/alpamayo-xavier/env.sh
source "$ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate"

# Everything the job needs was cached on the login node by setup_h100.sh and
# a0_prefetch.py, so refuse to reach the network and fail fast if something is missing.
export HF_HUB_OFFLINE=1

# --- preflight -----------------------------------------------------------
# Cheap, and it turns "the job died after 40 minutes" into "the job died in 20
# seconds and told you why".
python "$ALPAMAYO_REPO/h100/preflight.py"

# --- the actual work -----------------------------------------------------
python "$ALPAMAYO_REPO/h100/a1_golden.py" --clips 64 --max-new-tokens 256

echo
echo "=== trace statistics (the number that sets the Xavier decode budget) ==="
cat "$ALPAMAYO_ROOT/golden/trace_stats.json"
