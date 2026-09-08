# Alpamayo-1 on Jetson AGX Xavier

Running NVIDIA's 10B driving VLA (`nvidia/Alpamayo-R1-10B`) on a JetPack 5.1.7
AGX Xavier 32 GB, and measuring where the time goes.

Full write-up and reasoning:
- **Feasibility study** (what the model is, why the reference stack can't run, latency
  budget, quantize-vs-distil) — https://claude.ai/code/artifact/1b9756d7-f078-4e53-95aa-d036efd8e33f
- **Runbook** (every command, both machines, the three verification gates)
  — https://claude.ai/code/artifact/275137a9-d5f4-435f-b71e-511619ed2edf

## Paths

Set once in `env.sh`; every script reads them from the environment.

| Variable | Default | What lives there |
|---|---|---|
| `ALPAMAYO_REPO` | `/mnt/DISCL/work/bthapama/alpamayo-xavier` | This repository — code only |
| `ALPAMAYO_MODEL` | `/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B` | The released bf16 checkpoint, read-only |
| `ALPAMAYO_ROOT` | `/mnt/SHARED-SCRATCH/bthapama/alpamayo-work` | Everything derived: golden, fp16 ckpt, ONNX (~60–90 GB) |
| `HF_HOME` | `/mnt/SHARED-SCRATCH/bthapama/hf-cache` | The two Qwen config repos the reference code resolves at load |
| `ALPAMAYO_WORK` | `/mnt/ssdhome/models/alpamayo` | On the Jetson: engines, fixtures, results |

```bash
source /mnt/DISCL/work/bthapama/alpamayo-xavier/env.sh
alpamayo_check_paths          # verifies mounts and that the checkpoint has all 5 shards
```

Derived artefacts default to scratch rather than into `/work`, because the ONNX
graphs and their external-data files run to roughly 3–4× the checkpoint. Export
`ALPAMAYO_ROOT` before sourcing if your scratch gets purged on a schedule you
care about. Note the shell subtlety: `export VAR=... ; source env.sh` works,
`VAR=... source env.sh` does not persist and `alpamayo_check_paths` will say so.

**Cluster caveat:** run `h100/setup_h100.sh` on a login or data-transfer node the
first time. Even though the Alpamayo weights are already local, `base_model.py`
still calls `Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")` and
`AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")` at load time. Once
`HF_HOME` is populated, `export HF_HUB_OFFLINE=1` and run the rest on a compute node.

## The split, in one line

**The H100 produces ONNX. The Xavier produces engines.** A TensorRT `.plan` is
locked to the GPU, the compute capability and the TensorRT version it was built
on, so nothing compiled on an H100 (sm_90, TRT 10) will load on a Xavier
(sm_72, TRT 8.5.2). Calibration, however, needs the reference stack the Xavier
cannot run — so scales are computed on the H100 and baked into the ONNX.

```
H100  (py3.12 / torch 2.8 / flash-attn / bf16)      Xavier (py3.8 / torch 2.1 / CUDA 11.4)
──────────────────────────────────────────────     ──────────────────────────────────────
a1  golden run + CoC trace stats                 ┐
a2  bf16 -> fp16 audit and cast                  │  copy ONNX + fixtures (~12-22 GB)
a3  verify layer math, export 4 ONNX graphs      ├──────────────►  build_engines.sh
a4  INT8 calibration -> Q/DQ in the graph        ┘                 verify.py
                                                                   run_alpamayo.py
```

## Layout

| Path | Runs on | Purpose |
|---|---|---|
| `bench/xavier_hw_probe.py` | Xavier | Measure real FP16 TFLOP/s and GB/s at Alpamayo's GEMM shapes |
| `bench/alpamayo_stage_bench.py` | Xavier | Per-stage latency from shapes alone — no weights, no port |
| `bench/roofline.py` | anywhere | Analytic prediction; feed it the probe's constants |
| `env.sh` | both | Central path config + `alpamayo_check_paths` |
| `h100/setup_h100.sh` | H100 | Reference venv, Qwen configs, checkpoint verification |
| `h100/arch.py` | both | Architecture constants read from the released checkpoint |
| `h100/a1_golden.py` | H100 | Golden tensors, trace-length stats, fixtures |
| `h100/a2_cast_fp16.py` | H100 | bf16→fp16 overflow audit and cast |
| `h100/graphs.py` | H100 | The four export wrappers |
| `h100/a3_export_onnx.py` | H100 | Verify against reference, then export |
| `h100/a4_quantize_int8.py` | H100 | Explicit INT8 Q/DQ insertion |
| `xavier/setup_xavier.sh` | Xavier | venv **with** `--system-site-packages` (TensorRT is an apt package) |
| `xavier/build_engines.sh` | Xavier | `trtexec` builds + `--dumpProfile` |
| `xavier/verify.py` | Xavier | Stage-by-stage numeric check against golden |
| `xavier/run_alpamayo.py` | Xavier | End-to-end inference with CUDA-event stage timing |

## Order of work

```bash
# --- cluster ---------------------------------------------------------------
source /mnt/DISCL/work/bthapama/alpamayo-xavier/env.sh
bash "$ALPAMAYO_REPO/h100/setup_h100.sh"              # login node, first time only
source "$ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate"

python "$ALPAMAYO_REPO/h100/a1_golden.py"  --clips 64
python "$ALPAMAYO_REPO/h100/a2_cast_fp16.py" --audit-only
python "$ALPAMAYO_REPO/h100/a2_cast_fp16.py"
python "$ALPAMAYO_REPO/h100/a3_export_onnx.py"

rsync -avh --partial --progress "$ALPAMAYO_ROOT/onnx/" \
      bimal@<xavier>:/mnt/ssdhome/models/alpamayo/onnx/
rsync -avh --partial --progress "$ALPAMAYO_ROOT/fixtures/" "$ALPAMAYO_ROOT/golden/" \
      bimal@<xavier>:/mnt/ssdhome/models/alpamayo/

# --- Jetson ----------------------------------------------------------------
bash xavier/setup_xavier.sh
PRECISION=fp16 bash xavier/build_engines.sh
python xavier/verify.py --precision fp16
python xavier/run_alpamayo.py --images "$ALPAMAYO_WORK/frames/*.jpg" --precision fp16 --repeat 5

# --- only once FP16 is numerically correct ---------------------------------
python "$ALPAMAYO_REPO/h100/a4_quantize_int8.py" --calib-clips 64   # on the cluster
PRECISION=int8 bash xavier/build_engines.sh                         # on the Jetson
python xavier/verify.py --precision int8
```

Run `bench/` on the Xavier in parallel with all of this — it needs no ONNX, no
engines and no cluster, and gives you measured per-stage latency in three days.

Never quantize before the FP16 path verifies. An INT8 engine that is wrong and an
FP16 port that is wrong look identical from the outside.
