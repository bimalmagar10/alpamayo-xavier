"""Every measured number this analysis is allowed to plot, with its provenance.

One source of truth, for one reason: figures that quote the same quantity must
quote the same value, and a number whose origin nobody can name does not belong
in a paper. Each entry carries `source` -- the command, file or log line it came
from -- and `kind`:

    measured    read off an instrument on the machine it describes
    derived     arithmetic on measured values, with the arithmetic stated
    estimated   a model (FLOP counts, roofline) -- must be labelled in the figure
    spec        a datasheet or standard constant

If a value is not in here, a figure must compute it from data at run time
rather than hardcode it.
"""
from __future__ import annotations

FP16_MAX = 65504.0
# A naive RMSNorm squares its input before reducing, so x itself must stay under
# sqrt(65504) = 255.94 for x**2 to be representable at all. This single constant
# is the whole reason the language model could not be run in fp16 unmodified.
FP16_SQUARE_SAFE = FP16_MAX ** 0.5

BOARD = {
    "name": "Jetson AGX Xavier 32 GB",
    "arch": "sm_72 (Volta)",
    "dram_gbps": {"value": 136.5, "kind": "spec",
                  "source": "256-bit LPDDR4x @ 2133 MHz"},
    "fp16_tflops": {"value": 11.3, "kind": "spec",
                    "source": "512 CUDA + 64 tensor cores @ 1377 MHz, MAXN"},
    "jetpack": "5.1.7", "tensorrt": "8.5.2", "cuda": "11.4",
    "torch": "2.1.0a0", "python": "3.8",
}

MODEL = {
    "params_b": 11.076, "layers": 36, "hidden": 4096, "q_heads": 32,
    "kv_heads": 8, "head_dim": 128, "ffn": 12288, "vocab": 155697,
    "prefill_tokens": 3006, "cache_slots": 3584, "visual_tokens": 2880,
    "vit_patches": 11520, "waypoints": 64, "dt_s": 0.1, "flow_steps": 10,
    "lm_weight_gb": 15.17, "vision_weight_gb": 1.15, "expert_weight_gb": 4.56,
}

# ---------------------------------------------------------------------------
# the four exported graphs. params_b x 2 bytes reproduces the fp16 weight size,
# and the four plus the embedding fixture reproduce the checkpoint's 11.076 B.
# ---------------------------------------------------------------------------
GRAPHS = [
    {"name": "vision", "params_b": 0.576, "gb": 1.15, "calls": 1,
     "weights": "vision tower", "shape": "11\u2009520 \u00d7 1\u2009536 patches"},
    {"name": "prefill", "params_b": 7.583, "gb": 15.17, "calls": 1,
     "weights": "language model", "shape": "3\u2009006 positions"},
    {"name": "decode", "params_b": 7.583, "gb": 15.17, "calls": 21,
     "weights": "language model", "shape": "1 position + 3\u2009584 cache"},
    {"name": "expert", "params_b": 2.279, "gb": 4.56, "calls": 10,
     "weights": "action expert", "shape": "64 queries + 3\u2009584 cache"},
]
EMBEDDING = {"params_b": 0.638, "gb": 1.28,
             "note": "shipped as a fixture, never a graph"}
# Proven, not assumed: analysis/check_export_weights.py compared every byte.
EXPORT_IDENTITY = {
    "tensors": 326,
    "bytes": 15_168_200_704,
    "differing": 0,
    "result": "326 tensors, every byte identical",
    "kind": "measured",
    "source": "analysis/check_export_weights.py, full byte comparison, 2026-09-15",
}
GRAPHS_SOURCE = ("h100/a3_export_onnx.py; sizes are the .onnx.data files on disk, "
                 "call counts from run_alpamayo.py")

# ---------------------------------------------------------------------------
# verification: xavier/verify.py, run on the board, 2026-09-15
# ---------------------------------------------------------------------------
VERIFY = [
    {"stage": "preprocessing", "against": "golden pixels", "cos": 1.000000, "colour": "muted"},
    {"stage": "vision tower", "against": "golden embeds", "cos": 0.999762, "colour": "vision"},
    {"stage": "prefill (13 engines)", "against": "golden hidden", "cos": 0.999642, "colour": "prefill"},
    {"stage": "decode (PyTorch)", "against": "prefill, same position", "cos": 0.999998, "colour": "decode"},
    {"stage": "decode (PyTorch)", "against": "fp32 ONNX Runtime", "cos": 0.999982, "colour": "decode"},
    {"stage": "action expert", "against": "fp32 ONNX Runtime", "cos": 0.999999, "colour": "expert"},
]
VERIFY_SOURCE = "xavier/verify.py, fp16, Xavier, 2026-09-15 -- ALL CHECKED STAGES PASS"
ACCEPT_THRESHOLD = 0.999   # what verify.py treats as a pass

# ---------------------------------------------------------------------------
# defects this project actually shipped and then caught, and the controls that
# were run to prove the checks have teeth
# ---------------------------------------------------------------------------
DEFECTS = [
    {"name": "fp16 RMSNorm overflow",
     "cos": 0.000000,
     "stage": "prefill",
     "note": "x**2 reaches 6.8e8; fp16 saturates at 65 504, then NaN",
     "visible_without_reference": False,
     "source": "xavier/verify.py before h100/a3c_decompose_layernorm.py grew rmsnorm_fp16safe()"},
    {"name": "DeepStack at the input, not layers 0-2",
     "cos": None,
     "stage": "prefill",
     "note": "model still wrote fluent reasoning; trajectory quietly worse",
     "visible_without_reference": False,
     "source": "caught by comparison with the golden hidden state"},
    {"name": "trajectory integrated from v0 = 0",
     "cos": None,
     "stage": "postprocess",
     "note": "smooth, well-formed unicycle path, 10x too short",
     "visible_without_reference": True,
     "source": "run_alpamayo.py output 0.7 / 2.7 / 5.4 m against a 57.1 m reference"},
]

# Deliberate mis-assignments, run to show the check separates right from wrong.
CONTROLS = {"label": "weight roles deliberately swapped",
            "cos_range": (0.23, 0.96),
            "source": "h100/a7_weight_map.py control experiment"}

# ---------------------------------------------------------------------------
# trajectory endpoints, metres ahead / metres lateral
# ---------------------------------------------------------------------------
TRAJECTORY = {
    "golden": {"xy": (57.1, 1.02), "kind": "measured",
               "source": "golden/activations.npz pred_xyz, H100"},
    "fixed": {"xy": [(50.4, 0.5), (51.1, 1.1), (55.9, 0.7)], "kind": "measured",
              "source": "run_alpamayo.py --repeat 3, Xavier, 2026-09-15"},
    "v0_bug": {"dist": [0.7, 2.7, 5.4], "kind": "measured",
               "source": "run_alpamayo.py before postprocess took v0 from meta.json"},
}

# ---------------------------------------------------------------------------
# latency, milliseconds, CUDA-event timed, run of 2026-09-15
# ---------------------------------------------------------------------------
LATENCY = {
    "source": "run_alpamayo.py --precision fp16 --repeat 3 --residency auto",
    "runs": [
        {"run": 1, "vision": 2734.0, "embed": 63.8, "prefill": 10614.0,
         "decode": 10929.6, "tokens": 21, "expert": 1784.2,
         "compute": 26125.7, "load": 122656.6, "cpu": 1849.3, "peak_mib": 15764},
        {"run": 2, "vision": 2730.9, "embed": 169.0, "prefill": 10710.6,
         "decode": 9270.2, "tokens": 18, "expert": 1813.8,
         "compute": 24694.4, "load": 67256.9, "cpu": 1767.1, "peak_mib": 18124},
        {"run": 3, "vision": 2736.7, "embed": 51.0, "prefill": 10749.3,
         "decode": 10803.5, "tokens": 21, "expert": 1819.3,
         "compute": 26159.8, "load": 62084.5, "cpu": 1821.0, "peak_mib": 18124},
    ],
}

DECODE_PER_TOKEN_MS = {
    "PyTorch reference": {"value": 318, "kind": "measured",
                          "source": "baseline run on the board"},
    "TensorRT engines": {"value": 350, "kind": "measured",
                         "source": "measured before the engines were set aside"},
    "this implementation": {"value": 515, "kind": "derived",
                            "source": "10929.6 ms / 21 tokens, run 1"},
}

# ---------------------------------------------------------------------------
# memory. The free-memory counter on a unified-memory board is not reliable --
# recorded here with its spread so no figure can quote a single value as fact.
# ---------------------------------------------------------------------------
MEMORY = {
    "free_drop_gb": {"vision": [2.6, 1.5, 1.1], "expert": [8.4, 5.0, 4.6],
                     "decode_torch": [10.9],
                     "kind": "measured, unreliable",
                     "source": "cudaMemGetInfo deltas; page cache counts as free"},
    "torch_peak_mib": {"value": 18124, "kind": "measured",
                       "source": "torch.cuda.max_memory_allocated()"},
    "probe": {"engine_ceiling_gb": 13.9, "plain_cuda_gb": 27.0,
              "kind": "measured",
              "source": "xavier/mem_probe.py -- engines fail long before plain allocations"},
}
