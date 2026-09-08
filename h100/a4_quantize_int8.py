#!/usr/bin/env python3
"""Stage A4 (H100) -- insert explicit INT8 Q/DQ nodes into the exported graphs.

Why this runs on the H100 and not on the Xavier: calibration needs real driving
clips pushed through the full bfloat16 model to observe activation ranges, which
requires the reference stack (Python 3.12 / torch 2.8 / flash-attn). The Xavier
can run none of that. What the Xavier receives is an ONNX graph with the scales
already baked in, so `trtexec` needs no calibration data at all.

Explicit quantization (Q/DQ in the graph) is preferred over TensorRT's implicit
calibrator here because it is reproducible, inspectable, and does not depend on
TRT 8.5's calibrator behaving identically to a modern one.

    python a4_quantize_int8.py --calib-clips 64

Note on expectations: INT8 helps decode a lot (it is weight-bandwidth-bound) and
vision/prefill somewhat. If TensorRT cannot fuse a Qwen3 block it will scatter
reformat nodes between Q and DQ pairs and INT8 can come out *slower*. Always
compare against the FP16 engine and read `--dumpProfile` before believing it.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

# Layers that stay in FP16 regardless. The LM head is a single huge GEMM whose
# INT8 error lands directly on token sampling; the norms and the action heads are
# tiny and numerically sensitive. Excluding them costs almost no bandwidth.
KEEP_FP16 = [
    "*lm_head*", "*norm*", "*action_in_proj*", "*action_out_proj*",
    "*/Softmax*", "*/Add_rope*",
]


def collect_calibration(graph: str, golden: Path, n: int) -> dict:
    """Load per-graph calibration inputs dumped by a1_golden.py --dump-calib."""
    path = golden / f"calib_{graph}.npz"
    if not path.exists():
        raise SystemExit(
            f"missing {path}\n"
            f"Run:  python a1_golden.py --clips {n} --dump-calib  first.\n"
            f"Calibration inputs must come from real clips -- random tensors "
            f"produce ranges that are wrong in both directions.")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    root = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")
    ap.add_argument("--onnx", default=os.path.join(root, "onnx"))
    ap.add_argument("--golden", default=os.path.join(root, "golden"))
    ap.add_argument("--calib-clips", type=int, default=64)
    ap.add_argument("--graphs", default="vision,prefill,decode,expert")
    ap.add_argument("--per-channel", action="store_true", default=True)
    args = ap.parse_args()

    from modelopt.onnx.quantization import quantize

    onnx_dir, golden = Path(args.onnx), Path(args.golden)
    manifest = {}

    for graph in (g.strip() for g in args.graphs.split(",") if g.strip()):
        src = onnx_dir / f"{graph}.onnx"
        dst = onnx_dir / f"{graph}.int8.onnx"
        if not src.exists():
            print(f"[skip] {src} not found")
            continue

        calib = collect_calibration(graph, golden, args.calib_clips)
        print(f"[{graph}] calibrating on {next(iter(calib.values())).shape[0]} samples")

        quantize(
            onnx_path=str(src),
            output_path=str(dst),
            calibration_data=calib,
            calibration_method="entropy",       # better than minmax for transformer activations
            quantize_mode="int8",
            op_types_to_exclude=["Softmax", "LayerNormalization"],
            nodes_to_exclude=KEEP_FP16,
            use_external_data_format=True,
        )
        size = sum(f.stat().st_size for f in onnx_dir.glob(dst.name + "*")) / 1e9
        manifest[graph] = dict(src=src.name, dst=dst.name, size_gb=round(size, 2))
        print(f"[{graph}] -> {dst.name}  ({size:.2f} GB)")

    (onnx_dir / "quant_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nCopy the .int8.onnx graphs (plus their .data files) to the Xavier.")
    print("Engines must still be built ON the Xavier -- see xavier/build_engines.sh.")


if __name__ == "__main__":
    main()
