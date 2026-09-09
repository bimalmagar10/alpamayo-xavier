#!/usr/bin/env python3
"""Stage A3 (H100) -- verify the export wrappers against the reference model,
then write four static-shape ONNX graphs for the Xavier to compile.

Verification is not optional and runs first. `graphs.decoder_layer` re-expresses a
Qwen3 block in ONNX-friendly ops; if the QK-norm order, the RoPE convention or the
GQA expansion is wrong, the trajectories will still look plausible and be quietly
incorrect. The check compares against the real Hugging Face layer and refuses to
export on mismatch.

    source $ALPAMAYO_REPO/env.sh
    python $ALPAMAYO_REPO/h100/a3_export_onnx.py
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import onnx
import torch

import arch
import graphs
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

# Local checkpoint by default -- the released weights already live on scratch, so
# nothing here reaches for the hub. Override with $ALPAMAYO_MODEL or --model.
MODEL_DIR = os.environ.get(
    "ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B")
WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")

OPSET = 17


def report(name, a, b, tol):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    mae = (a - b).abs().mean().item()
    peak = (a - b).abs().max().item()
    ok = cos > 1 - tol and peak < 1e-1
    print(f"  [{'ok ' if ok else 'BAD'}] {name:28s} cos {cos:.6f}  mae {mae:.3e}  max {peak:.3e}")
    return ok


def verify_layer(model, dtype):
    """One real decoder layer, HF forward vs. graphs.decoder_layer."""
    print("verifying decoder_layer against the reference implementation")
    lm = model.vlm.model.language_model
    layer, cfg, seq = lm.layers[0], arch.LLM, 32
    x = torch.randn(1, seq, cfg["hidden"], device="cuda", dtype=dtype) * 0.02
    pos = torch.arange(seq, device="cuda").view(1, 1, seq).expand(3, 1, seq)
    cos, sin = graphs.rope_tables(pos.cpu(), dtype=dtype)
    cos, sin = cos.cuda(), sin.cuda()

    with torch.inference_mode(), graphs.export_friendly_sdpa():
        ours, _, _ = graphs.decoder_layer(
            layer, x, cos, sin, cfg, mask=graphs.causal_mask(seq, dtype).cuda())
        theirs = layer(hidden_states=x, position_embeddings=(cos, sin),
                       attention_mask=graphs.causal_mask(seq, dtype).cuda())
        theirs = theirs[0] if isinstance(theirs, tuple) else theirs
    if not report("decoder_layer", ours, theirs, tol=2e-4):
        raise SystemExit(
            "decoder_layer does not match the reference. Do NOT export.\n"
            "Check, in order: QK-norm applied per head before RoPE; rotate_half "
            "convention; GQA repeat_interleave (not repeat); rms_eps.")
    print()


def export(module, inputs, names_in, names_out, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.onnx")
    with graphs.export_friendly_sdpa():
        _export_inner(module, inputs, names_in, names_out, tmp)
    m = onnx.load(str(tmp))
    onnx.checker.check_model(m, full_check=False)
    onnx.save_model(m, str(path), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=path.name + ".data",
                    size_threshold=1024)
    tmp.unlink(missing_ok=True)
    Path(str(tmp) + ".data").unlink(missing_ok=True)
    size = sum(f.stat().st_size for f in path.parent.glob(path.name + "*")) / 1e9
    free, total = torch.cuda.mem_get_info()
    print(f"  wrote {path.name}  ({size:.2f} GB)   gpu free {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
    torch.cuda.empty_cache()


def _export_inner(module, inputs, names_in, names_out, tmp: Path):
    torch.onnx.export(
        module, inputs, str(tmp),
        input_names=names_in, output_names=names_out,
        opset_version=OPSET, do_constant_folding=True, dynamic_axes=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(WORK_ROOT, "golden"))
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "onnx"))
    ap.add_argument("--model", default=os.environ.get("ALPAMAYO_MODEL", MODEL_DIR),
                    help="local checkpoint directory (default: $ALPAMAYO_MODEL)")
    ap.add_argument("--prefill", type=int, default=None, help="defaults to the golden prompt length")
    ap.add_argument("--max-seq", type=int, default=arch.MAX_SEQ)
    ap.add_argument("--skip", default="", help="comma-separated: vision,prefill,decode,expert")
    args = ap.parse_args()

    golden = Path(args.golden)
    out = Path(args.out)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    dtype = torch.float16

    g = np.load(golden / "inputs.npz", allow_pickle=True)
    grid = torch.tensor(g["image_grid_thw"], dtype=torch.long, device="cuda")
    prefill = args.prefill or int(g["input_ids"].shape[1]) + arch.HISTORY_TRAJ_TOKENS
    print(f"prefill sequence {prefill}, cache {args.max_seq}, dtype float16\n")

    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to("cuda").to(dtype).eval()
    verify_layer(model, dtype)

    lm = model.vlm.model.language_model
    zeros = lambda *s: torch.zeros(*s, device="cuda", dtype=dtype)
    pos = torch.arange(prefill).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = (t.cuda() for t in graphs.rope_tables(pos, dtype=dtype))

    if "vision" not in skip:
        print("exporting vision")
        px = torch.tensor(g["pixel_values"], device="cuda", dtype=dtype)
        export(graphs.VisionGraph(model.vlm.model.visual, grid).eval(), (px,),
               ["pixel_values"], ["visual_embeds", "deepstack0", "deepstack1", "deepstack2"],
               out / "vision.onnx")

    if "prefill" not in skip:
        # no later graph references the vision tower; 1.15 GB back
        model.vlm.model.visual = None
        torch.cuda.empty_cache()
        print("exporting prefill")
        mod = graphs.PrefillGraph(lm, prefill, dtype).cuda().eval()
        inp = (zeros(1, prefill, 4096), cos, sin,
               zeros(1, prefill, 4096), zeros(1, prefill, 4096), zeros(1, prefill, 4096))
        export(mod, inp,
               ["inputs_embeds", "cos", "sin", "deepstack0", "deepstack1", "deepstack2"],
               ["last_hidden", "k_cache", "v_cache"], out / "prefill.onnx")

    if "decode" not in skip:
        print("exporting decode")
        mod = graphs.DecodeGraph(lm, model.vlm.lm_head, args.max_seq).cuda().eval()
        L, KV, HD = arch.LLM["layers"], arch.LLM["kv_heads"], arch.LLM["head_dim"]
        inp = (zeros(1, 1, 4096), cos[:, :1], sin[:, :1],
               zeros(L, 1, KV, args.max_seq, HD), zeros(L, 1, KV, args.max_seq, HD),
               zeros(1, 1, 1, args.max_seq + 1))
        export(mod, inp, ["hidden", "cos", "sin", "past_k", "past_v", "mask"],
               ["logits", "new_k", "new_v"], out / "decode.onnx")

    if "expert" not in skip:
        print("exporting expert")
        mod = graphs.ExpertGraph(model.expert, model.action_in_proj,
                                 model.action_out_proj, args.max_seq).cuda().eval()
        L, KV, HD, W = (arch.EXPERT["layers"], arch.EXPERT["kv_heads"],
                        arch.EXPERT["head_dim"], arch.N_WAYPOINTS)
        wpos = torch.arange(W).view(1, 1, -1).expand(3, 1, -1)
        wcos, wsin = (t.cuda() for t in graphs.rope_tables(wpos, dtype=dtype))
        inp = (zeros(1, W, 2), zeros(1, 1, 1), wcos, wsin,
               zeros(L, 1, KV, args.max_seq, HD), zeros(L, 1, KV, args.max_seq, HD),
               zeros(1, 1, W, args.max_seq + W))
        export(mod, inp, ["noisy_action", "timestep", "cos", "sin", "past_k", "past_v", "mask"],
               ["velocity"], out / "expert.onnx")

    (out / "shapes.json").write_text(json.dumps(
        dict(prefill=prefill, max_seq=args.max_seq, opset=OPSET, dtype="float16",
             vit_tokens=arch.VIT_TOKENS, visual_tokens=arch.VISUAL_TOKENS,
             waypoints=arch.N_WAYPOINTS, flow_steps=arch.FLOW_STEPS), indent=2))
    print(f"\nshapes.json written. Copy {out} to the Xavier and build engines there.")


if __name__ == "__main__":
    main()
