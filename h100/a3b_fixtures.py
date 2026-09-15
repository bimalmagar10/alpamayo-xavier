#!/usr/bin/env python3
"""Stage A3b (H100) -- build the fixtures the Jetson cannot compute for itself.

run_alpamayo.py needs five things that require transformers 4.57 and the Qwen3-VL
processor, neither of which exists for Python 3.8 on JetPack 5:

  embed_tokens.fp16.npy   [155697, 4096]  token embedding table (1.28 GB)
  input_ids.npy           [S]   prompt ids AFTER fuse_traj_tokens
  position_ids.npy        [3, S] Qwen3-VL 3D mRoPE positions from get_rope_index
  visual_mask.npy         [S]   bool, True where a visual token sits
  meta.json               prefill length, rope_deltas, max_seq

The embedding table is shared. Prompt IDs, position IDs and initial speed belong
to a particular clip and timestamp and must be regenerated for new inputs.

    python h100/a3b_fixtures.py
"""
import argparse
import json
import os

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

MODEL_DIR = os.environ.get(
    "ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B")
WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")
DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"
IMAGE_TOKEN_ID = 151655


def prepare_inputs(model, processor, data, clip, t0_us, max_seq):
    """Build sample-specific prompt/history/position inputs without model inference."""
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    tok = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")

    raw_ids = tok["input_ids"]
    fused = model.fuse_traj_tokens(raw_ids, {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"]})
    n = int(fused.shape[1])
    print("prompt ids      :", tuple(raw_ids.shape))
    print("after fusing    :", tuple(fused.shape),
          "  (masked_scatter -- length must be unchanged)")
    assert fused.shape == raw_ids.shape, "fuse_traj_tokens changed the length"

    # Qwen3-VL assigns 3D (temporal, height, width) positions; the vision segments
    # compress them, so this cannot be reproduced as a plain arange on the Jetson.
    pos, deltas = model.vlm.model.get_rope_index(
        input_ids=fused, image_grid_thw=tok["image_grid_thw"],
        attention_mask=tok.get("attention_mask"))
    pos = pos[:, 0].cpu().numpy().astype(np.int64)         # [3, S]
    rope_deltas = int(np.asarray(deltas.cpu()).reshape(-1)[0])
    print("position_ids    :", pos.shape, " rope_deltas:", rope_deltas)
    print("  t/h/w ranges  :", [(int(pos[i].min()), int(pos[i].max())) for i in range(3)])

    # The expert outputs acceleration and curvature; postprocess.py integrates them
    # from the speed at t=0, which the reference estimates by a least-squares fit
    # over the ego history. Compute it here, with the reference's own code.
    t0 = model.action_space.estimate_t0_states(data["ego_history_xyz"], data["ego_history_rot"])
    v0 = float(np.asarray(t0["v"].detach().cpu()).reshape(-1)[-1])
    print("ego speed v0    : %.3f m/s" % v0)

    visual_mask = (fused[0] == IMAGE_TOKEN_ID).cpu().numpy()
    print("visual tokens   :", int(visual_mask.sum()), " (expected 2880)")

    arrays = dict(input_ids=fused[0].cpu().numpy().astype(np.int64), position_ids=pos,
                  visual_mask=visual_mask, image_grid_thw=tok["image_grid_thw"].cpu().numpy())
    meta = dict(prefill=n, max_seq=max_seq, rope_deltas=rope_deltas, v0=v0,
                visual_tokens=int(visual_mask.sum()),
                vocab=int(model.vlm.model.language_model.embed_tokens.weight.shape[0]),
                image_token_id=IMAGE_TOKEN_ID, clip=clip, t0_us=int(t0_us))
    return arrays, meta, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "fixtures"))
    ap.add_argument("--max-seq", type=int, default=3584)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    processor = helper.get_processor(model.tokenizer)
    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    arrays, meta, tok = prepare_inputs(model, processor, data, args.clip, args.t0_us, args.max_seq)
    n = meta["prefill"]
    embed = model.vlm.model.language_model.embed_tokens.weight
    embed = embed.detach().to(torch.float16).cpu().numpy()
    print("embed_tokens    :", embed.shape, "%.2f GB" % (embed.nbytes / 1e9))

    for name, value in arrays.items():
        np.save(os.path.join(args.out, name + ".npy"), value)
    np.save(os.path.join(args.out, "embed_tokens.fp16.npy"), embed)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    # Export the actual runtime tokenizer, including Alpamayo's added tokens.
    from a8_token_strings import bytes_to_unicode, encode_literal, write_vocab
    table = bytes_to_unicode()
    token_strings = {i: tok for tok, i in model.tokenizer.get_vocab().items()}
    for i, token in model.tokenizer.added_tokens_decoder.items():
        token_strings[i] = encode_literal(str(token), table)
    write_vocab(token_strings, os.path.join(args.out, "vocab.json"), "model.tokenizer",
                size=int(embed.shape[0]))

    print("\nwrote fixtures to", args.out)
    print(json.dumps(meta, indent=2))

    shapes = os.path.join(WORK_ROOT, "onnx", "shapes.json")
    if os.path.exists(shapes):
        exported = json.load(open(shapes)).get("prefill")
        if exported != n:
            print("\n*** MISMATCH: prefill.onnx was exported at %d tokens, but the real "
                  "prompt is %d.\n    Re-export prefill:  a3_export_onnx.py --skip "
                  "vision,decode,expert" % (exported, n))
        else:
            print("\nprefill.onnx shape matches the prompt (%d tokens)." % n)


if __name__ == "__main__":
    main()
