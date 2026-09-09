#!/usr/bin/env python3
"""Stage A3b (H100) -- build the fixtures the Jetson cannot compute for itself.

run_alpamayo.py needs five things that require transformers 4.57 and the Qwen3-VL
processor, neither of which exists for Python 3.8 on JetPack 5:

  embed_tokens.fp16.npy   [155697, 4096]  token embedding table (1.28 GB)
  input_ids.npy           [S]   prompt ids AFTER fuse_traj_tokens
  position_ids.npy        [3, S] Qwen3-VL 3D mRoPE positions from get_rope_index
  visual_mask.npy         [S]   bool, True where a visual token sits
  meta.json               prefill length, rope_deltas, max_seq

They are deterministic for a fixed camera rig and prompt, so they are computed
once here and shipped alongside the ONNX.

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

    visual_mask = (fused[0] == IMAGE_TOKEN_ID).cpu().numpy()
    print("visual tokens   :", int(visual_mask.sum()), " (expected 2880)")

    embed = model.vlm.model.language_model.embed_tokens.weight
    embed = embed.detach().to(torch.float16).cpu().numpy()
    print("embed_tokens    :", embed.shape, "%.2f GB" % (embed.nbytes / 1e9))

    np.save(os.path.join(args.out, "input_ids.npy"), fused[0].cpu().numpy().astype(np.int64))
    np.save(os.path.join(args.out, "position_ids.npy"), pos)
    np.save(os.path.join(args.out, "visual_mask.npy"), visual_mask)
    np.save(os.path.join(args.out, "embed_tokens.fp16.npy"), embed)
    meta = dict(prefill=n, max_seq=args.max_seq, rope_deltas=rope_deltas,
                visual_tokens=int(visual_mask.sum()), vocab=int(embed.shape[0]),
                image_token_id=IMAGE_TOKEN_ID, clip=args.clip)
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)

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
