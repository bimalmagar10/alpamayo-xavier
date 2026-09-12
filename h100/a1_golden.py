#!/usr/bin/env python3
"""Stage A1 (H100) -- run the reference model and capture everything the Xavier
side will later be checked against.

Produces, under $ALPAMAYO_ROOT/golden/ (default /mnt/SHARED-SCRATCH/bthapama/alpamayo-work):
  inputs.npz        pixel_values, grid_thw, input_ids, position ids, ego history
  activations.npz   visual embeds, DeepStack features, hidden states at layers
                    0/8/16/24/35, the prefill KV cache, and the final waypoints
  trace_stats.json  measured chain-of-causation trace lengths -- the single
                    largest unknown in the Xavier latency budget
  h100_latency.json per-stage H100 timings, as the reference point for speedups

Run inside the reference venv:
    source $ALPAMAYO_REPO/env.sh
    source $ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate
    python $ALPAMAYO_REPO/h100/a1_golden.py --clips 64
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

# Local checkpoint by default -- the released weights already live on scratch, so
# nothing here reaches for the hub. Override with $ALPAMAYO_MODEL or --model.
MODEL_DIR = os.environ.get(
    "ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B")
WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")

DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"
CAPTURE_LAYERS = (0, 8, 16, 24, 35)


def _np(t):
    return t.detach().to(torch.float32).cpu().numpy()


class Tap:
    """Collects module outputs by forward hook, then releases them."""

    def __init__(self):
        self.store, self._handles = {}, []

    def watch(self, name, module, pick=lambda o: o):
        # generate() calls every module once for the whole prompt, then once per new
        # token. Only the FIRST call is the prefill we compare against; without this
        # guard the last decode step overwrites it.
        def hook(_m, _i, o):
            if name not in self.store:
                self.store[name] = pick(o)
        self._handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_inputs(model, processor, clip_id, t0_us):
    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    tokenized = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {"tokenized_data": tokenized,
         "ego_history_xyz": data["ego_history_xyz"],
         "ego_history_rot": data["ego_history_rot"]},
        "cuda",
    )
    return data, model_inputs


def time_stage(fn, warmup=1, iters=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "golden"))
    ap.add_argument("--model", default=os.environ.get("ALPAMAYO_MODEL", MODEL_DIR),
                    help="local checkpoint directory (default: $ALPAMAYO_MODEL)")
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--clips", type=int, default=16, help="clips to sample for trace-length stats")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not os.path.isdir(args.model):
        raise SystemExit("checkpoint not found: %s\n"
                         "Set ALPAMAYO_MODEL or pass --model." % args.model)
    print("loading %s" % args.model)
    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    data, model_inputs = build_inputs(model, processor, args.clip, args.t0_us)

    # ---- 1. golden inputs -------------------------------------------------
    tok = model_inputs["tokenized_data"]
    np.savez_compressed(
        out / "inputs.npz",
        pixel_values=_np(tok["pixel_values"]),
        image_grid_thw=tok["image_grid_thw"].cpu().numpy(),
        input_ids=tok["input_ids"].cpu().numpy(),
        ego_history_xyz=_np(model_inputs["ego_history_xyz"]),
        ego_history_rot=_np(model_inputs["ego_history_rot"]),
        clip_id=np.array(args.clip),
    )
    n_prefill = tok["input_ids"].shape[1] + 48   # + fused history-trajectory tokens
    print(f"[inputs] pixel_values {tuple(tok['pixel_values'].shape)}  "
          f"grid {tok['image_grid_thw'].tolist()}  prefill ~{n_prefill} tokens")

    # ---- 2. golden activations -------------------------------------------
    tap = Tap()
    lm = model.vlm.model.language_model
    tap.watch("visual", model.vlm.model.visual, lambda o: o[0] if isinstance(o, tuple) else o)
    tap.watch("deepstack", model.vlm.model.visual, lambda o: o[1] if isinstance(o, tuple) else None)
    for i in CAPTURE_LAYERS:
        tap.watch(f"layer{i}", lm.layers[i], lambda o: o[0] if isinstance(o, tuple) else o)
    tap.watch("prefill_norm", lm.norm)          # what prefill.onnx's last_hidden must match

    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6, num_traj_samples=1,
            max_generation_length=args.max_new_tokens, return_extra=True,
        )
    tap.close()

    saved = {k: _np(v) for k, v in tap.store.items()
             if isinstance(v, torch.Tensor)}
    ds = tap.store.get("deepstack")
    if isinstance(ds, (list, tuple)):
        for i, f in enumerate(ds):
            saved[f"deepstack{i}"] = _np(f)
    saved["pred_xyz"] = _np(pred_xyz)
    saved["pred_rot"] = _np(pred_rot)
    np.savez_compressed(out / "activations.npz", **saved)
    print(f"[golden] wrote {len(saved)} tensors -> {out/'activations.npz'}")
    print(f"[golden] CoC trace:\n{extra['cot'][0]}\n")

    # ---- 3. trace-length statistics --------------------------------------
    lengths = []
    for i in range(args.clips):
        try:
            _, mi = build_inputs(model, processor, args.clip, args.t0_us + i * 200_000)
        except Exception as exc:                       # clip window ran off the end
            print(f"[trace] clip offset {i} skipped: {exc}")
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, ex = model.sample_trajectories_from_data_with_vlm_rollout(
                data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=args.max_new_tokens, return_extra=True,
            )
        n = len(model.tokenizer(str(ex["cot"][0][0][0]))["input_ids"])
        lengths.append(n)
        print(f"[trace] sample {i:3d}: {n:4d} tokens")

    if lengths:
        a = np.array(lengths)
        stats = dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)),
                     p95=float(np.percentile(a, 95)), max=int(a.max()),
                     max_new_tokens=args.max_new_tokens, lengths=lengths)
        (out / "trace_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"\n[trace] p50 {stats['p50']:.0f}  p95 {stats['p95']:.0f}  max {stats['max']}")
        print(f"[trace] at 92.6 ms/token (Xavier INT8) the p50 trace alone costs "
              f"{stats['p50'] * 0.0926:.1f} s")

    # ---- 4. H100 reference latency ---------------------------------------
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
        vis_ms = time_stage(lambda: model.vlm.model.visual(
            tok["pixel_values"], tok["image_grid_thw"]))
    lat = dict(vision_ms=vis_ms, device=torch.cuda.get_device_name(0))
    (out / "h100_latency.json").write_text(json.dumps(lat, indent=2))
    print(f"[h100] vision tower: {vis_ms:.1f} ms")


if __name__ == "__main__":
    main()
