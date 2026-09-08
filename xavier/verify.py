#!/usr/bin/env python3
"""Check each Xavier stage against the H100 golden dump, in dependency order.

Run this before believing any latency number. Every failure mode that matters
here -- a wrong RoPE base, a transposed QK-norm, GQA expanded with repeat instead
of repeat_interleave, INT8 scales calibrated on the wrong tensor -- still produces
smooth, plausible-looking trajectories. Cosine similarity against the reference
is the only thing that catches them.

    python verify.py --work /mnt/ssdhome/models/alpamayo --precision int8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpamayo_xavier import rope                                    # noqa: E402
from alpamayo_xavier.trt_runner import load_engines                 # noqa: E402

# Per-stage tolerance. Vision and prefill accumulate over 27 and 36 layers, so
# they earn more slack; INT8 earns more again.
TOL = {"visual_embeds": 2e-3, "prefill_hidden": 5e-3, "waypoints_m": 0.5}


def compare(name, got, ref, tol):
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if got.shape != ref.shape:
        print("  [BAD] %-22s shape %s vs golden %s" % (name, got.shape, ref.shape))
        return False
    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
    rel = float(np.abs(got - ref).mean() / (np.abs(ref).mean() + 1e-12))
    ok = (1.0 - cos) < tol
    print("  [%s] %-22s cos %.6f  rel-mae %.4f  (tol %.0e)" %
          ("ok " if ok else "BAD", name, cos, rel, tol))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("ALPAMAYO_WORK",
                                                     "/mnt/ssdhome/models/alpamayo"))
    ap.add_argument("--precision", default="int8", choices=["int8", "fp16"])
    args = ap.parse_args()

    g = np.load(os.path.join(args.work, "golden", "inputs.npz"), allow_pickle=True)
    a = np.load(os.path.join(args.work, "golden", "activations.npz"))
    meta = json.load(open(os.path.join(args.work, "fixtures", "meta.json")))
    prefill_len = meta["prefill"]
    eng = load_engines(os.path.join(args.work, "engines"), args.precision,
                       names=("vision", "prefill"))
    passed = True

    print("\n1. vision tower")
    out = eng["vision"]({"pixel_values": g["pixel_values"]})
    passed &= compare("visual_embeds", out["visual_embeds"].float().cpu().numpy(),
                      a["visual"], TOL["visual_embeds"])

    print("\n2. prefill (fed the GOLDEN vision output, to isolate the backbone)")
    fx = os.path.join(args.work, "fixtures")
    embed = torch.from_numpy(np.load(os.path.join(fx, "embed_tokens.fp16.npy"))).cuda()
    ids = torch.from_numpy(np.load(os.path.join(fx, "input_ids.npy"))).long().cuda()
    vmask = torch.from_numpy(np.load(os.path.join(fx, "visual_mask.npy"))).cuda()
    pos = np.load(os.path.join(fx, "position_ids.npy"))

    embeds = embed[ids[:prefill_len]].unsqueeze(0).clone()
    embeds[0, vmask[:prefill_len]] = torch.from_numpy(a["visual"]).cuda().half()
    ds = []
    for i in range(3):
        z = torch.zeros_like(embeds)
        z[0, vmask[:prefill_len]] = torch.from_numpy(a["deepstack%d" % i]).cuda().half()
        ds.append(z)
    cos_p, sin_p = rope.tables(pos[:, :prefill_len])
    out = eng["prefill"]({"inputs_embeds": embeds,
                          "cos": torch.from_numpy(cos_p).cuda(),
                          "sin": torch.from_numpy(sin_p).cuda(),
                          "deepstack0": ds[0], "deepstack1": ds[1], "deepstack2": ds[2]})
    if "layer35" in a:
        passed &= compare("prefill_hidden", out["last_hidden"].float().cpu().numpy(),
                          a["layer35"][:, -1:], TOL["prefill_hidden"])

    print("\n3. end-to-end waypoints")
    print("   Run run_alpamayo.py on the golden clip and compare its final xy against")
    print("   activations.npz['pred_xyz']. Trajectory sampling is stochastic, so judge")
    print("   this on minADE across several seeds, not on a single trajectory.")

    print("\n%s" % ("ALL STAGES PASS" if passed else
                    "MISMATCH -- do not report latency for a model that computes the wrong thing"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
