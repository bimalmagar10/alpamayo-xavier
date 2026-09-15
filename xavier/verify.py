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
import gc
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpamayo_xavier import pieces as piecelib                      # noqa: E402
from alpamayo_xavier import preprocess, refdata, rope               # noqa: E402
from alpamayo_xavier.trt_runner import SCRATCH, Engine              # noqa: E402

# Per-stage tolerance. Vision and prefill accumulate over 27 and 36 layers, so
# they earn more slack; INT8 earns more again.
LAYERS, KV_HEADS, HEAD_DIM = 36, 8, 128
NEG_INF = -65504.0
TOL = {"pixel_values": 1e-3, "visual_embeds": 2e-3, "expert_velocity": 2e-3,
       "decode_logits": 2e-3, "decode_new_kv": 2e-3, "prefill_hidden": 5e-3, "waypoints_m": 0.5}


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
    ap.add_argument("--precision", default="fp16", choices=["int8", "fp16"])
    ap.add_argument("--decode", default="auto", choices=["auto", "trt", "torch"])
    args = ap.parse_args()
    work = args.work

    g = np.load(os.path.join(work, "golden", "inputs.npz"), allow_pickle=True)
    a = np.load(os.path.join(work, "golden", "activations.npz"))
    eng_dir = os.path.join(work, "engines")
    graphs = piecelib.load(eng_dir) or piecelib.load(os.path.join(work, "onnx"))
    refs = os.path.join(work, "refs")
    results = []                          # (stage, True/False); skipped stages are absent

    def plan(name):
        return os.path.join(eng_dir, "%s.%s.plan" % (name, args.precision))

    def stage(name):
        """A stage over its engines -- one, or the pieces a3d split it into -- or None
        if any are not built yet. One stage at a time: prefill and decode are 15 GB."""
        specs = piecelib.specs(graphs, name)
        missing = [p["name"] for p in specs if not os.path.exists(plan(p["name"]))]
        if missing:
            print("  [--] %d of %d %s engine(s) not built yet (e.g. %s) -- skipped"
                  % (len(missing), len(specs), name, missing[0]))
            return None
        # One engine at a time. A verify pass runs each piece once, and holding all 13
        # prefill engines at once left the board too fragmented to load decode after.
        st = piecelib.Stage(name, specs, lambda p, skip=(): Engine(plan(p), skip=skip),
                            sequential=True)
        print("  %d engine(s), loaded one at a time · %.1f GB free"
              % (len(specs), torch.cuda.mem_get_info()[0] / 1e9))
        return st

    def decode_stage():
        """PyTorch decode when the weight map is there, else the engines."""
        want_torch = args.decode == "torch" or (
            args.decode == "auto" and any(os.path.exists(os.path.join(work, d, "weight_map.json"))
                                          for d in ("engines", "onnx")))
        if not want_torch:
            return stage("decode")
        from alpamayo_xavier.torch_decode import TorchDecode
        td = TorchDecode(work)
        print("  PyTorch decode, weights from %s.onnx.data" % td.graph)
        return td

    print("\n0. preprocessing (frames/*.png -> pixel_values, NumPy on this board)")
    frames = sorted(glob.glob(os.path.join(work, "frames", "*.png")))
    if frames:
        from PIL import Image
        px, _ = preprocess.preprocess_images(
            [np.asarray(Image.open(f).convert("RGB")) for f in frames])
        results.append(("preprocessing", compare("pixel_values", px, g["pixel_values"],
                                                 TOL["pixel_values"])))
    else:
        print("  [--] no frames/*.png -- copy the frames/ that h100/a5_export_frames.py wrote")

    print("\n1. vision tower (vs the H100 golden output)")
    st = stage("vision")
    if st:
        out, _ = st.run({"pixel_values": g["pixel_values"]})
        results.append(("vision", compare("visual_embeds", out["visual_embeds"].float().cpu().numpy(),
                                          a["visual"], TOL["visual_embeds"])))
        st.close()
        gc.collect()
        print("  released · scratch %.2f GB · %.1f GB free"
              % (SCRATCH.size / 1e9, torch.cuda.mem_get_info()[0] / 1e9))

    print("\n2. prefill (fed the GOLDEN vision output, to isolate the backbone)")
    ps = None                             # what the decode check reuses
    st = stage("prefill")
    if st:
        meta = json.load(open(os.path.join(work, "fixtures", "meta.json")))
        S, max_seq = meta["prefill"], meta["max_seq"]
        fx = os.path.join(work, "fixtures")
        embed = np.load(os.path.join(fx, "embed_tokens.fp16.npy"), mmap_mode="r")   # 1.28 GB
        ids = np.load(os.path.join(fx, "input_ids.npy"))
        vmask = torch.from_numpy(np.load(os.path.join(fx, "visual_mask.npy"))).cuda()
        pos = np.load(os.path.join(fx, "position_ids.npy"))
        embeds = torch.from_numpy(np.ascontiguousarray(embed[ids[:S]])).unsqueeze(0).cuda()
        embeds[0, vmask[:S]] = torch.from_numpy(a["visual"]).cuda().half()
        ds = []
        for i in range(3):
            z = torch.zeros_like(embeds)
            z[0, vmask[:S]] = torch.from_numpy(a["deepstack%d" % i]).cuda().half()
            ds.append(z)
        cos_p, sin_p = rope.tables(pos[:, :S])
        out, kv = st.run({"inputs_embeds": embeds, "cos": cos_p, "sin": sin_p,
                          "deepstack0": ds[0], "deepstack1": ds[1], "deepstack2": ds[2]})
        if "prefill_norm" in a.files:
            results.append(("prefill", compare("prefill_hidden", out["last_hidden"].float().cpu().numpy(),
                                               a["prefill_norm"][:, -1:], TOL["prefill_hidden"])))
        else:
            print("  [--] golden has no prefill_norm -- re-run  a1_golden.py --clips 0  on the H100")
            results.append(("prefill", False))
        # keep what decode needs: the cache of the first S-1 tokens, and token S-1
        t = S - 1
        past_k = torch.zeros(LAYERS, 1, KV_HEADS, max_seq, HEAD_DIM, dtype=torch.float16, device="cuda")
        past_v = torch.zeros_like(past_k)
        k_t, v_t = [], []
        for a_, b_, k, v in kv:
            past_k[a_:b_ + 1, :, :, :t].copy_(k[:, :, :, :t])
            past_v[a_:b_ + 1, :, :, :t].copy_(v[:, :, :, :t])
            k_t.append(k[:, :, :, t:t + 1].float().cpu().numpy())
            v_t.append(v[:, :, :, t:t + 1].float().cpu().numpy())
        ps = dict(t=t, max_seq=max_seq, pos=pos, past_k=past_k, past_v=past_v,
                  k_t=np.concatenate(k_t), v_t=np.concatenate(v_t),
                  logits=out["logits"].float().reshape(-1).cpu().numpy(),
                  hidden=embeds[:, t:t + 1].clone())
        del embeds, ds, kv
        st.close()
        gc.collect()
        print("  released · scratch %.2f GB · %.1f GB free"
              % (SCRATCH.size / 1e9, torch.cuda.mem_get_info()[0] / 1e9))

    print("\n2b. decode (the last prompt token again, against prefill's cache of the others)")
    if ps is None:
        print("  [--] needs the prefill engines -- skipped")
    else:
        st = decode_stage()
        if st:
            t, max_seq = ps["t"], ps["max_seq"]
            mask = torch.full((1, 1, 1, max_seq + 1), NEG_INF, dtype=torch.float16, device="cuda")
            mask[..., :t] = 0.0
            mask[..., max_seq] = 0.0
            c1, s1 = rope.tables(ps["pos"][:, t:t + 1])
            st.bind_kv(ps["past_k"], ps["past_v"])
            out, dkv = st.run({"hidden": ps["hidden"], "cos": c1, "sin": s1, "mask": mask})
            logits = out["logits"].float().reshape(-1).cpu().numpy()
            ok = compare("decode_vs_prefill", logits, ps["logits"], TOL["decode_logits"])
            same = int(logits.argmax()) == int(ps["logits"].argmax())
            print("  [%s] next token             decode %d, prefill %d"
                  % ("ok " if same else "BAD", int(logits.argmax()), int(ps["logits"].argmax())))
            k_d = np.concatenate([k.float().cpu().numpy() for _, _, k, _ in dkv])
            v_d = np.concatenate([v.float().cpu().numpy() for _, _, _, v in dkv])
            ok &= compare("decode_new_kv", np.concatenate([k_d.ravel(), v_d.ravel()]),
                          np.concatenate([ps["k_t"].ravel(), ps["v_t"].ravel()]), TOL["decode_new_kv"])
            ref_path = os.path.join(refs, "decode_ref.npz")
            if os.path.exists(ref_path):
                ok &= compare("decode_vs_fp32_ref", logits, np.load(ref_path)["decode_logits"],
                              TOL["decode_logits"])
            else:
                print("  [--] no refs/decode_ref.npz -- run h100/a6_reference_io.py and copy refs/")
            results.append(("decode", bool(ok and same)))
            st.close()
            gc.collect()
            print("  released · scratch %.2f GB · %.1f GB free"
                  % (SCRATCH.size / 1e9, torch.cuda.mem_get_info()[0] / 1e9))

    print("\n3. action expert (fixed inputs vs an fp32 ONNX Runtime run of the same graph)")
    ref_path = os.path.join(refs, "expert_ref.npz")
    if not os.path.exists(ref_path):
        print("  [--] no refs/expert_ref.npz -- run h100/a6_reference_io.py and copy refs/")
    else:
        st = stage("expert")
        if st:
            r = np.load(ref_path)
            k, v = refdata.kv_cache(int(r["seed"]), tuple(int(x) for x in r["kv_shape"]))
            if refdata.checksum(k, v) != str(r["kv_sha256"]):
                print("  [BAD] the regenerated KV cache differs from the reference machine's")
                results.append(("expert", False))
            else:
                pk, pv = torch.from_numpy(k).cuda(), torch.from_numpy(v).cuda()
                del k, v
                st.bind_kv(pk, pv)
                out, _ = st.run({n: r[n] for n in ("noisy_action", "timestep", "cos", "sin", "mask")})
                results.append(("expert", compare("expert_velocity",
                                                  out["velocity"].float().cpu().numpy(),
                                                  r["velocity"], TOL["expert_velocity"])))
                del pk, pv
            st.close()
            gc.collect()
            print("  released · scratch %.2f GB · %.1f GB free"
                  % (SCRATCH.size / 1e9, torch.cuda.mem_get_info()[0] / 1e9))

    print("\n4. end-to-end waypoints")
    print("   Run run_alpamayo.py on the golden clip and compare its final xy against")
    print("   activations.npz['pred_xyz']. Trajectory sampling is stochastic, so judge")
    print("   this on minADE across several seeds, not on a single trajectory.")

    passed = bool(results) and all(ok for _, ok in results)
    print("\nchecked: %s" % (", ".join("%s %s" % (s, "ok" if ok else "BAD") for s, ok in results)
                             or "nothing"))
    print("%s" % ("ALL CHECKED STAGES PASS" if passed else
                  "MISMATCH -- do not report latency for a model that computes the wrong thing"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
