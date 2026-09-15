#!/usr/bin/env python3
"""Stage A6 (Mac or cluster, CPU; needs onnx + onnxruntime) -- fp32 references and
split checks, run through the SAME piece routing the Xavier uses
(xavier/alpamayo_xavier/pieces.py), with ONNX Runtime standing in for TensorRT.

  expert   the full expert graph on fixed inputs          -> refs/expert_ref.npz
           then its pieces chained on the same inputs     -> must match that
  prefill  its pieces chained on the golden inputs        -> vs golden prefill_norm
  decode   one step for the LAST prompt token, fed the prefill pieces' own K/V for
           the first 3005 tokens. It is the same token at the same position reached
           the other way, so its logits and K/V must equal prefill's
                                                          -> refs/decode_ref.npz

verify.py runs the same three checks on the Xavier's engines and compares against
these files. Pieces are converted to fp32 one at a time (a few GB each).

    python h100/a6_reference_io.py --payload ~/alpamayo-payload
    python h100/a6_reference_io.py --payload ~/alpamayo-payload --only expert
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "xavier"))
from alpamayo_xavier import pieces as piecelib                # noqa: E402
from alpamayo_xavier import refdata, rope                     # noqa: E402

LAYERS, KV_HEADS, HEAD_DIM, W = 36, 8, 128, 64
NEG_INF = -65504.0                   # the runner's mask value
SEED = 20260912
TRACE_TOKENS = 16                    # measured p50 chain-of-causation length


def fp32_copy(src, dst):
    """Write an fp32 copy of an fp16 graph (weights, constants, casts, I/O)."""
    m = onnx.load(src)
    g = m.graph
    for i, t in enumerate(g.initializer):
        if t.data_type == TensorProto.FLOAT16:
            g.initializer[i].CopyFrom(numpy_helper.from_array(
                numpy_helper.to_array(t).astype(np.float32), t.name))
    for n in g.node:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32)))
            if n.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
    for v in list(g.input) + list(g.output) + list(g.value_info):
        if v.type.tensor_type.elem_type == TensorProto.FLOAT16:
            v.type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(dst) + ".data")
    del m
    gc.collect()


class OrtPiece(object):
    """ONNX Runtime stand-in for trt_runner.Engine: an fp32 copy of one ONNX file."""

    def __init__(self, onnx_path, scratch):
        self.dir = tempfile.mkdtemp(prefix="piece_", dir=scratch)
        path = os.path.join(self.dir, "m.onnx")
        fp32_copy(onnx_path, path)
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count()
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.bound = {}

    def bind(self, name, array):
        self.bound[name] = array

    def __call__(self, feed):
        f = {k: np.asarray(v, dtype=np.float32) for k, v in feed.items()}
        f.update(self.bound)
        return dict(zip(self.out_names, self.sess.run(None, f)))

    def close(self):
        del self.sess
        shutil.rmtree(self.dir, ignore_errors=True)
        gc.collect()


def cos(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_set(payload, rel):
    man = os.path.join(payload, "MANIFEST.sha256")
    if not os.path.exists(man):
        return
    lines = [l for l in open(man).read().splitlines() if not l.endswith("  " + rel)]
    lines.append("%s  %s" % (sha256(os.path.join(payload, rel)), rel))
    open(man, "w").write("\n".join(lines) + "\n")


def stage(payload, graphs, name, scratch):
    onnx_dir = os.path.join(payload, "onnx")
    return piecelib.Stage(name, graphs[name]["pieces"],
                          lambda p, skip=(): OrtPiece(os.path.join(onnx_dir, p + ".onnx"), scratch),
                          sequential=True)


# ------------------------------------------------------------------------------
# expert
# ------------------------------------------------------------------------------
def expert_inputs(meta):
    """One flow-matching step, placed as run_alpamayo.py would place it."""
    max_seq, prefill = meta["max_seq"], meta["prefill"]
    pos = prefill + TRACE_TOKENS
    rs = np.random.RandomState(SEED + 1)
    wpos = np.arange(W) + pos + int(meta.get("rope_deltas", 0))
    c, s = rope.tables(np.broadcast_to(wpos, (3, W)))
    mask = np.full((1, 1, W, max_seq + W), NEG_INF, np.float16)
    mask[..., :pos] = 0.0
    mask[..., max_seq:] = 0.0
    small = dict(noisy_action=refdata.grid16(rs, (1, W, 2)),
                 timestep=np.full((1, 1, 1), 0.5, np.float16), cos=c, sin=s, mask=mask)
    k, v = refdata.kv_cache(SEED, (LAYERS, 1, KV_HEADS, max_seq, HEAD_DIM))
    return small, k, v


def expert(payload, meta, graphs, scratch):
    ref_path = os.path.join(payload, "refs", "expert_ref.npz")
    small, k, v = expert_inputs(meta)
    k32, v32 = k.astype(np.float32), v.astype(np.float32)
    if not os.path.exists(ref_path):
        print("expert: fp32 reference from the FULL graph")
        full = OrtPiece(os.path.join(payload, "onnx", "expert.onnx"), scratch)
        full.bind("past_k", k32)
        full.bind("past_v", v32)
        velocity = full(small)["velocity"]
        full.close()
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        np.savez_compressed(ref_path, velocity=velocity, seed=np.int64(SEED),
                            kv_shape=np.array(k.shape, np.int64),
                            kv_sha256=np.array(refdata.checksum(k, v)), **small)
        manifest_set(payload, "refs/expert_ref.npz")
    ref = np.load(ref_path)
    if "expert" not in graphs:
        return True
    st = stage(payload, graphs, "expert", scratch)
    st.bind_kv(k32, v32)
    out, _ = st.run(small)
    st.close()
    c = cos(out["velocity"], ref["velocity"])
    print("expert : pieces chained vs full graph      cos %.7f  %s" % (c, "ok" if c > 0.99999 else "MISMATCH"))
    return c > 0.99999


# ------------------------------------------------------------------------------
# prefill, then decode against prefill's own cache
# ------------------------------------------------------------------------------
def prefill_inputs(payload, meta):
    fx = os.path.join(payload, "fixtures")
    ga = np.load(os.path.join(payload, "golden", "activations.npz"))
    S = meta["prefill"]
    embed = np.load(os.path.join(fx, "embed_tokens.fp16.npy"), mmap_mode="r")
    ids = np.load(os.path.join(fx, "input_ids.npy"))[:S]
    vmask = np.load(os.path.join(fx, "visual_mask.npy"))[:S].astype(bool)
    pos = np.load(os.path.join(fx, "position_ids.npy"))
    embeds = np.asarray(embed[ids], np.float32)[None]
    embeds[0, vmask] = ga["visual"].astype(np.float32)
    feed = dict(inputs_embeds=embeds)
    for i in range(3):
        z = np.zeros_like(embeds)
        z[0, vmask] = ga["deepstack%d" % i].astype(np.float32)
        feed["deepstack%d" % i] = z
    feed["cos"], feed["sin"] = rope.tables(pos[:, :S])
    return feed, pos, ga


def prefill_and_decode(payload, meta, graphs, scratch):
    S, max_seq = meta["prefill"], meta["max_seq"]
    feed, pos, ga = prefill_inputs(payload, meta)
    t0 = time.time()
    st = stage(payload, graphs, "prefill", scratch)
    out, kv = st.run(feed)
    st.close()
    ok = True
    if "prefill_norm" in ga.files:
        c = cos(out["last_hidden"], ga["prefill_norm"][:, -1:])
        ok &= c > 0.999
        print("prefill: pieces chained vs H100 golden     cos %.6f  %s  (%.0fs)"
              % (c, "ok" if c > 0.999 else "MISMATCH", time.time() - t0))
    else:
        print("prefill: golden has no prefill_norm -- re-run a1_golden.py --clips 0")
        ok = False
    prefill_logits = out["logits"].reshape(-1).copy()
    if "decode" not in graphs:
        return ok

    # decode the LAST prompt token against the first S-1 cached tokens
    t = S - 1
    K = np.zeros((LAYERS, 1, KV_HEADS, max_seq, HEAD_DIM), np.float32)
    V = np.zeros_like(K)
    for a, b, k, v in kv:
        K[a:b + 1, :, :, :t] = k[:, :, :, :t]
        V[a:b + 1, :, :, :t] = v[:, :, :, :t]
    # K/V of token t for all 36 layers, in layer order -- prefill and decode are cut
    # at different layers, so compare whole stacks, not piece by piece
    k_t = np.concatenate([k[:, :, :, t:t + 1] for _, _, k, _ in kv])
    v_t = np.concatenate([v[:, :, :, t:t + 1] for _, _, _, v in kv])
    del kv
    mask = np.full((1, 1, 1, max_seq + 1), NEG_INF, np.float32)
    mask[..., :t] = 0.0
    mask[..., max_seq] = 0.0
    c1, s1 = rope.tables(pos[:, t:t + 1])
    t0 = time.time()
    st = stage(payload, graphs, "decode", scratch)
    st.bind_kv(K, V)
    dout, dkv = st.run({"hidden": feed["inputs_embeds"][:, t:t + 1], "cos": c1, "sin": s1, "mask": mask})
    st.close()
    decode_logits = dout["logits"].reshape(-1).copy()
    c_log = cos(decode_logits, prefill_logits)
    same = int(decode_logits.argmax()) == int(prefill_logits.argmax())
    c_kv = min(cos(np.concatenate([k for _, _, k, _ in dkv]), k_t),
               cos(np.concatenate([v for _, _, _, v in dkv]), v_t))
    good = c_log > 0.9999 and same and c_kv > 0.9999
    ok &= good
    print("decode : last prompt token, two ways       logits cos %.6f  argmax %s  K/V cos %.6f  %s  (%.0fs)"
          % (c_log, "same" if same else "DIFFERENT", c_kv, "ok" if good else "MISMATCH", time.time() - t0))
    out_path = os.path.join(payload, "refs", "decode_ref.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, prefill_logits=prefill_logits, decode_logits=decode_logits,
                        token_index=np.int64(t))
    manifest_set(payload, "refs/decode_ref.npz")
    print("wrote %s" % out_path)
    return ok


def main():
    ap = argparse.ArgumentParser(description="fp32 references and split checks.")
    ap.add_argument("--payload", default=os.environ.get(
        "ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work"),
        help="folder holding onnx/, fixtures/, golden/ (the Mac's ~/alpamayo-payload works)")
    ap.add_argument("--only", choices=["expert", "prefill"], default=None,
                    help="expert, or prefill (which also runs the decode check)")
    ap.add_argument("--scratch", default=None, help="where fp32 copies go (deleted after)")
    args = ap.parse_args()
    payload = os.path.expanduser(args.payload)
    meta = json.load(open(os.path.join(payload, "fixtures", "meta.json")))
    graphs = piecelib.load(os.path.join(payload, "onnx"))
    ok = True
    if args.only in (None, "expert"):
        ok &= expert(payload, meta, graphs, args.scratch)
    if args.only in (None, "prefill"):
        ok &= prefill_and_decode(payload, meta, graphs, args.scratch)
    print("\n%s" % ("ALL REFERENCE CHECKS PASS" if ok else "MISMATCH"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
