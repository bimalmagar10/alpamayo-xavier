"""Decode one token at a time in PyTorch, reading the weights out of the ONNX file.

Why not TensorRT here, when every other stage is an engine: decode reads all 15.2 GB
of the language model for each token, so it is bound by memory bandwidth, and
TensorRT cannot make that faster -- measured on this Xavier, 350 ms/token through
the engines against 318 ms in PyTorch. Worse, each engine costs about twice its
weights in memory on this board (measured, xavier/mem_probe.py), so decode's ten
engines never fitted at all. Plain torch tensors cost exactly their size.

The maths is the same graph the engines were built from (h100/graphs.py
decoder_layer): RMSNorm in fp32, QK-norm per head, interleaved-mRoPE rotation,
grouped-query attention over [cache ; this token], SwiGLU. verify.py checks it
against prefill and against the Mac's fp32 reference, exactly as it checks an engine.

Weights come from h100/a7_weight_map.py, which records where each one sits inside
<graph>.onnx.data -- so nothing is exported, converted or copied beforehand.
"""
import base64
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM = 36, 32, 8, 128
_NP = {1: np.float32, 10: np.float16}


def _tensor(spec, mm, device):
    if "inline" in spec:
        raw = base64.b64decode(spec["inline"])
    else:
        raw = mm[spec["offset"]:spec["offset"] + spec["length"]]
    a = np.frombuffer(raw, dtype=_NP[spec["dtype"]]).reshape(spec["dims"])
    return torch.from_numpy(a.copy()).to(device)      # copy: the mapping is read-only


def rms_norm(x, weight, eps):
    """Exactly h100/graphs.py: the statistics in fp32, so nothing overflows."""
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


def apply_rope(x, cos, sin):
    """x: [B, H, S, D]; cos/sin: [B, S, D]."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


class TorchDecode(object):
    """Stands in for a decode Stage: same bind_kv / load / run / close."""

    sequential = False

    def __init__(self, work, device="cuda", graph=None):
        self.work, self.device = work, device
        self.pieces = [{"name": "torch-decode"}]
        self.runners = {}
        self.load_seconds = 0.0
        self._kv = None
        path = os.path.join(work, "engines", "weight_map.json")
        if not os.path.exists(path):
            path = os.path.join(work, "onnx", "weight_map.json")
        if not os.path.exists(path):
            raise SystemExit("no weight_map.json -- run h100/a7_weight_map.py and copy it over")
        self.doc = json.load(open(path))
        self.eps = float(self.doc.get("eps", 1e-6))
        self.graph = graph or self._pick()

    def _pick(self):
        """Whichever graph's weights file is actually on this board."""
        for name, spec in self.doc["graphs"].items():
            if os.path.exists(os.path.join(self.work, "onnx", spec["source"])):
                return name
        have = sorted(os.listdir(os.path.join(self.work, "onnx")))
        raise SystemExit("none of %s is in %s/onnx (have: %s)"
                         % ([s["source"] for s in self.doc["graphs"].values()], self.work,
                            ", ".join(h for h in have if h.endswith(".data")) or "no .data files"))

    def bind_kv(self, past_k, past_v):
        self._kv = (past_k, past_v)

    def load(self):
        if self.runners:
            return
        t0 = time.perf_counter()
        spec = self.doc["graphs"][self.graph]
        data = os.path.join(self.work, "onnx", spec["source"])
        mm = np.memmap(data, dtype=np.uint8, mode="r")
        self.layers = [{r: _tensor(w, mm, self.device) for r, w in L.items()} for L in spec["layers"]]
        self.head = {r: _tensor(w, mm, self.device) for r, w in spec["head"].items()}
        del mm
        self.runners["torch-decode"] = self
        self.load_seconds = time.perf_counter() - t0
        print("  torch decode: %.1f GB of weights from %s in %.0f s"
              % (spec["bytes"] / 1e9, spec["source"], self.load_seconds))

    def close(self):
        for L in getattr(self, "layers", []):
            L.clear()
        self.layers, self.head = [], {}
        self.runners.clear()
        torch.cuda.empty_cache()

    def _to(self, x, dtype=torch.float16):
        if torch.is_tensor(x):
            return x.to(self.device, dtype)
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device, dtype)

    @torch.no_grad()
    def run(self, feed):
        """feed: hidden [1,1,4096], cos/sin [1,1,128], mask [1,1,1,max_seq+1].

        Returns ({"logits": [1,1,vocab]}, [(0, 35, new_k, new_v)]) -- the same shape
        of answer a decode Stage gives, so the caller writes K/V into the cache itself.
        """
        self.load()
        past_k, past_v = self._kv
        h = self._to(feed["hidden"])
        cos, sin = self._to(feed["cos"]), self._to(feed["sin"])
        mask = self._to(feed["mask"])
        new_k = torch.empty(LAYERS, 1, KV_HEADS, 1, HEAD_DIM, dtype=h.dtype, device=self.device)
        new_v = torch.empty_like(new_k)
        rep = Q_HEADS // KV_HEADS
        for i, w in enumerate(self.layers):
            x = rms_norm(h, w["input_ln"], self.eps)
            q = (x @ w["q"]).view(1, 1, Q_HEADS, HEAD_DIM)
            k = (x @ w["k"]).view(1, 1, KV_HEADS, HEAD_DIM)
            v = (x @ w["v"]).view(1, 1, KV_HEADS, HEAD_DIM)
            q = rms_norm(q, w["q_norm"], self.eps).transpose(1, 2)      # [1, 32, 1, 128]
            k = rms_norm(k, w["k_norm"], self.eps).transpose(1, 2)      # [1,  8, 1, 128]
            v = v.transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            new_k[i], new_v[i] = k, v
            kk = torch.cat((past_k[i], k), dim=2)                       # [1, 8, max_seq+1, 128]
            vv = torch.cat((past_v[i], v), dim=2)
            att = F.scaled_dot_product_attention(
                q, kk.repeat_interleave(rep, dim=1), vv.repeat_interleave(rep, dim=1),
                attn_mask=mask, dropout_p=0.0)
            h = h + (att.transpose(1, 2).reshape(1, 1, Q_HEADS * HEAD_DIM) @ w["o"])
            x = rms_norm(h, w["post_ln"], self.eps)
            h = h + ((F.silu(x @ w["gate"]) * (x @ w["up"])) @ w["down"])
        h = rms_norm(h, self.head["final_ln"], self.eps)
        return {"logits": h @ self.head["lm_head"]}, [(0, LAYERS - 1, new_k, new_v)]
