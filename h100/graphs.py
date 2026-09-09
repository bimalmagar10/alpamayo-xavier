"""Export wrappers that turn Alpamayo-1 into four static-shape ONNX graphs.

Design rules, all driven by what TensorRT 8.5.2 on sm_72 can actually consume:

1. **Nothing dynamic.** Shapes are fixed at export time. Your camera rig does not
   change between frames, so the vision grid is baked in as a constant.
2. **Rotary embeddings are computed on the host.** Qwen3-VL uses interleaved
   3D mRoPE, whose index arithmetic exports into a thicket of Gather/Slice that
   TRT 8.5 will not fuse. Passing cos/sin in as plain tensors costs 1.5 MB per
   prefill and removes the whole problem.
3. **DeepStack injection arrives pre-scattered.** The reference code writes vision
   features into hidden states at visual-token positions after layers 8/16/24.
   The host builds a full-length additive tensor that is zero elsewhere, turning
   a scatter into an Add.
4. **Decode emits only its own KV slice.** TensorRT 8.5 cannot alias an input
   buffer to an output, so returning a whole updated cache would copy ~490 MB
   per token. Instead the graph reads a persistent MAX_SEQ cache and returns the
   single new 1-token slice; the driver memcpy's 147 KB into place.
5. **Attention is eager.** No SDPA, no FlashAttention -- explicit matmul/softmax
   so the exporter emits ops TRT recognises.

These wrappers hold references to the real Hugging Face submodules, so weights
are never remapped by name and cannot silently transpose.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

import arch


# --------------------------------------------------------------------------
# SDPA shim for export
# --------------------------------------------------------------------------
# transformers >= 4.5x calls scaled_dot_product_attention(..., enable_gqa=True)
# instead of materialising the repeated KV heads. torch.onnx's TorchScript
# exporter has no symbolic for that flag and asserts:
#
#   conversion of scaled_dot_product_attention not implemented if enable_gqa is True
#
# Expanding the KV heads by hand and dropping the flag produces an identical
# result and exports cleanly. Doing it here rather than forcing eager attention
# matters: eager would materialise an 11,520 x 11,520 score matrix per vision
# layer (~4 GB) during tracing, whereas SDPA keeps the memory-efficient kernel
# and the ONNX symbolic still emits plain MatMul/Softmax/MatMul, which is exactly
# what TensorRT 8.5 wants.
_ORIG_SDPA = F.scaled_dot_product_attention


def _sdpa_expand_gqa(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, scale=None, enable_gqa=False, **kwargs):
    if enable_gqa:
        rep = query.shape[-3] // key.shape[-3]
        if rep > 1:
            key = key.repeat_interleave(rep, dim=-3)
            value = value.repeat_interleave(rep, dim=-3)
    return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                      is_causal=is_causal, scale=scale, **kwargs)


@contextlib.contextmanager
def export_friendly_sdpa():
    """Make every scaled_dot_product_attention call in the model exportable."""
    F.scaled_dot_product_attention = _sdpa_expand_gqa
    torch.nn.functional.scaled_dot_product_attention = _sdpa_expand_gqa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _ORIG_SDPA
        torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA


# --------------------------------------------------------------------------
# rotary helpers -- run on the host, both at export time and on the Xavier
# --------------------------------------------------------------------------
def interleaved_mrope(freqs: torch.Tensor, section=arch.LLM["mrope_section"]) -> torch.Tensor:
    """Qwen3-VL interleaves the temporal/height/width bands rather than chunking.

    freqs: [3, B, S, head_dim // 2] -> [B, S, head_dim // 2]
    """
    out = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):          # height, width
        out[..., offset:section[axis] * 3:3] = freqs[axis][..., offset:section[axis] * 3:3]
    return out


def rope_tables(position_ids: torch.Tensor, head_dim=arch.LLM["head_dim"],
                theta=arch.LLM["rope_theta"], dtype=torch.float16):
    """position_ids: [3, B, S] -> cos, sin each [B, S, head_dim]."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    freqs = position_ids.to(torch.float64)[..., None] * inv          # [3, B, S, hd/2]
    merged = interleaved_mrope(freqs)
    emb = torch.cat((merged, merged), dim=-1)                        # [B, S, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, D]; cos/sin: [B, S, D]."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


# --------------------------------------------------------------------------
# one Qwen3 decoder layer, expressed in ops TensorRT 8.5 understands
# --------------------------------------------------------------------------
def decoder_layer(layer, x, cos, sin, cfg, mask=None, past_k=None, past_v=None):
    """Returns (hidden, new_k, new_v). new_k/new_v cover only this call's tokens."""
    nq, nkv, hd = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    B, S, _ = x.shape
    a = layer.self_attn

    h = rms_norm(x, layer.input_layernorm.weight, cfg["rms_eps"])
    q = a.q_proj(h).view(B, S, nq, hd)
    k = a.k_proj(h).view(B, S, nkv, hd)
    v = a.v_proj(h).view(B, S, nkv, hd)
    q = rms_norm(q, a.q_norm.weight, cfg["rms_eps"]).transpose(1, 2)   # Qwen3 QK-norm,
    k = rms_norm(k, a.k_norm.weight, cfg["rms_eps"]).transpose(1, 2)   # per head, before RoPE
    v = v.transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    new_k, new_v = k, v

    if past_k is not None:
        k = torch.cat((past_k, k), dim=2)
        v = torch.cat((past_v, v), dim=2)

    # Expand the KV heads by hand, then call SDPA WITHOUT enable_gqa.
    #
    # This deliberately does not spell out matmul/softmax/matmul. Writing it out
    # materialises a [1, 32, S, S] score tensor per layer -- 3.3 GiB at S=3054 once
    # the fp32 softmax cast is counted -- and the JIT tracer retains every one of
    # them, so a 36-layer export needs ~120 GiB before the weights. SDPA keeps the
    # memory-efficient kernel (24 MiB/layer) and torch.onnx's symbolic still emits
    # plain MatMul/Softmax/MatMul, so TensorRT sees an identical graph.
    rep = nq // nkv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

    x = x + a.o_proj(attn.transpose(1, 2).reshape(B, S, nq * hd))
    h = rms_norm(x, layer.post_attention_layernorm.weight, cfg["rms_eps"])
    m = layer.mlp
    return x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h)), new_k, new_v


def causal_mask(seq: int, dtype: torch.dtype) -> torch.Tensor:
    m = torch.full((seq, seq), torch.finfo(dtype).min, dtype=dtype)
    return m.triu(1)[None, None]


# --------------------------------------------------------------------------
# graph 1 -- vision tower (the real HF module, grid baked in)
# --------------------------------------------------------------------------
class VisionGraph(nn.Module):
    """pixel_values [N_patches, 1536] -> visual embeds [N_merged, 4096] + 3 DeepStack maps."""

    def __init__(self, visual, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual
        self.register_buffer("grid_thw", grid_thw, persistent=False)

    def forward(self, pixel_values):
        out = self.visual(pixel_values, self.grid_thw)
        embeds, deepstack = (out if isinstance(out, tuple) else (out, None))
        if deepstack is None:
            return embeds
        return embeds, deepstack[0], deepstack[1], deepstack[2]


# --------------------------------------------------------------------------
# graph 2 -- prefill: 36 layers over the full prompt, emitting the KV cache
# --------------------------------------------------------------------------
class PrefillGraph(nn.Module):
    def __init__(self, language_model, seq_len: int, dtype=torch.float16):
        super().__init__()
        self.lm, self.seq, self.dtype = language_model, seq_len, dtype
        self.cfg = arch.LLM
        self.deepstack_at = arch.VISION["deepstack_indexes"]
        self.register_buffer("mask", causal_mask(seq_len, dtype), persistent=False)

    def forward(self, inputs_embeds, cos, sin, ds0, ds1, ds2):
        """inputs_embeds [1, S, 4096]; ds* [1, S, 4096], zero off visual positions."""
        deepstack = {self.deepstack_at[0]: ds0, self.deepstack_at[1]: ds1,
                     self.deepstack_at[2]: ds2}
        h = inputs_embeds
        ks, vs = [], []
        for i, layer in enumerate(self.lm.layers):
            h, k, v = decoder_layer(layer, h, cos, sin, self.cfg, mask=self.mask)
            if i in deepstack:
                h = h + deepstack[i]
            ks.append(k)
            vs.append(v)
        h = rms_norm(h, self.lm.norm.weight, self.cfg["rms_eps"])
        return h[:, -1:], torch.stack(ks), torch.stack(vs)


# --------------------------------------------------------------------------
# graph 3 -- one decode step against a persistent MAX_SEQ cache
# --------------------------------------------------------------------------
class DecodeGraph(nn.Module):
    def __init__(self, language_model, lm_head, max_seq: int):
        super().__init__()
        self.lm, self.lm_head, self.max_seq = language_model, lm_head, max_seq
        self.cfg = arch.LLM

    def forward(self, hidden, cos, sin, past_k, past_v, mask):
        """hidden [1,1,4096]; past_k/v [36,1,8,MAX_SEQ,128]; mask [1,1,1,MAX_SEQ+1]."""
        h = hidden
        ks, vs = [], []
        for i, layer in enumerate(self.lm.layers):
            h, k, v = decoder_layer(layer, h, cos, sin, self.cfg, mask=mask,
                                    past_k=past_k[i], past_v=past_v[i])
            ks.append(k)
            vs.append(v)
        h = rms_norm(h, self.lm.norm.weight, self.cfg["rms_eps"])
        return self.lm_head(h), torch.stack(ks), torch.stack(vs)


# --------------------------------------------------------------------------
# graph 4 -- one flow-matching step of the action expert
# --------------------------------------------------------------------------
class ExpertGraph(nn.Module):
    """The expert attends over [backbone KV ; its own KV] at every layer, which is
    why its KV geometry (8 heads x 128) matches the backbone exactly. It reads the
    same persistent cache buffers the decode engine writes."""

    def __init__(self, expert, action_in_proj, action_out_proj, max_seq: int,
                 dtype=torch.float16):
        super().__init__()
        self.expert = expert
        # PerWaypointActionInProjV2.forward hard-casts its inputs with .float(),
        # because the Fourier encoder takes sin/cos of arguments up to 2*pi*100 and
        # genuinely needs the precision. The reference survives the resulting
        # fp32-activation/bf16-weight mismatch only because it runs under autocast;
        # an ONNX trace does not. Keep this module in fp32 -- it is 1.35 M params,
        # so the cost is 2.7 MB and one Cast node -- and hand fp16 to the expert.
        self.action_in_proj = action_in_proj.float()
        self.action_out_proj = action_out_proj
        self.max_seq = max_seq
        self.dtype = dtype
        self.cfg = arch.EXPERT

    def forward(self, noisy_action, timestep, cos, sin, past_k, past_v, mask):
        """noisy_action [1,64,2]; timestep [1,1,1]; mask [1,1,64,MAX_SEQ+64]."""
        h = self.action_in_proj(noisy_action.float(), timestep.float()).to(self.dtype)
        if h.dim() == 2:
            h = h.view(1, arch.N_WAYPOINTS, -1)
        for i, layer in enumerate(self.expert.layers):
            h, _, _ = decoder_layer(layer, h, cos, sin, self.cfg, mask=mask,
                                    past_k=past_k[i], past_v=past_v[i])
        h = rms_norm(h, self.expert.norm.weight, self.cfg["rms_eps"])
        return self.action_out_proj(h)
