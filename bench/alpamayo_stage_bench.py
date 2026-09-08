#!/usr/bin/env python3
"""
alpamayo_stage_bench.py -- shape-faithful, per-stage latency bench for
NVIDIA Alpamayo-1 (Alpamayo-R1-10B) on Jetson AGX Xavier.

Rebuilds the exact tensor shapes of nvidia/Alpamayo-R1-10B -- read off its
safetensors headers and config.json -- with RANDOM fp16 weights, and times
each pipeline stage separately with CUDA events.

Why random weights are legitimate here: Alpamayo-1 is fully dense (no MoE,
no early exit, no data-dependent routing). Wall-clock latency of a dense
transformer is a function of shapes, dtype and kernel choice only. So these
are the real Xavier latencies -- obtained without a 22 GB download, without
Python 3.12, without bfloat16 and without flash-attn, none of which exist
on JetPack 5.

Verified architecture (nvidia/Alpamayo-R1-10B, 11.08 B params, bf16):
  vision  27 blocks, d=1152, mlp=4304, 16 heads x 72, patch (2,16,16)
  llm     36 layers, d=4096, 32 q-heads x 128, 8 kv-heads x 128, mlp=12288
          SwiGLU + RMSNorm + per-head q/k norm  (Qwen3-VL-8B topology)
  expert  36 layers, d=2048, 16 q-heads x 128, 8 kv-heads x 128, mlp=8256
          attends over [backbone KV ; expert KV]   (pi-0 style action expert)
  vocab   155697, untied embed / lm_head

Target runtime: JetPack 5.1.x, Python 3.8, torch 2.1.0a0+41361538.nv23.06, sm_72.

Run `sudo nvpmodel -m 0 && sudo jetson_clocks` first, and log tegrastats
alongside. See --help for the knobs that sweep the design space.
"""
import argparse
import json
import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Architecture constants, verified from the released checkpoint.
# ----------------------------------------------------------------------------
VIS = dict(layers=27, dim=1152, mlp=4304, heads=16, head_dim=72, out_dim=4096)
LLM = dict(layers=36, dim=4096, mlp=12288, q_heads=32, kv_heads=8, head_dim=128)
EXP = dict(layers=36, dim=2048, mlp=8256, q_heads=16, kv_heads=8, head_dim=128)
VOCAB = 155697

# Default workload: 4 cameras x 4 frames @ 320x576, temporal patch 2.
#   720 patches per (16x16) frame-slot, 2 slots per camera, 4 cameras
#   -> 5760 ViT tokens -> 2x2 merge -> 1440 LLM visual tokens
DEF_VIT_SLOTS, DEF_VIT_TOK = 8, 720
DEF_PREFILL = 1550          # 1440 visual + 48 history-traj + ~62 text
DEF_WAYPOINTS = 64
DEF_FLOW_STEPS = 10         # euler, dt=0.1


def rms_norm(x, w, eps=1e-6):
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * w.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x):
        return rms_norm(x, self.weight)


def build_rope(head_dim, max_len, device, dtype, base=1e6):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    f = torch.outer(t, inv)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x, cos, sin):
    # x: [B, H, S, D]
    d = x.shape[-1] // 2
    rot = torch.cat((-x[..., d:], x[..., :d]), dim=-1)
    return x * cos + rot * sin


def sdpa(q, k, v, causal=False):
    # torch 2.1 has no enable_gqa; expand kv heads explicitly.
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class DecoderLayer(nn.Module):
    """Qwen3-style: SwiGLU MLP, RMSNorm, per-head q/k RMSNorm, GQA."""

    def __init__(self, cfg, dtype):
        super().__init__()
        d, hd = cfg["dim"], cfg["head_dim"]
        self.nq, self.nkv, self.hd = cfg["q_heads"], cfg["kv_heads"], hd
        kw = dict(bias=False, dtype=dtype)
        self.q_proj = nn.Linear(d, self.nq * hd, **kw)
        self.k_proj = nn.Linear(d, self.nkv * hd, **kw)
        self.v_proj = nn.Linear(d, self.nkv * hd, **kw)
        self.o_proj = nn.Linear(self.nq * hd, d, **kw)
        self.q_norm = RMSNorm(hd, dtype)
        self.k_norm = RMSNorm(hd, dtype)
        self.gate_proj = nn.Linear(d, cfg["mlp"], **kw)
        self.up_proj = nn.Linear(d, cfg["mlp"], **kw)
        self.down_proj = nn.Linear(cfg["mlp"], d, **kw)
        self.in_norm = RMSNorm(d, dtype)
        self.post_norm = RMSNorm(d, dtype)

    def project_qkv(self, x, cos, sin, pos):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.nq, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nkv, self.hd).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        c, s = cos[pos:pos + S], sin[pos:pos + S]
        return apply_rope(q, c, s), apply_rope(k, c, s), v

    def mlp(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def forward(self, x, cos, sin, pos=0, kv=None, causal=True, extra_kv=None):
        h = self.in_norm(x)
        q, k, v = self.project_qkv(h, cos, sin, pos)
        if kv is not None:                       # incremental decode
            kc, vc = kv
            kc[:, :, pos:pos + k.shape[2]] = k
            vc[:, :, pos:pos + v.shape[2]] = v
            k, v = kc[:, :, :pos + q.shape[2]], vc[:, :, :pos + q.shape[2]]
            causal = False
        if extra_kv is not None:                 # action expert reads backbone KV
            k = torch.cat([extra_kv[0], k], dim=2)
            v = torch.cat([extra_kv[1], v], dim=2)
            causal = False
        a = sdpa(q, k, v, causal=causal)
        B, _, S, _ = a.shape
        x = x + self.o_proj(a.transpose(1, 2).reshape(B, S, self.nq * self.hd))
        return x + self.mlp(self.post_norm(x))


class VisionBlock(nn.Module):
    """Qwen3-VL ViT block: pre-LayerNorm, biased qkv, GELU MLP."""

    def __init__(self, dtype):
        super().__init__()
        d, m, hd = VIS["dim"], VIS["mlp"], VIS["head_dim"]
        self.h, self.hd = VIS["heads"], hd
        self.norm1 = nn.LayerNorm(d, dtype=dtype)
        self.norm2 = nn.LayerNorm(d, dtype=dtype)
        self.qkv = nn.Linear(d, 3 * d, bias=True, dtype=dtype)
        self.proj = nn.Linear(d, d, bias=True, dtype=dtype)
        self.fc1 = nn.Linear(d, m, bias=True, dtype=dtype)
        self.fc2 = nn.Linear(m, d, bias=True, dtype=dtype)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, S, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, D))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def init_small(mod):
    for p in mod.parameters():
        if p.dim() >= 2:
            nn.init.normal_(p, std=0.02)
    return mod


def bench(fn, warmup, iters, label):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ev = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn()
        e.record()
        ev.append((s, e))
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in ev)
    p50 = statistics.median(ms)
    p95 = ms[int(0.95 * (len(ms) - 1))]
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  {label:34s} p50 {p50:9.2f} ms   p95 {p95:9.2f} ms   peak {peak:8.0f} MiB")
    if isinstance(out, torch.Tensor) and not torch.isfinite(out).all():
        print(f"  {'':34s} WARNING: non-finite output (fp16 overflow in random weights)")
    return p50, p95


# ----------------------------------------------------------------------------
def stage_vision(a, dt, res):
    n = min(a.vision_layers, VIS["layers"])
    blocks = nn.Sequential(*[init_small(VisionBlock(dt)) for _ in range(n)]).cuda().eval()
    x = torch.randn(a.vit_slots, a.vit_tokens, VIS["dim"], device="cuda", dtype=dt) * 0.02
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        p50, p95 = bench(lambda: blocks(x), a.warmup, a.iters, f"vision  {n}/{VIS['layers']} blocks")
    scale = VIS["layers"] / n
    res["vision"] = dict(p50_ms=p50 * scale, p95_ms=p95 * scale, layers_run=n, extrapolated=scale != 1.0)
    del blocks, x
    torch.cuda.empty_cache()


def stage_prefill(a, dt, res):
    n = min(a.llm_layers, LLM["layers"])
    layers = nn.ModuleList([init_small(DecoderLayer(LLM, dt)) for _ in range(n)]).cuda().eval()
    cos, sin = build_rope(LLM["head_dim"], a.max_len, "cuda", dt)
    x = torch.randn(1, a.prefill, LLM["dim"], device="cuda", dtype=dt) * 0.02
    head = nn.Linear(LLM["dim"], VOCAB, bias=False, dtype=dt).cuda() if a.with_head else None

    def run():
        h = x
        for l in layers:
            h = l(h, cos, sin, causal=True)
        return head(h[:, -1:]) if head is not None else h

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        p50, p95 = bench(run, a.warmup, a.iters, f"prefill {n}/{LLM['layers']} layers @ {a.prefill} tok")
    scale = LLM["layers"] / n
    res["prefill"] = dict(p50_ms=p50 * scale, p95_ms=p95 * scale, tokens=a.prefill,
                          layers_run=n, extrapolated=scale != 1.0)
    del layers, x, head
    torch.cuda.empty_cache()


def stage_decode(a, dt, res):
    n = min(a.llm_layers, LLM["layers"])
    layers = nn.ModuleList([init_small(DecoderLayer(LLM, dt)) for _ in range(n)]).cuda().eval()
    cos, sin = build_rope(LLM["head_dim"], a.max_len, "cuda", dt)
    kv = [(torch.zeros(1, LLM["kv_heads"], a.max_len, LLM["head_dim"], device="cuda", dtype=dt),
           torch.zeros(1, LLM["kv_heads"], a.max_len, LLM["head_dim"], device="cuda", dtype=dt))
          for _ in range(n)]
    head = init_small(nn.Linear(LLM["dim"], VOCAB, bias=False, dtype=dt)).cuda()
    x = torch.randn(1, 1, LLM["dim"], device="cuda", dtype=dt) * 0.02
    pos = a.prefill

    def step():
        h = x
        for l, c in zip(layers, kv):
            h = l(h, cos, sin, pos=pos, kv=c)
        return head(h)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        p50, p95 = bench(step, a.warmup, a.iters, f"decode  {n}/{LLM['layers']} layers, ctx {pos}")
    scale = LLM["layers"] / n
    tok = p50 * scale
    res["decode"] = dict(per_token_ms=tok, p95_ms=p95 * scale, ctx=pos, layers_run=n,
                         extrapolated=scale != 1.0, tokens_per_s=1000.0 / tok)
    del layers, kv, head, x
    torch.cuda.empty_cache()


def stage_expert(a, dt, res):
    n = min(a.expert_layers, EXP["layers"])
    layers = nn.ModuleList([init_small(DecoderLayer(EXP, dt)) for _ in range(n)]).cuda().eval()
    cos, sin = build_rope(EXP["head_dim"], a.max_len, "cuda", dt)
    # backbone KV the expert cross-attends into: 8 kv-heads x 128, prefill long
    bkv = [(torch.randn(1, EXP["kv_heads"], a.prefill, EXP["head_dim"], device="cuda", dtype=dt) * 0.02,
            torch.randn(1, EXP["kv_heads"], a.prefill, EXP["head_dim"], device="cuda", dtype=dt) * 0.02)
           for _ in range(n)]
    x = torch.randn(1, a.waypoints, EXP["dim"], device="cuda", dtype=dt) * 0.02

    def flow():
        h = x
        for _ in range(a.flow_steps):
            for l, bk in zip(layers, bkv):
                h = l(h, cos, sin, causal=False, extra_kv=bk)
        return h

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        p50, p95 = bench(flow, max(2, a.warmup // 4), max(5, a.iters // 5),
                         f"expert  {n}/{EXP['layers']} x {a.flow_steps} flow steps")
    scale = EXP["layers"] / n
    res["expert"] = dict(p50_ms=p50 * scale, p95_ms=p95 * scale, flow_steps=a.flow_steps,
                         layers_run=n, extrapolated=scale != 1.0)
    del layers, bkv, x
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="vision,prefill,decode,expert")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--vision-layers", type=int, default=27)
    p.add_argument("--llm-layers", type=int, default=4,
                   help="layers actually built; result is scaled to 36. Use 36 for a true "
                        "full-stack run (needs ~16 GiB free -- run headless).")
    p.add_argument("--expert-layers", type=int, default=4)
    p.add_argument("--prefill", type=int, default=DEF_PREFILL)
    p.add_argument("--vit-slots", type=int, default=DEF_VIT_SLOTS)
    p.add_argument("--vit-tokens", type=int, default=DEF_VIT_TOK)
    p.add_argument("--waypoints", type=int, default=DEF_WAYPOINTS)
    p.add_argument("--flow-steps", type=int, default=DEF_FLOW_STEPS)
    p.add_argument("--gen-tokens", type=int, default=300, help="CoC + trajectory tokens to bill for")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--with-head", action="store_true", help="include lm_head in prefill")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--sdp", default="mem_efficient", choices=["mem_efficient", "math", "auto"])
    p.add_argument("--json", default=None)
    a = p.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable -- refusing to benchmark on CPU"
    dt = getattr(torch, a.dtype)
    cap = torch.cuda.get_device_capability()
    free, total = torch.cuda.mem_get_info()
    print(f"device {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}   "
          f"torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"memory {free / 2**30:.1f} GiB free / {total / 2**30:.1f} GiB   dtype {a.dtype}   sdp {a.sdp}\n")

    res = dict(meta=dict(device=torch.cuda.get_device_name(0), sm=f"{cap[0]}{cap[1]}",
                         torch=torch.__version__, cuda=torch.version.cuda, dtype=a.dtype,
                         sdp=a.sdp, prefill=a.prefill, gen_tokens=a.gen_tokens,
                         flow_steps=a.flow_steps, ts=time.strftime("%Y-%m-%dT%H:%M:%S")))

    ctx = torch.backends.cuda.sdp_kernel(
        enable_flash=False,                       # sm_72 has no FlashAttention kernel
        enable_mem_efficient=a.sdp != "math",
        enable_math=a.sdp != "mem_efficient",
    ) if a.sdp != "auto" else torch.backends.cuda.sdp_kernel()

    want = set(s.strip() for s in a.stages.split(","))
    with ctx:
        if "vision" in want:
            stage_vision(a, dt, res)
        if "prefill" in want:
            stage_prefill(a, dt, res)
        if "decode" in want:
            stage_decode(a, dt, res)
        if "expert" in want:
            stage_expert(a, dt, res)

    if want >= {"vision", "prefill", "decode", "expert"}:
        v = res["vision"]["p50_ms"]
        pf = res["prefill"]["p50_ms"]
        dc = res["decode"]["per_token_ms"] * a.gen_tokens
        ex = res["expert"]["p50_ms"]
        tot = v + pf + dc + ex
        res["end_to_end"] = dict(total_ms=tot, hz=1000.0 / tot,
                                 breakdown_ms=dict(vision=v, prefill=pf, decode=dc, expert=ex))
        print(f"\n  end-to-end @ {a.gen_tokens} generated tokens")
        for k, val in (("vision", v), ("prefill", pf), (f"decode x{a.gen_tokens}", dc), ("expert", ex)):
            print(f"    {k:24s} {val:10.1f} ms  {val / tot:6.1%}")
        print(f"    {'TOTAL':24s} {tot:10.1f} ms  ({1000.0 / tot:.3f} Hz)")
        print(f"    {'vs 99 ms on-vehicle':24s} {tot / 99.0:10.1f} x slower")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
