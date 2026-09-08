# Alpamayo-1 (Alpamayo-R1-10B) on Jetson AGX Xavier 32GB -- analytic roofline
# All architecture numbers verified from the safetensors headers + config.json.

GBs   = 136.5e9      # LPDDR4x bandwidth, AGX Xavier
FP16  = 11.3e12      # Volta tensor-core FP16 peak @1.377 GHz (64 TC)
INT8  = 22.6e12

MFU   = 0.45         # realistic large-GEMM efficiency, Volta iGPU + thermals
MBU   = 0.60         # realistic memory-bandwidth utilisation, decode

# ---------------- architecture ----------------
# ViT: 27 blocks, d=1152, mlp=4304
vit_layer = 1152*3456 + 1152*1152 + 1152*4304 + 4304*1152          # qkv, proj, fc1, fc2
VIT_P     = 27*vit_layer + 1152*3*2*16*16 + 4*(4608*4608+4608*4096) + 2304*1152
# LLM: 36 layers, d=4096, q 32x128, kv 8x128, mlp 12288 SwiGLU
llm_layer = 4096*4096 + 4096*1024*2 + 4096*4096 + 3*4096*12288
LLM_P     = 36*llm_layer
EMB_P     = 155697*4096
HEAD_P    = 155697*4096
# Expert: 36 layers, d=2048, q 16x128, kv 8x128, mlp 8256 SwiGLU
exp_layer = 2048*2048 + 2048*1024*2 + 2048*2048 + 3*2048*8256
EXP_P     = 36*exp_layer

# ---------------- workload ----------------
VIT_TOK   = 5760     # 4 cams x 4 frames -> 2 temporal slots x 720 patches x 4 cams
PREFILL   = 1550     # 1440 visual + 48 history-traj + ~62 text
WAYPTS    = 64
FLOW      = 10       # Euler, dt=0.1
KV_B      = 36*8*128*2*2   # bytes/token of backbone KV cache (fp16)

def gemm_s(fl, peak=FP16, mfu=MFU): return fl/(peak*mfu)
def mem_s(by): return by/(GBs*MBU)

print(f"{'component':22s} {'params':>10s}  {'bf16 GB':>8s}  {'int8 GB':>8s}  {'w4 GB':>8s}")
for n,p in [("vision encoder",VIT_P),("LLM 36 layers",LLM_P),("embed_tokens",EMB_P),
            ("lm_head",HEAD_P),("action expert",EXP_P)]:
    print(f"{n:22s} {p/1e6:9.1f}M  {p*2/1e9:8.2f}  {p/1e9:8.2f}  {p*0.5625/1e9:8.2f}")
tot = VIT_P+LLM_P+EMB_P+HEAD_P+EXP_P
print(f"{'TOTAL':22s} {tot/1e6:9.1f}M  {tot*2/1e9:8.2f}  {tot/1e9:8.2f}  {tot*0.5625/1e9:8.2f}")
print(f"\nKV cache: {KV_B/1024:.0f} KiB/token -> {KV_B*PREFILL/1e6:.0f} MB at {PREFILL} tokens\n")

# ---------------- compute ----------------
vit_gemm  = 2*VIT_P*VIT_TOK
vit_attn  = 27 * 4 * (720**2) * 1152 * 8          # attention within each of 8 image-slots
pre_gemm  = 2*LLM_P*PREFILL
pre_attn  = 36 * 4 * (PREFILL**2) * 4096
exp_gemm  = 2*EXP_P*WAYPTS*FLOW
print(f"vision  FLOPs : {(vit_gemm+vit_attn)/1e12:6.2f} T  -> {gemm_s(vit_gemm+vit_attn):6.2f} s")
print(f"prefill FLOPs : {(pre_gemm+pre_attn)/1e12:6.2f} T  -> {gemm_s(pre_gemm+pre_attn):6.2f} s")
print(f"expert  FLOPs : {exp_gemm/1e12:6.2f} T (compute-bound? no)\n")

exp_mem = FLOW*(EXP_P*2 + KV_B*PREFILL)
print(f"expert  mem   : {exp_mem/1e9:6.2f} GB -> {mem_s(exp_mem):6.2f} s  (fp16, {FLOW} flow steps)\n")

print("decode, per generated token:")
for lbl, bpw in [("fp16",2.0),("int8",1.0),("w4 (4.5b)",0.5625)]:
    by = (LLM_P+HEAD_P)*bpw
    print(f"  {lbl:10s} weights {by/1e9:5.2f} GB  floor {by/GBs*1e3:6.1f} ms  realistic {mem_s(by)*1e3:6.1f} ms")

print()
for lbl,bpw in [("FP16",2.0),("INT8 weights",1.0),("W4 weights",0.5625)]:
    dec = mem_s((LLM_P+HEAD_P)*bpw)
    ex  = mem_s(FLOW*(EXP_P*bpw + KV_B*PREFILL))
    for ntok,tag in [(128,"no reasoning (128 traj tok)"),(300,"short CoC (~172+128)"),(500,"long CoC (~372+128)")]:
        v = gemm_s(vit_gemm+vit_attn); p = gemm_s(pre_gemm+pre_attn)
        tot_s = v+p+dec*ntok+ex
        print(f"{lbl:13s} {tag:28s}  vision {v:4.1f} + prefill {p:4.1f} + decode {dec*ntok:6.1f} + expert {ex:4.1f} = {tot_s:6.1f} s   ({1/tot_s:5.2f} Hz)")
    print()
