#!/usr/bin/env python3
"""
xavier_hw_probe.py -- calibrate the Jetson AGX Xavier roofline at Alpamayo's
actual GEMM shapes, so every later latency prediction rests on measured
constants instead of datasheet peaks.

Reports:
  * achievable fp16 GEMM throughput (TFLOP/s) at the M,N,K triples that
    Alpamayo-1 actually issues, for prefill (M=1550) and decode (M=1)
  * achievable memory bandwidth (GB/s) for large streaming reads, which is
    what sets decode latency
  * the derived MFU / MBU constants to plug into the roofline

Target: JetPack 5.1.x, Python 3.8, torch 2.1.0a0+...nv23.06, sm_72.
Run `sudo nvpmodel -m 0 && sudo jetson_clocks` first.
"""
import argparse
import statistics
import torch

PEAK_FP16_TFLOPS = 11.3    # 64 Volta tensor cores @ 1.377 GHz
PEAK_BW_GBS = 136.5        # 256-bit LPDDR4x-2133


def time_cuda(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    evs = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        evs.append((s, e))
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in evs)
    return statistics.median(ms), ms[int(0.95 * (len(ms) - 1))]


def gemm_sweep():
    # (label, M, N, K) -- the GEMMs Alpamayo-1 actually issues, batch 1.
    shapes = [
        ("prefill  qkv-ish  ", 1550, 4096, 4096),
        ("prefill  kv proj   ", 1550, 1024, 4096),
        ("prefill  mlp gate  ", 1550, 12288, 4096),
        ("prefill  mlp down  ", 1550, 4096, 12288),
        ("vision   fc1       ", 5760, 4304, 1152),
        ("expert   mlp gate  ", 64, 8256, 2048),
        ("decode   qkv       ", 1, 4096, 4096),
        ("decode   mlp gate  ", 1, 12288, 4096),
        ("lm_head            ", 1, 155697, 4096),
    ]
    print("\n== fp16 GEMM (median of 50) ==")
    print(f"{'shape':22s} {'M':>6s} {'N':>7s} {'K':>6s} {'ms':>9s} {'TFLOP/s':>9s} {'MFU':>7s}")
    best = 0.0
    for label, m, n, k in shapes:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        med, _ = time_cuda(lambda: torch.mm(a, b))
        tflops = 2 * m * n * k / (med * 1e-3) / 1e12
        mfu = tflops / PEAK_FP16_TFLOPS
        if m >= 1024:
            best = max(best, mfu)
        print(f"{label:22s} {m:6d} {n:7d} {k:6d} {med:9.3f} {tflops:9.2f} {mfu:6.1%}")
        del a, b
        torch.cuda.empty_cache()
    return best


def bandwidth_probe():
    print("\n== streaming memory bandwidth ==")
    best = 0.0
    for mb in (256, 512, 1024, 2048):
        n = mb * 1024 * 1024 // 2          # fp16 elements
        x = torch.empty(n, device="cuda", dtype=torch.float16).normal_()
        y = torch.empty_like(x)
        med, _ = time_cuda(lambda: y.copy_(x), warmup=5, iters=20)
        gbs = 2 * x.numel() * 2 / (med * 1e-3) / 1e9   # read + write
        best = max(best, gbs)
        print(f"  copy {mb:5d} MiB  {med:8.3f} ms  {gbs:7.1f} GB/s  ({gbs / PEAK_BW_GBS:5.1%} of peak)")
        del x, y
        torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable -- do not benchmark on CPU"
    cap = torch.cuda.get_device_capability()
    print(f"device      : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
    print(f"torch       : {torch.__version__}  (cuda {torch.version.cuda})")
    print(f"bf16 usable : {torch.cuda.is_bf16_supported()}   <- expect False-ish on sm_72")
    free, total = torch.cuda.mem_get_info()
    print(f"gpu memory  : {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")

    mfu = gemm_sweep()
    mbu = bandwidth_probe()

    print("\n== calibrated roofline constants ==")
    print(f"  MFU (large fp16 GEMM) : {mfu:.3f}   -> effective {PEAK_FP16_TFLOPS * mfu:.2f} TFLOP/s")
    print(f"  MBU (streaming)       : {mbu / PEAK_BW_GBS:.3f}   -> effective {mbu:.1f} GB/s")
    print("\nPlug these into roofline.py and re-run the prediction table.")


if __name__ == "__main__":
    main()
