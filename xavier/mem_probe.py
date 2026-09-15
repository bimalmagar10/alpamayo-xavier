#!/usr/bin/env python3
"""How much memory can this Xavier really give out? (diagnostic, ~2 minutes)

run_alpamayo.py could not load decode's 10 engines (15.2 GB) even with 20.8 GB
reported free, failing at ~12.3 GB. This asks two questions in a fresh process:

  1. TensorRT engines: how many plans load before NvMap says ENOMEM?
  2. plain CUDA memory: how many blocks of the same size can torch allocate?

If torch reaches far past where the engines stopped, the limit is specific to
TensorRT's allocations rather than to the board's free memory.

    python xavier/mem_probe.py
    python xavier/mem_probe.py --engines "$ALPAMAYO_WORK/engines/prefill.p*.fp16.plan"
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WORK = os.environ.get("ALPAMAYO_WORK", "/mnt/ssdhome/models/alpamayo")


def sysmem():
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            d[k] = int(v.split()[0]) * 1024
        return "sys free %5.1f, avail %5.1f, cached %4.1f GB" % (
            d.get("MemFree", 0) / 1e9, d.get("MemAvailable", 0) / 1e9, d.get("Cached", 0) / 1e9)
    except (OSError, ValueError, IndexError):
        return "sys ?"


def sysfree():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemFree"):
                return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return 0.0


def gpu_free():
    return torch.cuda.mem_get_info()[0] / 1e9


def probe_engines(paths, skip=(), make_context=True):
    from alpamayo_xavier.trt_runner import Engine
    keep, total, before = [], 0.0, sysfree()
    for p in paths:
        try:
            keep.append(Engine(p, skip=skip, make_context=make_context))
        except RuntimeError as exc:
            print("  STOPPED at %s after %.1f GB of plans" % (os.path.basename(p), total))
            print("    %s" % str(exc).split("\n")[0][:110])
            break
        total += os.path.getsize(p) / 1e9
        now = sysfree()
        print("  %-24s %5.1f GB of plans · this engine cost %4.2f GB of system memory · %s"
              % (os.path.basename(p), total, before - now, sysmem()))
        before = now
    for e in keep:
        e.close()
    del keep
    torch.cuda.empty_cache()
    return total


def probe_torch(block_gb, limit_gb=30.0):
    blocks, total = [], 0.0
    while total < limit_gb:
        try:
            blocks.append(torch.empty(int(block_gb * 1e9), dtype=torch.uint8, device="cuda"))
        except RuntimeError as exc:
            print("  STOPPED after %.1f GB: %s" % (total, str(exc).split("\n")[0][:80]))
            break
        total += block_gb
        print("  %5.1f GB of %.1f GB blocks · gpu free %5.1f · %s"
              % (total, block_gb, gpu_free(), sysmem()))
    del blocks
    torch.cuda.empty_cache()
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default=os.path.join(WORK, "engines", "decode.p*.fp16.plan"))
    ap.add_argument("--block-gb", type=float, default=1.5)
    args = ap.parse_args()
    paths = sorted(glob.glob(args.engines))
    print("%s · %s\n" % (torch.cuda.get_device_name(0), sysmem()))
    print("1. TensorRT engines (%d plans, %.1f GB in total)"
          % (len(paths), sum(os.path.getsize(p) for p in paths) / 1e9))
    eng = probe_engines(paths)
    print("\n1b. the same engines, without allocating the KV buffers (bind() supplies those)")
    eng_skip = probe_engines(paths, skip=("past_k", "past_v"))
    print("\n1c. engines only, no execution context (is Myelin copying the weights per context?)")
    eng_noctx = probe_engines(paths, skip=("past_k", "past_v"), make_context=False)
    print("\n2. plain CUDA blocks of %.1f GB" % args.block_gb)
    tor = probe_torch(args.block_gb)
    print("\nengines: %.1f GB   ·   no KV buffers: %.1f GB   ·   no context either: %.1f GB   ·   plain CUDA: %.1f GB"
          % (eng, eng_skip, eng_noctx, tor))
    print("plans are %.1f GB in total; whichever column reaches that can hold the whole stage"
          % (sum(os.path.getsize(p) for p in paths) / 1e9))


if __name__ == "__main__":
    main()
