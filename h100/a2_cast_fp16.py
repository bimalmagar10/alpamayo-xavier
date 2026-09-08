#!/usr/bin/env python3
"""Stage A2 (H100) -- audit the bfloat16 -> float16 cast, then perform it.

Volta (sm_72) has no bfloat16 arithmetic, so every Alpamayo weight has to become
float16 before it can reach the Xavier. bfloat16 carries float32's exponent
range; float16 saturates at 65504 and goes subnormal below 6.1e-5. A blind
`.half()` on 22 GB is therefore not safe -- it is just usually lucky.

This script reports every tensor at risk, keeps the unsafe ones in float32
(they are small), and writes a float16 checkpoint plus a manifest recording
exactly which tensors were promoted and why.

    python a2_cast_fp16.py --src "$ALPAMAYO_MODEL" --dst "$ALPAMAYO_ROOT/ckpt/alpamayo-fp16"
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP16_MAX = 65504.0
FP16_TINY = 6.104e-5          # smallest normal float16
OVERFLOW_HEADROOM = 0.5       # promote if max|w| exceeds half of FP16_MAX
SUBNORMAL_LIMIT = 0.02        # promote if >2% of values would go subnormal


def audit(t: torch.Tensor) -> dict:
    f = t.detach().to(torch.float32)
    a = f.abs()
    nz = a[a > 0]
    return dict(
        max=float(a.max()) if a.numel() else 0.0,
        subnormal_frac=float((nz < FP16_TINY).float().mean()) if nz.numel() else 0.0,
        numel=int(t.numel()),
    )


def verdict(st: dict) -> tuple[str, str]:
    if st["max"] > FP16_MAX:
        return "float32", f"overflows fp16 (max {st['max']:.1f})"
    if st["max"] > FP16_MAX * OVERFLOW_HEADROOM:
        return "float32", f"within {OVERFLOW_HEADROOM:.0%} of fp16 max ({st['max']:.1f})"
    if st["subnormal_frac"] > SUBNORMAL_LIMIT:
        return "float32", f"{st['subnormal_frac']:.1%} of values subnormal in fp16"
    return "float16", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get(
        "ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B"),
        help="directory of bf16 safetensors shards (default: $ALPAMAYO_MODEL)")
    ap.add_argument("--dst", default=os.path.join(os.environ.get(
        "ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work"), "ckpt/alpamayo-fp16"))
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors under {src}")

    manifest, promoted, index, total_bytes = {}, [], {}, 0
    for shard in shards:
        out_tensors = {}
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if not t.is_floating_point():
                    out_tensors[name] = t
                    continue
                st = audit(t)
                target, why = verdict(st)
                manifest[name] = dict(**st, dtype=target, reason=why)
                if target == "float32":
                    promoted.append((name, why, st["numel"]))
                out_tensors[name] = t.to(getattr(torch, target))

        if not args.audit_only:
            out_path = dst / shard.name
            save_file(out_tensors, str(out_path), metadata={"format": "pt"})
            for name, t in out_tensors.items():
                index[name] = shard.name
                total_bytes += t.numel() * t.element_size()
            print(f"[cast] {shard.name} -> {out_path.name}")
        del out_tensors

    print(f"\n{len(promoted)} tensor(s) held in float32:")
    for name, why, numel in sorted(promoted, key=lambda x: -x[2])[:40]:
        print(f"  {name:70s} {numel:>12,}  {why}")
    extra_mb = sum(n for _, _, n in promoted) * 2 / 2**20
    print(f"\nfloat32 promotions cost {extra_mb:.1f} MiB over pure fp16.")

    (dst / "cast_manifest.json").write_text(json.dumps(manifest, indent=1))
    if not args.audit_only:
        (dst / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total_bytes}, "weight_map": index}, indent=1))
        print(f"checkpoint: {total_bytes / 1e9:.2f} GB -> {dst}")


if __name__ == "__main__":
    main()
