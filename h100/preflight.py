#!/usr/bin/env python3
"""Preflight check for an H100 compute node -- answers "do I need `module load cuda`?"

PyTorch from PyPI ships its own CUDA runtime, cuBLAS and cuDNN as nvidia-*-cu12
wheels inside the venv. At runtime it needs only the *driver* (libcuda.so.1),
which comes from the kernel module, not from an environment module. Loading a
system CUDA can therefore be a no-op -- or worse, put a different minor version of
cuBLAS/cuDNN ahead of the ones torch was built against.

This reads /proc/self/maps after importing torch to show which shared objects were
actually loaded, so you can see whether the venv's libraries or the module's won.

    python h100/preflight.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

WATCH = ("libcuda.so", "libcudart", "libcublas", "libcudnn", "libnvrtc", "libcufft")


def section(title):
    print("\n" + title)
    print("-" * len(title))


def loaded_libraries():
    """Shared objects mapped into this process, filtered to the CUDA stack."""
    found = {}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.rstrip().rpartition(" ")[2]
                if not path.startswith("/"):
                    continue
                base = os.path.basename(path)
                for w in WATCH:
                    if base.startswith(w):
                        found.setdefault(base, path)
    except OSError:
        pass
    return found


def classify(path):
    if "/site-packages/nvidia/" in path:
        return "venv wheel"
    if "/site-packages/torch/" in path:
        return "torch bundle"
    if path.startswith(("/usr/lib", "/lib")):
        return "driver / system"
    return "MODULE or system CUDA"


def main():
    section("node")
    print("host        :", os.uname().nodename)
    print("python      :", sys.executable)
    print("venv        :", os.environ.get("VIRTUAL_ENV", "(none)"))

    section("driver (this is the part that actually matters)")
    if shutil.which("nvidia-smi"):
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True)
        print(q.stdout.strip() or q.stderr.strip())
    else:
        print("nvidia-smi NOT FOUND -- no GPU visible on this node")

    section("environment modules / CUDA paths")
    for v in ("LOADEDMODULES", "CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH"):
        val = os.environ.get(v)
        if not val:
            print("%-16s (unset)" % v)
        elif v == "LD_LIBRARY_PATH":
            print("%-16s" % v)
            for p in val.split(":"):
                if p:
                    print("                 ", p)
        else:
            print("%-16s %s" % (v, val))

    section("torch")
    import torch
    print("torch       :", torch.__version__)
    print("built for   : CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
    print("cuda avail  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        torch.zeros(8, device="cuda")           # force the runtime to load
        print("device      :", torch.cuda.get_device_name(0),
              "sm_%d%d" % torch.cuda.get_device_capability())
        free, total = torch.cuda.mem_get_info()
        print("memory      : %.1f / %.1f GiB free" % (free / 2**30, total / 2**30))
        if total / 2**30 < 24:
            print("  WARNING: Alpamayo needs >=24 GB for bf16 inference")

    section("which CUDA libraries actually got loaded")
    libs = loaded_libraries()
    if not libs:
        print("(could not read /proc/self/maps)")
    for base in sorted(libs):
        print("  %-28s %-18s %s" % (base, classify(libs[base]), libs[base]))
    module_libs = [b for b, p in libs.items()
                   if classify(p) == "MODULE or system CUDA" and not b.startswith("libcuda.so")]
    if module_libs:
        print("\n  NOTE: %s came from outside the venv. torch was built against CUDA %s;"
              % (", ".join(module_libs), torch.version.cuda))
        print("  if you hit cuBLAS/cuDNN errors, `module unload` the CUDA module and retry.")
    else:
        print("\n  All CUDA libraries came from the venv or the driver -- no CUDA module needed.")

    section("packages the pipeline needs")
    for name, stage in (("transformers", "a1"), ("flash_attn", "a1"),
                        ("physical_ai_av", "a0/a1"), ("onnx", "a3"),
                        ("onnxruntime", "a4"), ("modelopt", "a4")):
        try:
            m = __import__(name)
            print("  %-16s %-12s %s" % (name, getattr(m, "__version__", "?"), "(%s)" % stage))
        except Exception as exc:
            print("  %-16s %-12s (%s)  <- %s" % (name, "MISSING", stage, type(exc).__name__))

    try:
        import onnxruntime as ort
        try:
            ort.preload_dlls()
        except Exception:
            pass
        provs = ort.get_available_providers()
        print("\n  onnxruntime providers:", ", ".join(provs))
        if "CUDAExecutionProvider" not in provs:
            print("  WARNING: no CUDAExecutionProvider -- a4 calibration would run on CPU.")
    except ImportError:
        pass

    print("\nOK")


if __name__ == "__main__":
    main()
