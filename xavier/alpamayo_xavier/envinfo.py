"""Best-effort runtime and artifact provenance for reproducible measurements.

Probe failures are represented as unavailable values or explicit errors. Never
infer a JetPack version or import unused heavyweight libraries during a benchmark.
"""
from __future__ import print_function

import ctypes
import glob
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys


def _read(path, limit=4096):
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", "replace").strip("\x00\n ")
    except (OSError, IOError):
        return None


def _run(cmd, timeout=10):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, check=False)
        if result.returncode != 0:
            return None
        return result.stdout.decode("utf-8", "replace").strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _dpkg(name):
    return _run(["dpkg-query", "--showformat=${Version}", "--show", name])


def driver_version():
    """CUDA driver version through libcuda, the one thing torch does not expose."""
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(v)) != 0:
            return None
        return "%d.%d" % (v.value // 1000, (v.value % 1000) // 10)
    except Exception:
        return None


def jetpack():
    """L4T and JetPack as the board itself reports them.

    No L4T -> JetPack table is hardcoded on purpose: the mapping changes with point
    releases and a wrong constant here would quietly mislabel every result. The
    nvidia-jetpack package version is the authoritative answer when it is installed.
    """
    rel = _read("/etc/nv_tegra_release")
    out = dict(l4t_release=rel and rel.splitlines()[0],
               l4t_core_package=_dpkg("nvidia-l4t-core"),
               jetpack_package=_dpkg("nvidia-jetpack"),
               model=_read("/proc/device-tree/model"),
               soc_family=_read("/sys/devices/soc0/family"),
               machine=_read("/proc/device-tree/compatible"))
    for key, cmd in (("power_mode", ["nvpmodel", "-q"]),
                     ("clocks", ["jetson_clocks", "--show"])):
        out[key] = _run(cmd, timeout=10)
    return out


def gpu():
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return dict(available=False)
    p = torch.cuda.get_device_properties(0)
    cc = torch.cuda.get_device_capability(0)
    return dict(available=True, name=p.name,
                compute_capability="sm_%d%d" % cc,
                architecture={(7, 0): "Volta", (7, 2): "Volta", (7, 5): "Turing",
                              (8, 0): "Ampere", (8, 6): "Ampere", (8, 7): "Ampere",
                              (8, 9): "Ada", (9, 0): "Hopper"}.get(tuple(cc), "unknown"),
                multiprocessors=p.multi_processor_count,
                total_memory_bytes=p.total_memory,
                unified_memory=True if "Xavier" in (p.name or "") or "Orin" in (p.name or "") else None,
                driver_cuda_api_version=driver_version())


def libraries():
    """Separate loaded modules, installed distributions, and CUDA build versions."""
    out = {"loaded": {}}
    for name in ("torch", "numpy", "tensorrt", "PIL", "onnx", "onnxruntime", "cv2"):
        mod = sys.modules.get(name)
        if mod is not None:
            out["loaded"][name] = dict(version=getattr(mod, "__version__", None),
                                       path=getattr(mod, "__file__", None))
    torch = sys.modules.get("torch")
    if torch is not None:
        out["torch_cuda_build"] = torch.version.cuda
        out["cudnn_runtime"] = torch.backends.cudnn.version()
        out["torch_git"] = getattr(torch.version, "git_version", None)
        out["torch_build_config"] = torch.__config__.show()
        out["torch_execution"] = dict(cpu_threads=torch.get_num_threads(),
                                      interop_threads=torch.get_num_interop_threads(),
                                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                                      cudnn_deterministic=torch.backends.cudnn.deterministic)
    out["cuda_toolkit_nvcc"] = _run(["nvcc", "--version"])
    out["cuda_version_file"] = _read("/usr/local/cuda/version.json") or _read("/usr/local/cuda/version.txt")
    out["cuda_driver_api"] = driver_version()
    # These are the libraries mapped by THIS process, including TensorRT's native
    # allocations which PyTorch's memory counters do not account for.
    mapped = _read("/proc/self/maps", 8 * 1024 * 1024) or ""
    out["mapped_accelerator_libraries"] = sorted(set(
        line.split()[-1] for line in mapped.splitlines()
        if "/" in line and any(n in line for n in
                                ("libcuda", "libcudnn", "libnvinfer", "libcublas", "libnvonnxparser"))))
    out["cuda_runtime_api"] = None
    for path in out["mapped_accelerator_libraries"]:
        if "libcudart" in os.path.basename(path):
            try:
                lib = ctypes.CDLL(path)
                value = ctypes.c_int()
                if lib.cudaRuntimeGetVersion(ctypes.byref(value)) == 0:
                    out["cuda_runtime_api"] = "%d.%d" % (value.value // 1000, (value.value % 1000) // 10)
                    break
            except (OSError, AttributeError):
                pass
    packages = _run(["dpkg-query", "-W", "-f=${binary:Package}=${Version}\n"])
    out["nvidia_packages"] = ([line for line in packages.splitlines()
                               if line.startswith(("nvidia-", "libnvinfer", "libcudnn", "cuda-"))]
                              if packages else None)
    return out


def code(repo_dir):
    """Which revision produced this result, and whether the tree was clean."""
    def git(*a):
        return _run(["git", "-C", repo_dir] + list(a), timeout=10)
    head = git("rev-parse", "HEAD")
    if head is None:
        return dict(dir=repo_dir, source_sha256=source_hashes(repo_dir))
    return dict(dir=repo_dir, commit=head, branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(git("status", "--porcelain")),
                described=git("describe", "--always", "--dirty"), source_sha256=source_hashes(repo_dir))


def source_hashes(repo_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(repo_dir, "xavier", "**", "*.py"), recursive=True)):
        try:
            with open(path, "rb") as f:
                out[os.path.relpath(path, repo_dir)] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            pass
    return out


def artifacts(engine_dir, precision, work):
    """Inventory only; selected backend artifacts are listed in configuration.

    Size and mtime are recorded without reading multi-GB files into the page cache.
    They are NOT content identities. Small metadata files carry SHA256 hashes.
    """
    out = {"engine_dir": engine_dir, "plans": {}, "total_plan_bytes": 0}
    for p in sorted(glob.glob(os.path.join(engine_dir, "*.%s.plan" % precision))):
        try:
            st = os.stat(p)
        except OSError:
            continue
        out["plans"][os.path.basename(p)] = dict(bytes=st.st_size, mtime=int(st.st_mtime))
        out["total_plan_bytes"] += st.st_size
    for label, rel in (("weight_map", "engines/weight_map.json"), ("pieces", "engines/pieces.json"),
                       ("vocab", "fixtures/vocab.json"), ("meta", "fixtures/meta.json")):
        path = os.path.join(work, rel)
        if label == "weight_map" and not os.path.exists(path):
            path = os.path.join(work, "onnx", "weight_map.json")
        try:
            with open(path, "rb") as f:
                blob = f.read()
            out[label] = dict(path=path, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
        except OSError:
            out[label] = None
    out["filesystem"] = _run(["df", "-hP", work])
    return out


def host():
    u = platform.uname()
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in ("MemTotal", "SwapTotal"):
                mem[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return dict(hostname=u.node, kernel=u.release, machine=u.machine, processor=u.processor or None,
                architecture=platform.architecture()[0], os=_read("/etc/os-release", 512),
                cpu_count=os.cpu_count(), memory_total_bytes=mem.get("MemTotal"),
                swap_total_bytes=mem.get("SwapTotal"),
                cpu_model=next((l.split(":", 1)[1].strip()
                                for l in (_read("/proc/cpuinfo", 8192) or "").splitlines()
                                if l.startswith(("model name", "CPU part"))), None))


def collect(work, engine_dir, precision, repo_dir=None, pip_freeze=True):
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(python=dict(version=platform.python_version(), executable=sys.executable,
                          implementation=platform.python_implementation(),
                          virtualenv=os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")),
               probe_errors={}, unavailable_semantics="null means unavailable or probe failed; no version is guessed")
    env["execution_environment"] = {k: os.environ.get(k) for k in
                                    ("CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING",
                                     "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    probes = dict(host=host, jetson=jetpack, gpu=gpu, libraries=libraries,
                  code=lambda: code(repo_dir), artifacts=lambda: artifacts(engine_dir, precision, work))
    for name, probe in probes.items():
        try:
            env[name] = probe()
        except Exception as exc:
            env[name] = None
            env["probe_errors"][name] = str(exc)
    if pip_freeze:
        # Inventory names/versions only: pip freeze can expose credential-bearing
        # direct URLs and imports of unused libraries can change benchmark memory.
        try:
            env["installed_packages"] = sorted(
                ({"name": d.metadata.get("Name", "unknown"), "version": d.version}
                 for d in importlib.metadata.distributions()), key=lambda d: d["name"].lower())
        except Exception as exc:
            env["probe_errors"]["installed_packages"] = str(exc)
    return env


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ssdhome/models/alpamayo"
    print(json.dumps(collect(w, os.path.join(w, "engines"), "fp16"), indent=2, default=str))
