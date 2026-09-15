"""Minimal TensorRT 8.5 runner backed by torch CUDA tensors.

Using torch for device memory avoids pycuda entirely -- one less aarch64 build to
fight -- and lets engine outputs be sliced and copied with ordinary tensor ops.
On Xavier host and device memory are the same physical LPDDR4x, so a tensor
handed to TensorRT costs no transfer.
"""
import gc
import os
from typing import Dict, Optional

import numpy as np
import tensorrt as trt
import torch

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
}


class _SharedScratch(object):
    """One activation ("device memory") buffer shared by every engine.

    By default each TensorRT execution context allocates its own scratch, sized for
    the biggest intermediate it will ever hold. For a 3,006-token prefill piece
    that is the attention score matrix -- GBs -- and 13 pieces loaded together ran
    the Xavier out of memory. Our engines only ever run one after another on one
    CUDA stream, so they can all use the same buffer, sized to the largest.
    """

    def __init__(self):
        self.buf = None
        self.size = 0
        self.live = {}                        # engine id -> bytes it needs

    def reserve(self, key, nbytes):
        self.live[key] = nbytes
        need = max(self.live.values())
        if need > self.size:
            torch.cuda.synchronize()          # nothing may still be running on the old one
            self.buf = None
            torch.cuda.empty_cache()
            self.buf = torch.empty(need, dtype=torch.uint8, device="cuda")
            self.size = need

    def drop(self, key):
        """Give the buffer back once a stage's last engine is closed. Vision sizes it
        at 2.5 GB; decode needs far less and needs that memory for its weights."""
        self.live.pop(key, None)
        if not self.live and self.buf is not None:
            torch.cuda.synchronize()
            self.buf = None
            self.size = 0
            torch.cuda.empty_cache()

    @property
    def ptr(self):
        return int(self.buf.data_ptr()) if self.buf is not None else 0


SCRATCH = _SharedScratch()


class Engine(object):
    """One deserialized .plan with persistent, pre-allocated bindings."""

    def __init__(self, path, logger_severity=trt.Logger.WARNING, skip=(), make_context=True):
        """skip: bindings that bind() will point at someone else's memory, so no
        buffer is allocated for them here. The KV cache is 528 MB per engine."""
        self.logger = trt.Logger(logger_severity)
        with open(path, "rb") as f, trt.Runtime(self.logger) as rt:
            # Map the plan instead of reading it. f.read() puts a second copy of every
            # engine's weights in anonymous memory, which on a Xavier comes out of the
            # same pool as the GPU's -- measured at ~2x the plan size per engine, so
            # decode's 10 engines never fitted. Mapped pages are file-backed and the
            # kernel can drop them as soon as TensorRT has copied the weights.
            blob = np.memmap(f, dtype=np.uint8, mode="r")
            self.engine = rt.deserialize_cuda_engine(memoryview(blob))
            if self.engine is None:
                # Out of memory. Hand back everything not in use and try once more.
                gc.collect()
                torch.cuda.empty_cache()
                self.engine = rt.deserialize_cuda_engine(memoryview(blob))
            del blob
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass
        if self.engine is None:
            raise RuntimeError(
                "failed to deserialize %s. If TensorRT reported 'out of memory' just "
                "above, too much is loaded at once -- free memory or load fewer engines. "
                "Otherwise the plan was built for another GPU or TensorRT version: "
                "rebuild it on this device." % path)
        # No private scratch: this context runs in SCRATCH, set just before each call.
        self.scratch_bytes = int(self.engine.device_memory_size)
        self.context = None
        if make_context:                      # make_context=False is for mem_probe only
            self.context = self.engine.create_execution_context_without_device_memory()
            SCRATCH.reserve(id(self), self.scratch_bytes)
        self.path = path

        self.inputs, self.outputs, self.shapes = {}, {}, {}
        self._bindings = [0] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = _TRT_TO_TORCH[self.engine.get_binding_dtype(i)]
            self.shapes[name] = shape
            if name in skip:
                self._bindings[i] = 0            # bind() must supply this one
                self.inputs[name] = None
                continue
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            self._bindings[i] = int(buf.data_ptr())
            (self.inputs if self.engine.binding_is_input(i) else self.outputs)[name] = buf

    def bind(self, name, tensor):
        """Point an input binding at an existing device tensor (no copy).

        Use this for the persistent KV cache so it is never re-uploaded.
        """
        idx = self.engine.get_binding_index(name)
        if idx < 0:
            raise KeyError("%s has no binding %r" % (self.path, name))
        expected = self.shapes[name]
        if tuple(tensor.shape) != expected:
            raise ValueError("%s expects %s, got %s" % (name, expected, tuple(tensor.shape)))
        if not tensor.is_contiguous():
            raise ValueError("%s must be contiguous" % name)
        self._bindings[idx] = int(tensor.data_ptr())
        self.inputs[name] = tensor

    def __call__(self, feed=None, stream=None):
        """Copy `feed` into the input buffers, run, return the output buffers."""
        if 0 in self._bindings:
            missing = [n for n, i in ((n, self.engine.get_binding_index(n)) for n in self.shapes)
                       if self._bindings[i] == 0]
            raise RuntimeError("%s: bind() these first: %s" % (self.path, ", ".join(missing)))
        for name, value in (feed or {}).items():
            dst = self.inputs[name]
            src = value if torch.is_tensor(value) else torch.from_numpy(np.ascontiguousarray(value))
            dst.copy_(src.to(dst.dtype), non_blocking=True)
        handle = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        if self.scratch_bytes:
            self.context.device_memory = SCRATCH.ptr
        if not self.context.execute_async_v2(bindings=self._bindings, stream_handle=handle):
            raise RuntimeError("TensorRT execution failed for %s" % self.path)
        return self.outputs

    def close(self):
        """Release the engine and its bindings.

        The four FP16 engines total ~35 GB of weights against ~25 GiB free on a
        Xavier, so they cannot all be resident. Freeing each after its stage keeps
        the peak at one engine plus the KV cache. INT8 halves the total and fits
        everything at once.
        """
        for buf in [b for b in list(self.inputs.values()) + list(self.outputs.values())
                    if b is not None]:
            del buf
        self.inputs.clear()
        self.outputs.clear()
        self._bindings = []
        if self.context is not None:
            del self.context
        del self.engine
        SCRATCH.drop(id(self))
        gc.collect()
        torch.cuda.empty_cache()

    def __repr__(self):
        ins = ", ".join("%s%s" % (k, v) for k, v in self.shapes.items())
        return "<Engine %s: %s, scratch %.0f MB>" % (self.path.split("/")[-1], ins,
                                                      self.scratch_bytes / 1e6)


def engine_path(engine_dir, name, precision):
    import os
    p = os.path.join(engine_dir, "%s.%s.plan" % (name, precision))
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run build_engines.sh first" % p)
    return p


def plan_bytes(engine_dir, precision):
    """Bytes of every plan of this precision -- whole graphs and pieces alike."""
    import glob
    import os
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(engine_dir, "*.%s.plan" % precision)))


def load_engines(engine_dir, precision="int8", names=("vision", "prefill", "decode", "expert")):
    out = {}
    for n in names:
        out[n] = Engine(engine_path(engine_dir, n, precision))
        print("loaded %s" % out[n])
    return out
