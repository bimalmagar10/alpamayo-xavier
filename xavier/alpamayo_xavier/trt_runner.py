"""Minimal TensorRT 8.5 runner backed by torch CUDA tensors.

Using torch for device memory avoids pycuda entirely -- one less aarch64 build to
fight -- and lets engine outputs be sliced and copied with ordinary tensor ops.
On Xavier host and device memory are the same physical LPDDR4x, so a tensor
handed to TensorRT costs no transfer.
"""
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


class Engine(object):
    """One deserialized .plan with persistent, pre-allocated bindings."""

    def __init__(self, path, logger_severity=trt.Logger.WARNING):
        self.logger = trt.Logger(logger_severity)
        with open(path, "rb") as f, trt.Runtime(self.logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                "failed to deserialize %s -- a plan is locked to the GPU, the "
                "TensorRT version and the compute capability it was built on. "
                "Rebuild it on this device." % path)
        self.context = self.engine.create_execution_context()
        self.path = path

        self.inputs, self.outputs, self.shapes = {}, {}, {}
        self._bindings = [0] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = _TRT_TO_TORCH[self.engine.get_binding_dtype(i)]
            self.shapes[name] = shape
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
        for name, value in (feed or {}).items():
            dst = self.inputs[name]
            src = value if torch.is_tensor(value) else torch.from_numpy(np.ascontiguousarray(value))
            dst.copy_(src.to(dst.dtype), non_blocking=True)
        handle = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        if not self.context.execute_async_v2(bindings=self._bindings, stream_handle=handle):
            raise RuntimeError("TensorRT execution failed for %s" % self.path)
        return self.outputs

    def __repr__(self):
        ins = ", ".join("%s%s" % (k, v) for k, v in self.shapes.items())
        return "<Engine %s: %s>" % (self.path.split("/")[-1], ins)


def load_engines(engine_dir, precision="int8", names=("vision", "prefill", "decode", "expert")):
    import os
    out = {}
    for n in names:
        p = os.path.join(engine_dir, "%s.%s.plan" % (n, precision))
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run build_engines.sh first" % p)
        out[n] = Engine(p)
        print("loaded %s" % out[n])
    return out
