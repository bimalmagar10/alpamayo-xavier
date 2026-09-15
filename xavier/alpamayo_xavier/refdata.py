"""Deterministic test inputs shared by h100/a6_reference_io.py and xavier/verify.py.

The expert engine has no golden activations, so it is checked against an fp32
ONNX Runtime run of its own graph on fixed inputs. Its KV caches are 264 MB each,
too big to ship, so both machines regenerate them from a seed.

Values are k/1024 for integers k in [-2048, 2048): exactly representable in fp16,
so no rounding differs between machines. They are drawn with the legacy
RandomState, whose stream NumPy keeps frozen across versions, so the Mac (NumPy
2.x) and the Xavier (NumPy 1.x, Python 3.8) produce the same bytes. `checksum`
lets verify.py prove that rather than assume it.
"""
import hashlib

import numpy as np


def grid16(rs, shape):
    """Uniform values on a 1/1024 grid in [-2, 2), as fp16."""
    k = rs.randint(-2048, 2048, size=shape, dtype=np.int16)
    return (k.astype(np.float32) / 1024.0).astype(np.float16)


def kv_cache(seed, shape):
    """Key and value caches of `shape` = [layers, 1, kv_heads, max_seq, head_dim]."""
    rs = np.random.RandomState(seed)
    k = np.empty(shape, np.float16)
    v = np.empty(shape, np.float16)
    for i in range(shape[0]):                 # layer by layer keeps the temporaries small
        k[i] = grid16(rs, shape[1:])
        v[i] = grid16(rs, shape[1:])
    return k, v


def checksum(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
