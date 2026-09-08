"""Qwen3-VL interleaved 3D mRoPE tables, computed on the host.

The rotary index arithmetic is deliberately kept out of the TensorRT graphs: it
exports into a large Gather/Slice subgraph that TRT 8.5 will not fuse, and it is
cheap here. A 3,000-token prefill table is 1.5 MB.

`mrope_section = (24, 20, 20)` sums to head_dim // 2 = 64. Qwen3-VL interleaves
the temporal / height / width bands as T H W T H W ... rather than concatenating
them in three chunks, which is what `mrope_interleaved: true` selects.
"""
import numpy as np

HEAD_DIM = 128
ROPE_THETA = 5_000_000.0
MROPE_SECTION = (24, 20, 20)


def _interleave(freqs, section=MROPE_SECTION):
    """freqs: [3, S, HEAD_DIM//2] -> [S, HEAD_DIM//2]."""
    out = freqs[0].copy()
    for axis, offset in ((1, 1), (2, 2)):          # height, width
        idx = slice(offset, section[axis] * 3, 3)
        out[:, idx] = freqs[axis][:, idx]
    return out


def tables(position_ids, head_dim=HEAD_DIM, theta=ROPE_THETA, dtype=np.float16):
    """position_ids: [3, S] int -> (cos, sin), each [1, S, head_dim].

    Computed in float64 then cast: at position ~3000 with theta 5e6 the low
    frequencies need the precision, and getting this subtly wrong shifts the
    attention pattern in a way that still produces plausible trajectories.
    """
    position_ids = np.asarray(position_ids, dtype=np.float64)
    if position_ids.ndim == 1:
        position_ids = np.broadcast_to(position_ids, (3, position_ids.shape[0]))
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    freqs = position_ids[..., None] * inv           # [3, S, hd/2]
    merged = _interleave(freqs)
    emb = np.concatenate([merged, merged], axis=-1)  # [S, hd]
    return (np.cos(emb).astype(dtype)[None], np.sin(emb).astype(dtype)[None])
