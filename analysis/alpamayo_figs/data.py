"""Find and load the project's artefacts, wherever this script is being run.

The same figure script has to work on the H100 (where the golden rollout lives),
on the Mac (where only the payload is), and on a laptop with nothing at all. So
loading is explicit about what it found: every loader returns (data, provenance)
and never silently substitutes something else.
"""
from __future__ import annotations

import os

import numpy as np

CANDIDATE_ROOTS = [
    os.environ.get("ALPAMAYO_ROOT"),
    os.environ.get("ALPAMAYO_WORK"),
    "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work",
    os.path.expanduser("~/alpamayo-payload"),
    "/mnt/ssdhome/models/alpamayo",
]


def find_root(explicit=None):
    """The first directory that actually holds a golden/ or onnx/ tree.

    An explicit --root is never quietly replaced by a fallback: silently
    plotting a different machine's data than the one that was asked for is
    exactly the failure this whole analysis exists to prevent.
    """
    if explicit:
        ok = os.path.isdir(explicit) and (
            os.path.isdir(os.path.join(explicit, "golden"))
            or os.path.isdir(os.path.join(explicit, "onnx")))
        return explicit if ok else None
    for root in CANDIDATE_ROOTS:
        if root and os.path.isdir(root) and (
                os.path.isdir(os.path.join(root, "golden"))
                or os.path.isdir(os.path.join(root, "onnx"))):
            return root
    return None


def load_golden(root=None):
    """(arrays, path) from golden/activations.npz + inputs.npz, or (None, reason).

    Keys written by h100/a1_golden.py:
        visual, deepstack0..2, layer0, layer8, layer16, layer24, layer35,
        prefill_norm, pred_xyz, pred_rot        (activations.npz)
        pixel_values, input_ids, ego_history_xyz, ...   (inputs.npz)
    """
    root = find_root(root)
    if root is None:
        return None, "no ALPAMAYO_ROOT with a golden/ directory on this machine"
    path = os.path.join(root, "golden", "activations.npz")
    if not os.path.exists(path):
        return None, "%s does not exist" % path
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            out[k] = z[k]
    inputs = os.path.join(root, "golden", "inputs.npz")
    if os.path.exists(inputs):
        with np.load(inputs, allow_pickle=True) as z:
            for k in z.files:
                out["in_" + k] = z[k]
    return out, path


def layer_keys(golden):
    """The captured residual-stream taps, in layer order."""
    keys = [k for k in golden if k.startswith("layer") and k[5:].isdigit()]
    return sorted(keys, key=lambda k: int(k[5:]))


def synthetic_golden(seed=0):
    """A stand-in with roughly the right shape and scale, for layout work only.

    Deliberately not a good imitation: the point is to exercise the plotting code
    when the cluster is not reachable. Anything drawn from this is stamped.
    """
    rng = np.random.default_rng(seed)
    out = {}
    # Residual streams that grow with depth, which is the qualitative fact the
    # real capture shows; the magnitudes here are invented.
    for i, scale in zip((0, 8, 16, 24, 35), (7.5e3, 1.1e4, 1.5e4, 2.0e4, 2.6e4)):
        x = rng.standard_normal((1, 256, 4096)).astype(np.float32)
        x = x / np.abs(x).max() * scale
        out["layer%d" % i] = x
    out["prefill_norm"] = rng.standard_normal((1, 256, 4096)).astype(np.float32)
    t = np.linspace(0, 1, 64)
    out["pred_xyz"] = np.stack([57.1 * t, 1.02 * t ** 2, np.zeros_like(t)], -1)[None]
    return out


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))
