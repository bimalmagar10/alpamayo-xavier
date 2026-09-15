"""Select a matched image/ego-history sample while sharing the model artifacts."""
import json
import math
import os

import numpy as np


def read_meta(fixtures):
    with open(os.path.join(fixtures, "meta.json")) as f:
        return json.load(f)


def validate_contract(meta, reference, grid=None, reference_grid=None):
    for name in ("prefill", "max_seq", "visual_tokens", "vocab"):
        if name not in meta or name not in reference:
            raise ValueError("sample and reference metadata must specify " + name)
        if meta[name] != reference[name]:
            raise ValueError("sample %s=%s differs from engine fixtures (%s); "
                             "this sample cannot reuse these fixed-shape engines"
                             % (name, meta[name], reference[name]))
    if grid is not None and reference_grid is not None and not np.array_equal(grid, reference_grid):
        raise ValueError("image grid differs from the grid baked into the vision engine")


def load_sample(sample, shared_fixtures):
    """Return ordered images and sample fixtures; never borrow another sample's v0."""
    sample = os.path.abspath(sample)
    fixtures = os.path.join(sample, "fixtures")
    meta = read_meta(fixtures)
    validate_contract(meta, read_meta(shared_fixtures))
    if not isinstance(meta.get("v0"), (int, float)) or not math.isfinite(meta["v0"]):
        raise ValueError("sample metadata must contain its own finite v0")
    if not meta.get("clip") or "t0_us" not in meta:
        raise ValueError("sample metadata must identify its clip and t0_us")
    names = meta.get("image_files", [])
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError("sample metadata must list exactly 16 distinct images in camera/time order")
    images = []
    for name in names:
        path = os.path.abspath(os.path.join(sample, name))
        if os.path.commonpath([sample, path]) != sample or not os.path.isfile(path):
            raise ValueError("missing image or image outside sample folder: " + name)
        images.append(path)
    for name in ("input_ids.npy", "position_ids.npy", "visual_mask.npy", "image_grid_thw.npy"):
        if not os.path.isfile(os.path.join(fixtures, name)):
            raise ValueError("sample is missing " + name)
    return fixtures, meta, images


def validate_arrays(meta, ids, positions, visual_mask, embedding_rows):
    n = meta["prefill"]
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("input_ids must be an integer vector of prefill length")
    if positions.shape != (3, n) or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("position_ids must have integer shape [3, prefill]")
    if visual_mask.shape != (n,) or visual_mask.dtype != np.bool_:
        raise ValueError("visual_mask must be a boolean vector of prefill length")
    if ids.min() < 0 or ids.max() >= embedding_rows:
        raise ValueError("sample token IDs exceed the shared embedding table")
    if meta.get("visual_tokens") is not None and int(visual_mask.sum()) != meta["visual_tokens"]:
        raise ValueError("visual mask does not match the sample's visual-token count")
    if meta.get("image_token_id") is not None and not np.array_equal(visual_mask, ids == meta["image_token_id"]):
        raise ValueError("visual mask and image placeholder token IDs disagree")


def expected_grid(fixtures, work):
    path = os.path.join(fixtures, "image_grid_thw.npy")
    if os.path.isfile(path):
        return np.load(path, allow_pickle=False)
    # Older exports stored this only in golden inputs. Read just that NPZ entry.
    golden = os.path.join(work, "golden", "inputs.npz")
    if os.path.isfile(golden):
        with np.load(golden, allow_pickle=False) as doc:
            if "image_grid_thw" in doc:
                return doc["image_grid_thw"].copy()
    return None
