#!/usr/bin/env python3
"""Prepare new PhysicalAI clips/timestamps for existing Xavier engines.

Loads the reference model once, but performs no VLM forward pass or ONNX export.
Each output folder contains 16 PNGs and small sample-specific fixtures. The Xavier
continues to use WORK/fixtures/embed_tokens.fp16.npy and its existing engines.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "xavier"))
from alpamayo_xavier import preprocess, sample_inputs
from a5_export_frames import to_hwc_uint8

WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")
MODEL_DIR = os.environ.get("ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B")


def sample_name(clip, t0_us):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", clip) or t0_us <= 0:
        raise ValueError("clip must be an identifier and t0_us must be positive")
    return "%s__%d" % (clip, t0_us)


def compare_preprocessing(images, reference_pixels, reference_grid):
    pixels, grid = preprocess.preprocess_images(images)
    if pixels.shape != reference_pixels.shape or not np.array_equal(grid, reference_grid):
        raise ValueError("Xavier preprocessing shape/grid differs from the reference processor")
    a, b = pixels.reshape(-1), reference_pixels.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(a @ b / denom) if denom else float(np.array_equal(a, b))
    if not np.isfinite(cosine) or cosine < 0.999:
        raise ValueError("Xavier preprocessing differs from the reference (cosine %.6f)" % cosine)
    return dict(cosine=cosine, mean_absolute_error=float(np.abs(pixels - reference_pixels).mean()),
                shape=list(pixels.shape))


def export_sample(destination, data, arrays, meta, reference_pixels, reference_meta, reference_grid):
    """Publish a complete sample folder only after its compatibility checks pass."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("sample already exists: " + str(destination))
    if tuple(data["image_frames"].shape[:2]) != (4, 4):
        raise ValueError("existing engines require four cameras and four timestamps per camera")
    camera_ids = data["camera_indices"].cpu().numpy().tolist()
    if camera_ids != [0, 1, 2, 6]:
        raise ValueError("unexpected camera order; expected camera indices [0, 1, 2, 6]")
    sample_inputs.validate_contract(meta, reference_meta, arrays["image_grid_thw"], reference_grid)
    sample_inputs.validate_arrays(meta, arrays["input_ids"], arrays["position_ids"],
                                  arrays["visual_mask"], reference_meta["vocab"])
    images = [to_hwc_uint8(data["image_frames"][c, t]) for c in range(4) for t in range(4)]
    prep = compare_preprocessing(images, reference_pixels, arrays["image_grid_thw"])
    meta = dict(meta, camera_indices=camera_ids, preprocessing_check=prep,
                image_order="camera-major, chronological within each camera",
                image_files=["frames/%02d_cam%d_t%d.png" % (c * 4 + t, c, t)
                             for c in range(4) for t in range(4)])
    for key in ("absolute_timestamps", "relative_timestamps"):
        if key in data:
            meta[key] = data[key].cpu().numpy().tolist()
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=".preparing-", dir=str(destination.parent)))
    try:
        (scratch / "frames").mkdir()
        (scratch / "fixtures").mkdir()
        for name, image in zip(meta["image_files"], images):
            Image.fromarray(image).save(scratch / name)
        for name, value in arrays.items():
            np.save(scratch / "fixtures" / (name + ".npy"), value)
        # Retain the real history for audit; runtime uses its already-fused tokens.
        np.savez_compressed(scratch / "fixtures" / "ego_history.npz",
                            ego_history_xyz=data["ego_history_xyz"].float().cpu().numpy(),
                            ego_history_rot=data["ego_history_rot"].float().cpu().numpy())
        (scratch / "fixtures" / "meta.json").write_text(json.dumps(meta, indent=2))
        scratch.rename(destination)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", nargs="+", required=True, help="one or more PhysicalAI clip IDs")
    ap.add_argument("--t0-us", nargs="+", type=int, required=True,
                    help="sample each clip at these timestamps in microseconds")
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--work", default=WORK_ROOT, help="existing H100 export root, including fixtures/ and golden/")
    ap.add_argument("--out", default=None, help="default: WORK/samples")
    args = ap.parse_args()
    output = Path(args.out or os.path.join(args.work, "samples"))
    jobs = [(clip, ts, output / sample_name(clip, ts)) for clip in args.clip for ts in args.t0_us]
    if len(set(p for _, _, p in jobs)) != len(jobs):
        ap.error("duplicate clip/timestamp combinations")
    for _, _, path in jobs:
        if path.exists():
            ap.error("sample already exists; choose a new output folder: " + str(path))
    reference_fixtures = os.path.join(args.work, "fixtures")
    reference_meta = sample_inputs.read_meta(reference_fixtures)
    reference_grid = sample_inputs.expected_grid(reference_fixtures, args.work)
    if reference_grid is None:
        ap.error("missing original image grid in fixtures/image_grid_thw.npy or golden/inputs.npz")

    # Import the reference stack only after argument validation, so --help works
    # on the Mac. The model is loaded once on the host, as in a3b_fixtures.py.
    import torch
    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from a3b_fixtures import prepare_inputs

    print("Loading reference model once to prepare input fixtures (no inference or engine export).", flush=True)
    model = AlpamayoR1.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    processor = helper.get_processor(model.tokenizer)
    with torch.inference_mode():
        for clip, ts, destination in jobs:
            print("Preparing %s at %.3f s" % (clip, ts / 1e6), flush=True)
            data = load_physical_aiavdataset(clip, t0_us=ts)
            arrays, meta, tok = prepare_inputs(model, processor, data, clip, ts, reference_meta["max_seq"])
            meta["model_source"] = args.model
            export_sample(destination, data, arrays, meta, tok["pixel_values"].float().cpu().numpy(),
                          reference_meta, reference_grid)
            print("Ready: %s" % destination, flush=True)
    print("Transfer the sample folders into Xavier's WORK/samples, then run with --sample PATH.")


if __name__ == "__main__":
    main()
