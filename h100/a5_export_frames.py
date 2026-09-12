#!/usr/bin/env python3
"""Stage A5 (cluster, CPU only) -- export the golden clip's 16 camera frames as PNG.

run_alpamayo.py on the Xavier takes real image files, but a1_golden.py keeps only
the already-preprocessed pixel_values. This writes the raw frames that produced
them, in the order the model sees them -- camera-major, then time, exactly as
`helper.create_message(image_frames.flatten(0, 1))` -- so a sorted glob on the
Jetson reproduces that order:

    frames/00_cam0_t0.png ... frames/15_cam3_t3.png

PNG, not JPEG: lossless, so the Jetson preprocesses the same pixels the H100 did.
The script then runs the Jetson's own NumPy preprocessing
(xavier/alpamayo_xavier/preprocess.py) on these frames and compares the result with
golden/inputs.npz, so a porting error shows up here instead of as a vague vision
mismatch on the Xavier.

The fixtures (input_ids with the fused ego-history tokens) belong to this clip and
t0. Frames from any other clip need their own a1_golden.py + a3b_fixtures.py run.

    python h100/a5_export_frames.py          # same clip and t0 as a1_golden.py
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "xavier"))
from alpamayo_xavier import preprocess                      # noqa: E402

WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")
DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"       # a1_golden.DEFAULT_CLIP
DEFAULT_T0_US = 5_100_000                                  # a1_golden.py --t0-us default


def to_hwc_uint8(frame):
    """(3, H, W) tensor or array -> (H, W, 3) uint8."""
    x = frame.cpu().numpy() if hasattr(frame, "cpu") else np.asarray(frame)
    x = np.transpose(x, (1, 2, 0))
    if x.dtype != np.uint8:
        x = x * 255.0 if x.max() <= 1.0 else x
        x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(x)


def main():
    ap = argparse.ArgumentParser(description="Export the golden clip's frames as PNG.")
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=DEFAULT_T0_US)
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "frames"))
    ap.add_argument("--golden", default=os.path.join(WORK_ROOT, "golden", "inputs.npz"))
    args = ap.parse_args()

    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    data = load_physical_aiavdataset(args.clip, t0_us=args.t0_us)
    frames = data["image_frames"]                    # [cameras, timesteps, 3, H, W]
    n_cam, n_t = frames.shape[:2]
    print("image_frames :", tuple(frames.shape), frames.dtype)

    os.makedirs(args.out, exist_ok=True)
    for old in os.listdir(args.out):
        if old.endswith(".png"):
            os.remove(os.path.join(args.out, old))
    images = []
    for c in range(n_cam):
        for t in range(n_t):
            img = to_hwc_uint8(frames[c, t])
            name = "%02d_cam%d_t%d.png" % (c * n_t + t, c, t)
            Image.fromarray(img).save(os.path.join(args.out, name))
            images.append(img)
    print("wrote %d frames -> %s" % (len(images), args.out))

    # The preprocessing the Jetson will run, checked against the H100 processor's.
    px, grid = preprocess.preprocess_images(images)
    g = np.load(args.golden, allow_pickle=True)
    ref = g["pixel_values"].astype(np.float32)
    print("pixel_values : jetson %s  vs  golden %s" % (px.shape, ref.shape))
    if px.shape != ref.shape:
        sys.exit("SHAPE MISMATCH -- fix preprocess.py before going further")
    if not np.array_equal(grid, g["image_grid_thw"]):
        sys.exit("grid_thw mismatch: %s vs %s" % (grid.tolist(), g["image_grid_thw"].tolist()))
    diff = np.abs(px - ref)
    cos = float(px.ravel() @ ref.ravel() / (np.linalg.norm(px) * np.linalg.norm(ref)))
    print("max |diff| %.4f   mean |diff| %.5f   cosine %.6f" % (diff.max(), diff.mean(), cos))
    print("OK" if cos > 0.999 else
          "WARNING: preprocessing differs from the reference -- expect a vision mismatch")


if __name__ == "__main__":
    main()
