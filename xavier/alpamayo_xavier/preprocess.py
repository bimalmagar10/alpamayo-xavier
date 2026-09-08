"""Qwen3-VL image preprocessing, reimplemented for Python 3.8 / numpy.

The Xavier cannot install transformers 4.57, so the processor has to be
reproduced here. It is only three steps -- resize onto a patch-aligned grid,
normalise, patchify -- and `verify.py` checks the output against the tensor the
H100 produced for the same frames, so an error here cannot pass silently.

For a 1080x1920 camera under Alpamayo's pixel bounds this resolves to 320x576,
a 20x36 patch grid, 720 ViT patches per frame and 180 LLM tokens after the 2x2
spatial merge -- 2,880 visual tokens across 4 cameras x 4 frames.
"""
import math

import numpy as np
from PIL import Image

PATCH = 16
TEMPORAL_PATCH = 2
MERGE = 2
MIN_PIXELS = 163_840
MAX_PIXELS = 196_608
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape(3, 1, 1)


def smart_resize(height, width, factor=PATCH * MERGE,
                 min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Nearest patch-aligned size whose area falls inside [min_pixels, max_pixels]."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio beyond 200:1")
    h_bar = max(factor, int(round(height / factor)) * factor)
    w_bar = max(factor, int(round(width / factor)) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, int(math.floor(height / beta / factor)) * factor)
        w_bar = max(factor, int(math.floor(width / beta / factor)) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = int(math.ceil(height * beta / factor)) * factor
        w_bar = int(math.ceil(width * beta / factor)) * factor
    return h_bar, w_bar


def _to_chw_float(image, size):
    """PIL or HWC uint8 array -> normalised CHW float32 at `size`."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert("RGB").resize((size[1], size[0]), Image.BICUBIC)
    x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (x - IMAGE_MEAN) / IMAGE_STD


def patchify(frames_chw):
    """[T, C, H, W] -> [grid_t * grid_h * grid_w, C * temporal * patch * patch].

    Mirrors Qwen2VLImageProcessor: the merge dimensions are folded into the token
    order so the ViT's 2x2 spatial merge reads contiguous groups.
    """
    t, c, h, w = frames_chw.shape
    if t % TEMPORAL_PATCH != 0:
        pad = TEMPORAL_PATCH - (t % TEMPORAL_PATCH)
        frames_chw = np.concatenate([frames_chw, np.repeat(frames_chw[-1:], pad, axis=0)])
        t = frames_chw.shape[0]
    grid_t = t // TEMPORAL_PATCH
    grid_h, grid_w = h // PATCH, w // PATCH

    p = frames_chw.reshape(
        grid_t, TEMPORAL_PATCH, c,
        grid_h // MERGE, MERGE, PATCH,
        grid_w // MERGE, MERGE, PATCH,
    )
    p = p.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = p.reshape(grid_t * grid_h * grid_w, c * TEMPORAL_PATCH * PATCH * PATCH)
    return np.ascontiguousarray(flat, dtype=np.float32), (grid_t, grid_h, grid_w)


def preprocess_images(images):
    """A list of frames (one per camera-timestep) -> (pixel_values, grid_thw).

    Each frame is passed as its own image, exactly as `helper.create_message`
    does with `image_frames.flatten(0, 1)` -- so each contributes grid_t = 1 and
    the temporal patch dimension is filled by repeating the frame.
    """
    if not images:
        raise ValueError("no images given")
    first = images[0]
    h, w = (first.shape[:2] if isinstance(first, np.ndarray) else (first.height, first.width))
    size = smart_resize(h, w)

    all_patches, grids = [], []
    for img in images:
        chw = _to_chw_float(img, size)[None]                 # [1, C, H, W]
        patches, grid = patchify(chw)
        all_patches.append(patches)
        grids.append(grid)
    return np.concatenate(all_patches, 0), np.array(grids, dtype=np.int64)


def token_counts(grid_thw):
    """(ViT tokens, LLM tokens after the spatial merge) for a grid_thw array."""
    vit = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    return vit, vit // (MERGE * MERGE)
