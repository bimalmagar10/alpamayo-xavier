"""Alpamayo-1 architecture constants.

Every value here was read out of the released artefacts, not from prose:
  * tensor shapes from the safetensors headers of nvidia/Alpamayo-R1-10B
  * scalars from that repo's config.json
  * backbone/vision hyper-parameters from Qwen/Qwen3-VL-8B-Instruct, which
    NVlabs/alpamayo names as `vlm_name_or_path`

Keep this module dependency-free: it is imported on the H100 (Python 3.12)
and on the Xavier (Python 3.8) alike.
"""

# --- Qwen3-VL-8B text backbone -------------------------------------------
LLM = dict(
    layers=36,
    hidden=4096,
    heads=32,
    kv_heads=8,
    head_dim=128,
    intermediate=12288,
    rms_eps=1e-6,
    rope_theta=5_000_000.0,
    mrope_section=(24, 20, 20),      # sums to head_dim // 2
    mrope_interleaved=True,
)

# --- Qwen3-VL vision tower ------------------------------------------------
VISION = dict(
    depth=27,
    hidden=1152,
    heads=16,
    head_dim=72,
    intermediate=4304,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    num_position_embeddings=2304,    # 48 x 48, bicubically resampled per grid
    out_hidden=4096,
    deepstack_indexes=(8, 16, 24),   # LLM layers that receive DeepStack features
)

# --- Alpamayo action expert (Qwen3 text block, narrower) ------------------
EXPERT = dict(
    layers=36,
    hidden=2048,
    heads=16,
    kv_heads=8,                      # inherited -> KV geometry matches the backbone
    head_dim=128,
    intermediate=8256,
    rms_eps=1e-6,
    non_causal=True,
)

# --- Alpamayo heads and vocabulary ---------------------------------------
VOCAB = 155_697
TRAJ_TOKEN_START = 151_669
TRAJ_VOCAB = 4_000
TRAJ_TOKEN_IDS = dict(
    history_start=155_674, history_end=155_676, history=155_684,
    future_start=155_681, future_end=155_683, future=155_685,
)

# --- Inference workload ---------------------------------------------------
N_CAMERAS = 4
N_FRAMES = 4
N_IMAGES = N_CAMERAS * N_FRAMES      # each frame is passed as its own image
MIN_PIXELS = 163_840
MAX_PIXELS = 196_608
IMAGE_HW = (320, 576)                # smart_resize(1080, 1920, factor=32) under those bounds
GRID_HW = (20, 36)                   # 320/16, 576/16
VIT_TOKENS_PER_IMAGE = 720           # 20 * 36
LLM_TOKENS_PER_IMAGE = 180           # after the 2x2 spatial merge
VIT_TOKENS = VIT_TOKENS_PER_IMAGE * N_IMAGES     # 11_520
VISUAL_TOKENS = LLM_TOKENS_PER_IMAGE * N_IMAGES  # 2_880

N_WAYPOINTS = 64
HISTORY_TRAJ_TOKENS = 48
FLOW_STEPS = 10                      # FlowMatching(num_inference_steps=10), Euler
ACTION_DIMS = (N_WAYPOINTS, 2)       # acceleration, curvature

# Static shapes the TensorRT engines are built for. MAX_SEQ must exceed
# prefill + the longest reasoning trace you intend to allow.
MAX_SEQ = 3584
DEFAULT_MAX_NEW_TOKENS = 256         # matches test_inference.py

PARAMS = dict(vision=576_400_000, llm=6_945_800_000, embed=637_700_000,
              lm_head=637_700_000, expert=2_279_100_000, total=11_078_526_194)
