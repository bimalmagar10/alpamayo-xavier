#!/usr/bin/env python3
"""Stage A0 (login node, CPU) -- cache the driving clip before anything needs a GPU.

`load_physical_aiavdataset` streams video from the Hugging Face dataset
nvidia/PhysicalAI-Autonomous-Vehicles on first use. That dataset is **gated**, and
compute nodes usually have no route to huggingface.co, so a1_golden.py will hang
or fail there unless the clip is already in $HF_HOME.

This script pulls exactly the frames a1 will ask for -- one clip, sampled at the
same t0 offsets -- and verifies the shapes match what the rest of the pipeline
assumes. No GPU, no model load.

Prerequisites, once:
    1. Accept the terms at
       https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
    2. hf auth login

    source $ALPAMAYO_REPO/env.sh
    source $ALPAMAYO_ROOT/alpamayo/ar1_venv/bin/activate
    python $ALPAMAYO_REPO/h100/a0_prefetch.py --clips 64
"""
import argparse
import os
import socket
import subprocess
import sys
import time

DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"
EXPECTED_CAMERAS = 4
EXPECTED_FRAMES = 4


def check_network(host="huggingface.co", port=443, timeout=10):
    """Compute nodes are frequently walled off. Say so plainly rather than letting
    every clip fail with a stack trace that looks like an auth problem."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
    except OSError as exc:
        sys.exit(
            "Cannot reach %s:%d (%s).\n\n"
            "  This node has no route to the internet, so the clips cannot be streamed.\n"
            "  Run this script on a LOGIN or DATA-TRANSFER node instead -- it needs no\n"
            "  GPU. Note that a1_golden.py streams the same video, so it also needs a\n"
            "  compute node with network access.\n"
            % (host, port, exc))
    print("network : %s reachable" % host)


def check_auth():
    try:
        who = subprocess.run(["hf", "auth", "whoami"], capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("could not run `hf auth whoami` -- continuing, but if the dataset is "
              "gated you will get a 401/403 below")
        return
    if who.returncode != 0 or "Not logged in" in (who.stdout + who.stderr):
        sys.exit(
            "Not logged in to Hugging Face.\n\n"
            "  nvidia/PhysicalAI-Autonomous-Vehicles is gated (gated: auto), so it\n"
            "  needs an accepted licence and a token:\n"
            "    1. accept at https://huggingface.co/datasets/"
            "nvidia/PhysicalAI-Autonomous-Vehicles\n"
            "    2. hf auth login\n")
    print("hf auth :", who.stdout.strip().splitlines()[0] if who.stdout.strip() else "ok")


def cache_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--t0-us", type=int, default=5_100_000)
    ap.add_argument("--clips", type=int, default=64,
                    help="how many shifted samples a1 will take from this clip")
    ap.add_argument("--stride-us", type=int, default=200_000)
    args = ap.parse_args()

    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        print("HF_HOME is unset -- did you `source env.sh`? Caching to the default "
              "~/.cache/huggingface, which compute nodes may not share.")
    else:
        print("HF_HOME :", hf_home)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        sys.exit("HF_HUB_OFFLINE=1 is set. Unset it for this step -- it must reach the hub.")

    print("host    :", socket.gethostname())
    check_network()
    check_auth()
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

    before = cache_size(hf_home) if hf_home else 0
    ok, failed, t_start = 0, [], time.time()

    for i in range(args.clips):
        t0 = args.t0_us + i * args.stride_us
        try:
            data = load_physical_aiavdataset(args.clip, t0_us=t0)
        except Exception as exc:                     # clip window ran past the end
            failed.append((i, type(exc).__name__, str(exc)[:120]))
            continue
        if ok == 0:
            frames = data["image_frames"]
            print("\nfirst sample:")
            print("  image_frames    :", tuple(frames.shape))
            print("  ego_history_xyz :", tuple(data["ego_history_xyz"].shape))
            print("  ego_future_xyz  :", tuple(data["ego_future_xyz"].shape))
            n_cam, n_frame = frames.shape[0], frames.shape[1]
            h, w = frames.shape[-2:]
            print("  -> %d cameras x %d frames = %d images at %dx%d"
                  % (n_cam, n_frame, n_cam * n_frame, h, w))
            if (n_cam, n_frame) != (EXPECTED_CAMERAS, EXPECTED_FRAMES):
                print("  WARNING: pipeline assumes %d cameras x %d frames. Token counts "
                      "and every static ONNX shape derive from this."
                      % (EXPECTED_CAMERAS, EXPECTED_FRAMES))
        ok += 1
        if ok % 8 == 0:
            print("  cached %d/%d samples (%.0fs)" % (ok, args.clips, time.time() - t_start))

    after = cache_size(hf_home) if hf_home else 0
    print("\ncached %d of %d samples in %.0fs" % (ok, args.clips, time.time() - t_start))
    if hf_home:
        print("cache grew by %.2f GB (now %.2f GB)"
              % ((after - before) / 1e9, after / 1e9))
    if failed:
        print("\n%d sample(s) unavailable -- normal once t0 runs past the end of the clip:"
              % len(failed))
        for i, kind, msg in failed[:3]:
            print("  sample %d: %s: %s" % (i, kind, msg))
    if ok == 0:
        sys.exit("\nNothing cached. The network and auth checks passed, so this is most\n"
                 "likely a bad --clip id or a t0 outside the clip. Try --clips 1 first.")

    print("\nThe clip is readable: %d samples. physical_ai_av caches only the dataset" % ok)
    print("metadata and streams the camera video on every run, so a1_golden.py,")
    print("a3b_fixtures.py and a5_export_frames.py need network access on the compute")
    print("node too. Keep HF_HUB_OFFLINE=0, and use --clips %d for a1." % ok)


if __name__ == "__main__":
    main()
