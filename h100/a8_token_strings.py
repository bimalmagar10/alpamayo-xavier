#!/usr/bin/env python3
"""Stage A8 -- ship the tokenizer's surface forms so the Jetson can read its own output.

run_alpamayo.py samples token ids and, until now, could only report how many. The
runtime does not need `transformers`/`tokenizers`. Export a flat list, one surface form
per id -- about 2.5 MB of JSON -- which xavier/alpamayo_xavier/detok.py turns back
into text with no dependencies at all.

Every entry is stored in byte-level BPE's stand-in encoding (GPT-2's
bytes_to_unicode, which Qwen inherits), added tokens included, so the Jetson has
exactly one decoding rule for the whole table.

    python h100/a8_token_strings.py                      # $ALPAMAYO_MODEL
    python h100/a8_token_strings.py --tokenizer /path/to/tokenizer.json
    python h100/a8_token_strings.py --verify             # decode the prompt back

Reads tokenizer.json directly when it is there (no dependencies, so this also runs
on the Mac), and falls back to transformers only if it is not.
"""
import argparse
import glob
import hashlib
import json
import os
import sys

MODEL_DIR = os.environ.get(
    "ALPAMAYO_MODEL", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-models/Alpamayo-R1-10B")
WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")
VOCAB_SIZE = 155_697                     # the embedding table's rows, not the tokenizer's
# Order used by NVIDIA's ReasoningVLAConfig._build_processor. Existing base
# tokens (image_pad) retain their IDs. Validate every trajectory ID from config.
# https://github.com/NVlabs/alpamayo/blob/main/src/alpamayo_r1/models/base_model.py
SPECIAL_KEYS = ("prompt_start prompt_end image_start image_pre_tkn image_end "
                "traj_history_start traj_history_pre_tkn traj_history_end cot_start cot_end "
                "meta_action_start meta_action_end traj_future_start traj_future_pre_tkn "
                "traj_future_end traj_history traj_future image_pad vectorized_wm "
                "vectorized_wm_start vectorized_wm_end vectorized_wm_pre_tkn route_start "
                "route_pad route_end question_start question_end answer_start answer_end").split()
TRAJ_KEYS = ("history", "future", "history_start", "future_start", "history_end", "future_end")


def bytes_to_unicode():
    """byte -> printable stand-in character (GPT-2's table, verbatim)."""
    keep = (list(range(ord("!"), ord("~") + 1))
            + list(range(ord("\xa1"), ord("\xac") + 1))
            + list(range(ord("\xae"), ord("\xff") + 1)))
    codes, n = list(keep), 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            codes.append(256 + n)
            n += 1
    return dict(zip(keep, [chr(c) for c in codes]))


def encode_literal(s, table):
    """A literal string (an added token) in the same stand-in encoding as the rest."""
    return "".join(table[b] for b in s.encode("utf-8"))


def from_tokenizer_json(path, table):
    """{id: surface form} out of a HuggingFace fast-tokenizer file, using only json."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("model", {}).get("type") != "BPE" or doc.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("This exporter requires a BPE tokenizer with a ByteLevel decoder")
    out = {}
    vocab = doc.get("model", {}).get("vocab", {})
    if isinstance(vocab, dict):
        for tok, i in vocab.items():
            out[int(i)] = tok                       # already stand-in encoded
    for a in doc.get("added_tokens", []):
        out[int(a["id"])] = encode_literal(a["content"], table)
    return out


def extend_alpamayo(ids, config, table):
    """Reproduce runtime-added tokens without loading the 10B model."""
    ids = dict(ids)
    reverse = {v: k for k, v in ids.items()}
    next_id = max(ids, default=-1) + 1
    def add(text):
        nonlocal next_id
        piece = encode_literal(text, table)
        if piece not in reverse:
            idx = next_id
            next_id += 1
            ids[idx] = piece
            reverse[piece] = idx
        return reverse[piece]
    size = config.get("traj_vocab_size")
    if size is not None:
        for i in range(size):
            idx = add("<i%d>" % i)
            expected = config.get("traj_token_start_idx")
            if expected is not None and idx != expected + i:
                raise ValueError("trajectory token ID does not match checkpoint: <i%d>" % i)
    keys = SPECIAL_KEYS if config.get("add_special_tokens") else ["traj_" + k for k in TRAJ_KEYS]
    for key in keys:
        add("<|" + key + "|>")
    for key, expected in config.get("traj_token_ids", {}).items():
        if reverse.get(encode_literal("<|traj_" + key + "|>", table)) != expected:
            raise ValueError("special token ID does not match checkpoint: " + key)
    if config.get("vocab_size") is not None and len(ids) != config["vocab_size"]:
        raise ValueError("extended vocabulary size does not match checkpoint")
    return ids


def write_vocab(ids, out, source, size=VOCAB_SIZE, provenance=None):
    size = max(size, max(ids, default=-1) + 1)
    tokens = [ids.get(i) for i in range(size)]
    doc = dict(tokens=tokens, size=size, source=source, encoding="byte-level-bpe/gpt2",
               provenance=provenance or {})
    blob = json.dumps(doc, ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(blob)
    print("wrote %s (%d entries; sha256 %s)" %
          (out, size, hashlib.sha256(blob.encode("utf-8")).hexdigest()))
    return doc


def from_transformers(model_dir, table):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    out = {}
    for tok, i in tk.get_vocab().items():
        out[int(i)] = tok
    for i, tok in getattr(tk, "added_tokens_decoder", {}).items():
        out[int(i)] = encode_literal(str(tok), table)
    return out


def locate(arg):
    """A tokenizer.json from --tokenizer, whether it names the file or a directory."""
    if arg and arg.endswith(".json") and os.path.exists(arg):
        return arg
    for d in ([arg] if arg else []) + [MODEL_DIR]:
        if d and os.path.isdir(d):
            hit = sorted(glob.glob(os.path.join(d, "**", "tokenizer.json"), recursive=True))
            if hit:
                return hit[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json, or a directory holding one")
    ap.add_argument("--config", default=None,
                    help="Alpamayo config.json to reproduce/validate its added tokens; defaults to MODEL/config.json")
    ap.add_argument("--out", default=os.path.join(WORK_ROOT, "fixtures", "vocab.json"))
    ap.add_argument("--size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--verify", action="store_true",
                    help="decode fixtures/input_ids.npy back to text as a self-check")
    args = ap.parse_args()

    table = bytes_to_unicode()
    config_path = args.config or os.path.join(MODEL_DIR, "config.json")
    config = None
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    elif args.config:
        ap.error("missing --config file: " + config_path)
    path = locate(args.tokenizer)
    if path:
        ids, source = from_tokenizer_json(path, table), path
    else:
        print("no tokenizer.json found -- falling back to transformers")
        source = args.tokenizer or (config or {}).get("vlm_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
        ids = from_transformers(source, table)
    provenance = {}
    for label, file in (("tokenizer_sha256", path), ("config_sha256", config_path if config else None)):
        if file:
            with open(file, "rb") as f:
                provenance[label] = hashlib.sha256(f.read()).hexdigest()
    if config:
        ids = extend_alpamayo(ids, config, table)
    write_vocab(ids, args.out, source, args.size, provenance)

    if args.verify:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xavier"))
        import numpy as np
        from alpamayo_xavier.detok import Detokenizer
        d = Detokenizer.find(args.out)
        prompt = np.load(os.path.join(os.path.dirname(args.out), "input_ids.npy"))
        head, tail = prompt[:48].tolist(), prompt[-48:].tolist()
        print("\nprompt head     : %r" % d.decode(head))
        print("prompt tail     : %r" % d.decode(tail))
        if "�" in d.decode(head) + d.decode(tail):
            print("\nWARNING: replacement characters in the prompt -- the table may be wrong")
        else:
            print("\nround-trip looks correct")


if __name__ == "__main__":
    main()
