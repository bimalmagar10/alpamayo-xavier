#!/usr/bin/env python3
"""Stage A7 -- map the language model's weights inside the ONNX weights file, so the
Xavier can run decode in PyTorch (runs anywhere; needs only `onnx`).

Why PyTorch for decode: decode reads all 15.2 GB of the language model's weights for
every token, so it is limited by memory bandwidth. TensorRT measured 350 ms/token
against PyTorch's 318, and each TensorRT engine costs about twice its weights in
memory on a Xavier, which does not fit. Plain torch tensors do.

Nothing is exported or copied. Every weight already sits in <graph>.onnx.data; this
records where. torch.onnx emits one layer as a fixed sequence of weight-consuming
nodes, which is what the roles below are read from:

    Mul [4096]      input_layernorm         MatMul [4096, 4096]    o_proj
    MatMul [4096, 4096]  q_proj             Mul [4096]             post_attention_layernorm
    MatMul [4096, 1024]  k_proj             MatMul [4096, 12288]   gate_proj
    MatMul [4096, 1024]  v_proj             MatMul [4096, 12288]   up_proj
    Mul [128]       q_norm                  MatMul [12288, 4096]   down_proj
    Mul [128]       k_norm
  then, in the head piece: Mul [4096] final norm, MatMul [4096, 155697] lm_head

    python h100/a7_weight_map.py --onnx ~/alpamayo-payload/onnx
    -> onnx/weight_map.json  (a few KB, one entry per graph that is present)
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import base64

import onnx
from onnx import numpy_helper

LAYERS = 36
ROLES = [("input_ln", [4096]), ("q", [4096, 4096]), ("k", [4096, 1024]), ("v", [4096, 1024]),
         ("q_norm", [128]), ("k_norm", [128]), ("o", [4096, 4096]), ("post_ln", [4096]),
         ("gate", [4096, 12288]), ("up", [4096, 12288]), ("down", [12288, 4096])]
HEAD = [("final_ln", [4096]), ("lm_head", [4096, 155697])]
WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")


def where(t):
    """Where this weight's bytes are: in <graph>.onnx.data, or inline in the .onnx.

    torch.onnx leaves small tensors (the 128- and 4096-element norms) inside the
    file, so those travel in this map as base64 -- about 400 KB in total.
    """
    d = {e.key: e.value for e in t.external_data}
    common = dict(dims=list(t.dims), dtype=int(t.data_type), name=t.name)
    if d:
        common.update(location=d["location"], offset=int(d.get("offset", 0)),
                      length=int(d.get("length", 0)))
        return common
    raw = t.raw_data or numpy_helper.to_array(t).tobytes()
    common.update(inline=base64.b64encode(raw).decode("ascii"), length=len(raw))
    return common


def weights_in_order(paths):
    """Every initializer consumed by a node, in node order, across the pieces in order."""
    out = []
    for path in paths:
        g = onnx.load(path, load_external_data=False).graph
        init = {t.name: t for t in g.initializer}
        for n in g.node:
            for i in n.input:
                t = init.get(i)
                if t is None:
                    continue
                out.append(where(t))
    return out


def build(onnx_dir, graph):
    pieces = sorted(glob.glob(os.path.join(onnx_dir, graph + ".p[0-9][0-9].onnx")))
    paths = pieces or [os.path.join(onnx_dir, graph + ".onnx")]
    if not os.path.exists(paths[0]):
        return None
    found = weights_in_order(paths)
    expect = LAYERS * len(ROLES) + len(HEAD)
    if len(found) != expect:
        raise SystemExit("%s: found %d weights, expected %d -- has the export changed?"
                         % (graph, len(found), expect))
    layers, at = [], 0
    for L in range(LAYERS):
        layer = {}
        for role, dims in ROLES:
            w = found[at]
            if w["dims"] != dims:
                raise SystemExit("%s layer %d: expected %s for %s, found %s (%s)"
                                 % (graph, L, dims, role, w["dims"], w["name"]))
            layer[role] = w
            at += 1
        layers.append(layer)
    head = {}
    for role, dims in HEAD:
        w = found[at]
        if w["dims"] != dims:
            raise SystemExit("%s head: expected %s for %s, found %s" % (graph, dims, role, w["dims"]))
        head[role] = w
        at += 1
    every = [w for L in layers for w in L.values()] + list(head.values())
    data = {w["location"] for w in every if "location" in w}
    if len(data) != 1:
        raise SystemExit("%s: weights spread over %s" % (graph, data))
    inline = [w for w in every if "inline" in w]
    total = sum(w["length"] for w in every)
    print("%-8s %d layers x %d weights + head · %.2f GB in %s, %d small ones inline (%.0f KB)"
          % (graph, LAYERS, len(ROLES), total / 1e9, list(data)[0], len(inline),
             sum(w["length"] for w in inline) / 1e3))
    return dict(source=list(data)[0], bytes=total, layers=layers, head=head)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Where the language model's weights live.")
    ap.add_argument("--onnx", default=os.path.join(WORK_ROOT, "onnx"))
    ap.add_argument("--graphs", default="decode,prefill")
    args = ap.parse_args()
    onnx_dir = os.path.expanduser(args.onnx)
    doc = {"format": 1, "eps": 1e-6, "graphs": {}}
    for graph in args.graphs.split(","):
        got = build(onnx_dir, graph)
        if got:
            doc["graphs"][graph] = got
    if not doc["graphs"]:
        sys.exit("no graphs found in %s" % onnx_dir)
    out = os.path.join(onnx_dir, "weight_map.json")
    json.dump(doc, open(out, "w"))
    print("wrote %s (%.0f KB)" % (out, os.path.getsize(out) / 1e3))
    man = os.path.join(os.path.dirname(os.path.abspath(onnx_dir)), "MANIFEST.sha256")
    if os.path.exists(man):
        rel = "onnx/weight_map.json"
        lines = [l for l in open(man).read().splitlines() if not l.endswith("  " + rel)]
        lines.append("%s  %s" % (sha256(out), rel))
        open(man, "w").write("\n".join(lines) + "\n")
        print("MANIFEST.sha256: %s entry written" % rel)


if __name__ == "__main__":
    main()
