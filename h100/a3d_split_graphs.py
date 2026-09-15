#!/usr/bin/env python3
"""Stage A3d -- split the expert, decode and prefill graphs into engines small enough
for TensorRT 8.5 to build on a 32 GB Xavier (runs anywhere; needs only `onnx`).

Measured on the Xavier (2026-09-12): TensorRT's Myelin compiler fuses a whole
transformer graph into one node and, to time it, asks for ONE block of GPU memory
bigger than all of that graph's weights, rounded up to a power of two. For the
expert (4.6 GB of fp16 weights) that is 8 GiB, and the request fails even with
28 GB free and 23 GB of swap. The vision tower (1.15 GB) builds. Decode and
prefill carry 15 GB each.

So each graph is cut at layer boundaries into pieces of a few layers, plus a head
piece (final norm + lm_head, or action_out_proj). A piece is an ordinary ONNX file
whose weights stay in the ORIGINAL <graph>.onnx.data at the same offsets: only a
few MB of .onnx files are new; nothing is re-exported or re-transferred.

Piece I/O (recorded per piece in onnx/pieces.json for the runner):
  first piece   the graph's own inputs; for the expert it also holds action_in_proj
  every piece   hidden_in -> hidden_out, the residual stream between layers
  KV            decode/expert pieces read the full past_k/past_v (bound, not copied);
                prefill/decode pieces emit their own layers' K/V
                (k_cache/v_cache, new_k/new_v) with the layer range in pieces.json
  head piece    the graph's final outputs (last_hidden+logits, logits, velocity)

    python h100/a3d_split_graphs.py --onnx ~/alpamayo-payload/onnx
    python h100/a3d_split_graphs.py --layers expert=9 decode=4 prefill=3
"""
import argparse
import collections
import copy
import glob
import hashlib
import json
import os
import re
import sys

import onnx
from onnx import helper, shape_inference

LAYERS = 36
DEFAULT_LAYERS = {"expert": 9, "decode": 4, "prefill": 3}
KV_OUT = {"prefill": ("k_cache", "v_cache"), "decode": ("new_k", "new_v"), "expert": None}
FINAL = {"prefill": ["last_hidden", "logits"], "decode": ["logits"], "expert": ["velocity"]}
WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")


def down_proj_node(i):
    """torch.onnx names repeated module calls by call order: /down_proj, /down_proj_1, ..."""
    return "/down_proj/MatMul" if i == 0 else "/down_proj_%d/MatMul" % i


def layer_outputs(g):
    """The residual-stream tensor leaving each layer (after DeepStack's add in prefill)."""
    by_name = {n.name: n for n in g.node}
    cons = collections.defaultdict(list)
    for n in g.node:
        for t in n.input:
            cons[t].append(n)
    graph_inputs = {i.name for i in g.input}
    out = []
    for i in range(LAYERS):
        dp = by_name.get(down_proj_node(i))
        if dp is None:
            raise ValueError("no node %s -- has the export's naming changed?" % down_proj_node(i))
        adds = [c for c in cons[dp.output[0]] if c.op_type == "Add"]
        if len(adds) != 1:
            raise ValueError("layer %d: expected one residual Add after down_proj, found %d"
                             % (i, len(adds)))
        t = adds[0].output[0]
        while True:                              # h = h + deepstack[i], layers 0-2 of prefill
            ds = [c for c in cons[t] if c.op_type == "Add" and any(x in graph_inputs for x in c.input)]
            if not ds:
                break
            t = ds[0].output[0]
        out.append(t)
    return out


def extract(g, stop, outputs, extra_nodes=()):
    """Nodes needed to compute `outputs` from `stop` tensors, graph inputs and weights."""
    nodes = list(g.node) + list(extra_nodes)
    producer = {}
    for idx, n in enumerate(nodes):
        for o in n.output:
            producer[o] = idx
    weights = {t.name for t in g.initializer}
    graph_inputs = {i.name for i in g.input}
    keep, seen, used_inputs = set(), set(), []
    todo = list(outputs)
    while todo:
        t = todo.pop()
        if not t or t in seen:
            continue
        seen.add(t)
        if t in stop or t in weights:
            continue
        if t in graph_inputs:
            used_inputs.append(t)
            continue
        if t not in producer:
            raise ValueError("tensor %r has no producer inside the piece" % t)
        idx = producer[t]
        if idx not in keep:
            keep.add(idx)
            todo.extend(nodes[idx].input)
    kept = sorted(keep)
    used_weights = sorted({x for i in kept for x in nodes[i].input if x in weights})
    return kept, [copy.deepcopy(nodes[i]) for i in kept], used_weights, used_inputs


def rename(nodes, old, new):
    for n in nodes:
        for i, x in enumerate(n.input):
            if x == old:
                n.input[i] = new
        for i, x in enumerate(n.output):
            if x == old:
                n.output[i] = new


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_graph(onnx_dir, graph, per):
    src = os.path.join(onnx_dir, graph + ".onnx")
    model = onnx.load(src, load_external_data=False)
    g = model.graph
    info = {v.name: v for v in shape_inference.infer_shapes(model).graph.value_info}
    info.update({v.name: v for v in list(g.input) + list(g.output)})
    weights = {t.name: t for t in g.initializer}
    bounds = layer_outputs(g)
    for t in bounds:
        if t not in info:
            raise ValueError("no shape for residual tensor %s" % t)

    kv_concat = {}
    if KV_OUT[graph]:
        for name in KV_OUT[graph]:
            node = [n for n in g.node if name in n.output][0]
            if node.op_type != "Concat" or len(node.input) != LAYERS:
                raise ValueError("%s is not a %d-way Concat" % (name, LAYERS))
            kv_concat[name] = node

    for old in glob.glob(os.path.join(onnx_dir, graph + ".p*.onnx")):
        os.remove(old)

    ranges = [(a, min(a + per, LAYERS) - 1) for a in range(0, LAYERS, per)]
    ranges.append(None)                                    # the head piece
    pieces, covered, counts = [], collections.Counter(), []
    for k, rng in enumerate(ranges):
        name = "%s.p%02d" % (graph, k)
        extra, outputs = [], []
        if rng is None:
            stop, outputs = {bounds[-1]}, list(FINAL[graph])
        else:
            a, b = rng
            stop = set() if a == 0 else {bounds[a - 1]}
            outputs = [bounds[b]]
            for kv_name, node in kv_concat.items():
                tmp = "__piece_" + kv_name
                extra.append(helper.make_node("Concat", list(node.input[a:b + 1]), [tmp],
                                              name="/piece_" + kv_name, axis=0))
                outputs.append(tmp)
        idx, nodes, used_w, used_in = extract(g, stop, outputs, extra)
        covered.update(i for i in idx if i < len(g.node))

        inputs_vi, outputs_vi = [], []
        for t in sorted(stop):
            rename(nodes, t, "hidden_in")
            vi = copy.deepcopy(info[t])
            vi.name = "hidden_in"
            inputs_vi.append(vi)
        order = [i.name for i in g.input]
        for t in sorted(set(used_in), key=order.index):
            inputs_vi.append(copy.deepcopy(info[t]))
        if rng is None:
            outputs_vi = [copy.deepcopy(info[t]) for t in FINAL[graph]]
        else:
            rename(nodes, bounds[rng[1]], "hidden_out")
            vi = copy.deepcopy(info[bounds[rng[1]]])
            vi.name = "hidden_out"
            outputs_vi.append(vi)
            for kv_name in kv_concat:
                rename(nodes, "__piece_" + kv_name, kv_name)
                vi = copy.deepcopy(info[kv_name])
                vi.type.tensor_type.shape.dim[0].dim_value = rng[1] - rng[0] + 1
                outputs_vi.append(vi)

        piece = helper.make_model(
            helper.make_graph(nodes, name, inputs_vi, outputs_vi,
                              initializer=[weights[w] for w in used_w]),
            opset_imports=model.opset_import, producer_name="a3d_split_graphs")
        piece.ir_version = model.ir_version
        path = os.path.join(onnx_dir, name + ".onnx")
        onnx.save(piece, path)                  # weights stay in <graph>.onnx.data
        onnx.checker.check_model(path)          # by path: resolves that external data
        nbytes = sum(int(e.value) for w in used_w for e in weights[w].external_data if e.key == "length")
        counts.append((name, rng, len(nodes), nbytes))
        pieces.append(dict(name=name, file=name + ".onnx",
                           layers=list(rng) if rng else None, head=rng is None,
                           inputs=[v.name for v in inputs_vi],
                           outputs=[v.name for v in outputs_vi],
                           kv=list(KV_OUT[graph]) if (rng and KV_OUT[graph]) else None,
                           weight_bytes=nbytes))

    dropped = [g.node[i].name for i in range(len(g.node)) if i not in covered]
    shared = sum(1 for c in covered.values() if c > 1)
    print("%s: %d pieces of %d layers + head, from %s" % (graph, len(ranges) - 1, per, os.path.basename(src)))
    for name, rng, n_nodes, nbytes in counts:
        print("  %-12s %-9s %5d nodes  %6.2f GB weights"
              % (name, "head" if rng is None else "L%d-%d" % rng, n_nodes, nbytes / 1e9))
    print("  original nodes in no piece: %d %s" % (len(dropped), dropped[:4]))
    print("  nodes repeated in several pieces (RoPE/mask prep): %d" % shared)
    return dict(source=graph + ".onnx", layers_per_piece=per, pieces=pieces)


def update_manifest(onnx_dir, files):
    root = os.path.dirname(os.path.abspath(onnx_dir))
    man = os.path.join(root, "MANIFEST.sha256")
    if not os.path.exists(man):
        return
    rels = {os.path.relpath(os.path.abspath(f), root) for f in files}
    piece = re.compile(r"\.p\d\d\.onnx$")            # stale pieces from an earlier split
    lines = [l for l in open(man).read().splitlines()
             if l.split(None, 1)[1].strip() not in rels
             and not piece.search(l.split(None, 1)[1].strip())]
    for rel in sorted(rels):
        lines.append("%s  %s" % (sha256(os.path.join(root, rel)), rel))
    open(man, "w").write("\n".join(lines) + "\n")
    print("MANIFEST.sha256: %d piece entries written" % len(rels))


def main():
    ap = argparse.ArgumentParser(description="Split graphs into buildable pieces.")
    ap.add_argument("--onnx", default=os.path.join(WORK_ROOT, "onnx"))
    ap.add_argument("--layers", nargs="*", default=[],
                    help="graph=N overrides, e.g. decode=4 prefill=3 (default %s)" % DEFAULT_LAYERS)
    ap.add_argument("--graphs", default="expert,decode,prefill")
    args = ap.parse_args()
    onnx_dir = os.path.expanduser(args.onnx)
    per = dict(DEFAULT_LAYERS)
    for spec in args.layers:
        k, v = spec.split("=")
        per[k] = int(v)

    manifest_path = os.path.join(onnx_dir, "pieces.json")
    doc = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"format": 1, "graphs": {}}
    written = []
    for graph in args.graphs.split(","):
        if not os.path.exists(os.path.join(onnx_dir, graph + ".onnx")):
            print("%s.onnx not found -- skipped" % graph)
            continue
        doc["graphs"][graph] = split_graph(onnx_dir, graph, per[graph])
        written += [os.path.join(onnx_dir, p["file"]) for p in doc["graphs"][graph]["pieces"]]
    json.dump(doc, open(manifest_path, "w"), indent=1)
    written.append(manifest_path)
    print("wrote %s" % manifest_path)
    update_manifest(onnx_dir, written)


if __name__ == "__main__":
    main()
