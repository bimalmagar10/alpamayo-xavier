#!/usr/bin/env python3
"""Stage A3c -- make the ONNX graphs readable by TensorRT 8.5 (runs anywhere, no GPU).

torch.onnx at opset 17 writes every nn.LayerNorm as one LayerNormalization node.
TensorRT 8.5's ONNX parser has no importer for that op -- it arrived in 8.6
(onnx-tensorrt release/8.5-GA vs release/8.6-GA, builtin_op_importers.cpp) -- so
8.5 falls back to looking for a plugin of that name, finds none, and the build
fails. vision.onnx has 58 of these nodes and expert.onnx has one.

This rewrites each into ops 8.5 does import -- ReduceMean, Sub, Mul, Add, Sqrt,
Div -- computed in float32 and cast back, as torch's own pre-opset-17 export did.
Only the small .onnx file changes. The weights stay in <name>.onnx.data under the
same names and offsets, so nothing large is re-exported or re-transferred.

    python h100/a3c_decompose_layernorm.py                         # $ALPAMAYO_ROOT/onnx/*.onnx
    python h100/a3c_decompose_layernorm.py ~/alpamayo-payload/onnx/vision.onnx
    python h100/a3c_decompose_layernorm.py --selftest              # needs onnxruntime
"""
import argparse
import glob
import hashlib
import os
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")


def decompose(node, dtype, uid):
    """Nodes equivalent to one LayerNormalization over the last axis."""
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attrs.get("axis", -1) != -1:
        raise ValueError("%s: only axis=-1 is handled, got %s" % (node.name, attrs["axis"]))
    eps = float(attrs.get("epsilon", 1e-5))
    x, scale = node.input[0], node.input[1]
    bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
    y = node.output[0]
    half = dtype == TensorProto.FLOAT16
    p = "%s/decomposed%d" % (node.name or "LayerNorm", uid)
    out = []

    def op(kind, inputs, name, **kw):
        out.append(helper.make_node(kind, inputs, [p + "/" + name], name=p + "/" + name, **kw))
        return p + "/" + name

    x32 = op("Cast", [x], "x_fp32", to=TensorProto.FLOAT) if half else x
    mean = op("ReduceMean", [x32], "mean", axes=[-1], keepdims=1)
    d = op("Sub", [x32, mean], "centered")
    var = op("ReduceMean", [op("Mul", [d, d], "squared")], "var", axes=[-1], keepdims=1)
    eps_c = op("Constant", [], "eps", value=numpy_helper.from_array(np.array(eps, np.float32)))
    std = op("Sqrt", [op("Add", [var, eps_c], "var_eps")], "std")
    xn = op("Div", [d, std], "normalized")
    s = op("Cast", [scale], "scale_fp32", to=TensorProto.FLOAT) if half else scale
    yv = op("Mul", [xn, s], "scaled")
    if bias:
        b = op("Cast", [bias], "bias_fp32", to=TensorProto.FLOAT) if half else bias
        yv = op("Add", [yv, b], "shifted")
    if half:
        out.append(helper.make_node("Cast", [yv], [y], name=p + "/to_fp16", to=TensorProto.FLOAT16))
    else:
        out[-1].output[0] = y                       # the last op writes the original output
    return out


def rewrite_graph(graph):
    """Replace LayerNormalization in place, keeping topological order. Returns count."""
    types = {t.name: t.data_type for t in graph.initializer}
    nodes, n = [], 0
    for node in graph.node:
        if node.op_type != "LayerNormalization":
            nodes.append(node)
            continue
        if len(node.output) > 1 and any(node.output[1:]):
            raise ValueError("%s: Mean/InvStdDev outputs are used; not handled" % node.name)
        dtype = types.get(node.input[1])
        if dtype is None:
            raise ValueError("%s: scale %s is not an initializer" % (node.name, node.input[1]))
        nodes.extend(decompose(node, dtype, n))
        n += 1
    if n:
        del graph.node[:]
        graph.node.extend(nodes)
    return n


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rewrite_file(path, keep_original=False):
    m = onnx.load(path, load_external_data=False)   # structure only; weights stay on disk
    n = rewrite_graph(m.graph)
    if n == 0:
        print("  %-14s no LayerNormalization -- unchanged" % os.path.basename(path))
        return False
    data = path + ".data"
    before = os.stat(data) if os.path.exists(data) else None
    tmp = path + ".a3c.tmp"
    onnx.save(m, tmp)                               # same dir, so the .data reference holds
    try:
        onnx.checker.check_model(tmp)               # by path: also resolves external data
    except Exception:
        os.remove(tmp)
        raise
    if keep_original:
        shutil.copy2(path, path + ".orig")
    os.replace(tmp, path)
    if before is not None:
        after = os.stat(data)
        assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime), \
            "%s changed -- it must not" % data
    left = sum(nd.op_type == "LayerNormalization"
               for nd in onnx.load(path, load_external_data=False).graph.node)
    print("  %-14s %d LayerNormalization -> primitives, %d left, weights untouched"
          % (os.path.basename(path), n, left))
    return True


def update_manifest(changed):
    """Refresh the MANIFEST.sha256 entries of rewritten files, if a manifest exists."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(changed[0])))
    man = os.path.join(root, "MANIFEST.sha256")
    if not os.path.exists(man):
        return
    want = {os.path.abspath(p) for p in changed}
    lines, hit = [], 0
    for line in open(man).read().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.strip().lstrip("*")
        if os.path.abspath(os.path.join(root, rel)) in want:
            digest, hit = sha256(os.path.join(root, rel)), hit + 1
        lines.append("%s  %s" % (digest, rel))
    open(man, "w").write("\n".join(lines) + "\n")
    print("  MANIFEST.sha256: %d entr%s refreshed" % (hit, "y" if hit == 1 else "ies"))


def selftest():
    """The rewrite against a float64 reference, in fp16 and fp32, with an outlier channel."""
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    ok = True
    for dtype, np_t, tol in ((TensorProto.FLOAT16, np.float16, 2e-2),
                             (TensorProto.FLOAT, np.float32, 1e-4)):
        C, eps = 1152, 1e-6
        x = (rng.standard_normal((4, 9, C)) * 3 + 0.5).astype(np_t)
        x[..., 7] *= 60                                   # like a ViT outlier channel
        w = rng.standard_normal(C).astype(np_t)
        b = rng.standard_normal(C).astype(np_t)
        g = helper.make_graph(
            [helper.make_node("LayerNormalization", ["x", "w", "b"], ["y"],
                              axis=-1, epsilon=eps, name="ln")], "t",
            [helper.make_tensor_value_info("x", dtype, x.shape)],
            [helper.make_tensor_value_info("y", dtype, x.shape)],
            initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8
        assert rewrite_graph(m.graph) == 1
        onnx.checker.check_model(m)
        got = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]
                                   ).run(None, {"x": x})[0].astype(np.float64)
        x64 = x.astype(np.float64)
        mu = x64.mean(-1, keepdims=True)
        ref = (x64 - mu) / np.sqrt(((x64 - mu) ** 2).mean(-1, keepdims=True) + eps) * w + b
        err = float(np.abs(got - ref).max())
        good = err < tol and not np.isnan(got).any()
        ok &= good
        print("  selftest %-7s max |err| %.2e  (tol %.0e)  %s"
              % (TensorProto.DataType.Name(dtype), err, tol, "ok" if good else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Replace LayerNormalization for TensorRT 8.5.")
    ap.add_argument("paths", nargs="*", help="default: $ALPAMAYO_ROOT/onnx/*.onnx")
    ap.add_argument("--keep-original", action="store_true", help="save <name>.onnx.orig")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    paths = args.paths or sorted(glob.glob(os.path.join(WORK_ROOT, "onnx", "*.onnx")))
    paths = [p for p in paths if not p.endswith(".int8.onnx")]
    if not paths:
        sys.exit("no .onnx files found")
    changed = [p for p in paths if rewrite_file(p, args.keep_original)]
    if changed:
        update_manifest(changed)


if __name__ == "__main__":
    main()
