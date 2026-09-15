#!/usr/bin/env python3
"""Stage A3c -- make the ONNX graphs buildable AND correct on TensorRT 8.5 fp16.

Two problems, one rewrite:

1. torch.onnx at opset 17 writes every nn.LayerNorm as one LayerNormalization node.
   TensorRT 8.5's ONNX parser has no importer for it (added in 8.6; onnx-tensorrt
   release/8.5-GA vs release/8.6-GA), so the build fails. vision.onnx has 58 such
   nodes, expert.onnx one.

2. Written out naively in fp16, LayerNorm overflows on this model. Measured in an
   fp32 run of vision.onnx: from block 10 on, activations reach ~535 (x^2 ~ 2.9e5)
   and the merger's input reaches ~13,000 (row variance ~1.5e5), both beyond fp16's
   65,504. fp32 casts in the graph do not save it: TensorRT's Myelin compiler fuses
   the whole tower into one node and ran the statistics in fp16 anyway -- the first
   engine came out with cosine 0.29 against the reference.

So each LayerNorm becomes primitives TensorRT 8.5 imports (Abs, ReduceMax,
ReduceMean, Sub, Mul, Add, Div, Sqrt) in a form that cannot overflow in fp16:
every row is first divided by its own max |x|. LayerNorm is invariant to scaling
its input -- (x/s - mean(x/s)) / sqrt(var(x/s) + eps/s^2) is exactly LayerNorm(x)
-- so this changes nothing in fp32, and in fp16 it keeps every intermediate in
[-2, 2]. The fp32 casts stay as well, for engines that do honour them.

3. The language model's RMSNorm (and the expert's) has the same problem, worse. It
   was exported as Cast(fp32) -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div ->
   Mul(x); measured in fp32, the residual stream reaches 7,517 at layer 4 and 26,127
   from layer 17, so x^2 reaches ~6.8e8. In fp16 mean(x^2) becomes inf, 1/sqrt(inf)
   is 0, and every normalised output is exactly zero: the first prefill engines
   produced last_hidden == 0. Each such chain is rewired to divide by the row's
   max |x| first -- (x/s) / sqrt(mean((x/s)^2) + eps/s^2), again exact in fp32.

Only the small .onnx file changes; weights stay in <name>.onnx.data under the same
names and offsets. Files rewritten by the first version of this script (nodes named
.../decomposedN/...) are collapsed back and rewritten in the safe form.

    python h100/a3c_decompose_layernorm.py                        # $ALPAMAYO_ROOT/onnx/*.onnx
    python h100/a3c_decompose_layernorm.py ~/alpamayo-payload/onnx/vision.onnx
    python h100/a3c_decompose_layernorm.py --selftest [rows.npy ...]   # needs onnxruntime
"""
import argparse
import collections
import glob
import hashlib
import os
import re
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

WORK_ROOT = os.environ.get("ALPAMAYO_ROOT", "/mnt/SHARED-SCRATCH/bthapama/alpamayo-work")
TAG = "ln_fp16safe"                                   # names of nodes this version writes
OLD = re.compile(r"^(?P<base>.*)/decomposed(?P<k>\d+)/(?P<part>\w+)$")   # first version


def decompose(node, dtype, uid):
    """Nodes equivalent to one LayerNormalization over the last axis, fp16-safe."""
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    if attrs.get("axis", -1) != -1:
        raise ValueError("%s: only axis=-1 is handled, got %s" % (node.name, attrs["axis"]))
    eps = float(attrs.get("epsilon", 1e-5))
    x, scale = node.input[0], node.input[1]
    bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
    y = node.output[0]
    half = dtype == TensorProto.FLOAT16
    p = "%s/%s%d" % (node.name or "LayerNorm", TAG, uid)
    out = []

    def op(kind, inputs, name, **kw):
        out.append(helper.make_node(kind, inputs, [p + "/" + name], name=p + "/" + name, **kw))
        return p + "/" + name

    def const(name, value):
        return op("Constant", [], name, value=numpy_helper.from_array(np.array(value, np.float32)))

    x32 = op("Cast", [x], "x_fp32", to=TensorProto.FLOAT) if half else x
    amax = op("ReduceMax", [op("Abs", [x32], "abs")], "row_absmax", axes=[-1], keepdims=1)
    s = op("Add", [amax, const("tiny", 1e-4)], "row_scale")        # no 0/0 on an all-zero row
    xs = op("Div", [x32, s], "x_scaled")                           # every element in [-1, 1]
    mean = op("ReduceMean", [xs], "mean", axes=[-1], keepdims=1)
    d = op("Sub", [xs, mean], "centered")
    var = op("ReduceMean", [op("Mul", [d, d], "squared")], "var", axes=[-1], keepdims=1)
    # eps / s^2: if s^2 overflows fp16 it becomes inf and this term 0 -- correct, since
    # a row that large has a variance far above eps.
    eps_s = op("Div", [const("eps", eps), op("Mul", [s, s], "scale_sq")], "eps_scaled")
    std = op("Sqrt", [op("Add", [var, eps_s], "var_eps")], "std")
    xn = op("Div", [d, std], "normalized")
    w = op("Cast", [scale], "scale_fp32", to=TensorProto.FLOAT) if half else scale
    yv = op("Mul", [xn, w], "scaled")
    if bias:
        b = op("Cast", [bias], "bias_fp32", to=TensorProto.FLOAT) if half else bias
        yv = op("Add", [yv, b], "shifted")
    if half:
        out.append(helper.make_node("Cast", [yv], [y], name=p + "/to_fp16", to=TensorProto.FLOAT16))
    else:
        out[-1].output[0] = y                       # the last op writes the original output
    return out


def collapse_previous(graph):
    """Turn first-version decompositions back into LayerNormalization nodes."""
    groups = collections.OrderedDict()
    for i, nd in enumerate(graph.node):
        m = OLD.match(nd.name)
        if m:
            groups.setdefault((m.group("base"), m.group("k")), {})[m.group("part")] = (i, nd)
    if not groups:
        return 0
    at, drop = {}, set()
    for (base, _), parts in groups.items():
        get = lambda part: parts[part][1]                           # noqa: E731
        half = "x_fp32" in parts
        x = get("x_fp32").input[0] if half else get("mean").input[0]
        scale = get("scale_fp32").input[0] if half else get("scaled").input[1]
        bias = None
        if "shifted" in parts:
            bias = get("bias_fp32").input[0] if half else get("shifted").input[1]
        eps = float(numpy_helper.to_array(get("eps").attribute[0].t))
        y = get("to_fp16").output[0] if half else get("shifted" if bias else "scaled").output[0]
        at[min(i for i, _ in parts.values())] = helper.make_node(
            "LayerNormalization", [x, scale] + ([bias] if bias else []), [y],
            name=base, axis=-1, epsilon=eps)
        drop.update(i for i, _ in parts.values())
    nodes = []
    for i, nd in enumerate(graph.node):
        if i in at:
            nodes.append(at[i])
        if i not in drop:
            nodes.append(nd)
    del graph.node[:]
    graph.node.extend(nodes)
    return len(groups)


RMS_TAG = "rms_fp16safe"


def _const(graph, name):
    """Value of an initializer or Constant output, or None."""
    for t in graph.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    for n in graph.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    return numpy_helper.to_array(a.t)
    return None


def rmsnorm_fp16safe(graph):
    """Rewire every exported RMSNorm so it cannot overflow in fp16. Returns count.

    Matches   x -> Pow(2) -> ReduceMean -> Add(eps) -> Sqrt -> Div(_, sqrt) -> Mul(x, _)
    and makes Pow and that Mul read x/s instead of x, and Add read eps/s^2, with
    s = max|x| over the row (+1e-4 so an all-zero row stays finite). Idempotent.
    """
    cons = {}
    for n in graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    producer = {o: n for n in graph.node for o in n.output}

    def only(tensor, op):
        c = [n for n in cons.get(tensor, []) if n.op_type == op]
        return c[0] if len(c) == 1 else None

    before, count = {}, 0
    for pw in graph.node:
        if pw.op_type != "Pow" or ("/" + RMS_TAG) in pw.input[0]:
            continue
        e = _const(graph, pw.input[1])
        if e is None or float(np.asarray(e).ravel()[0]) != 2.0:
            continue
        x = pw.input[0]
        rm = only(pw.output[0], "ReduceMean")
        add = rm and only(rm.output[0], "Add")
        sq = add and only(add.output[0], "Sqrt")
        dv = sq and only(sq.output[0], "Div")
        mu = dv and only(dv.output[0], "Mul")
        if not (rm and add and sq and dv and mu) or dv.input[1] != sq.output[0] or x not in mu.input:
            continue
        src = producer.get(x)
        dtype = TensorProto.FLOAT
        if src is not None and src.op_type == "Cast":
            dtype = [a.i for a in src.attribute if a.name == "to"][0]
        np_t = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
        p = "%s/%s%d" % (pw.name or "RMSNorm", RMS_TAG, count)
        eps = [i for i in add.input if i != rm.output[0]][0]
        mk = helper.make_node
        head = [mk("Abs", [x], [p + "/abs"], name=p + "/abs"),
                mk("ReduceMax", [p + "/abs"], [p + "/row_absmax"], name=p + "/row_absmax", axes=[-1], keepdims=1),
                mk("Constant", [], [p + "/tiny"], name=p + "/tiny", value=numpy_helper.from_array(np.array(1e-4, np_t))),
                mk("Add", [p + "/row_absmax", p + "/tiny"], [p + "/row_scale"], name=p + "/row_scale"),
                mk("Div", [x, p + "/row_scale"], [p + "/x_scaled"], name=p + "/x_scaled"),
                mk("Mul", [p + "/row_scale", p + "/row_scale"], [p + "/scale_sq"], name=p + "/scale_sq")]
        before.setdefault(id(pw), []).extend(head)
        before.setdefault(id(add), []).append(
            mk("Div", [eps, p + "/scale_sq"], [p + "/eps_scaled"], name=p + "/eps_scaled"))
        pw.input[0] = p + "/x_scaled"
        mu.input[list(mu.input).index(x)] = p + "/x_scaled"
        add.input[list(add.input).index(eps)] = p + "/eps_scaled"
        count += 1
    if count:
        nodes = []
        for n in graph.node:
            nodes.extend(before.get(id(n), []))
            nodes.append(n)
        del graph.node[:]
        graph.node.extend(nodes)
    return count


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
    old = collapse_previous(m.graph)
    n = rewrite_graph(m.graph)
    r = rmsnorm_fp16safe(m.graph)
    if n == 0 and r == 0:
        print("  %-16s nothing to rewrite -- unchanged" % os.path.basename(path))
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
    print("  %-16s LayerNorm %d%s, RMSNorm %d -> fp16-safe; %d LayerNormalization left; weights untouched"
          % (os.path.basename(path), n, " (%d upgraded)" % old if old else "", r, left))
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


# ------------------------------------------------------------------------------
# self-test
# ------------------------------------------------------------------------------
def _reference(x, w, b, eps):
    x = x.astype(np.float64)
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * w + b


def _fp16_arithmetic(x, w, b, eps, safe):
    """Both formulations with every intermediate stored in fp16 -- what an engine
    that ignores the fp32 casts computes. Reductions accumulate in fp32 and round
    the result to fp16, as GPU kernels do."""
    h = np.float16
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        x = x.astype(h)
        if safe:
            s = (np.abs(x).max(-1, keepdims=True) + h(1e-4)).astype(h)
            xs = (x / s).astype(h)
            eps_t = (h(eps) / (s * s).astype(h)).astype(h)
        else:
            xs, eps_t = x, h(eps)
        mean = xs.mean(-1, keepdims=True, dtype=np.float32).astype(h)
        d = (xs - mean).astype(h)
        var = (d * d).astype(h).mean(-1, keepdims=True, dtype=np.float32).astype(h)
        y = (d / np.sqrt((var + eps_t).astype(h)).astype(h)).astype(h)
        return (y * w.astype(h) + b.astype(h)).astype(h)


def _cos(a, b):
    a = np.nan_to_num(a.astype(np.float64), posinf=0, neginf=0).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def selftest(npy_files=()):
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    ok = True

    print("  1. the rewritten graph in ONNX Runtime vs a float64 reference")
    for dtype, np_t, tol in ((TensorProto.FLOAT16, np.float16, 2e-2),
                             (TensorProto.FLOAT, np.float32, 1e-4)):
        C, eps = 1152, 1e-6
        x = (rng.standard_normal((4, 9, C)) * 3 + 0.5).astype(np_t)
        x[..., 7] *= 60
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
        err = float(np.abs(got - _reference(x, w, b, eps)).max())
        good = err < tol and not np.isnan(got).any()
        ok &= good
        print("     %-7s max |err| %.2e  (tol %.0e)  %s"
              % (TensorProto.DataType.Name(dtype), err, tol, "ok" if good else "FAIL"))

    print("  1b. an exported-style RMSNorm, rewritten, in ONNX Runtime (fp32) with a 26,000 outlier")
    C, eps = 4096, 1e-6
    x = (rng.standard_normal((2, 5, C)) * 2).astype(np.float16)
    x[..., 11] = 26000.0
    w = (1 + 0.1 * rng.standard_normal(C)).astype(np.float16)
    mk = helper.make_node
    g = helper.make_graph([
        mk("Cast", ["x"], ["x32"], to=TensorProto.FLOAT),
        mk("Pow", ["x32", "two"], ["sq"]), mk("ReduceMean", ["sq"], ["ms"], axes=[-1], keepdims=1),
        mk("Add", ["ms", "eps"], ["mse"]), mk("Sqrt", ["mse"], ["rt"]), mk("Div", ["one", "rt"], ["rs"]),
        mk("Mul", ["x32", "rs"], ["xn"]), mk("Cast", ["w"], ["w32"], to=TensorProto.FLOAT),
        mk("Mul", ["xn", "w32"], ["y32"]), mk("Cast", ["y32"], ["y"], to=TensorProto.FLOAT16)], "rms",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, x.shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, x.shape)],
        initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(np.array(2.0, np.float32), "two"),
                     numpy_helper.from_array(np.array(eps, np.float32), "eps"),
                     numpy_helper.from_array(np.array(1.0, np.float32), "one")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    assert rmsnorm_fp16safe(m.graph) == 1 and rmsnorm_fp16safe(m.graph) == 0   # idempotent
    onnx.checker.check_model(m)
    got = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"]
                               ).run(None, {"x": x})[0].astype(np.float64)
    x64 = x.astype(np.float64)
    ref = x64 / np.sqrt((x64 ** 2).mean(-1, keepdims=True) + eps) * w
    c = _cos(got, ref)
    ok &= c > 0.9999
    print("     rewritten graph vs float64 reference   cos %.6f  %s" % (c, "ok" if c > 0.9999 else "FAIL"))
    h16 = np.float16
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xh = x.astype(h16)
        ms = (xh * xh).astype(h16).mean(-1, keepdims=True, dtype=np.float32).astype(h16)
        naive = (xh * (h16(1) / np.sqrt((ms + h16(eps)).astype(h16))).astype(h16)).astype(h16) * w
        s = (np.abs(xh).max(-1, keepdims=True) + h16(1e-4)).astype(h16)
        xs = (xh / s).astype(h16)
        ms2 = (xs * xs).astype(h16).mean(-1, keepdims=True, dtype=np.float32).astype(h16)
        e2 = (h16(eps) / (s * s).astype(h16)).astype(h16)
        safe = (xs * (h16(1) / np.sqrt((ms2 + e2).astype(h16))).astype(h16)).astype(h16) * w
    c_n, c_s = _cos(naive, ref), _cos(safe, ref)
    ok &= c_s > 0.9999
    print("     pure fp16: naive cos %.4f (max |y| %.1f)   safe cos %.6f  %s"
          % (c_n, float(np.nan_to_num(np.abs(naive.astype(np.float64))).max()), c_s, "ok" if c_s > 0.9999 else "FAIL"))

    print("  2. pure fp16 arithmetic (an engine that ignores the casts), naive vs safe")
    cases = []
    for label, amp in (("block-like, outliers ~535", 535.0), ("merger-like, outliers ~13000", 13000.0)):
        x = rng.standard_normal((64, 1152)).astype(np.float32) * 3
        x[:, rng.choice(1152, 4, replace=False)] = amp * rng.choice([-1, 1], (64, 4))
        cases.append((label, x))
    for f in npy_files:
        x = np.load(f).astype(np.float32).reshape(-1, np.load(f).shape[-1])
        cases.append(("real: " + os.path.basename(f), x[:: max(1, len(x) // 2048)]))
    for label, x in cases:
        C = x.shape[-1]
        w = (1 + 0.1 * rng.standard_normal(C)).astype(np.float32)
        b = (0.1 * rng.standard_normal(C)).astype(np.float32)
        ref = _reference(x, w, b, 1e-6)
        naive = _fp16_arithmetic(x, w, b, 1e-6, safe=False)
        safe = _fp16_arithmetic(x, w, b, 1e-6, safe=True)
        c_n, c_s = _cos(naive, ref), _cos(safe, ref)
        good = c_s > 0.9999
        ok &= good
        print("     %-44s naive cos %.4f (%d non-finite)   safe cos %.6f  %s"
              % (label[:44], c_n, int((~np.isfinite(naive)).sum()), c_s, "ok" if good else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser(description="Replace LayerNormalization for TensorRT 8.5 fp16.")
    ap.add_argument("paths", nargs="*", help="default: $ALPAMAYO_ROOT/onnx/*.onnx "
                                             "(with --selftest: optional .npy rows to test)")
    ap.add_argument("--keep-original", action="store_true", help="save <name>.onnx.orig")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(args.paths) else 1)
    paths = args.paths or sorted(glob.glob(os.path.join(WORK_ROOT, "onnx", "*.onnx")))
    paths = [p for p in paths if not p.endswith(".int8.onnx")]
    if not paths:
        sys.exit("no .onnx files found")
    changed = [p for p in paths if rewrite_file(p, args.keep_original)]
    if changed:
        update_manifest(changed)


if __name__ == "__main__":
    main()
