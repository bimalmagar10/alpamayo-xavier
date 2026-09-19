#!/usr/bin/env python3
"""Are prefill.onnx and decode.onnx really carrying the same weights?

The export writes the language model twice, once traced over the whole prompt and
once traced over a single position against a cache. The claim that both carry the
same 15.17 GB of parameters underpins two decisions -- decode reading prefill's
weights file in PyTorch, and decode's own engines being discarded -- so it is
worth proving rather than assuming.

Three levels of evidence, cheapest first:

  structure   every weight in both maps has the same role, shape, offset and
              length. Cheap: reads only weight_map.json.
  layout      both .data files are the same size, and the recorded ranges tile
              them the same way.
  content     the bytes at those ranges are equal. This reads both files in
              full (about 30 GB) unless --sample is given.

Needs nothing but numpy and the weight map -- no onnx, no torch, no cluster.

    python analysis/check_export_weights.py                  # structure + layout + bytes
    python analysis/check_export_weights.py --sample 65536    # 64 kB per tensor
    python analysis/check_export_weights.py --structure-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpamayo_figs import data                                    # noqa: E402

CHUNK = 16 << 20            # 16 MB: large enough to stream, small enough to stay in cache


def load_map(root):
    for rel in ("onnx/weight_map.json", "engines/weight_map.json"):
        path = os.path.join(root, rel)
        if os.path.exists(path):
            return json.load(open(path)), path
    raise SystemExit("no weight_map.json under %s -- run h100/a7_weight_map.py" % root)


def flatten(spec):
    """[(label, entry)] for every weight the map records, in layer order."""
    out = []
    for i, layer in enumerate(spec["layers"]):
        for role, w in layer.items():
            out.append(("layer%02d.%s" % (i, role), w))
    for role, w in spec["head"].items():
        out.append(("head.%s" % role, w))
    return out


def compare_structure(a, b):
    """Role-by-role: same dims, same dtype, same place in the file?"""
    fa, fb = flatten(a), flatten(b)
    if len(fa) != len(fb):
        return None, "different number of weights: %d vs %d" % (len(fa), len(fb))
    rows, bad = [], []
    for (la, wa), (lb, wb) in zip(fa, fb):
        same = (la == lb and wa["dims"] == wb["dims"] and wa["dtype"] == wb["dtype"]
                and wa.get("offset") == wb.get("offset")
                and wa.get("length") == wb.get("length")
                and ("inline" in wa) == ("inline" in wb))
        rows.append((la, wa, wb, same))
        if not same:
            bad.append(la)
    return rows, bad


def compare_bytes(rows, path_a, path_b, sample=0):
    """Read the recorded ranges out of both .data files and compare them."""
    ma = np.memmap(path_a, dtype=np.uint8, mode="r")
    mb = np.memmap(path_b, dtype=np.uint8, mode="r")
    checked = mismatched = 0
    bad = []
    t0 = time.time()
    for i, (label, wa, wb, _) in enumerate(rows):
        if "inline" in wa:                          # travels in the .onnx, not the .data
            same = wa["inline"] == wb["inline"]
            n = wa["length"]
        else:
            off, ln = wa["offset"], wa["length"]
            n = min(ln, sample) if sample else ln
            same = True
            for start in range(0, n, CHUNK):
                stop = min(start + CHUNK, n)
                if not np.array_equal(ma[off + start:off + stop], mb[off + start:off + stop]):
                    same = False
                    break
        checked += n
        if not same:
            mismatched += 1
            bad.append(label)
        if i % 50 == 0 or i == len(rows) - 1:
            done = (i + 1) / len(rows)
            sys.stderr.write("\r  comparing %5.1f%%  (%.1f GB, %.0f s)"
                             % (100 * done, checked / 1e9, time.time() - t0))
            sys.stderr.flush()
    sys.stderr.write("\n")
    del ma, mb
    return checked, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=None)
    ap.add_argument("--a", default="prefill")
    ap.add_argument("--b", default="decode")
    ap.add_argument("--sample", type=int, default=0,
                    help="compare only the first N bytes of each tensor (0 = all)")
    ap.add_argument("--structure-only", action="store_true")
    args = ap.parse_args()

    root = data.find_root(args.root)
    if root is None:
        raise SystemExit("no alpamayo root found; pass --root")
    doc, map_path = load_map(root)
    graphs = doc["graphs"]
    for g in (args.a, args.b):
        if g not in graphs:
            raise SystemExit("%s is not in %s (have: %s)"
                             % (g, map_path, ", ".join(sorted(graphs))))
    A, B = graphs[args.a], graphs[args.b]
    pa = os.path.join(root, "onnx", A["source"])
    pb = os.path.join(root, "onnx", B["source"])

    print("weight map    : %s" % map_path)
    print("comparing     : %s  vs  %s\n" % (args.a, args.b))

    # ---- structure ------------------------------------------------------
    rows, bad = compare_structure(A, B)
    if rows is None:
        raise SystemExit("structure differs: %s" % bad)
    ext = [r for r in rows if "inline" not in r[1]]
    inl = [r for r in rows if "inline" in r[1]]
    print("structure")
    print("  weights recorded        %d  (%d external, %d inline in the .onnx)"
          % (len(rows), len(ext), len(inl)))
    print("  role, shape, dtype      %s" % ("identical" if not bad else "DIFFER: %s" % bad[:5]))
    print("  offset and length       %s" % ("identical" if not bad else "see above"))
    ext_bytes = sum(r[1]["length"] for r in ext)
    inl_bytes = sum(r[1]["length"] for r in inl)
    print("  external bytes          %s  (%.2f GB)"
          % (format(ext_bytes, ",d"), ext_bytes / 1e9))
    print("  inline bytes            %s  (the 128- and 4096-element norms)"
          % format(inl_bytes, ",d"))

    # ---- layout ---------------------------------------------------------
    sa, sb = os.path.getsize(pa), os.path.getsize(pb)
    print("\nlayout")
    print("  %-22s %s bytes" % (A["source"], format(sa, ",d")))
    print("  %-22s %s bytes" % (B["source"], format(sb, ",d")))
    print("  same size               %s" % ("yes" if sa == sb else "NO"))
    print("  ranges cover            %.4f%% of the file" % (100.0 * ext_bytes / sa))
    print("  unaccounted             %s bytes" % format(sa - ext_bytes, ",d"))

    if args.structure_only:
        return 0 if not bad else 1

    # ---- content --------------------------------------------------------
    print("\ncontent  (%s)" % ("first %s bytes of each tensor"
                               % format(args.sample, ",d") if args.sample else "every byte"))
    checked, cbad = compare_bytes(rows, pa, pb, args.sample)
    print("  compared                %.2f GB per file" % (checked / 1e9))
    print("  tensors differing       %d" % len(cbad))
    if cbad:
        print("  first differing         %s" % ", ".join(cbad[:5]))

    ok = not bad and not cbad
    print("\n%s" % ("VERDICT: prefill and decode carry the same weights, "
                    "at the same offsets, byte for byte."
                    if ok else "VERDICT: the two graphs DIFFER -- see above."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
