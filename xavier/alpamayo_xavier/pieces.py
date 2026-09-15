"""Run a graph that h100/a3d_split_graphs.py cut into pieces, as one stage.

Engine-agnostic: a piece runner only needs `.bind(name, buffer)`, `__call__(feed)`
returning {output name: array} and `.close()`. On the Xavier that is
trt_runner.Engine; on the Mac, h100/a6_reference_io.py wraps ONNX Runtime the same
way. So the routing that verify.py and run_alpamayo.py rely on is exactly the code
the fp32 reference was produced with.

pieces.json (written next to the ONNX files) lists, per graph, each piece's file,
layer range, inputs and outputs. Routing is by name:
    hidden_in              <- the previous piece's hidden_out
    past_k / past_v        <- bound once to the shared KV cache (never copied)
    anything else          <- the stage's own inputs (cos, sin, mask, ...)
Pieces that emit K/V (prefill, decode) report (first_layer, last_layer, k, v) so
the caller writes them into its cache.
"""
import json
import os
import time
from contextlib import nullcontext


# I/O of each graph when it was NOT split, as a one-piece spec, so every stage
# runs through the same Stage code whether it is one engine or several.
WHOLE = {
    "vision": dict(inputs=["pixel_values"],
                   outputs=["visual_embeds", "deepstack0", "deepstack1", "deepstack2"], kv=None),
    "prefill": dict(inputs=["inputs_embeds", "cos", "sin", "deepstack0", "deepstack1", "deepstack2"],
                    outputs=["last_hidden", "logits", "k_cache", "v_cache"], kv=["k_cache", "v_cache"]),
    "decode": dict(inputs=["hidden", "cos", "sin", "past_k", "past_v", "mask"],
                   outputs=["logits", "new_k", "new_v"], kv=["new_k", "new_v"]),
    "expert": dict(inputs=["noisy_action", "timestep", "cos", "sin", "past_k", "past_v", "mask"],
                   outputs=["velocity"], kv=None),
}


def specs(graphs, name, layers=36):
    """The pieces of `name` from pieces.json, or a single whole-graph piece."""
    if name in graphs:
        return graphs[name]["pieces"]
    w = WHOLE[name]
    return [dict(name=name, file=name + ".onnx", layers=[0, layers - 1] if w["kv"] else None,
                 head=True, inputs=w["inputs"], outputs=w["outputs"], kv=w["kv"])]


def _own(x):
    """A copy that outlives the runner it came from: torch tensor or numpy array."""
    return x.clone() if hasattr(x, "clone") else x.copy()


def load(onnx_or_engine_dir):
    """The pieces.json next to the ONNX files, or {} when nothing was split."""
    path = os.path.join(onnx_or_engine_dir, "pieces.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path))["graphs"]


class Stage(object):
    def __init__(self, graph, pieces, factory, sequential=False, observer=None):
        """factory(piece_name, skip) -> runner; skip names the bindings bind() will supply. sequential=True loads one piece at a time
        and frees it after use (the Mac's fp32 reference); False keeps them all."""
        self.graph = graph
        self.pieces = pieces
        self.factory = factory
        self.sequential = sequential
        self.runners = {}
        self._kv = None
        self.load_seconds = 0.0      # what run() spent reading plans, in sequential mode
        self.observer = observer

    def _observe(self, kind, p):
        return (self.observer(kind, self.graph, piece=p["name"], sequential=self.sequential)
                if self.observer else nullcontext())

    @property
    def names(self):
        return [p["name"] for p in self.pieces]

    def load(self):
        for p in self.pieces:
            self._get(p)

    def _get(self, p):
        r = self.runners.get(p["name"])
        if r is None:
            skip = ("past_k", "past_v") if self._kv is not None and "past_k" in p["inputs"] else ()
            with self._observe("engine_load", p):
                r = self.runners[p["name"]] = self.factory(p["name"], skip)
                if self._kv is not None and "past_k" in p["inputs"]:
                    r.bind("past_k", self._kv[0])
                    r.bind("past_v", self._kv[1])
        return r

    def bind_kv(self, past_k, past_v):
        """The full [layers, 1, kv_heads, max_seq, head_dim] caches, shared by every piece."""
        self._kv = (past_k, past_v)
        for p in self.pieces:
            if p["name"] in self.runners and "past_k" in p["inputs"]:
                self.runners[p["name"]].bind("past_k", past_k)
                self.runners[p["name"]].bind("past_v", past_v)

    def close(self):
        for r in self.runners.values():
            r.close()
        self.runners.clear()

    def run(self, feed):
        """feed: the stage's inputs by name. Returns (final outputs, kv list)."""
        hidden, kv, out = None, [], None
        self.load_seconds = 0.0
        for p in self.pieces:
            t0 = time.perf_counter()
            fresh = p["name"] not in self.runners
            r = self._get(p)
            if fresh:
                self.load_seconds += time.perf_counter() - t0
            args = {}
            for name in p["inputs"]:
                if name == "hidden_in":
                    args[name] = hidden
                elif name not in ("past_k", "past_v"):
                    args[name] = feed[name]
            with self._observe("engine_execute", p):
                out = r(args)
            if p.get("kv"):
                k_name, v_name = p["kv"]
                kv.append((p["layers"][0], p["layers"][1], out[k_name], out[v_name]))
            if "hidden_out" in p["outputs"]:
                hidden = out["hidden_out"]
            if self.sequential:
                if p.get("kv"):                      # detach from the runner before freeing it
                    a, b, k, v = kv[-1]
                    kv[-1] = (a, b, _own(k), _own(v))
                if "hidden_out" in p["outputs"]:
                    hidden = _own(hidden)
                if not p["head"]:
                    with self._observe("engine_release", p):
                        self.runners.pop(p["name"]).close()
        return out, kv
