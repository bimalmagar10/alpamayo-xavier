"""Exercise the real runner with CPU tensors and simulated engines, not GPU validation."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "xavier"))


@unittest.skipIf(torch is None, "CPU PyTorch required for simulated runner test")
class RunnerTests(unittest.TestCase):
    def run_runner(self, extra=(), fail_second_frame=False, missing_vocab=False, prepared_sample=False):
        from alpamayo_xavier import telemetry
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as stack:
            work = Path(folder)
            fx = work / "fixtures"
            fx.mkdir()
            (work / "engines").mkdir()
            n_prefill, visual_tokens = (17, 16) if prepared_sample else (2, 1)
            meta = dict(prefill=n_prefill, max_seq=n_prefill + 6, v0=8.71,
                        visual_tokens=visual_tokens, vocab=155697)
            (fx / "meta.json").write_text(json.dumps(meta))
            np.save(fx / "embed_tokens.fp16.npy", np.zeros((155697, 2), dtype=np.float16))
            np.save(fx / "input_ids.npy", np.array([1] * visual_tokens + [2]))
            np.save(fx / "position_ids.npy", np.zeros((3, n_prefill), dtype=np.int64))
            np.save(fx / "visual_mask.npy", np.array([True] * visual_tokens + [False]))
            if not missing_vocab:
                shutil.copyfile(ROOT / "artifacts/telemetry/vocab.json", fx / "vocab.json")
            from PIL import Image
            for i in range(16):
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(work / ("%02d.png" % i))
            grid = None
            if prepared_sample:
                case = work / "samples/new_clip__6100000"
                (case / "fixtures").mkdir(parents=True)
                (case / "frames").mkdir()
                grid = np.tile([1, 2, 2], (16, 1))
                np.save(fx / "image_grid_thw.npy", grid)
                for name in ("input_ids.npy", "position_ids.npy", "visual_mask.npy", "image_grid_thw.npy"):
                    shutil.copyfile(fx / name, case / "fixtures" / name)
                names = ["frames/%02d.png" % i for i in reversed(range(16))]
                for name in names:
                    shutil.copyfile(work / Path(name).name, case / name)
                (case / "fixtures/meta.json").write_text(json.dumps(dict(
                    meta, v0=12.34, clip="new_clip", t0_us=6100000, image_files=names)))

            loads = {}
            class Engine:
                def __init__(self, path, skip=()):
                    self.name = Path(path).name
                    loads[self.name] = loads.get(self.name, 0) + 1
                    if fail_second_frame and self.name == "vision" and loads[self.name] >= 2:
                        raise RuntimeError("simulated second-frame load failure")

                def bind(self, *args):
                    pass

                def close(self):
                    pass

                def __call__(self, args):
                    if self.name == "vision":
                        return {key: torch.zeros(visual_tokens, 2, dtype=torch.float16) for key in
                                ("visual_embeds", "deepstack0", "deepstack1", "deepstack2")}
                    if self.name in ("prefill", "decode"):
                        n = n_prefill if self.name == "prefill" else 1
                        keys = ("k_cache", "v_cache") if self.name == "prefill" else ("new_k", "new_v")
                        out = {k: torch.zeros(36, 1, 8, n, 128, dtype=torch.float16) for k in keys}
                        out["logits"] = torch.zeros(1, 1, 155697, dtype=torch.float16)
                        return out
                    return {"velocity": torch.zeros(1, 64, 2, dtype=torch.float16)}

            fake_trt = types.ModuleType("alpamayo_xavier.trt_runner")
            fake_trt.Engine = Engine
            fake_trt.engine_path = lambda directory, name, precision: name
            fake_trt.plan_bytes = lambda *args: 0
            stack.enter_context(patch.dict(sys.modules, {"alpamayo_xavier.trt_runner": fake_trt}))
            spec = importlib.util.spec_from_file_location("runner_under_test", ROOT / "xavier/run_alpamayo.py")
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            class TorchDecoder:
                sequential = False
                graph = "prefill"
                pieces = [{"name": "torch-decode"}]

                def __init__(self, work):
                    self.runners = {}
                    self.doc = {"graphs": {"prefill": {"source": "prefill.onnx.data"}}}

                def bind_kv(self, *args):
                    pass

                def load(self):
                    if not self.runners:
                        self.engine = Engine("decode")
                        self.runners["torch-decode"] = self.engine

                def run(self, feed):
                    self.load()
                    out = self.engine(feed)
                    return out, [(0, 35, out["new_k"], out["new_v"])]

                def close(self):
                    self.runners.clear()
            stack.enter_context(patch.object(runner, "TorchDecode", TorchDecoder))
            stack.enter_context(patch.object(runner.envinfo, "collect", return_value={"test_backend": "simulated"}))
            stack.enter_context(patch.object(runner.preprocess, "preprocess_images",
                                             return_value=(np.zeros((1, 2), np.float16), grid)))
            stack.enter_context(patch.object(runner.preprocess, "token_counts", return_value=(4 * visual_tokens, visual_tokens)))
            tokens = iter([58289, 155681] * 10)  # "Slow" and the trajectory marker
            stack.enter_context(patch.object(runner, "sample", side_effect=lambda *args: next(tokens)))
            stack.enter_context(patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self))
            generator = torch.Generator
            stack.enter_context(patch.object(torch, "Generator", lambda **kw: generator(device="cpu")))
            for name in ("zeros", "full", "randn"):
                original = getattr(torch, name)
                def cpu_call(*args, _original=original, **kw):
                    kw.pop("device", None)
                    return _original(*args, **kw)
                stack.enter_context(patch.object(torch, name, cpu_call))

            class Event:
                def __init__(self, **kw):
                    self.time = 0

                def record(self):
                    self.time = time.perf_counter()

                def elapsed_time(self, end):
                    return (end.time - self.time) * 1000

            for name, value in dict(Event=Event, synchronize=lambda: None, empty_cache=lambda: None,
                                    reset_peak_memory_stats=lambda: None,
                                    get_device_name=lambda *a: "SIMULATED Xavier",
                                    get_device_capability=lambda *a: (7, 2),
                                    mem_get_info=lambda: (24 * 2**30, 32 * 2**30),
                                    memory_stats=lambda: {}, memory_allocated=lambda: 0,
                                    memory_reserved=lambda: 0, max_memory_allocated=lambda: 0,
                                    max_memory_reserved=lambda: 0).items():
                stack.enter_context(patch.object(torch.cuda, name, value))
            result = work / "results" / "run.json"
            source_args = ["--sample", str(case)] if prepared_sample else ["--images", str(work / "*.png")]
            argv = ["run_alpamayo.py", "--work", str(work)] + source_args + [
                    "--precision", "fp16", "--decode", "trt", "--repeat", "2",
                    "--max-new-tokens", "4", "--flow-steps", "2", "--telemetry-interval", "0",
                    "--json", str(result)] + list(extra)
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            error = None
            try:
                runner.main()
            except RuntimeError as exc:
                error = exc
            with result.open() as f:
                data = json.load(f)
            # Save a clearly labeled example from simulated execution for reviewers.
            if not extra and not fail_second_frame and not missing_vocab and not prepared_sample:
                data["example_only"] = "SIMULATED CPU backend; timings are not Xavier measurements"
                telemetry.write_json(str(ROOT / "artifacts/telemetry/example-results.json"), data)
            return data, loads, error

    def test_two_frames_and_text_and_timings(self):
        data, loads, error = self.run_runner()
        self.assertIsNone(error)
        self.assertEqual(data["status"], "complete")
        self.assertEqual(loads["decode"], 1)
        self.assertEqual(data["summary"]["subsequent_frames"]["count"], 1)
        for run in data["runs"]:
            self.assertEqual(run["reasoning_tokens"], 2)
            self.assertEqual(run["decode_steps"], 2)
            self.assertEqual(run["reasoning_token_ids"], [58289, 155681])
            self.assertTrue(run["reasoning_text"].endswith("<|traj_future_start|>"))
            self.assertGreaterEqual(run["frame_wall_ms"], run["total_ms"])
            t = run["telemetry"]["timing"]
            self.assertAlmostEqual(sum(t["exclusive_wall_ms"].values()) + t["uninstrumented_wall_ms"],
                                   run["frame_wall_ms"], places=4)

    def test_token_limit_does_not_record_an_unprocessed_token(self):
        data, _, _ = self.run_runner(extra=["--max-new-tokens", "1"])
        for run in data["runs"]:
            self.assertEqual(run["reasoning_tokens"], 1)
            self.assertEqual(run["decode_steps"], 1)

    def test_no_reasoning_does_not_load_decoder(self):
        data, loads, _ = self.run_runner(extra=["--no-reasoning"])
        self.assertNotIn("decode", loads)
        self.assertEqual(data["runs"][0]["reasoning_text"], "")
        self.assertEqual(data["runs"][0]["decode_steps"], 0)

    def test_missing_vocab_keeps_ids_and_explicit_null_text(self):
        data, _, _ = self.run_runner(missing_vocab=True)
        self.assertIsNone(data["runs"][0]["reasoning_text"])
        self.assertEqual(data["runs"][0]["reasoning_text_status"], "vocabulary_missing")

    def test_completed_frame_survives_later_failure(self):
        data, _, error = self.run_runner(fail_second_frame=True)
        self.assertIsNotNone(error)
        self.assertEqual(data["status"], "failed")
        self.assertEqual(len(data["runs"]), 1)
        self.assertEqual(data["error"]["completed_frames"], 1)

    def test_hybrid_run_reports_torch_fp16_and_reuses_decoder(self):
        data, loads, error = self.run_runner(extra=["--decode", "torch", "--precision", "int8"])
        self.assertIsNone(error)
        self.assertEqual(loads["decode"], 1)
        stages = data["configuration"]["selected_stages"]
        self.assertEqual(stages["decode"]["backend"], "torch")
        self.assertEqual(stages["decode"]["precision"], "fp16")
        self.assertEqual(stages["prefill"]["precision"], "int8")
        self.assertEqual(data["runs"][0]["reasoning_text"], "Slow<|traj_future_start|>")

    def test_lazy_mode_reloads_decoder_each_frame(self):
        data, loads, error = self.run_runner(extra=["--residency", "lazy"])
        self.assertIsNone(error)
        self.assertEqual(loads["decode"], 2)
        self.assertEqual(data["runs"][1]["telemetry"]["residency"]["before"], [])

    def test_new_sample_selects_matching_metadata_and_shared_embedding_and_vocabulary(self):
        data, _, error = self.run_runner(prepared_sample=True)
        self.assertIsNone(error)
        cfg = data["configuration"]
        self.assertEqual(cfg["fixtures"]["v0"], 12.34)
        self.assertEqual(cfg["fixtures"]["clip"], "new_clip")
        self.assertEqual(cfg["fixtures"]["t0_us"], 6100000)
        self.assertTrue(cfg["images"][0].endswith("15.png"))
        self.assertEqual(data["runs"][0]["reasoning_text"], "Slow<|traj_future_start|>")


if __name__ == "__main__":
    unittest.main()
