"""CPU checks for results accounting, fixtures, and optional hardware probes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "xavier"), str(ROOT / "h100")]
from alpamayo_xavier import envinfo, sysmon, telemetry
from alpamayo_xavier.detok import Detokenizer
from a8_token_strings import bytes_to_unicode, encode_literal, extend_alpamayo, write_vocab


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class TimingTests(unittest.TestCase):
    def test_streamed_load_is_counted_once_and_exclusive_times_partition_wall(self):
        clock = Clock()
        trace = telemetry.Trace(clock=clock)
        with trace("prefill"):
            with trace.scope("engine_load", "prefill"):
                clock.advance(7)
            with trace.scope("engine_execute", "prefill"):
                clock.advance(2)
            clock.advance(1)
        self.assertEqual(trace.stages, {"prefill": [3000.0], "engine load": [7000.0]})
        data = trace.export(10000)
        self.assertEqual(sum(data["exclusive_wall_ms"].values()), 10000)
        self.assertEqual(data["uninstrumented_wall_ms"], 0)

    def test_explicit_load_is_not_counted_twice(self):
        clock = Clock()
        trace = telemetry.Trace(clock=clock)
        with trace("engine load"):
            with trace.scope("engine_load", "expert"):
                clock.advance(3)
        self.assertEqual(trace.total(), 3000)

    def test_exception_unwinds_scopes(self):
        trace = telemetry.Trace()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with trace.scope("load"):
                with trace.scope("piece"):
                    raise RuntimeError("failed")
        self.assertEqual(trace.stack, [])
        self.assertIn("error", trace.export(10)["records"][1])

    def test_percentiles_and_first_frame_separation(self):
        result = telemetry.summary([{"frame_wall_ms": n} for n in [1000, 100, 200]])
        self.assertEqual(result["first_frame"]["p50_ms"], 1000)
        self.assertEqual(result["subsequent_frames"]["p50_ms"], 150)
        self.assertEqual(result["subsequent_frames"]["p95_ms"], 195)

    def test_failed_json_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nested", "results.json")
            telemetry.write_json(path, {"reasoning_text": "नेपाल", "runs": [1]})
            with self.assertRaises(ValueError):
                telemetry.write_json(path, {"bad": float("nan")})
            with open(path) as f:
                self.assertEqual(json.load(f)["runs"], [1])
            self.assertEqual(os.listdir(os.path.dirname(path)), ["results.json"])

    def test_faults_are_not_mislabeled_as_swap(self):
        delta = telemetry.counter_delta({"process": {"major_faults": 2}, "system_vm": {"pswpin": 4}},
                                        {"process": {"major_faults": 6}, "system_vm": {"pswpin": 4}})
        notes = telemetry.observations({"exclusive_wall_ms": {}}, delta)
        self.assertEqual(len(notes), 1)
        self.assertIn("memory-mapped", notes[0]["message"])


class TokenTests(unittest.TestCase):
    def test_multibyte_characters_are_joined_before_utf8_decode(self):
        table = bytes_to_unicode()
        d = Detokenizer([table[0xc3], table[0xa9], encode_literal("<|traj_future_start|>", table)])
        self.assertEqual(d.decode([0, 1, 2]), "é<|traj_future_start|>")
        self.assertEqual(d.pieces([0, 1]), ["�", "�"])

    def test_missing_and_corrupt_vocab_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vocab.json"
            self.assertFalse(Detokenizer.find(folder).available)
            path.write_text('{"encoding": "unknown", "tokens": ["hello"]}')
            result = Detokenizer.find(folder)
            self.assertFalse(result.available)
            self.assertIn("unsupported", result.error)

    def test_checkpoint_id_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "trajectory token ID"):
            extend_alpamayo({0: "a"}, {"traj_vocab_size": 2, "traj_token_start_idx": 99}, bytes_to_unicode())

    def test_vocab_can_be_written_to_current_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "vocab.json")
            write_vocab({0: "hello"}, path, "test", size=1)
            self.assertEqual(Detokenizer.find(path).decode([0]), "hello")

    def test_prepared_r1_vocab_matches_stop_and_embedding_size(self):
        d = Detokenizer.find(str(ROOT / "artifacts/telemetry/vocab.json"))
        self.assertEqual(len(d.vocab), 155697)
        self.assertEqual(d.decode([151669, 155668, 155677, 155681]),
                         "<i0><i3999><|cot_start|><|traj_future_start|>")


class EnvironmentTests(unittest.TestCase):
    def test_failed_probe_never_becomes_a_version_string(self):
        failure = subprocess.CompletedProcess([], 1, stdout=b"package missing", stderr=b"error")
        with patch.object(subprocess, "run", return_value=failure):
            self.assertIsNone(envinfo._dpkg("missing"))
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 1)):
            self.assertIsNone(envinfo._run(["test"]))

    def test_no_git_is_reported_without_fake_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertNotIn("commit", envinfo.code(folder))

    def test_no_sensors_and_permission_failures_are_nonfatal(self):
        with patch.object(sysmon, "_channels", return_value=[]):
            sampler = sysmon.Sampler().start()
            self.assertFalse(sampler.report()["available"])
            sampler.stop()
        self.assertIsNone(sysmon._value("/not/a/sensor", 1))

    def test_sensor_window_isolated_and_power_rails_not_summed(self):
        with patch.object(sysmon, "_channels", return_value=[
                ("power_1_0040_VIN_w", "a", "W", 1), ("power_1_0041_GPU_w", "b", "W", 1)]):
            sampler = sysmon.Sampler()
        sampler.points = [(1., [10., 2.]), (2., [20., 4.])]
        result = sampler.report(1.5, 3)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["channels"]["power_1_0040_VIN_w"]["mean"], 20)
        self.assertNotIn("power_total_w", result["channels"])

    def test_nonfinite_sensor_reading_does_not_break_json(self):
        with tempfile.NamedTemporaryFile(mode="w") as f:
            f.write("nan")
            f.flush()
            self.assertIsNone(sysmon._value(f.name, 1))


if __name__ == "__main__":
    unittest.main()
