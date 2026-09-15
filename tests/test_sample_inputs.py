"""Validate sample pairing and fixed-shape engine compatibility without a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "xavier"), str(ROOT / "h100")]
from alpamayo_xavier import preprocess, sample_inputs
from a9_prepare_samples import sample_name


class SampleTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(prefill=17, max_seq=32, visual_tokens=16, vocab=20,
                         clip="new_clip", t0_us=6100000, v0=12.34, image_token_id=7)
        self.ids = np.array([7] * 16 + [2], dtype=np.int64)
        self.positions = np.zeros((3, 17), dtype=np.int64)
        self.mask = self.ids == 7
        self.grid = np.tile([1, 2, 2], (16, 1))

    def test_grid_change_rejected_even_with_same_total_tokens(self):
        with self.assertRaisesRegex(ValueError, "grid"):
            sample_inputs.validate_contract(self.meta, self.meta,
                                            np.tile([1, 20, 36], (16, 1)),
                                            np.tile([1, 36, 20], (16, 1)))

    def test_prompt_length_change_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefill"):
            sample_inputs.validate_contract(dict(self.meta, prefill=18), self.meta)

    def test_mask_must_match_fused_ids(self):
        sample_inputs.validate_arrays(self.meta, self.ids, self.positions, self.mask, 20)
        wrong_mask = self.mask.copy()
        wrong_mask[0], wrong_mask[-1] = False, True
        with self.assertRaisesRegex(ValueError, "placeholder"):
            sample_inputs.validate_arrays(self.meta, self.ids, self.positions, wrong_mask, 20)

    def test_new_sample_uses_its_own_speed_and_preserves_manifest_order(self):
        with tempfile.TemporaryDirectory() as root:
            shared, case = Path(root) / "shared", Path(root) / "case"
            shared.mkdir()
            (case / "fixtures").mkdir(parents=True)
            (case / "frames").mkdir()
            (shared / "meta.json").write_text(json.dumps(dict(self.meta, v0=8.71, clip="old_clip")))
            # Reverse lexical order intentionally: the explicit manifest is authoritative.
            names = ["frames/%02d.png" % i for i in reversed(range(16))]
            for name in names:
                (case / name).write_bytes(b"test")
            meta = dict(self.meta, image_files=names)
            (case / "fixtures/meta.json").write_text(json.dumps(meta))
            for name, array in dict(input_ids=self.ids, position_ids=self.positions,
                                    visual_mask=self.mask, image_grid_thw=self.grid).items():
                np.save(case / "fixtures" / (name + ".npy"), array)
            fixtures, selected, images = sample_inputs.load_sample(str(case), str(shared))
            self.assertEqual(selected["v0"], 12.34)
            self.assertEqual(selected["clip"], "new_clip")
            self.assertTrue(images[0].endswith("15.png"))
            self.assertFalse((Path(fixtures) / "embed_tokens.fp16.npy").exists())
            del meta["v0"]
            (case / "fixtures/meta.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "own finite v0"):
                sample_inputs.load_sample(str(case), str(shared))

    def test_old_golden_grid_is_read_without_loading_pixel_values(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "golden").mkdir()
            np.savez(Path(root) / "golden/inputs.npz", image_grid_thw=self.grid)
            grid = sample_inputs.expected_grid(str(Path(root) / "fixtures"), root)
            np.testing.assert_array_equal(grid, self.grid)

    def test_each_image_is_resized_using_its_own_aspect_ratio(self):
        _, grids = preprocess.preprocess_images([np.zeros((32, 64, 3), dtype=np.uint8),
                                                 np.zeros((64, 32, 3), dtype=np.uint8)])
        self.assertEqual(grids[0, 1], grids[1, 2])
        self.assertNotEqual(grids[0, 1], grids[1, 1])

    def test_sample_names_are_stable_and_do_not_escape_output(self):
        self.assertEqual(sample_name("clip-123", 6100000), "clip-123__6100000")
        with self.assertRaises(ValueError):
            sample_name("../../wrong", 6100000)


if __name__ == "__main__":
    unittest.main()
