"""Verify portable sample files using synthetic camera/history tensors."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "h100"), str(ROOT / "xavier")]
import a9_prepare_samples as exporter


@unittest.skipIf(torch is None, "CPU PyTorch required for synthetic sample export")
class ExportTests(unittest.TestCase):
    def test_export_writes_all_frames_and_small_fixtures_without_model_weights(self):
        frames = torch.zeros(4, 4, 3, 8, 8, dtype=torch.uint8)
        for c in range(4):
            for t in range(4):
                frames[c, t].fill_(c * 4 + t)
        data = dict(image_frames=frames, camera_indices=torch.tensor([0, 1, 2, 6]),
                    ego_history_xyz=torch.zeros(1, 1, 16, 3),
                    ego_history_rot=torch.eye(3).expand(1, 1, 16, 3, 3),
                    absolute_timestamps=torch.arange(16).reshape(4, 4))
        meta = dict(prefill=17, max_seq=32, visual_tokens=16, vocab=20, image_token_id=7,
                    v0=12.34, clip="new_clip", t0_us=6100000)
        ids = np.array([7] * 16 + [2], dtype=np.int64)
        grid = np.tile([1, 2, 2], (16, 1))
        arrays = dict(input_ids=ids, position_ids=np.zeros((3, 17), dtype=np.int64),
                      visual_mask=ids == 7, image_grid_thw=grid)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case"
            with patch.object(exporter, "compare_preprocessing", return_value={"cosine": 1.0}):
                exporter.export_sample(path, data, arrays, meta, None, meta, grid)
            saved = json.loads((path / "fixtures/meta.json").read_text())
            self.assertEqual(saved["v0"], 12.34)
            self.assertEqual(saved["t0_us"], 6100000)
            self.assertEqual(len(saved["image_files"]), 16)
            for i, name in enumerate(saved["image_files"]):
                with Image.open(path / name) as im:
                    self.assertTrue(np.all(np.asarray(im) == i))
            np.testing.assert_array_equal(np.load(path / "fixtures/input_ids.npy"), ids)
            self.assertFalse((path / "fixtures/embed_tokens.fp16.npy").exists())
            self.assertEqual(list(Path(folder).iterdir()), [path])
            with self.assertRaises(FileExistsError):
                exporter.export_sample(path, data, arrays, meta, None, meta, grid)


if __name__ == "__main__":
    unittest.main()
