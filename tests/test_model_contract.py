"""Architecture and strict-load contract for the public v1 wrapper."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch

from models import build_model


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "bitrackswin_seg_v1_raw.pth"


class ModelContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = build_model(image_size=256, in_channels=5, num_classes=2)

    def test_public_architecture_count(self) -> None:
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 13_441_205)
        self.assertEqual(self.model.backbone.depths, [4, 6, 2])
        self.assertEqual(self.model.backbone.window_sizes, [8, 8, 32])
        self.assertFalse(self.model.backbone.use_multimodal)

    def test_release_weights_load_strictly(self) -> None:
        if not WEIGHTS.is_file():
            self.skipTest("GitHub Release weight is not present in this checkout")
        payload = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        result = self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
