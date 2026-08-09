"""Integrity checks for the v1 weights-only checkpoint.

Run from the repository root with:
    python -m unittest tests.test_checkpoint_artifact
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPOSITORY_ROOT / "weights" / "bitrackswin_seg_v1_raw.pth"
METADATA_PATH = REPOSITORY_ROOT / "release" / "v1.0.0-metadata.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    """Match the canonical RAW tensor hash produced by the export script."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


class ReleaseCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not WEIGHTS_PATH.is_file():
            raise unittest.SkipTest(f"Release weight artifact is unavailable: {WEIGHTS_PATH}")
        if not METADATA_PATH.is_file():
            raise unittest.SkipTest(f"Release metadata is unavailable: {METADATA_PATH}")
        cls.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        cls.payload = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)

    def test_only_weights_and_metadata_are_packaged(self) -> None:
        self.assertEqual(set(self.payload), {"model_state_dict", "metadata"})
        self.assertNotIn("ema_state_dict", self.payload)
        self.assertNotIn("optimizer_state_dict", self.payload)
        self.assertNotIn("scheduler_state_dict", self.payload)
        self.assertNotIn("rng_state", self.payload)

    def test_raw_weight_metadata_is_consistent(self) -> None:
        embedded = self.payload["metadata"]
        self.assertEqual(embedded["weight_variant"], "raw")
        self.assertEqual(embedded["source_checkpoint_key"], "model_state_dict")
        self.assertEqual(embedded["epoch"], 95)
        self.assertAlmostEqual(embedded["best_miou"], 0.8759134275714613)
        self.assertEqual(embedded["tensor_count"], 369)
        self.assertEqual(embedded["trainable_parameter_count"], 13_441_205)
        self.assertEqual(embedded["state_element_count"], 13_441_277)
        self.assertEqual(embedded, {key: self.metadata[key] for key in embedded})

    def test_file_hash_matches_release_metadata(self) -> None:
        self.assertEqual(self.metadata["artifact_filename"], WEIGHTS_PATH.name)
        self.assertEqual(self.metadata["artifact_bytes"], WEIGHTS_PATH.stat().st_size)
        self.assertEqual(self.metadata["artifact_sha256"], sha256_file(WEIGHTS_PATH))
        self.assertGreater(self.metadata["raw_tensors_differing_from_ema"], 0)

    def test_tensor_hash_matches_release_metadata(self) -> None:
        self.assertEqual(
            self.metadata["state_dict_sha256"],
            state_dict_fingerprint(self.payload["model_state_dict"]),
        )


if __name__ == "__main__":
    unittest.main()
