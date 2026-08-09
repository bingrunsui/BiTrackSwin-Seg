from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def read_raster(path: str | Path) -> np.ndarray:
    """Read a raster as HWC. GeoTIFF support requires rasterio."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as source:
            return source.read().transpose(1, 2, 0)
    from PIL import Image
    array = np.asarray(Image.open(path))
    return array[..., None] if array.ndim == 2 else array


class FiveBandSegmentationDataset(Dataset):
    def __init__(self, root: str | Path, manifest: str | Path, image_dir="images", label_dir="labels",
                 normalization: Mapping | None = None, image_size: Sequence[int] = (256, 256),
                 ignore_index: int = 255, nodata_threshold: float = -999.0):
        self.root, self.ignore_index = Path(root), ignore_index
        self.nodata_threshold = float(nodata_threshold)
        self.image_dir, self.label_dir = self.root / image_dir, self.root / label_dir
        manifest = Path(manifest)
        if not manifest.is_absolute():
            manifest = self.root / manifest
        self.ids = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        norm = normalization or {}
        self.mean = np.asarray(norm.get("mean", [0] * 5), dtype=np.float32)
        self.std = np.asarray(norm.get("std", [1] * 5), dtype=np.float32)
        self.clip = norm.get("clip", [-10.0, 10.0])
        self.size = tuple(map(int, image_size))

    def __len__(self): return len(self.ids)
    def _path(self, directory: Path, sample: str) -> Path:
        candidate = directory / sample
        if candidate.suffix: return candidate
        for suffix in (".tif", ".tiff", ".png"):
            if (directory / f"{sample}{suffix}").exists(): return directory / f"{sample}{suffix}"
        raise FileNotFoundError(sample)
    def __getitem__(self, index):
        sample = self.ids[index]
        image = read_raster(self._path(self.image_dir, sample)).astype(np.float32)
        label = read_raster(self._path(self.label_dir, sample))[..., 0].astype(np.int64)
        if image.shape[-1] != 5: raise ValueError(f"{sample}: expected five bands, got {image.shape[-1]}")
        invalid = (~np.isfinite(image).all(axis=-1)) | (image < self.nodata_threshold).any(axis=-1)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        label[invalid] = self.ignore_index
        image = np.clip((image - self.mean) / (self.std + 1e-8), *self.clip)
        x = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        y = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()
        x = torch.nn.functional.interpolate(x, self.size, mode="bilinear", align_corners=False)[0]
        y = torch.nn.functional.interpolate(y, self.size, mode="nearest")[0, 0].long()
        return {"image": x, "label": y, "name": Path(sample).stem}
