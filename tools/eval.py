from __future__ import annotations
import argparse, torch
from torch.utils.data import DataLoader
from common import read_config, model_from_config
from datasets import FiveBandSegmentationDataset
from engine import load_model_checkpoint, segmentation_metrics

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); cfg = read_config(args.config); device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model_from_config(cfg).to(device).eval(); load_model_checkpoint(model, args.checkpoint, device=device, strict=True)
    data = cfg["data"]; ds = FiveBandSegmentationDataset(data["root"], data["val_manifest"], data["image_dir"], data["label_dir"], data["normalization"], cfg["model"]["image_size"], cfg["model"].get("ignore_index", 255), data.get("nodata_threshold", -999.0))
    sums = {}; batches = 0
    for batch in DataLoader(ds, batch_size=cfg["train"]["batch_size"], num_workers=cfg["train"].get("num_workers", 0)):
        with torch.inference_mode(): metrics = segmentation_metrics(model(batch["image"].to(device)), batch["label"].to(device), cfg["model"].get("ignore_index", 255))
        for key, value in metrics.items(): sums[key] = sums.get(key, 0.) + value
        batches += 1
    print({key: value / max(batches, 1) for key, value in sums.items()})
if __name__ == "__main__": main()
