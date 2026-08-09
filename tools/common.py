from __future__ import annotations
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def read_config(path):
    with open(path, encoding="utf-8") as handle: return yaml.safe_load(handle)

def model_from_config(config):
    from models import build_model
    spec = config["model"]
    return build_model(image_size=int(spec["image_size"][0]), in_channels=int(spec["input_channels"]),
                       num_classes=int(spec["num_classes"]))
