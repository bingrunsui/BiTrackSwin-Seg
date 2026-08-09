"""Export a minimal, reproducible release checkpoint from a trusted training checkpoint.

The source checkpoint may contain optimizer, scheduler, EMA, scaler, and RNG state.
This script intentionally exports only the RAW ``model_state_dict`` selected by the
training run, together with small provenance metadata. It never uses
``ema_state_dict`` as the released model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch


REQUIRED_SOURCE_KEYS = {"epoch", "best_miou", "model_state_dict", "ema_state_dict", "config"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, shapes, dtypes, and raw CPU values deterministically."""
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"State-dict entry {name!r} is not a tensor.")
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Trusted full training checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Weights-only .pth output path.")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON sidecar output path.")
    parser.add_argument("--release-version", default="v1.0.0")
    parser.add_argument(
        "--trainable-parameter-count",
        type=int,
        required=True,
        help="Count obtained from the strict-load model implementation (buffers excluded).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Source checkpoint does not exist: {args.source}")

    # The source is a locally produced and trusted checkpoint. ``weights_only``
    # cannot load the full training payload on every supported PyTorch version.
    source: dict[str, Any] = torch.load(args.source, map_location="cpu", weights_only=False)
    missing = REQUIRED_SOURCE_KEYS.difference(source)
    if missing:
        raise KeyError(f"Source checkpoint is missing required keys: {sorted(missing)}")

    raw_state_dict = source["model_state_dict"]
    ema_state_dict = source["ema_state_dict"]
    if not isinstance(raw_state_dict, Mapping) or not raw_state_dict:
        raise TypeError("Source model_state_dict must be a non-empty mapping.")
    if not isinstance(ema_state_dict, Mapping):
        raise TypeError("Source ema_state_dict must be a mapping.")
    if set(raw_state_dict) != set(ema_state_dict):
        raise ValueError("RAW and EMA state dictionaries do not have the same keys.")

    raw_tensors = dict(raw_state_dict)
    tensor_count = len(raw_tensors)
    state_element_count = sum(value.numel() for value in raw_tensors.values() if isinstance(value, torch.Tensor))
    if tensor_count != sum(isinstance(value, torch.Tensor) for value in raw_tensors.values()):
        raise TypeError("Every RAW state-dict entry must be a tensor.")

    raw_fingerprint = state_dict_fingerprint(raw_tensors)
    differing_from_ema = sum(
        not torch.equal(raw_tensors[name].detach().cpu(), ema_state_dict[name].detach().cpu())
        for name in raw_tensors
    )
    if differing_from_ema == 0:
        raise RuntimeError("RAW and EMA weights are identical; cannot prove RAW selection.")

    embedded_metadata = {
        "format": "bitrackswin-seg-weights-only",
        "format_version": 1,
        "release_version": args.release_version,
        "source_checkpoint_key": "model_state_dict",
        "weight_variant": "raw",
        "epoch": int(source["epoch"]),
        "best_miou": float(source["best_miou"]),
        "architecture_config": source["config"],
        "tensor_count": tensor_count,
        "trainable_parameter_count": args.trainable_parameter_count,
        "state_element_count": state_element_count,
        "state_dict_sha256": raw_fingerprint,
    }
    release_payload = {"model_state_dict": raw_tensors, "metadata": embedded_metadata}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    torch.save(release_payload, args.output)

    # Reload using the safe weights-only mode to ensure the public artifact has
    # no training-state dependency and retains the selected RAW tensors exactly.
    exported = torch.load(args.output, map_location="cpu", weights_only=True)
    if set(exported) != {"model_state_dict", "metadata"}:
        raise RuntimeError("Release artifact contains unexpected top-level keys.")
    if state_dict_fingerprint(exported["model_state_dict"]) != raw_fingerprint:
        raise RuntimeError("Reloaded release artifact does not match RAW source weights.")

    sidecar = {
        **embedded_metadata,
        "artifact_filename": args.output.name,
        "artifact_bytes": args.output.stat().st_size,
        "artifact_sha256": sha256_file(args.output),
        "source_checkpoint": args.source.name,
        "source_checkpoint_bytes": args.source.stat().st_size,
        "source_checkpoint_contains": sorted(source.keys()),
        "ema_tensor_count": len(ema_state_dict),
        "raw_tensors_differing_from_ema": differing_from_ema,
        "verification": "Reloaded with torch.load(weights_only=True) and matched RAW state-dict SHA256.",
    }
    args.metadata.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(sidecar, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
