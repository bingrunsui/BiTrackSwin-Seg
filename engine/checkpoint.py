from __future__ import annotations
from pathlib import Path
import torch

def load_model_checkpoint(model, checkpoint: str | Path, device="cpu", strict=True):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=strict)
    return payload
