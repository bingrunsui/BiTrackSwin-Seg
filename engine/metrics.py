import torch

@torch.no_grad()
def segmentation_metrics(logits, target, ignore_index=255):
    prediction = logits.argmax(1); valid = target != ignore_index
    values = {}
    ious = []
    for cls in range(logits.shape[1]):
        tp = ((prediction == cls) & (target == cls) & valid).sum().float()
        fp = ((prediction == cls) & (target != cls) & valid).sum().float()
        fn = ((prediction != cls) & (target == cls) & valid).sum().float()
        iou = tp / (tp + fp + fn).clamp_min(1); values[f"iou_{cls}"] = iou.item(); ious.append(iou)
    values["miou"] = torch.stack(ious).mean().item(); return values
