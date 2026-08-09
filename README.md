# BiTrackSwin-Seg

BiTrackSwin-Seg v1 is a five-band, single-stream semantic-segmentation model for remote-sensing imagery. It combines a convolutional stem, a three-stage SwinV2-style encoder with a stage-1 BiLevel cross-window block, and a U-Net decoder.

> **Scope of v1.** The released reference model consumes one tensor of shape `[B, 5, H, W]`. It is **not** a two-stream or multimodal checkpoint. The name `multimodal` in the original experiment directory is historical only.

## Model Zoo

| Model | Input | Parameters | Training checkpoint | mIoU | Foreground IoU | Foreground F1/Dice |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| BiTrackSwin-Seg v1 | 5-band, single stream | 13,441,205 trainable | Epoch 95, RAW state dict | 0.8759 | 0.7534 | 0.8594 |

The model-zoo result is from the fixed experiment `exp_20260731_200227`, task `task1`. It uses the RAW model state, not EMA. The corresponding EMA result was mIoU 0.8732.

The checkpoint is intentionally not stored in Git. Download `bitrackswin_seg_v1_raw.pth` from the GitHub Release for `v1.0.0`, verify its SHA256 from the release notes, and place it under `weights/`.

## Quick Start

The release was validated locally with Python 3.10.19, PyTorch 2.10.0+cu130, NumPy 2.1.2, and an NVIDIA RTX 3080 Ti (12.9 GB reported by CUDA), using BF16 AMP after the first epoch. See [DL setup](docs/dl_setup.md) for a portable Deep Learning (DL) environment.

```bash
conda env create -f environment.yml
conda activate bitrackswin-seg
pip install -r requirements.txt

# Validate model construction without requiring the private dataset.
python tools/train.py --config configs/bitrackswin_seg_v1.yaml --dry-run

# Evaluate the RAW release checkpoint.
python tools/eval.py --config configs/bitrackswin_seg_v1.yaml \
  --checkpoint weights/bitrackswin_seg_v1_raw.pth

# Segment one 5-band GeoTIFF and write a PNG mask.
python tools/infer.py --config configs/bitrackswin_seg_v1.yaml \
  --checkpoint weights/bitrackswin_seg_v1_raw.pth \
  --input /path/to/image.tif --output predictions/image_mask.png
```

## Data and Reproducibility

The training data, labels, original checkpoints, TensorBoard events, and full logs are not distributed in this repository. Before training or evaluation, prepare the layout described in [the data contract](docs/data.md). In particular:

- v1 expects five bands in this order: `Red`, `Green`, `Blue`, `RedEdge`, `NIR`.
- Input samples are 256 x 256 GeoTIFF tiles; the binary label values are `0` (background), `1` (plastic-film residue), and `255` (ignore).
- Exact comparison with the reported result would require the original `splits/v1_train.txt` and `splits/v1_val.txt` membership. Those manifests are not currently available in this repository or release.
- The reference run sub-sampled 41,196 source samples to 24,717 (60%), then used a single 80/20 split: 19,773 training and 4,944 validation samples. The historical experiment did not preserve those manifests, so its exact sample membership cannot yet be independently reconstructed. This is a known reproducibility limitation; a release must not claim exact replication until the manifests are recovered or regenerated and versioned.

The documented per-band normalization statistics are part of the configuration. They were estimated on the 24,717 selected samples with a robust two-pass procedure and NoData exclusion; see [reproducibility notes](docs/reproducibility.md).

## Architecture

```text
5-band input
  -> convolutional stem
  -> 3-stage SwinV2-style encoder + BiLevel cross-window interaction
  -> U-Net decoder with stage-0 skip
  -> 2-class segmentation logits
```

The model has no auxiliary branch in v1, uses no pretrained Swin or auxiliary weights, and trains all 13.44 M parameters from scratch.

## Training Recipe

The release configuration records the historical epoch-95 recipe: 100 epochs, batch size 2 with four-step gradient accumulation, seed 42, WSD learning-rate schedule, BF16 AMP, CE + Dice + focal-OHEM loss, and EMA monitoring. `best.pth` selects the epoch-95 RAW model because its validation mIoU exceeded its EMA counterpart.

The current `tools/train.py` is a deliberately small training and smoke-test entry point. It validates dataset loading and model optimization with AdamW and cross-entropy, but does **not** yet implement every historical scheduling, augmentation, loss, accumulation, and EMA detail. It must not be used to claim reproduction of the model-zoo metrics. Strict checkpoint loading, evaluation plumbing, and inference are the release-verified paths; a full training-recipe implementation remains required for independently reproducible retraining.

For a complete parameter record, data assumptions, and verification limits, read [the v1 reproducibility notes](docs/reproducibility.md).

## Citation and License

Please cite the software metadata in [CITATION.cff](CITATION.cff). The project has no granted open-source license yet; see [LICENSE_PENDING.md](LICENSE_PENDING.md) before copying, redistributing, or modifying the code.

## Acknowledgements

The implementation uses concepts associated with Swin Transformer/Swin Transformer V2, U-Net, BiFormer-style routing, focal loss, and OHEM. These references do not by themselves establish redistribution rights for any borrowed code. The maintainer must complete source-level provenance and license review before public release.
