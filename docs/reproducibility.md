# v1 reproducibility record

## Reference checkpoint

- Source experiment: `outputs_simple/exp_20260731_200227/task1`
- Selected state: RAW model state from `best.pth`; not EMA
- Selection epoch: 95
- Trainable parameters: 13,441,205
- State-dict elements: 13,441,277 (the additional 72 elements are buffers)
- Hardware: NVIDIA GeForce RTX 3080 Ti (12.9 GB visible memory)
- Input: five-band, single-stream tensor; no auxiliary input
- Strict-load validation: passed locally with Python 3.10.19, PyTorch 2.10.0+cu130, and NumPy 2.1.2
- Checkpoint topology: encoder depths `[4, 6, 2]`; the BiLevel v24 cross-window block is present in stage 1 only

## Reported validation metrics

| State | mIoU | Background IoU | Foreground IoU | Foreground precision | Foreground recall | Foreground F1/Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW, epoch 95 | 0.8759 | 0.9984 | 0.7534 | 0.8444 | 0.8749 | 0.8594 |
| EMA, epoch 95 | 0.8732 | 0.9983 | 0.7480 | 0.8298 | 0.8836 | 0.8558 |

The unrounded RAW mIoU recorded by the experiment is 0.875913; foreground IoU is 0.753420.

## Normalization provenance

The reference run scanned 24,717 selected five-band tiles and excluded 714,030 NoData pixels. It used a two-pass, 3-sigma robust global estimate and applied `(x - mean) / std`, clipped to `[-10, 10]`. The values are recorded verbatim in `configs/bitrackswin_seg_v1.yaml`.

## What is reproducible now

With the release checkpoint, the configuration, and data that obey the documented contract, users should be able to reproduce model construction, strict checkpoint loading, and inference preprocessing.

The included training entry point is a minimal smoke-test/optimization path. It does not yet implement the complete historical WSD, gradient-accumulation, CE + Dice + focal-OHEM, augmentation, and EMA recipe. The YAML retains those values as provenance, rather than claiming that the current training command recreates the reported run.

## What is not yet independently reproducible

The original run did not retain its train/validation manifest files, dataset checksum, source-data version, or environment lockfile. Its complete training recipe is also not yet implemented in the public training entry point. The numerical validation row therefore remains an experiment record, not an independently reproducible benchmark until those artifacts are recovered/released and the recipe is implemented. New manifests or a new data version must be named separately and their results must not be represented as a bitwise or sample-identical reproduction of v1.

## Release checklist

1. Export the RAW `model_state_dict` from `best.pth`; do not upload the full optimizer checkpoint to Git.
2. Attach the state dict, its SHA256, file size, and this configuration to a GitHub Release tagged `v1.0.0`.
3. Add recovered split manifests and checksums, or explicitly publish a new split version.
4. Re-run strict-load evaluation in a clean DL environment and record the command output.
5. Complete the code/data/provenance review before adding an open-source license.
