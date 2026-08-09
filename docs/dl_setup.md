# DL environment setup

The reference training log was produced on an NVIDIA GeForce RTX 3080 Ti with 12.9 GB CUDA-visible memory. It used CUDA, BF16 automatic mixed precision after the first epoch, batch size 2, and gradient accumulation 4 (effective batch size 8). Local release validation used Python 3.10.19, PyTorch 2.10.0+cu130, and NumPy 2.1.2.

## Conda environment

```bash
conda env create -f environment.yml
conda activate bitrackswin-seg
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`environment.yml` provides a CUDA 12.1 baseline. If the DL machine uses another supported CUDA driver/runtime combination, install the matching PyTorch build from the official PyTorch selector first, then install the remaining packages from `requirements.txt`. The repository does not require or force the locally tested CUDA 13.0 build.

## Memory guidance

- Reference setting: 2 samples/GPU x 4 accumulation steps.
- If memory is insufficient, lower `train.batch_size` and increase `train.accumulation_steps` proportionally; this preserves the effective batch size but not bitwise-identical training.
- CPU-only execution is suitable only for import and smoke tests, not for the release training recipe.

## Verification

Before a public release, execute all of the following in a fresh DL environment:

```bash
python tools/train.py --config configs/bitrackswin_seg_v1.yaml --dry-run
python tools/eval.py --config configs/bitrackswin_seg_v1.yaml --checkpoint weights/bitrackswin_seg_v1_raw.pth
python tools/infer.py --config configs/bitrackswin_seg_v1.yaml --checkpoint weights/bitrackswin_seg_v1_raw.pth --input /path/to/image.tif --output predictions/smoke.png
```

The release checkpoint must load strictly and its SHA256 must match the value in the GitHub Release notes.
