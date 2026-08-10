# BiTrackSwin-Seg

[English](README.md) | **简体中文**

BiTrackSwin-Seg v1 是一个面向遥感影像的五波段、单流语义分割模型。模型由卷积式 Stem、带 Stage 1 BiLevel 跨窗口模块的三阶段 SwinV2 风格编码器，以及 U-Net 解码器组成。

> **v1 范围说明：**公开参考模型只接收一个形状为 `[B, 5, H, W]` 的张量。它并不是双流或多模态检查点；原实验目录中的 `multimodal` 只是历史命名。

## 模型库

| 模型 | 输入 | 参数量 | 训练检查点 | mIoU | 前景 IoU | 前景 F1/Dice |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| BiTrackSwin-Seg v1 | 五波段、单流 | 13,441,205 个可训练参数 | Epoch 95，RAW 状态字典 | 0.8759 | 0.7534 | 0.8594 |

模型库中的结果来自固定实验 `exp_20260731_200227` 的 `task1`。发布权重采用 RAW 模型状态，而不是 EMA；同一轮次的 EMA mIoU 为 0.8732。

检查点不会写入 Git 历史。请从 `v1.0.0` GitHub Release 下载 `bitrackswin_seg_v1_raw.pth`，按照 Release notes 校验 SHA256，然后将文件放入 `weights/` 目录。

## 快速开始

本地发布验证环境为 Python 3.10.19、PyTorch 2.10.0+cu130、NumPy 2.1.2，以及 NVIDIA RTX 3080 Ti（CUDA 报告显存 12.9 GB）。参考训练在第一个 epoch 后使用 BF16 AMP。可移植的深度学习环境说明见 [DL 环境配置](docs/dl_setup.md)。

```bash
conda env create -f environment.yml
conda activate bitrackswin-seg
pip install -r requirements.txt

# 无需私有数据集，仅验证模型能否根据公开配置正确构建。
python tools/train.py --config configs/bitrackswin_seg_v1.yaml --dry-run

# 使用 RAW 发布权重进行评估。
python tools/eval.py --config configs/bitrackswin_seg_v1.yaml \
  --checkpoint weights/bitrackswin_seg_v1_raw.pth

# 分割一幅五波段 GeoTIFF，并写出 PNG 掩膜。
python tools/infer.py --config configs/bitrackswin_seg_v1.yaml \
  --checkpoint weights/bitrackswin_seg_v1_raw.pth \
  --input /path/to/image.tif --output predictions/image_mask.png
```

## 数据与可复现性

本仓库不分发训练影像、标签、原始检查点、TensorBoard 事件或完整日志。训练或评估前，请按照[数据约定](docs/data.md)准备数据，重点要求如下：

- v1 使用五个波段，顺序固定为 `Red`、`Green`、`Blue`、`RedEdge`、`NIR`。
- 输入样本为 256 × 256 的 GeoTIFF 瓦片；二分类标签值为 `0`（背景）、`1`（残膜前景）和 `255`（忽略）。
- 若要与报告结果进行严格比较，必须拥有原始 `splits/v1_train.txt` 和 `splits/v1_val.txt` 的样本成员列表；当前仓库和 Release 均未包含这些清单。
- 参考实验先从 41,196 个源样本中抽取 24,717 个样本（60%），随后进行一次 80/20 划分，得到 19,773 个训练样本和 4,944 个验证样本。历史实验没有保留清单，因此目前无法独立重建完全相同的样本成员。在找回原清单，或生成并版本化新清单之前，不应声称精确复现该实验。

公开配置包含逐波段归一化统计量。这些统计量是在上述 24,717 个样本上，通过排除 NoData 的鲁棒两遍扫描估计得到的。详情见 [v1 可复现性说明](docs/reproducibility.md)。

## 模型结构

```text
五波段输入
  -> 卷积式 Stem
  -> 三阶段 SwinV2 风格编码器 + BiLevel 跨窗口交互
  -> 带 Stage 0 跳跃连接的 U-Net 解码器
  -> 二分类分割 logits
```

v1 没有辅助分支，不使用预训练 Swin 或辅助权重，全部 13.44 M 参数均从头训练。

## 历史训练配置

发布配置记录了 Epoch 95 参考模型的历史训练方案：共训练 100 个 epoch，batch size 为 2，梯度累积 4 步，随机种子为 42，使用 WSD 学习率调度、BF16 AMP、CE + Dice + focal-OHEM 损失，并监控 EMA。`best.pth` 选择 Epoch 95 的 RAW 模型，是因为它的验证集 mIoU 高于对应 EMA 模型。

当前 `tools/train.py` 是刻意保留的最小训练与安装冒烟测试入口。它可以用 AdamW 和交叉熵验证数据读取、模型构建与优化流程，但尚未实现历史训练中的全部调度、数据增强、损失函数、梯度累积和 EMA 细节。因此，不得使用该入口声称复现模型库中的指标。严格加载检查点、评估流程和推理流程已经通过发布验证；若要实现独立可复现的完整重训练，仍需补齐历史训练方案。

完整参数记录、数据假设与验证边界见 [v1 可复现性说明](docs/reproducibility.md)。

## 引用与许可证

引用本项目时，请使用 [CITATION.cff](CITATION.cff) 中的软件元数据。项目目前尚未授予开源许可证；复制、再分发或修改代码前，请先阅读 [LICENSE_PENDING.md](LICENSE_PENDING.md)。

## 致谢

本实现采用了与 Swin Transformer / Swin Transformer V2、U-Net、BiFormer 风格路由、Focal Loss 和 OHEM 相关的思想。仅在注释中引用这些工作，并不能自动证明所有代码片段均具有可再分发许可；维护者应在正式授予开源许可证前完成源码级来源与许可证审查。
