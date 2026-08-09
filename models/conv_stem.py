"""
Conv Stem + Peak-Preserving Downsample  (v24)
=============================================
替代 RGBStemStage0 的 W-MSA stem 与 PatchMerging.

【为什么 stage0 换卷积】
  · zero-stage0 ΔfgIoU = -0.4890 (占 fgIoU 96%): stage0 干的是局部高频细丝检测.
  · window=8 的 W-MSA 感受野 8x8, 与两层 3x3 卷积(5x5)相当, 却要物化
    1024 个窗口的 64x64 注意力矩阵 —— 用最贵的算子做最不需要它的事.
  · stem.proj.weight 是每个 epoch 的梯度榜首:
      7.14(shift 基线, 无 BiLevel) 中位 3.5~4.2, 全程裁剪率 0~2%
      0720(BiLevel)               中位 19 -> 4449, 全程裁剪率 33.6%
    Xiao et al., "Early Convolutions Help Transformers See Better", NeurIPS 2021
    正是针对这个现象: 卷积 stem 显著改善优化稳定性与对 LR 的敏感度.

【为什么 PatchMerging 要改】
  ① 输出侧无归一化: concat -> LN -> Linear -> (什么都没有).
     全图唯一一个输出尺度只由权重范数决定、无任何约束的模块.
  ② 更要紧的是 4C -> 2C 这个 2 倍压缩.
     PatchMerging 的 concat 本身是 space-to-depth, 信息无损;
     丢失发生在紧接着的 Linear(4C->2C).
     对 1~2 px 细丝: 一个 1px 细丝在 2x2 块里占 4 个位置之一,
     若该 Linear 接近平均则幅度掉 4x; 两次 merge = 16x, 三次 = 64x.
     这正是 zero-stage2 = -0.0001 / zero-stage3 = -0.0000 的物理来源,
     而"丢掉哪一半"完全由一个无约束的 Linear 决定.

【本模块的修法】
  · space_to_depth(2x2) -> 4C                (无损, 同 Swin 的 concat)
  · 并联 max-pool(2x2) -> C                  (★ 0 参数, 显式保留"块内最强响应")
  · concat -> 5C -> 1x1 conv -> out_dim
  · -> Norm                                   (★ 输出侧收口)

  max 分支是针对细丝的直接补丁: 平均会把 1/4 占空比的峰值稀释掉,
  max 不会. 代价是 0 参数 + 一个 [B,C,H/2,W/2] 中间量.

参考文献:
  Xiao et al., Early Convolutions Help Transformers See Better, NeurIPS 2021
  Dai et al., CoAtNet, NeurIPS 2021                    (C-C-T-T 设计原则)
  Liu et al., ConvNeXt, CVPR 2022                      (卷积块现代配方)
  Sunkara & Luo, SPD-Conv, ECML-PKDD 2022              (space-to-depth 无损下采样)
  Shi et al., ESPCN / PixelShuffle, CVPR 2016          (逆操作)
  Liu et al., Swin V2, CVPR 2022 §3.1                  (归一化放输出侧, 幅度有界)
  Zhang & Sennrich, RMSNorm, NeurIPS 2019
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LayerNorm2d",
    "ConvStemBlock",
    "ConvStem",
    "PeakPreservingDownsample",
]


class LayerNorm2d(nn.Module):
    """NCHW 上沿通道维的 LayerNorm (ConvNeXt 约定)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvStemBlock(nn.Module):
    """ConvNeXt 风格块: DWConv7x7 -> LN -> PW(expand) -> GELU -> PW -> LayerScale -> residual.

    ★ 256^2 上 expand_ratio 默认取 2 而不是 4:
      [B=2, 48*4, 256, 256] bf16 = 50 MB/份, 是 stem 的显存大头.
      取 2 省一半, 而 stem 的任务(局部高频)不需要很宽的 FFN.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expand_ratio: int = 2,
        ls_init: float = 1e-6,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(ls_init)))
        self.drop_path_p = float(drop_path)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_path_p
        mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep)
        return x * mask / keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.pw2(self.act(self.pw1(h)))
        return x + self._drop_path(self.gamma * h)


class ConvStem(nn.Module):
    """stage0: in_chans -> dim, 全程保持输入分辨率 (stride 1).

    返回 [B, H*W, C] 的 token 序列, 与原 RGBStemStage0 的 premerge 契约一致
    (供 UNetDecoder 的 256^2 级 skip 使用).
    """

    def __init__(
        self,
        in_chans: int = 5,
        dim: int = 48,
        depth: int = 4,
        kernel_size: int = 7,
        expand_ratio: int = 2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=3, stride=1, padding=1)
        self.proj_norm = LayerNorm2d(dim)
        dpr = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                ConvStemBlock(
                    dim, kernel_size=kernel_size, expand_ratio=expand_ratio, drop_path=dpr[i]
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, in_chans, H, W] -> (feat_nchw [B,C,H,W], tokens [B,H*W,C])"""
        x = self.proj_norm(self.proj(x))
        for blk in self.blocks:
            x = blk(x)
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return x, tokens


class PeakPreservingDownsample(nn.Module):
    """2x 下采样, 替代 PatchMerging.

        x_spd = space_to_depth_2x2(x)      # [B, 4C, H/2, W/2]   无损
        x_max = maxpool_2x2(x)             # [B,  C, H/2, W/2]   保峰值, 0 参数
        y     = conv1x1( cat([x_spd, x_max]) )                   # 5C -> out_dim
        y     = norm(y)                    # ★ 输出侧收口

    对照原 PatchMerging: concat(4C) -> LN -> Linear(4C->2C) -> 无归一化.
      本模块把归一化挪到输出侧(Swin V2 res-post-norm 的同一原则), 并加 max 分支.

    compress: out_dim 默认 2*C (与原 PatchMerging 同宽, 保证下游通道数不变).
      若要做"第一次下采样不压缩"的消融, 传 out_dim=4*C 即可 —— 但注意
      那会让 stage1/2/3 全部加宽, 参数约 4x, 需同时下调 depths.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int = -1,
        use_max_branch: bool = True,
        norm_layer: str = "layernorm",
    ):
        super().__init__()
        self.dim = dim
        self.out_dim = int(out_dim) if out_dim > 0 else dim * 2
        self.use_max_branch = bool(use_max_branch)
        in_ch = dim * 4 + (dim if self.use_max_branch else 0)
        self.reduction = nn.Conv2d(in_ch, self.out_dim, kernel_size=1, bias=False)
        if norm_layer == "layernorm":
            self.norm = LayerNorm2d(self.out_dim)
        elif norm_layer == "groupnorm":
            self.norm = nn.GroupNorm(min(32, self.out_dim), self.out_dim)
        else:
            raise ValueError(f"未知 norm_layer: {norm_layer}")

    @staticmethod
    def _space_to_depth(x: torch.Tensor) -> torch.Tensor:
        """[B,C,H,W] -> [B,4C,H/2,W/2], 与 Swin PatchMerging 的 concat 等价(无损)."""
        return torch.cat(
            [x[..., 0::2, 0::2], x[..., 1::2, 0::2], x[..., 0::2, 1::2], x[..., 1::2, 1::2]],
            dim=1,
        )

    def forward_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, out_dim, H/2, W/2]  (纯空间形态)"""
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        parts = [self._space_to_depth(x)]
        if self.use_max_branch:
            parts.append(F.max_pool2d(x, kernel_size=2, stride=2))
        y = self.reduction(torch.cat(parts, dim=1))
        return self.norm(y)

    def forward(self, x: torch.Tensor, H=None, W=None):
        """★ [2026-07-31] 与 patch_merging.PatchMerging.forward 逐项对齐的 token 契约.

        输入 3D [B, H*W, C] (需给 H, W) 或 4D [B, H, W, C];
        输出 (x, H_out, W_out), 格式与输入相同.
        这样 multimodal.py 的三个调用点只需改类名, 上下游一行不动
        (裁决: "先保持整个特征的稳定性").
        """
        is_3d = (x.dim() == 3)
        if is_3d:
            assert H is not None and W is not None, "3D 输入必须提供 H, W"
            B, N, C = x.shape
            assert N == H * W, f"N({N}) != H*W({H*W})"
            x = x.view(B, H, W, C)
        else:
            B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()          # [B,C,H,W]
        y = self.forward_nchw(x)                        # [B,out,H/2,W/2]
        Ho, Wo = y.shape[-2], y.shape[-1]
        y = y.permute(0, 2, 3, 1).contiguous()          # [B,Ho,Wo,out]
        if is_3d:
            y = y.view(B, Ho * Wo, self.out_dim)
        return y, Ho, Wo
