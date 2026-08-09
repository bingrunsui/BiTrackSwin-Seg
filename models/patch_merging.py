"""
Patch Merging Module
====================
空间下采样模块，参考Swin Transformer

功能:
- 将相邻的 2×2 patches 合并为 1 个
- 空间尺寸减半，通道数加倍
- 支持 4D [B, H, W, C] 和 3D [B, N, C] 两种输入格式
- 自动处理 H 或 W 为奇数的情况 (padding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PatchMerging(nn.Module):
    """
    Patch Merging Layer
    
    将相邻的 2×2 patches 合并，实现空间下采样
    
    操作流程:
    1. 取 2×2 邻域的 4 个位置
    2. Concat: [B, H/2, W/2, 4C]
    3. Linear: 4C → 2C
    4. LayerNorm
    
    Args:
        dim: 输入特征维度
        norm_layer: 归一化层
    
    Example:
        merge = PatchMerging(dim=96)
        
        # 4D输入
        x = torch.randn(2, 64, 64, 96)
        out, H_out, W_out = merge(x)  # [2, 32, 32, 192]
        
        # 3D输入
        x = torch.randn(2, 64*64, 96)
        out, H_out, W_out = merge(x, H=64, W=64)  # [2, 32*32, 192]
    """
    
    def __init__(
        self,
        dim: int,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.dim = dim
        
        # 线性层: 4C → 2C
        # 这个和swintransformer做法是一模一样的
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        
        # 归一化
        self.norm = norm_layer(4 * dim) if norm_layer else nn.LayerNorm(4 * dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        H: Optional[int] = None, 
        W: Optional[int] = None
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: 输入特征
               - 4D: [B, H, W, C]
               - 3D: [B, H*W, C] (需要提供 H, W)
            H, W: 空间尺寸 (仅 3D 输入时需要)
            
        Returns:
            x: 输出特征 (与输入格式相同)
            H_out: 输出高度
            W_out: 输出宽度
        """
        # 判断输入格式
        is_3d_input = (x.dim() == 3)
        
        if is_3d_input:
            # 3D输入: [B, N, C] → [B, H, W, C]
            assert H is not None and W is not None, \
                "3D输入时必须提供 H 和 W 参数"
            B, N, C = x.shape
            assert N == H * W, \
                f"序列长度({N})与空间尺寸({H}×{W}={H*W})不匹配"
            x = x.view(B, H, W, C)
        else:
            # 4D输入: [B, H, W, C]
            B, H, W, C = x.shape
        
        # 验证维度
        assert C == self.dim, \
            f"输入维度({C})与期望维度({self.dim})不匹配"
        
        # 处理奇数尺寸: padding
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            # [B, H, W, C] → padding在H和W方向
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
        
        # 取 2×2 邻域的 4 个位置
        # x0: 左上, x1: 右上, x2: 左下, x3: 右下
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        
        # Concat: [B, H/2, W/2, 4C]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        
        # 计算输出尺寸
        H_out = H // 2
        W_out = W // 2
        
        # LayerNorm
        x = self.norm(x)
        
        # Linear: 4C → 2C
        x = self.reduction(x)  # [B, H/2, W/2, 2C]
        
        # 如果输入是3D，转回3D
        if is_3d_input:
            x = x.view(B, H_out * W_out, -1)  # [B, N', 2C]
        
        return x, H_out, W_out


class PatchMergingV2(nn.Module):
    """
    Patch Merging V2 (使用卷积实现)
    
    与 PatchMerging 功能相同，但使用卷积实现
    可能在某些情况下更高效
    
    Args:
        dim: 输入特征维度
        norm_layer: 归一化层
    """
    
    def __init__(
        self,
        dim: int,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        self.dim = dim
        
        # 使用卷积实现 patch merging
        # kernel=2, stride=2 实现 2×2 合并
        self.reduction = nn.Conv2d(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=2,
            stride=2,
            bias=False
        )
        
        # 归一化 (在通道维度)
        self.norm = norm_layer(2 * dim) if norm_layer else nn.LayerNorm(2 * dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        H: Optional[int] = None, 
        W: Optional[int] = None
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: 输入特征
               - 4D: [B, H, W, C]
               - 3D: [B, H*W, C]
            H, W: 空间尺寸 (仅 3D 输入时需要)
            
        Returns:
            x: 输出特征
            H_out, W_out: 输出空间尺寸
        """
        is_3d_input = (x.dim() == 3)
        
        if is_3d_input:
            assert H is not None and W is not None
            B, N, C = x.shape
            x = x.view(B, H, W, C)
        else:
            B, H, W, C = x.shape
        
        # 处理奇数尺寸
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H = H + pad_h
            W = W + pad_w
        
        # [B, H, W, C] → [B, C, H, W] (Conv2d需要)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # 卷积下采样
        x = self.reduction(x)  # [B, 2C, H/2, W/2]
        
        H_out = H // 2
        W_out = W // 2
        
        # [B, 2C, H/2, W/2] → [B, H/2, W/2, 2C]
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # LayerNorm
        x = self.norm(x)
        
        # 如果输入是3D，转回3D
        if is_3d_input:
            x = x.view(B, H_out * W_out, -1)
        
        return x, H_out, W_out