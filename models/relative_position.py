"""
Relative Position Encoding Module (2D分解式)
============================================
用于Global Attention的可学习2D相对位置编码

设计思路:
- 将2D相对位置分解为H方向和W方向
- 参数量大幅减少 (从 O(H²W²) 降到 O(H+W))
- 天然支持任意输入尺寸

参考论文:
- Axial Attention (ECCV 2020)
- CSwin Transformer (CVPR 2022)
"""

import torch
import torch.nn as nn
from typing import Tuple


class RelativePosition2D(nn.Module):
    """
    分解式2D相对位置编码
    
    将完整的2D相对位置偏置分解为H方向和W方向的独立编码，
    最终偏置 = H方向偏置 + W方向偏置
    
    Args:
        num_heads: 注意力头数
        max_relative_position: 最大相对位置距离，默认64
    
    Example:
        rel_pos = RelativePosition2D(num_heads=8, max_relative_position=64)
        bias = rel_pos(H=16, W=16)  # [8, 256, 256]
    """
    
    def __init__(
        self,
        num_heads: int,
        max_relative_position: int = 64
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.max_relative_position = max_relative_position
        
        # 相对位置范围: [-(max-1), max-1]，共 2*max-1 个位置
        num_positions = 2 * max_relative_position - 1
        
        # H方向相对位置偏置表: [2*max-1, num_heads]
        self.relative_position_h = nn.Parameter(
            torch.zeros(num_positions, num_heads)
        )
        
        # W方向相对位置偏置表: [2*max-1, num_heads]
        self.relative_position_w = nn.Parameter(
            torch.zeros(num_positions, num_heads)
        )
        
        # 初始化
        nn.init.trunc_normal_(self.relative_position_h, std=0.02)
        nn.init.trunc_normal_(self.relative_position_w, std=0.02)
    
    def forward(self, H: int, W: int) -> torch.Tensor:
        """
        生成相对位置偏置矩阵
        
        Args:
            H: 特征图高度
            W: 特征图宽度
            
        Returns:
            relative_position_bias: [num_heads, H*W, H*W]
        """
        device = self.relative_position_h.device
        
        # 生成坐标
        coords_h = torch.arange(H, device=device)  # [H]
        coords_w = torch.arange(W, device=device)  # [W]
        
        # 计算H方向相对位置: [H, H]
        # relative_h[i,j] = i - j，范围 [-(H-1), H-1]
        relative_h = coords_h.unsqueeze(1) - coords_h.unsqueeze(0)
        
        # 计算W方向相对位置: [W, W]
        relative_w = coords_w.unsqueeze(1) - coords_w.unsqueeze(0)
        
        # 偏移到非负索引: [0, 2*max-2]
        relative_h = relative_h + self.max_relative_position - 1
        relative_w = relative_w + self.max_relative_position - 1
        
        # 限制在有效范围内 (处理超出max_relative_position的情况)
        relative_h = relative_h.clamp(0, 2 * self.max_relative_position - 2)
        relative_w = relative_w.clamp(0, 2 * self.max_relative_position - 2)
        
        # 查表获取偏置值
        # H方向偏置: [H, H, num_heads]
        bias_h = self.relative_position_h[relative_h]
        
        # W方向偏置: [W, W, num_heads]
        bias_w = self.relative_position_w[relative_w]
        
        # 扩展到2D网格
        # bias_h: [H, H, num_heads] -> [H, W, H, W, num_heads]
        # 对于位置(i1,j1)和(i2,j2)，H方向偏置只取决于i1-i2
        bias_h = bias_h.unsqueeze(1).unsqueeze(3)  # [H, 1, H, 1, num_heads]
        bias_h = bias_h.expand(H, W, H, W, self.num_heads)  # [H, W, H, W, num_heads]
        
        # bias_w: [W, W, num_heads] -> [H, W, H, W, num_heads]
        # 对于位置(i1,j1)和(i2,j2)，W方向偏置只取决于j1-j2
        bias_w = bias_w.unsqueeze(0).unsqueeze(2)  # [1, W, 1, W, num_heads]
        bias_w = bias_w.expand(H, W, H, W, self.num_heads)  # [H, W, H, W, num_heads]
        
        # 合并: H偏置 + W偏置
        bias = bias_h + bias_w  # [H, W, H, W, num_heads]
        
        # 重塑为attention需要的形状
        # [H, W, H, W, num_heads] -> [H*W, H*W, num_heads] -> [num_heads, H*W, H*W]
        bias = bias.reshape(H * W, H * W, self.num_heads)
        bias = bias.permute(2, 0, 1).contiguous()  # [num_heads, H*W, H*W]
        
        return bias


class RelativePositionBiasAdder(nn.Module):
    """
    便捷封装: 在Attention计算中添加相对位置偏置
    
    用法:
        self.rel_pos = RelativePositionBiasAdder(num_heads=8)
        
        # 在attention中
        attn = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        attn = self.rel_pos(attn, H, W)  # 添加偏置
        attn = attn.softmax(dim=-1)
    """
    
    def __init__(
        self,
        num_heads: int,
        max_relative_position: int = 64
    ):
        super().__init__()
        self.rel_pos_2d = RelativePosition2D(num_heads, max_relative_position)
    
    def forward(
        self, 
        attn: torch.Tensor, 
        H: int, 
        W: int
    ) -> torch.Tensor:
        """
        Args:
            attn: 注意力分数 [B, num_heads, H*W, H*W]
            H, W: 特征图尺寸
            
        Returns:
            attn + bias: [B, num_heads, H*W, H*W]
        """
        bias = self.rel_pos_2d(H, W)  # [num_heads, H*W, H*W]
        return attn + bias.unsqueeze(0)  # 广播到batch维度
    
    
    '''
    使用示例：
    # 在 Global Attention 中
    self.rel_pos = RelativePosition2D(num_heads=8, max_relative_position=64)

    # forward时
    bias = self.rel_pos(H=16, W=16)  # [8, 256, 256]
    attn = q @ k.transpose(-2, -1) + bias
    '''