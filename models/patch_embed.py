"""
Patch Embedding Module
======================
将图像切分成patches并投影到embedding空间

功能:
- 支持任意输入尺寸 (自动padding)
- 支持任意输入通道数
- 支持重叠patches (stride < patch_size)
- 返回padding信息 (用于分割任务还原)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from dataclasses import dataclass


@dataclass
class PatchEmbedOutput:
    """PatchEmbed的输出结构"""
    embeddings: torch.Tensor      # [B, num_patches, embed_dim]
    H: int                         # 输出高度 (patch网格)
    W: int                         # 输出宽度 (patch网格)
    pad_info: Dict[str, int]       # padding信息


class PatchEmbed(nn.Module):
    """
    通用Patch Embedding
    
    Args:
        patch_size: patch大小 (kernel_size)，默认4
        stride: 滑动步长，默认2 (stride < patch_size 产生重叠)
        in_chans: 输入通道数
        embed_dim: 输出嵌入维度
        bias: 卷积是否使用偏置
    
    Example:
        # 重叠patches: 512 → 256
        patch_embed = PatchEmbed(patch_size=4, stride=2, in_chans=3, embed_dim=96)
        
        # 非重叠patches: 512 → 128
        patch_embed = PatchEmbed(patch_size=4, stride=4, in_chans=3, embed_dim=96)
    """
    
    def __init__(
        self,
        patch_size: int = 4,
        stride: int = 2,
        in_chans: int = 3,
        embed_dim: int = 96,
        bias: bool = True
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.stride = stride
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 自动计算padding
        # 公式推导: 要使 output_size = input_size / stride
        # (input + 2*padding - kernel) / stride + 1 = input / stride
        # padding = (kernel - stride) / 2
        self.padding = (patch_size - stride) // 2
        
        # 卷积投影: patch切分 + 线性投影
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=self.padding,
            bias=bias
        )
        
        # 归一化
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> PatchEmbedOutput:
        """
        Args:
            x: [B, C, H, W] 输入图像
            
        Returns:
            PatchEmbedOutput: embeddings, H, W, pad_info
        """
        B, C, H, W = x.shape
        assert C == self.in_chans, f"通道数不匹配: 期望{self.in_chans}, 实际{C}"
        
        # 记录原始尺寸
        original_h, original_w = H, W
        
        # 计算需要的padding (使输出为整数)
        # 输出尺寸公式: (H + 2*self.padding - patch_size) / stride + 1
        # 需要 (H + 2*self.padding - patch_size) 能被 stride 整除
        H_temp = H + 2 * self.padding - self.patch_size
        W_temp = W + 2 * self.padding - self.patch_size
        
        pad_h = (self.stride - H_temp % self.stride) % self.stride
        pad_w = (self.stride - W_temp % self.stride) % self.stride
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        _, _, H_padded, W_padded = x.shape
        
        # 卷积投影
        x = self.proj(x)  # [B, embed_dim, H', W']
        H_out, W_out = x.shape[2], x.shape[3]
        
        # 展平 + 转置
        x = x.flatten(2).transpose(1, 2)  
        # flatten 在维度2上展平: [B, embed_dim, H'*W'] - [B, H'*W', embed_dim]
        # transpose : [B,C,H',W'] -> [B, H'*W', embed_dim]
        
        # [B, num_patches, embed_dim]
        
        # 归一化
        x = self.norm(x)
        
        # padding信息 (分割任务还原用)
        pad_info = {
            'pad_h': pad_h,
            'pad_w': pad_w,
            'original_h': original_h,
            'original_w': original_w,
            'padded_h': H_padded,
            'padded_w': W_padded,
        }
        
        return PatchEmbedOutput(
            embeddings=x,
            H=H_out,
            W=W_out,
            pad_info=pad_info
        )


def remove_padding(
    x: torch.Tensor, 
    pad_info: Dict[str, int],
    stride: int
) -> torch.Tensor:
    """
    移除padding，还原到原始尺寸 (分割任务用)
    
    Args:
        x: [B, C, H, W] 或 [B, H, W, C]
        pad_info: PatchEmbed返回的padding信息
        stride: PatchEmbed使用的stride
    """
    H_original = pad_info['original_h'] // stride
    W_original = pad_info['original_w'] // stride
    
    if x.dim() == 4:
        if x.shape[1] < x.shape[-1]:  # [B, H, W, C]
            return x[:, :H_original, :W_original, :]
        else:  # [B, C, H, W]
            return x[:, :, :H_original, :W_original]
    
    return x


'''
在这里面都是例子了
import torch
import torch.nn as nn
from patch_embedding_model import PatchEmbed

class MySegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Stage 1: Patch Embedding (224x224 → 112x112)
        self.patch_embed1 = PatchEmbed(
            patch_size=4, stride=2, in_chans=3, embed_dim=64
        )
        
        # Stage 2: 降采样 (112x112 → 56x56)
        self.patch_embed2 = PatchEmbed(
            patch_size=4, stride=2, in_chans=64, embed_dim=128
        )
        
        # Stage 3: 降采样 (56x56 → 28x28)
        self.patch_embed3 = PatchEmbed(
            patch_size=4, stride=2, in_chans=128, embed_dim=256
        )
        
        # 其他层...
        self.decoder = nn.Conv2d(256, 21, 1)  # 假设21类分割
        
    def forward(self, x):
        B = x.shape[0]
        
        # Stage 1
        out1 = self.patch_embed1(x)
        x1 = out1.embeddings  # [B, 112*112, 64]
        H1, W1 = out1.H, out1.W
        
        # 转回2D特征图
        x1 = x1.transpose(1, 2).reshape(B, 64, H1, W1)
        
        # Stage 2
        out2 = self.patch_embed2(x1)
        x2 = out2.embeddings.transpose(1, 2).reshape(B, 128, out2.H, out2.W)
        
        # Stage 3
        out3 = self.patch_embed3(x2)
        x3 = out3.embeddings.transpose(1, 2).reshape(B, 256, out3.H, out3.W)
        
        # 解码
        output = self.decoder(x3)
        
        # 上采样到原始尺寸
        output = nn.functional.interpolate(
            output, size=(x.shape[2], x.shape[3]), 
            mode='bilinear', align_corners=False
        )
        
        return output

# 测试
model = MySegmentationModel()
x = torch.randn(2, 3, 224, 224)
output = model(x)
print(f"输出形状: {output.shape}")  # [2, 21, 224, 224]

'''

'''
这个是不同的配置示例
# 1. 非重叠patches (传统ViT风格)
patch_embed = PatchEmbed(patch_size=16, stride=16, in_chans=3, embed_dim=768)
# 输入: [B, 3, 224, 224]
# 输出: [B, 196, 768]  (14x14 patches)

# 2. 重叠patches (Swin风格)
patch_embed = PatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=96)
# 输入: [B, 3, 224, 224]
# 输出: [B, 3136, 96]  (56x56 patches)

# 3. 作为降采样层
downsample = PatchEmbed(patch_size=2, stride=2, in_chans=64, embed_dim=128)
# 输入: [B, 64, 56, 56]
# 输出: [B, 784, 128]  (28x28 patches)
'''