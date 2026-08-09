"""
Global Attention Module
=======================
全局自注意力机制，用于Stage 3-4

特点:
- 接收PatchMerging后的token序列
- 全局范围内计算patch与patch之间的注意力
- 使用分解式2D相对位置编码 (来自relative_position.py)

修改记录:
  [2026-04-04] V14 升级:
    1. FlashAttention: 使用 PyTorch 2.0+ F.scaled_dot_product_attention
       自动选择 FlashAttention / Memory-Efficient 后端 (Dao et al., NeurIPS 2022)
    2. 2D RoPE: 旋转位置编码替代 RelativePosition2D
       (RoFormer — Su et al. 2021, EVA-02 — Fang et al. 2023, Hiera — Ryali et al. ICML 2023)
       X 轴编码 head_dim 前半部分, Y 轴编码后半部分, 天然支持平移不变性和分辨率外推
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .relative_position import RelativePosition2D


# =============================================================================
# ★ [2026-04-04] 2D Rotary Position Encoding (2D RoPE)
# =============================================================================

class RotaryPositionEncoding2D(nn.Module):
    """
    ★ [2026-04-04] 2D 旋转位置编码 (2D RoPE)
    
    将 1D RoPE 扩展到 2D: head_dim 前半部分编码 Y 坐标, 后半部分编码 X 坐标.
    
    优点:
      - 只关注相对距离和方向, 天然具有平移不变性
      - 支持分辨率外推 (训练 16×16, 推理 32×32 无需插值)
      - 遥感影像无固定上下左右方向, RoPE 比绝对位置编码更合适
    
    参考:
      - RoFormer (Su et al., 2021): 提出 RoPE
      - EVA-02 (Fang et al., 2023): 在 ViT 中使用 2D RoPE
      - Hiera (Ryali et al., ICML 2023): 层级视觉 Transformer 中使用 RoPE
    
    Args:
        dim: head_dim (每个注意力头的维度)
        theta: RoPE 基础频率 (默认 10000.0)
    """
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        
        # 每个坐标轴使用 dim//2 的一半: dim//4 对频率
        half_dim = dim // 4  # 每个轴的频率数
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer('freqs', freqs)
        
        # 缓存
        self._cache_key = None
        self._cache_cos = None
        self._cache_sin = None
    
    def _build_cache(self, H: int, W: int, device: torch.device):
        """预计算 cos/sin 缓存"""
        cache_key = (H, W, device)
        if self._cache_key == cache_key:
            return self._cache_cos, self._cache_sin
        
        freqs = self.freqs.to(device)
        
        # Y/X 坐标
        pos_y = torch.arange(H, device=device, dtype=torch.float32)
        pos_x = torch.arange(W, device=device, dtype=torch.float32)
        
        freqs_y = torch.outer(pos_y, freqs)  # [H, dim//4]
        freqs_x = torch.outer(pos_x, freqs)  # [W, dim//4]
        
        # 扩展到 2D 网格
        freqs_y = freqs_y.unsqueeze(1).expand(-1, W, -1)  # [H, W, dim//4]
        freqs_x = freqs_x.unsqueeze(0).expand(H, -1, -1)  # [H, W, dim//4]
        
        # 拼接: [H, W, dim//2] (y 前半, x 后半)
        freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1)  # [H, W, dim//2]
        freqs_2d = freqs_2d.reshape(H * W, -1)             # [N, dim//2]
        
        # cos/sin: [1, 1, N, dim//2]
        cos_cache = freqs_2d.cos().unsqueeze(0).unsqueeze(0)
        sin_cache = freqs_2d.sin().unsqueeze(0).unsqueeze(0)
        
        self._cache_key = cache_key
        self._cache_cos = cos_cache
        self._cache_sin = sin_cache
        
        return cos_cache, sin_cache
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, H: int, W: int):
        """
        对 q, k 应用 2D RoPE
        
        Args:
            q: [B, heads, N, head_dim]
            k: [B, heads, N, head_dim]
            H, W: 空间尺寸 (N = H * W)
            
        Returns:
            q_rotated, k_rotated: 同形状
        """
        cos, sin = self._build_cache(H, W, q.device)
        
        # RoPE 旋转: head_dim 分成两半
        # q = [q1, q2], 旋转后 = [q1*cos - q2*sin, q1*sin + q2*cos]
        d = q.shape[-1]
        half_d = d // 2
        
        q1, q2 = q[..., :half_d], q[..., half_d:]
        k1, k2 = k[..., :half_d], k[..., half_d:]
        
        q_rotated = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
        
        return q_rotated, k_rotated


class GlobalAttention(nn.Module):
    """
    全局多头自注意力
    
    与WindowAttention的区别:
    - WindowAttention: 只在窗口内计算注意力
    - GlobalAttention: 在整个特征图上计算注意力
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数
        qkv_bias: QKV投影是否使用偏置
        attn_drop: 注意力dropout
        proj_drop: 输出投影dropout
        max_relative_position: 相对位置编码的最大范围
        use_flash_attn: ★ [2026-04-04] 是否使用 FlashAttention (PyTorch 2.0+)
        use_rope: ★ [2026-04-04] 是否使用 2D RoPE 替代 RelativePosition2D
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        max_relative_position: int = 64,
        # ★ [2026-04-04] V14 新增
        use_flash_attn: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn
        self.use_rope = use_rope
        
        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # ★ [2026-04-04] 位置编码: 2D RoPE 或原始 RelativePosition2D
        if use_rope:
            self.rope = RotaryPositionEncoding2D(dim=self.head_dim)
            self.relative_position = None
        else:
            self.rope = None
            # 2D分解式相对位置编码 (原始方案)
            self.relative_position = RelativePosition2D(
                num_heads=num_heads,
                max_relative_position=max_relative_position
            )
        
        # Softmax (非 FlashAttn 时使用)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: token序列 [B, N, C]，N 可以是 H*W 或者 MAE 中的 N_visible
            H: 特征图高度 (用于相对位置编码)
            W: 特征图宽度 (用于相对位置编码)
            
        Returns:
            out: [B, N, C]
            
        Note:
            当 N != H*W 时 (如 MAE 随机遮罩后)，将跳过相对位置偏置
        """
        B, N, C = x.shape
        
        # QKV投影: [B, N, 3C] -> [B, N, 3, num_heads, head_dim]
        # 这里我的N（num_patches）与Vit不一样，因为这个要多一个cls_token，而我直接是做分割任务不需要这个东西
        # reshape:-> [batch_size, num_patches, 3, num_heads, embed_dim_per_head]
        # permute:-> [3, batch_size, num_heads, num_patches, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # ★ [2026-04-04] 2D RoPE: 对 q, k 施加旋转位置编码
        if self.use_rope and self.rope is not None and N == H * W:
            q, k = self.rope(q, k, H, W)
        
        # ★ [2026-04-04] FlashAttention 路径 (PyTorch 2.0+)
        #   使用 F.scaled_dot_product_attention, 自动选择 Flash / Memory-Efficient 后端
        #   注意: FlashAttention 不支持自定义 attn_bias, 所以仅在 use_rope 模式时走此路径
        #   (RoPE 已把位置信息编码进 q,k, 不需要额外 position bias)
        if self.use_flash_attn and self.use_rope and hasattr(F, 'scaled_dot_product_attention'):
            dropout_p = self.attn_drop.p if self.training else 0.0
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=dropout_p,
                is_causal=False,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # 标准注意力路径 (支持 relative position bias)
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
            
            # 添加相对位置偏置 (仅在使用 RelativePosition2D 时)
            if not self.use_rope and self.relative_position is not None and N == H * W:
                relative_position_bias = self.relative_position(H, W)
                attn = attn + relative_position_bias.unsqueeze(0)
            
            # Softmax + Dropout
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            
            # 加权求和
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
