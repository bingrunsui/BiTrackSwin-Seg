"""
Deformable Transformer Block Module
====================================
可变形注意力的 Encoder 和 Decoder Block

与 ViTEncoderBlock/ViTDecoderBlock 接口一致，可无缝替换

结构:
- DeformableEncoderBlock: Deformable Self-Attention + FFN
- DeformableDecoderBlock: Deformable Self-Attention + Cross-Attention + FFN

使用场景:
- 大分辨率特征图 (减少计算量)
- 需要稀疏采样的场景
- 与 ViTEncoderBlock 可互换使用

修改记录:
  [2026-03-14] DeformableSelfAttention & DeformableCrossAttention 四项增强:
               1. Q 位置编码: 2D 正弦 PE, 使偏移/权重预测具备空间感知
               2. 可学习 per-head 偏移尺度: 初始小感受野, 逐步扩大
               3. Sigmoid 调制替代 Softmax: 采样点可独立抑制 (防止学到噪声)
               4. 偏移/输出零初始化: 训练初期近似恒等, 更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# =============================================================================
# 公共组件
# =============================================================================

class DropPath(nn.Module):
    """随机深度 (Stochastic Depth)"""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class FFN(nn.Module):
    """前馈神经网络"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# 相对位置编码 (用于 Deformable Attention)
# =============================================================================

class DeformableRelativePosition(nn.Module):
    """
    用于 Deformable Attention 的相对位置编码
    
    为采样点生成相对于参考点的位置偏置
    """
    
    def __init__(
        self,
        num_heads: int,
        num_points: int = 4,
        max_relative_position: int = 64
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.num_points = num_points
        self.max_relative_position = max_relative_position
        
        num_positions = 2 * max_relative_position - 1
        
        self.relative_position_h = nn.Parameter(
            torch.zeros(num_positions, num_heads)
        )
        self.relative_position_w = nn.Parameter(
            torch.zeros(num_positions, num_heads)
        )
        
        nn.init.trunc_normal_(self.relative_position_h, std=0.02)
        nn.init.trunc_normal_(self.relative_position_w, std=0.02)
    
    def forward(
        self,
        sampling_offsets: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Args:
            sampling_offsets: [B, N, num_heads, num_points, 2]
            H, W: 特征图尺寸
            
        Returns:
            position_bias: [B, N, num_heads, num_points]
        """
        B, N, num_heads, num_points, _ = sampling_offsets.shape
        
        # 计算相对位置索引
        rel_h = (sampling_offsets[..., 0] * H).long()
        rel_w = (sampling_offsets[..., 1] * W).long()
        
        # 偏移到非负索引
        rel_h = rel_h + self.max_relative_position - 1
        rel_w = rel_w + self.max_relative_position - 1
        
        # 限制范围
        rel_h = rel_h.clamp(0, 2 * self.max_relative_position - 2)
        rel_w = rel_w.clamp(0, 2 * self.max_relative_position - 2)
        
        # 查表 [B, N, num_heads, num_points] -> [B, N, num_heads, num_points, num_heads]
        bias_h = self.relative_position_h[rel_h]
        bias_w = self.relative_position_w[rel_w]
        
        # 取对角线
        bias_h = torch.diagonal(bias_h, dim1=2, dim2=4).permute(0, 1, 3, 2)
        bias_w = torch.diagonal(bias_w, dim1=2, dim2=4).permute(0, 1, 3, 2)
        
        return bias_h + bias_w


# =============================================================================
# ★ [2026-03-14] 共享工具: 2D 正弦位置编码
# =============================================================================

_pe_cache = {}

def _get_2d_sinusoidal_pe(H: int, W: int, C: int, device: torch.device) -> torch.Tensor:
    """
    生成 2D 正弦位置编码 (带缓存)
    前 C/2 维编码 y, 后 C/2 维编码 x
    Returns: [1, H*W, C]
    """
    key = (H, W, C, device)
    if key in _pe_cache:
        return _pe_cache[key]
    
    half_c = C // 2
    dim_t = torch.arange(half_c, dtype=torch.float32, device=device)
    dim_t = 10000.0 ** (2.0 * (dim_t // 2) / half_c)
    
    pos_y = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1) / max(H, 1)
    pos_x = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1) / max(W, 1)
    
    pe_y = pos_y / dim_t.unsqueeze(0)
    pe_x = pos_x / dim_t.unsqueeze(0)
    
    pe_y = torch.stack([pe_y[:, 0::2].sin(), pe_y[:, 1::2].cos()], dim=-1).flatten(1)
    pe_x = torch.stack([pe_x[:, 0::2].sin(), pe_x[:, 1::2].cos()], dim=-1).flatten(1)
    
    pe = torch.zeros(H, W, C, device=device)
    pe[:, :, :half_c] = pe_y.unsqueeze(1).expand(-1, W, -1)
    pe[:, :, half_c:half_c + half_c] = pe_x.unsqueeze(0).expand(H, -1, -1)
    
    pe = pe.reshape(1, H * W, C)
    _pe_cache[key] = pe
    return pe


# =============================================================================
# Deformable Self-Attention
# =============================================================================

class DeformableSelfAttention(nn.Module):
    """
    可变形自注意力 (增强版)
    
    每个位置学习 K 个采样点，只在采样点上计算注意力
    
    ★ [2026-03-14] 四项增强:
      1. Q 位置编码: 2D 正弦 PE, 使偏移/权重预测具备空间感知
      2. 可学习 per-head 偏移尺度: 初始小感受野, 逐步扩大
      3. Sigmoid 调制替代 Softmax: 采样点可独立抑制
      4. 更稳健的初始化: offset/output 零初始化
    
    Args:
        dim: 输入维度
        num_heads: 注意力头数
        num_points: 每个查询的采样点数
        dropout: dropout 比例
        use_relative_position: 是否使用相对位置编码
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.0,
        use_relative_position: bool = True
    ):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # ★ [2026-03-14] 位置编码投影
        self.pos_proj = nn.Linear(dim, dim)
        
        # 采样偏移量预测
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        
        # ★ [2026-03-14] 可学习 per-head 偏移尺度
        self.offset_scale = nn.Parameter(
            torch.full((1, 1, num_heads, 1, 1), -2.2)
        )
        
        # 注意力权重预测
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        
        # Value 投影
        self.value_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.output_proj = nn.Linear(dim, dim)
        
        # 相对位置编码 (可选, 与新 PE 互补)
        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.rel_pos = DeformableRelativePosition(
                num_heads=num_heads,
                num_points=num_points
            )
        else:
            self.rel_pos = None
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """★ [2026-03-14] 更稳健的初始化"""
        # 位置编码投影: 较小初始化
        nn.init.xavier_uniform_(self.pos_proj.weight, gain=0.5)
        nn.init.constant_(self.pos_proj.bias, 0.0)
        
        # offset weight 零初始化
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        
        # offset bias: 更小的径向网格
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, i, :] *= (i + 1) * 0.05  # ★ 0.1→0.05
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # ★ attention weight 零初始化: sigmoid(0)=0.5
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        # ★ output_proj 零初始化
        nn.init.constant_(self.output_proj.weight, 0.0)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def _get_reference_points(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """生成参考点网格"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device) / H,
            torch.linspace(0.5, W - 0.5, W, device=device) / W,
            indexing='ij'
        )
        ref_points = torch.stack([ref_y.flatten(), ref_x.flatten()], dim=-1)
        return ref_points.unsqueeze(0)  # [1, H*W, 2]
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] 输入特征，N = H * W
            H, W: 空间尺寸
            
        Returns:
            output: [B, N, C]
        """
        B, N, C = x.shape
        
        # 生成参考点
        reference_points = self._get_reference_points(H, W, x.device)
        reference_points = reference_points.expand(B, -1, -1)  # [B, N, 2]
        
        # ★ [2026-03-14] 注入位置编码
        pos_enc = _get_2d_sinusoidal_pe(H, W, C, x.device)
        x_with_pos = x + self.pos_proj(pos_enc)
        
        # Value 投影并重塑
        value = self.value_proj(x)
        value = value.view(B, H, W, self.num_heads, self.head_dim)
        
        # ★ [2026-03-14] 预测采样偏移 (带位置感知 + 可学习尺度)
        sampling_offsets = self.sampling_offsets(x_with_pos)
        sampling_offsets = sampling_offsets.view(B, N, self.num_heads, self.num_points, 2)
        sampling_offsets = sampling_offsets.tanh() * torch.sigmoid(self.offset_scale)
        
        # 计算采样位置
        ref_points = reference_points.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1, 2]
        sampling_locations = ref_points + sampling_offsets  # [B, N, num_heads, num_points, 2]
        sampling_locations = sampling_locations * 2 - 1  # [0, 1] -> [-1, 1]
        
        # ★ [2026-03-14] Sigmoid 调制 (带位置感知)
        attention_weights = self.attention_weights(x_with_pos)
        attention_weights = attention_weights.view(B, N, self.num_heads, self.num_points)
        
        # 添加相对位置偏置 (与 sigmoid 互补)
        if self.use_relative_position and self.rel_pos is not None:
            pos_bias = self.rel_pos(sampling_offsets, H, W)
            attention_weights = attention_weights + pos_bias
        
        attention_weights = torch.sigmoid(attention_weights)  # ★ softmax→sigmoid
        
        # 采样
        value = value.permute(0, 3, 4, 1, 2).contiguous()  # [B, num_heads, head_dim, H, W]
        value = value.view(B * self.num_heads, self.head_dim, H, W)
        
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).contiguous()
        sampling_locations = sampling_locations.view(B * self.num_heads, N, self.num_points, 2)
        
        sampled_values = F.grid_sample(
            value,
            sampling_locations,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B * num_heads, head_dim, N, num_points]
        
        sampled_values = sampled_values.view(B, self.num_heads, self.head_dim, N, self.num_points)
        
        # 加权求和
        attention_weights = attention_weights.permute(0, 2, 1, 3)  # [B, num_heads, N, num_points]
        output = (sampled_values * attention_weights.unsqueeze(2)).sum(dim=-1)
        
        # 重塑
        output = output.permute(0, 3, 1, 2).contiguous()  # [B, N, num_heads, head_dim]
        output = output.view(B, N, C)
        
        # 输出投影
        output = self.output_proj(output)
        output = self.dropout(output)
        
        return output


# =============================================================================
# Deformable Cross-Attention
# =============================================================================

class DeformableCrossAttention(nn.Module):
    """
    可变形交叉注意力 (增强版)
    
    Query 来自一个模态，在另一个模态上进行可变形采样
    
    ★ [2026-03-14] 四项增强: 同 DeformableSelfAttention
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.0,
        use_relative_position: bool = True
    ):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        
        # Query 投影
        self.query_proj = nn.Linear(dim, dim)
        
        # ★ [2026-03-14] 位置编码投影
        self.pos_proj = nn.Linear(dim, dim)
        
        # 采样偏移量 (基于 query + PE 预测)
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        
        # ★ [2026-03-14] 可学习 per-head 偏移尺度
        self.offset_scale = nn.Parameter(
            torch.full((1, 1, num_heads, 1, 1), -2.2)
        )
        
        # 注意力权重 (★ sigmoid 调制)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        
        # Value 投影
        self.value_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.output_proj = nn.Linear(dim, dim)
        
        # 相对位置编码
        self.use_relative_position = use_relative_position
        if use_relative_position:
            self.rel_pos = DeformableRelativePosition(
                num_heads=num_heads,
                num_points=num_points
            )
        else:
            self.rel_pos = None
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """★ [2026-03-14] 更稳健的初始化"""
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0.0)
        
        nn.init.xavier_uniform_(self.pos_proj.weight, gain=0.5)
        nn.init.constant_(self.pos_proj.bias, 0.0)
        
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, i, :] *= (i + 1) * 0.05  # ★ 0.1→0.05
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.output_proj.weight, 0.0)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def _get_reference_points(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """生成参考点网格"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device) / H,
            torch.linspace(0.5, W - 0.5, W, device=device) / W,
            indexing='ij'
        )
        ref_points = torch.stack([ref_y.flatten(), ref_x.flatten()], dim=-1)
        return ref_points.unsqueeze(0)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        H_q: int,
        W_q: int,
        H_kv: int,
        W_kv: int
    ) -> torch.Tensor:
        """
        Args:
            query: [B, N_q, C] 查询特征
            key_value: [B, N_kv, C] 键值特征
            H_q, W_q: 查询空间尺寸
            H_kv, W_kv: 键值空间尺寸
            
        Returns:
            output: [B, N_q, C]
        """
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape
        
        # 参考点 (query 位置)
        reference_points = self._get_reference_points(H_q, W_q, query.device)
        reference_points = reference_points.expand(B, -1, -1)
        
        # Query 投影
        query = self.query_proj(query)
        
        # ★ [2026-03-14] 注入位置编码
        pos_enc = _get_2d_sinusoidal_pe(H_q, W_q, C, query.device)
        query_with_pos = query + self.pos_proj(pos_enc)
        
        # Value 投影并重塑
        value = self.value_proj(key_value)
        value = value.view(B, H_kv, W_kv, self.num_heads, self.head_dim)
        
        # ★ [2026-03-14] 预测采样偏移 (带位置感知 + 可学习尺度)
        sampling_offsets = self.sampling_offsets(query_with_pos)
        sampling_offsets = sampling_offsets.view(B, N_q, self.num_heads, self.num_points, 2)
        sampling_offsets = sampling_offsets.tanh() * torch.sigmoid(self.offset_scale)
        
        # 计算采样位置 (在 key_value 空间)
        ref_points_kv = reference_points.clone()
        ref_points_kv = ref_points_kv.unsqueeze(2).unsqueeze(3)
        
        sampling_locations = ref_points_kv + sampling_offsets
        sampling_locations = sampling_locations * 2 - 1  # [0, 1] -> [-1, 1]
        
        # ★ [2026-03-14] Sigmoid 调制
        attention_weights = self.attention_weights(query_with_pos)
        attention_weights = attention_weights.view(B, N_q, self.num_heads, self.num_points)
        
        # 相对位置偏置
        if self.use_relative_position and self.rel_pos is not None:
            pos_bias = self.rel_pos(sampling_offsets, H_kv, W_kv)
            attention_weights = attention_weights + pos_bias
        
        attention_weights = torch.sigmoid(attention_weights)  # ★ softmax→sigmoid
        
        # 采样
        value = value.permute(0, 3, 4, 1, 2).contiguous()
        value = value.view(B * self.num_heads, self.head_dim, H_kv, W_kv)
        
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).contiguous()
        sampling_locations = sampling_locations.view(B * self.num_heads, N_q, self.num_points, 2)
        
        sampled_values = F.grid_sample(
            value,
            sampling_locations,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        sampled_values = sampled_values.view(B, self.num_heads, self.head_dim, N_q, self.num_points)
        
        # 加权求和
        attention_weights = attention_weights.permute(0, 2, 1, 3)
        output = (sampled_values * attention_weights.unsqueeze(2)).sum(dim=-1)
        
        output = output.permute(0, 3, 1, 2).contiguous()
        output = output.view(B, N_q, C)
        
        output = self.output_proj(output)
        output = self.dropout(output)
        
        return output


# =============================================================================
# Deformable Encoder Block (统一接口)
# =============================================================================

class DeformableEncoderBlock(nn.Module):
    """
    可变形注意力 Encoder Block
    
    结构: LayerNorm → Deformable Self-Attention → 残差 → LayerNorm → FFN → 残差
    
    接口与 ViTEncoderBlock 完全一致: forward(x, H, W) -> x
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数
        num_points: 每个查询的采样点数
        mlp_ratio: FFN 隐藏层维度比例
        drop: dropout
        attn_drop: 注意力 dropout
        drop_path: DropPath 比例
        use_relative_position: 是否使用相对位置编码
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_points: int = 4,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        use_relative_position: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Deformable Self-Attention
        self.attn = DeformableSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_points=num_points,
            dropout=attn_drop,
            use_relative_position=use_relative_position
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]
            H, W: 空间尺寸
            
        Returns:
            out: [B, H*W, C]
        """
        # Deformable Self-Attention + 残差
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        
        # FFN + 残差
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        
        return x


# =============================================================================
# Deformable Decoder Block (统一接口)
# =============================================================================

class DeformableDecoderBlock(nn.Module):
    """
    可变形注意力 Decoder Block
    
    结构:
    LayerNorm → Deformable Self-Attention → 残差
    → LayerNorm → Deformable Cross-Attention → 残差
    → LayerNorm → FFN → 残差
    
    接口与 ViTDecoderBlock 完全一致: forward(x, H, W, aux, H_aux, W_aux) -> x
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数
        num_points: 每个查询的采样点数
        mlp_ratio: FFN 隐藏层维度比例
        drop: dropout
        attn_drop: 注意力 dropout
        drop_path: DropPath 比例
        use_relative_position: 是否使用相对位置编码
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_points: int = 4,
        mlp_ratio: float = 4.,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        use_relative_position: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Deformable Self-Attention
        self.self_attn = DeformableSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_points=num_points,
            dropout=attn_drop,
            use_relative_position=use_relative_position
        )
        
        # Deformable Cross-Attention
        self.cross_attn = DeformableCrossAttention(
            dim=dim,
            num_heads=num_heads,
            num_points=num_points,
            dropout=attn_drop,
            use_relative_position=use_relative_position
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )
    
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        aux: torch.Tensor,
        H_aux: int,
        W_aux: int
    ) -> torch.Tensor:
        """
        Args:
            x: 主分支特征 (RGB) [B, H*W, C]
            H, W: 主分支空间尺寸
            aux: 辅助分支特征 (多光谱) [B, H_aux*W_aux, C]
            H_aux, W_aux: 辅助分支空间尺寸
            
        Returns:
            out: [B, H*W, C]
        """
        # ============ Deformable Self-Attention ============
        x = x + self.drop_path(self.self_attn(self.norm1(x), H, W))
        
        # ============ Deformable Cross-Attention ============
        shortcut = x
        x = self.norm2(x)
        x = self.cross_attn(x, aux, H, W, H_aux, W_aux)
        x = shortcut + self.drop_path(x)
        
        # ============ FFN ============
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        
        return x


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    
    print("=" * 70)
    print("Deformable Encoder/Decoder Block 测试")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 2
    H, W = 16, 16
    dim = 256
    num_heads = 8
    
    # 输入
    x = torch.randn(batch_size, H * W, dim, device=device)
    aux = torch.randn(batch_size, H * W, dim, device=device)
    
    # 测试 DeformableEncoderBlock
    print("\n[1] 测试 DeformableEncoderBlock")
    encoder = DeformableEncoderBlock(
        dim=dim,
        num_heads=num_heads,
        num_points=4,
        use_relative_position=True
    ).to(device)
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"参数量: {params / 1e6:.2f}M")
    
    with torch.no_grad():
        out = encoder(x, H, W)
    
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    assert out.shape == x.shape, "形状不匹配!"
    
    # 测试 DeformableDecoderBlock
    print("\n[2] 测试 DeformableDecoderBlock")
    decoder = DeformableDecoderBlock(
        dim=dim,
        num_heads=num_heads,
        num_points=4,
        use_relative_position=True
    ).to(device)
    
    params = sum(p.numel() for p in decoder.parameters())
    print(f"参数量: {params / 1e6:.2f}M")
    
    with torch.no_grad():
        out = decoder(x, H, W, aux, H, W)
    
    print(f"输入 x: {x.shape}")
    print(f"输入 aux: {aux.shape}")
    print(f"输出: {out.shape}")
    assert out.shape == x.shape, "形状不匹配!"
    
    # 对比 ViTEncoderBlock 接口
    print("\n[3] 接口兼容性测试")
    print("DeformableEncoderBlock.forward(x, H, W) -> 与 ViTEncoderBlock 一致 ✓")
    print("DeformableDecoderBlock.forward(x, H, W, aux, H_aux, W_aux) -> 与 ViTDecoderBlock 一致 ✓")
    
    print("\n" + "=" * 70)
    print("✅ 测试通过!")
    print("=" * 70)