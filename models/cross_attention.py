"""
Cross Attention Module
======================
跨模态注意力机制，用于多模态数据融合

特点:
- Q来自主分支(RGB)，K/V来自辅助分支(多光谱/雷达等)
- 位置编码可选 (use_pos_embed参数)
- 维度验证，增强代码鲁棒性
- 支持不同空间尺寸的输入 (N != M)

使用建议:
- Stage 1-2 (细节融合): use_pos_embed=True
- Stage 3-4 (语义融合): use_pos_embed=False 或 True (可实验)
"""
'''
CrossAttention (通用版):
├── 两个模态空间尺寸可能不同
├── 例如: RGB是 64×64，多光谱是 32×32
├── 需要处理不对齐的情况
└── 更灵活，但稍慢

CrossAttentionFast (快速版):
├── 两个模态空间尺寸完全相同
├── 例如: RGB和多光谱都是 64×64
├── 假设已经对齐，跳过检查
└── 更高效，但要求更严格

'''

import torch
import torch.nn as nn
from typing import Optional

from .relative_position import RelativePosition2D


class CrossAttention(nn.Module):
    """
    跨模态注意力
    
    Args:
        dim: 特征维度 (Q/K/V必须相同)
        num_heads: 注意力头数
        qkv_bias: Q/K/V投影是否使用偏置
        attn_drop: 注意力dropout
        proj_drop: 输出投影dropout
        use_pos_embed: 是否使用位置编码
        max_relative_position: 相对位置编码最大范围 (仅use_pos_embed=True时有效)
    
    Example:
        # Stage 1-2: 使用位置编码
        cross_attn_local = CrossAttention(dim=192, use_pos_embed=True)
        
        # Stage 3-4: 不使用位置编码
        cross_attn_global = CrossAttention(dim=768, use_pos_embed=False)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        use_pos_embed: bool = True,
        max_relative_position: int = 64
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pos_embed = use_pos_embed
        
        assert dim % num_heads == 0, f"dim({dim})必须能被num_heads({num_heads})整除"
        
        # Q投影 (来自主分支)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # K/V投影 (来自辅助分支)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 位置编码 (可选)
        if use_pos_embed:
            self.pos_embed = RelativePosition2D(
                num_heads=num_heads,
                max_relative_position=max_relative_position
            )
        else:
            self.pos_embed = None
        
        # Softmax
        self.softmax = nn.Softmax(dim=-1)
        
        # ★ [2026-03-21] 专用初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        ★ [2026-03-21] 微量初始化 — 与 CrossAttentionFast 保持一致
        output proj 使用极小权重, 保证训练初期 cross-attn ≈ 恒等映射
        同时允许梯度穿过 proj → K/V 路径不被阻断
        """
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        nn.init.constant_(self.proj.bias, 0.0)
    
    def _check_dimensions(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        H_q: int, W_q: int,
        H_kv: int, W_kv: int
    ):
        """
        验证输入维度
        """
        B_q, N_q, C_q = query.shape
        B_k, N_k, C_k = key.shape
        B_v, N_v, C_v = value.shape
        
        # 检查batch size
        assert B_q == B_k == B_v, \
            f"Batch size不匹配: query({B_q}), key({B_k}), value({B_v})"
        
        # 检查特征维度
        assert C_q == C_k == C_v == self.dim, \
            f"特征维度不匹配: query({C_q}), key({C_k}), value({C_v}), expected({self.dim})"
        
        # 检查K和V序列长度
        assert N_k == N_v, \
            f"Key和Value序列长度不匹配: key({N_k}), value({N_v})"
        
        # 检查空间尺寸与序列长度
        assert N_q == H_q * W_q, \
            f"Query序列长度({N_q})与空间尺寸({H_q}×{W_q}={H_q*W_q})不匹配"
        
        assert N_k == H_kv * W_kv, \
            f"Key序列长度({N_k})与空间尺寸({H_kv}×{W_kv}={H_kv*W_kv})不匹配"
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        H_q: int,
        W_q: int,
        H_kv: int,
        W_kv: int
    ) -> torch.Tensor:
        """
        Args:
            query: 主分支特征 (RGB) [B, N, C]
            key: 辅助分支特征 (多光谱) [B, M, C]
            value: 辅助分支特征 (多光谱) [B, M, C]
            H_q, W_q: query的空间尺寸 (N = H_q * W_q)
            H_kv, W_kv: key/value的空间尺寸 (M = H_kv * W_kv)
            
        Returns:
            output: [B, N, C] 与query形状相同
        """
        # 维度验证
        self._check_dimensions(query, key, value, H_q, W_q, H_kv, W_kv)
        
        B, N, C = query.shape
        M = key.shape[1]
        
        # Q/K/V投影
        q = self.q_proj(query)  # [B, N, C]
        k = self.k_proj(key)    # [B, M, C]
        v = self.v_proj(value)  # [B, M, C]
        
        # 重塑为多头形式
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, num_heads, N, M]
        
        # 添加位置编码 (可选)
        if self.use_pos_embed and self.pos_embed is not None:
            if N == M and H_q == H_kv and W_q == W_kv:
                # 空间对齐: 直接使用位置偏置
                pos_bias = self.pos_embed(H_q, W_q)  # [num_heads, N, N]
                attn = attn + pos_bias.unsqueeze(0)
            else:
                # 空间不对齐: 构建跨空间位置偏置
                cross_pos_bias = self._build_cross_position_bias(H_q, W_q, H_kv, W_kv)
                attn = attn + cross_pos_bias.unsqueeze(0)
        
        # Softmax + Dropout
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # 加权求和
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 输出投影
        output = self.proj(output)
        output = self.proj_drop(output)
        
        return output
    
    def _build_cross_position_bias(
        self,
        H_q: int, W_q: int,
        H_kv: int, W_kv: int
    ) -> torch.Tensor:
        """
        构建跨空间位置偏置 (当Q和K/V空间尺寸不同时)
        """
        # 添加 None 检查
        if self.pos_embed is None:
            raise ValueError("pos_embed is None, cannot build position bias")
        
        device = self.pos_embed.relative_position_h.device
        max_rel = self.pos_embed.max_relative_position
        
        # 生成归一化坐标
        coords_q_h = torch.arange(H_q, device=device).float() / max(H_q - 1, 1)
        coords_q_w = torch.arange(W_q, device=device).float() / max(W_q - 1, 1)
        coords_k_h = torch.arange(H_kv, device=device).float() / max(H_kv - 1, 1)
        coords_k_w = torch.arange(W_kv, device=device).float() / max(W_kv - 1, 1)
        
        # 计算相对位置并映射到索引
        rel_h = coords_q_h.unsqueeze(1) - coords_k_h.unsqueeze(0)  # [H_q, H_kv]
        rel_w = coords_q_w.unsqueeze(1) - coords_k_w.unsqueeze(0)  # [W_q, W_kv]
        
        rel_h = (rel_h * (max_rel - 1) + max_rel - 1).long().clamp(0, 2 * max_rel - 2)
        rel_w = (rel_w * (max_rel - 1) + max_rel - 1).long().clamp(0, 2 * max_rel - 2)
        
        # 查表
        bias_h = self.pos_embed.relative_position_h[rel_h]  # [H_q, H_kv, num_heads]
        bias_w = self.pos_embed.relative_position_w[rel_w]  # [W_q, W_kv, num_heads]
        
        # 扩展为 [H_q, W_q, H_kv, W_kv, num_heads]
        bias_h = bias_h.unsqueeze(1).unsqueeze(3)  # [H_q, 1, H_kv, 1, num_heads]
        bias_w = bias_w.unsqueeze(0).unsqueeze(2)  # [1, W_q, 1, W_kv, num_heads]
        
        cross_pos_bias = bias_h + bias_w  # [H_q, W_q, H_kv, W_kv, num_heads]
        
        # 重塑为 [num_heads, N, M]
        N = H_q * W_q
        M = H_kv * W_kv
        cross_pos_bias = cross_pos_bias.reshape(N, M, self.num_heads)
        cross_pos_bias = cross_pos_bias.permute(2, 0, 1).contiguous()
        
        return cross_pos_bias


class CrossAttentionFast(nn.Module):
    """
    快速版跨模态注意力 (针对空间对齐情况优化)
    
    当两个模态空间完全对齐时使用，计算更高效
    
    Args:
        dim: 特征维度
        num_heads: 注意力头数
        qkv_bias: Q/K/V投影是否使用偏置
        attn_drop: 注意力dropout
        proj_drop: 输出投影dropout
        use_pos_embed: 是否使用位置编码
        max_relative_position: 相对位置编码最大范围
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        use_pos_embed: bool = True,
        max_relative_position: int = 64
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pos_embed = use_pos_embed
        
        assert dim % num_heads == 0, f"dim({dim})必须能被num_heads({num_heads})整除"
        
        # Q投影
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # K/V投影
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 位置编码 (可选)
        if use_pos_embed:
            self.pos_embed = RelativePosition2D(
                num_heads=num_heads,
                max_relative_position=max_relative_position
            )
        else:
            self.pos_embed = None
        
        # Softmax
        self.softmax = nn.Softmax(dim=-1)
        
        # ★ [2026-03-20] 专用初始化 (会被 multimodal._init_weights 覆盖后重新调用)
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        ★ [2026-03-21] 微量初始化 — 替代纯零初始化, 解决梯度死锁
        
        旧版 (3/20): output proj 纯零初始化 → cross_attn 输出 = 0
          问题: proj.weight=0 时, 反向传播 d_loss/d_kv = d_loss/d_output @ weight = 0
                → Aux K/V 路径完全没有梯度 → aux_s2/s3 永远学不到东西
                → "鸡生蛋"死锁: weight=0 → 梯度=0 → weight 不更新 → 梯度继续=0
        
        新版: xavier_uniform_(gain=0.01) → proj.weight 极小但非零
          效果: cross_attn 初期贡献仅为主干的 ~1% (≈ 零残差初始化)
                但反向梯度可以穿过 proj → 传回 K/V → Aux 深层能学习
          参考: GPT-3 用 1/√(2*n_layers) 缩放, 我们用 gain=0.01 更保守
        """
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        nn.init.constant_(self.proj.bias, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Args:
            query: 主分支特征 [B, H*W, C]
            key: 辅助分支特征 [B, H*W, C]
            value: 辅助分支特征 [B, H*W, C]
            H, W: 空间尺寸 (两个模态共享)
            
        Returns:
            output: [B, H*W, C]
        """
        B, N, C = query.shape
        
        # 维度验证
        assert key.shape == value.shape == query.shape, \
            f"空间对齐模式下，Q/K/V形状必须相同: query{query.shape}, key{key.shape}, value{value.shape}"
        assert N == H * W, \
            f"序列长度({N})与空间尺寸({H}×{W}={H*W})不匹配"
        assert C == self.dim, \
            f"特征维度({C})与期望维度({self.dim})不匹配"
        
        # Q/K/V投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头形式
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # [B, num_heads, N, N]
        
        # 添加位置偏置 (可选)
        if self.use_pos_embed and self.pos_embed is not None:
            pos_bias = self.pos_embed(H, W)  # [num_heads, N, N]
            attn = attn + pos_bias.unsqueeze(0)
        
        # Softmax + Dropout
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        # 加权求和
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # 输出投影
        output = self.proj(output)
        output = self.proj_drop(output)
        
        return output
