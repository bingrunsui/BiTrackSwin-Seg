"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ ★ [2026-06-29] 本次改动 (shift-window 对照实验支持):                          ║
║   SwinEncoderBlock 新增 shift_size 参数 + 标准 Swin SW-MSA:                  ║
║     _get_attn_mask(H,W): 构造并按分辨率缓存 shifted-window 注意力掩码;        ║
║     _windowed_attn: shift_size>0 时 torch.roll(-s) + 掩码注意 + roll(+s) 还原;║
║     shift_size=0 (默认) 时与改动前逐字节一致 (纯 W-MSA).                      ║
║   依赖: window_attention.py 的 WindowAttention.forward 需已支持 mask 形参     ║
║   (你已自行改完其余文件). LayerScale γ_1/γ_2 在 forward 中应用, 与本改动正交. ║
╚═══════════════════════════════════════════════════════════════════════════╝

Transformer Block Module
========================
包含 Encoder 和 Decoder Block
 
结构:
- 公共组件: FFN, DropPath
- Encoder: SwinEncoderBlock, ViTEncoderBlock (纯 Self-Attention)
- Decoder: SwinDecoderBlock, ViTDecoderBlock (Self-Attention + Cross-Attention)
- **新增**: SwinDeformableDecoderBlock (Self-Attention + Deformable Cross-Attention)
 
使用场景:
- Encoder: RGB/多光谱 各自的特征提取
- Decoder: 多模态融合 (Q=RGB, K,V=辅助模态)
- SwinDeformableDecoderBlock: 高分辨率多模态融合 (显存友好)
 
新增:
- LoRALinear: 通用的 LoRA 线性层实现
主干网络 + LoRA 统一管理
 
核心功能:
1. 统一在 __init__ 中设定 LoRA 参数 (lora_r, lora_alpha)。
2. 覆盖所有 Attention 权重: 
   - WindowAttention/GlobalAttention 的 'qkv'
   - CrossAttention 的 'q_proj', 'k_proj', 'v_proj'
   - 输出投影 'proj'
 
修改记录:
  [2026-03-14] DeformableCrossAttention 四项增强 (DCNv3/InternImage 风格):
               1. Q 位置编码: 2D 正弦 PE 注入, 使偏移/权重预测具备空间感知
               2. 可学习 per-head 偏移尺度: 替代固定 tanh()*0.5, 初始小→逐步扩大
               3. Sigmoid 调制替代 Softmax: 采样点可独立抑制 (防止学到噪声)
               4. 偏移/输出零初始化: 训练初期近似恒等映射, 更稳定的起步
  [2026-04-04] V14 现代化升级 (受 LLaMA / CaiT / EVA-02 启发):
               1. RMSNorm: 替代 LayerNorm, 省去均值计算, 前向/反向速度 +10~20%
                  (参考: T5, LLaMA, EVA-02)
               2. SwiGLU FFN: 门控 FFN, 非线性表达能力大幅提升
                  (参考: LLaMA, PaLM, EVA-02 — Fang et al. 2023)
               3. LayerScale: 残差分支乘可学习标量 γ (init=1e-4), 拯救深层 ViT 梯度
                  (参考: CaiT — Touvron et al. ICCV 2021, DeiT III — ECCV 2022)
               以上三项仅应用于 ViTEncoderBlock / ViTDecoderBlock (Aux ViT 从头训练),
               SwinEncoderBlock / SwinDecoderBlock 保持不变 (Swin-T 预训练兼容).
"""
 
import sys
from pathlib import Path
 
# 路径设置
_current_file = Path(__file__).resolve()
_current_dir = _current_file.parent
_project_root = _current_dir.parent
for _p in [str(_project_root), str(_current_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import math
 
from .window_attention import WindowAttention, window_partition, window_reverse
from .global_attention import GlobalAttention
from .cross_attention import CrossAttention, CrossAttentionFast
 
# =============================================================================
# LoRA 基础组件 (新增)
# =============================================================================
 
class LoRALinear(nn.Module):
    """
    [通用组件] LoRA 线性层
    
    用于替换标准的 nn.Linear，实现低秩适应 (Low-Rank Adaptation)。
    支持主干网络和辅助分支统一调用。
    
    Args:
        original_linear: 原始的 Linear 层 (权重将被冻结)
        r: LoRA 的秩 (Rank)
        alpha: LoRA 缩放系数
        dropout: LoRA 路径的 dropout
        
    公式: h = Wx + BAx * (alpha/r)
    """
    def __init__(
        self, 
        original_linear: nn.Linear, 
        r: int = 4, 
        alpha: int = 1, 
        dropout: float = 0.
    ):
        super().__init__()
        self.linear = original_linear
        
        # 冻结原参数
        for param in self.linear.parameters():
            param.requires_grad = False
            
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA 矩阵 (B, A)
        # A: [r, in], B: [out, r]
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        self.lora_dropout = nn.Dropout(p=dropout)
        
        self.reset_parameters()
 
    def reset_parameters(self):
        # A 使用 Kaiming 初始化 (便于训练开始时有梯度)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 使用 零初始化   (保证初始状态下输出与原模型完全一致)
        nn.init.zeros_(self.lora_B)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原路径 (冻结)
        result_base = self.linear(x)
        
        # LoRA 路径 (可训练): (x @ A^T @ B^T) * scaling
        # x: [B, N, in_dim]
        # A: [r, in_dim] -> A.T: [in_dim, r]
        # B: [out_dim, r] -> B.T: [r, out_dim]
        lora_out = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return result_base + lora_out
    
# =============================================================================
# ★ [2026-04-04] RMSNorm — 替代 LayerNorm (T5, LLaMA, EVA-02)
# =============================================================================

class RMSNorm(nn.Module):
    """
    ★ [2026-04-04] Root Mean Square Layer Normalization
    
    省去均值中心化, 只计算均方根归一化, 前向/反向速度 +10~20%.
    在 ViT 中已被 EVA-02 (Fang et al., 2023) 验证精度不掉甚至更稳定.
    
    参考:
      - T5 (Raffel et al., 2020): 首次在 Transformer 中使用
      - LLaMA (Touvron et al., 2023): LLM 标配
      - EVA-02 (Fang et al., 2023): Vision Transformer 中验证
    
    公式: x_norm = x / RMS(x) * γ, 其中 RMS(x) = sqrt(mean(x²) + ε)
    
    Args:
        dim: 归一化维度
        eps: 数值稳定性常量
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
    
    def extra_repr(self) -> str:
        return f'{self.dim}, eps={self.eps}'


# =============================================================================
# ★ [2026-04-04] SwiGLU FFN — 门控 FFN (LLaMA, PaLM, EVA-02)
# =============================================================================

class SwiGLUFFN(nn.Module):
    """
    ★ [2026-04-04] SwiGLU Feed-Forward Network (门控线性单元)
    
    传统 FFN: Linear → GELU → Linear
    SwiGLU:   out = W3(SiLU(W1(x)) ⊙ W2(x))
    
    用 SiLU(Swish) 激活函数做门控, 非线性表达能力远超 GELU.
    参数量增加 ~33% (两个投影矩阵), 但效果提升显著.
    
    参考:
      - GLU Variants (Shazeer, 2020): 提出 SwiGLU
      - LLaMA (Touvron et al., 2023): SwiGLU 是核心组件
      - PaLM (Chowdhery et al., 2022): 同样使用 SwiGLU
      - EVA-02 (Fang et al., 2023): 在 ViT 中验证 SwiGLU 有效
    
    Args:
        in_features: 输入维度
        hidden_features: 隐藏层维度 (默认 = in_features * 8/3, 保持参数量与 4x FFN 相当)
        out_features: 输出维度 (默认 = in_features)
        drop: Dropout 比例
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        # SwiGLU 标准做法: hidden_dim = 2/3 * 4 * dim ≈ 2.67 * dim
        # 这样总参数量与标准 4x FFN 相当 (因为有两个投影)
        hidden_features = hidden_features or int(in_features * 8 / 3)
        
        self.w1 = nn.Linear(in_features, hidden_features)   # gate 投影
        self.w2 = nn.Linear(in_features, hidden_features)   # value 投影
        self.w3 = nn.Linear(hidden_features, out_features)  # 输出投影
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: W3(SiLU(W1(x)) * W2(x))
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))
    
# =============================================================================
# 公共组件
# =============================================================================
 
class DropPath(nn.Module):
    """
    随机深度 (Stochastic Depth)
    
    训练时以一定概率丢弃整个路径
    """
    
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
    """
    前馈神经网络 (Feed-Forward Network)
    
    结构: Linear → GELU → Dropout → Linear → Dropout
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
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
# Deformable Cross Attention (用于 SwinDeformableDecoderBlock)
# =============================================================================
 
class DeformableCrossAttention(nn.Module):
    """
    可变形交叉注意力 (增强版, DCNv3/InternImage 风格)
    
    Query 来自主分支，在辅助分支上进行可变形采样
    复杂度: O(N × K) 而非 O(N × M)，显存友好
    
    ★ [2026-03-14] 四项关键改进:
      1. Q 位置编码: 2D 正弦位置编码注入 Query, 使偏移/权重预测具备空间感知
         (旧版: Q 不知道自己在哪 → 相似特征预测相同偏移, 采样点无空间区分)
      2. 可学习 per-head 偏移尺度: 替代固定 tanh()*0.5, 初始小感受野逐步扩大
         (旧版: 偏移范围 ±0.5 → Stage1 几乎覆盖整个 16×16 KV 图, 失去局部稀疏优势)
      3. Sigmoid 调制替代 Softmax: 每个采样点可独立抑制 (权重可全为 0)
         (旧版: softmax 强制分配权重 → 即使全采到噪声也会注入到输出中)
      4. 偏移初始化改进: 更小初始偏置 + 零初始化 weight, 训练初期近似恒等映射
    
    Args:
        dim: 特征维度
        num_heads: 注意力头数
        num_points: 每个查询的采样点数
        dropout: dropout 比例
    """
    
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        
        # Query 投影
        self.query_proj = nn.Linear(dim, dim)
        
        # ★ [2026-03-14] 改进1: 位置编码投影层
        #   2D 正弦 PE (dim 维) → 投影到 dim → 与 query 相加
        #   使 sampling_offsets 和 attention_weights 能感知 Q 的空间位置
        self.pos_proj = nn.Linear(dim, dim)
        
        # 采样偏移量 (基于 query + 位置编码 预测)
        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        
        # ★ [2026-03-14] 改进2: 可学习 per-head 偏移尺度
        #   替代固定 tanh() * 0.5, 初始化为较小值 (≈0.1)
        #   公式: offsets = tanh(raw_offsets) * sigmoid(offset_scale)
        #   sigmoid(offset_scale_init) ≈ 0.1 → offset_scale_init = log(0.1/0.9) ≈ -2.2
        self.offset_scale = nn.Parameter(
            torch.full((1, 1, num_heads, 1, 1), -2.2)
        )
        
        # ★ [2026-03-14] 改进3: Sigmoid 调制替代 Softmax
        #   注意力权重: Linear → sigmoid (每个采样点独立 [0,1])
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        
        # Value 投影
        self.value_proj = nn.Linear(dim, dim)
        
        # 输出投影
        self.output_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 位置编码缓存 (避免重复生成)
        self._pe_cache = {}
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        ★ [2026-03-14] 改进4: 更稳健的初始化
          - offset weight 零初始化: 训练初期偏移 ≈ 仅由 bias 决定 (网格采样)
          - offset bias: 更小的径向网格 (0.05 步长, 而非 0.1)
          - attention weight: 零初始化, sigmoid(0)=0.5 → 均匀权重开始
          - output_proj: 零初始化 → 训练初期 cross-attn 输出 ≈ 0, 不破坏残差主干
        """
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0.0)
        
        # 位置编码投影: 较小初始化, 初期位置信号不要太强
        nn.init.xavier_uniform_(self.pos_proj.weight, gain=0.5)
        nn.init.constant_(self.pos_proj.bias, 0.0)
        
        # ★ offset weight 零初始化 → 初期偏移仅由 bias 控制 (稳定)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        
        # ★ offset bias: 径向网格初始化, 步长更小 (0.05)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, i, :] *= (i + 1) * 0.05  # ★ 0.1→0.05, 更小初始感受野
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # ★ attention weight 零初始化: sigmoid(0)=0.5, 训练起点为均匀权重
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        
        # ★ output_proj 零初始化: 初期 cross-attn 对残差主干 ≈ 无影响, 逐步学习
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
    
    def _get_pos_encoding(self, H: int, W: int, C: int, device: torch.device) -> torch.Tensor:
        """
        ★ [2026-03-14] 改进1: 生成 2D 正弦位置编码
        
        每个空间位置 (y, x) → C 维正弦编码
        前 C/2 维编码 y 坐标, 后 C/2 维编码 x 坐标
        
        Args:
            H, W: 空间尺寸
            C: 编码维度 (= self.dim)
            device: 设备
        Returns:
            pe: [1, H*W, C]
        """
        cache_key = (H, W, C, device)
        if cache_key in self._pe_cache:
            return self._pe_cache[cache_key]
        
        half_c = C // 2
        # 温度衰减频率 (与 ViT/DETR 一致)
        dim_t = torch.arange(half_c, dtype=torch.float32, device=device)
        dim_t = 10000.0 ** (2.0 * (dim_t // 2) / half_c)
        
        pos_y = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1) / max(H, 1)  # [H, 1]
        pos_x = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1) / max(W, 1)  # [W, 1]
        
        # [H, half_c] 和 [W, half_c]
        pe_y = pos_y / dim_t.unsqueeze(0)  # [H, half_c]
        pe_x = pos_x / dim_t.unsqueeze(0)  # [W, half_c]
        
        # sin/cos 交替
        pe_y = torch.stack([pe_y[:, 0::2].sin(), pe_y[:, 1::2].cos()], dim=-1).flatten(1)  # [H, half_c]
        pe_x = torch.stack([pe_x[:, 0::2].sin(), pe_x[:, 1::2].cos()], dim=-1).flatten(1)  # [W, half_c]
        
        # 组合: [H, W, C]
        pe = torch.zeros(H, W, C, device=device)
        pe[:, :, :half_c] = pe_y.unsqueeze(1).expand(-1, W, -1)
        pe[:, :, half_c:half_c + half_c] = pe_x.unsqueeze(0).expand(H, -1, -1)
        
        pe = pe.reshape(1, H * W, C)  # [1, H*W, C]
        
        self._pe_cache[cache_key] = pe
        return pe
    
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
            query: [B, N_q, C] 查询特征 (主分支)
            key_value: [B, N_kv, C] 键值特征 (辅助分支)
            H_q, W_q: 查询空间尺寸
            H_kv, W_kv: 键值空间尺寸
            
        Returns:
            output: [B, N_q, C]
        """
        B, N_q, C = query.shape
        _, N_kv, _ = key_value.shape
        
        # 参考点 (query 位置，归一化到 0-1)
        reference_points = self._get_reference_points(H_q, W_q, query.device)
        reference_points = reference_points.expand(B, -1, -1)  # [B, N_q, 2]
        
        # Query 投影
        query = self.query_proj(query)
        
        # ★ [2026-03-14] 改进1: 注入位置编码
        #   生成 2D 正弦 PE → 投影 → 与 query 相加
        #   使后续的 offset/weight 预测具备空间感知能力
        pos_enc = self._get_pos_encoding(H_q, W_q, C, query.device)  # [1, N_q, C]
        query_with_pos = query + self.pos_proj(pos_enc)
        
        # Value 投影并重塑
        value = self.value_proj(key_value)
        value = value.view(B, H_kv, W_kv, self.num_heads, self.head_dim)
        
        # ★ [2026-03-14] 改进2: 预测采样偏移 (基于带位置的 query)
        #   偏移尺度由可学习 offset_scale 控制, 初始 sigmoid(-2.2) ≈ 0.1
        sampling_offsets = self.sampling_offsets(query_with_pos)
        sampling_offsets = sampling_offsets.view(B, N_q, self.num_heads, self.num_points, 2)
        sampling_offsets = sampling_offsets.tanh() * torch.sigmoid(self.offset_scale)
        
        # 计算采样位置 (在 key_value 空间)
        # 如果 query 和 key_value 尺寸不同，需要缩放参考点
        scale_h = H_kv / H_q
        scale_w = W_kv / W_q
        
        ref_points_scaled = reference_points.clone()
        ref_points_scaled[..., 0] = ref_points_scaled[..., 0] * scale_h / H_kv * H_q
        ref_points_scaled[..., 1] = ref_points_scaled[..., 1] * scale_w / W_kv * W_q
        
        ref_points_expanded = ref_points_scaled.unsqueeze(2).unsqueeze(3)  # [B, N_q, 1, 1, 2]
        
        sampling_locations = ref_points_expanded + sampling_offsets  # [B, N_q, num_heads, num_points, 2]
        sampling_locations = sampling_locations * 2 - 1  # [0, 1] -> [-1, 1] for grid_sample
        
        # ★ [2026-03-14] 改进3: Sigmoid 调制 (DCNv3 风格)
        #   每个采样点独立 sigmoid → 权重可全为 ~0 (抑制噪声) 或全为 ~1
        #   不再强制 4 个点权重和为 1
        attention_weights = self.attention_weights(query_with_pos)
        attention_weights = attention_weights.view(B, N_q, self.num_heads, self.num_points)
        attention_weights = torch.sigmoid(attention_weights)
        
        # 采样
        value = value.permute(0, 3, 4, 1, 2).contiguous()  # [B, num_heads, head_dim, H_kv, W_kv]
        value = value.view(B * self.num_heads, self.head_dim, H_kv, W_kv)
        
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4).contiguous()
        sampling_locations = sampling_locations.view(B * self.num_heads, N_q, self.num_points, 2)
        
        # 双线性插值采样
        sampled_values = F.grid_sample(
            value,
            sampling_locations,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B * num_heads, head_dim, N_q, num_points]
        
        sampled_values = sampled_values.view(B, self.num_heads, self.head_dim, N_q, self.num_points)
        
        # 加权求和
        attention_weights = attention_weights.permute(0, 2, 1, 3)  # [B, num_heads, N_q, num_points]
        output = (sampled_values * attention_weights.unsqueeze(2)).sum(dim=-1)
        # [B, num_heads, head_dim, N_q]
        
        output = output.permute(0, 3, 1, 2).contiguous()  # [B, N_q, num_heads, head_dim]
        output = output.view(B, N_q, C)
        
        output = self.output_proj(output)
        output = self.dropout(output)
        
        return output
 
 
# =============================================================================
# Swin Encoder Block
# =============================================================================
 
class SwinEncoderBlock(nn.Module):
    """
    Swin Transformer Encoder Block
    
    结构 (随 swin_version 切换):
      V1 (Pre-Norm):  LN → Attn → 残差 → LN → FFN → 残差
      V2 (Post-Norm): Attn → LN → 残差 → FFN → LN → 残差
    
    ═══════════════════════════════════════════════════════════════════════════
    ★ [2026-05-09] v19 升级: 把 V14 的 LayerScale/SwiGLU 三件套引入 Swin
    ═══════════════════════════════════════════════════════════════════════════
    
    改动理由:
    ─────────
    v18 之前 SwinEncoderBlock 只用最朴素的 LayerNorm + GELU FFN, 没有现代化
    设计 (LayerScale/SwiGLU). 这导致 ViTEncoderBlock (Stage 4) 跟 SwinEncoderBlock
    (Stage 1-3) 的"特征行为风格"不一致, 从训练日志看也有以下证据:
      - Stage 1 std=0.90, Stage 2 std=1.17 (Epoch 199), 仍偏大且不稳
      - Stage 4 ViT 因为有 LayerScale γ 收尾稳定在 std=0.42
    
    v19 把 LayerScale + SwiGLU 也扩到 Swin block, 跟 ViT 保持一致:
      1. LayerScale γ (CaiT, Touvron et al., ICCV 2021):
         - per-channel 可学习参数, 初始化 layerscale_init (建议 1e-2)
         - V1: x = x + drop_path(γ * attn(LN(x)))
         - V2: x = shortcut + drop_path(γ * LN(attn(x)))
         - 防止深层梯度消失 + 抑制特征 std 漂移
      2. SwiGLU FFN (Shazeer 2020, LLaMA 2023, EVA-02 2023):
         - 门控 FFN: W3(SiLU(W1(x)) ⊙ W2(x)), hidden_dim = 8/3 * dim
         - 参数量基本不变, 表达能力提升
    
    ★ [2026-05-09] v19 升级: WindowAttention 双开关 (use_qk_norm + use_rope)
    ─────────────────────────────────────────────────────────────────────────
    SwinEncoderBlock 仅做参数透传, 实际逻辑在 WindowAttention.
    
    ═══════════════════════════════════════════════════════════════════════════
    ★ [2026-04-26 修改] 加入 swin_version 切换 Pre/Post Norm
    ═══════════════════════════════════════════════════════════════════════════
    
    修改原因:
    ─────────
    上一轮训练 (4.23 SwinV2 全套) mIoU 崩到 0.50, 根因之一:
    SwinV2 论文 (Liu et al., CVPR 2022) 必须三件套捆绑使用:
      1. Scaled Cosine Attention   (✓ 已做)
      2. Log-spaced CPB            (✓ 已做)
      3. Residual Post-Norm        (✗ 没做 ← 这次修复的就是这个)
    微软研究院实测: V1 Pre-Norm 一个 block 后激活幅值放大 61x,
    V2 Post-Norm 仅放大 3x. 上轮日志 stage1 std=1.70 就是 Pre-Norm 累积证据.
    
    更关键: V2 ckpt 的 norm1.weight 训练时归一化的是 attn 输出 (Post-Norm),
    但你套在 Pre-Norm 架构里, 它收到的是 attn 输入分布 → 语义错位 → 训练崩.
    
    "权重风格" 必须和 "forward 风格" 匹配:
      V1 ckpt + Pre-Norm forward  → ✅ 正确 (你之前 v15 的 0.8165 走这条)
      V2 ckpt + Post-Norm forward → ✅ 正确 (这次修复目标)
      V2 ckpt + Pre-Norm forward  → ❌ 上次崩盘的配置
      V1 ckpt + Post-Norm forward → ❌ 同样会崩
    
    设计:
    ─────
    - 新增 swin_version 参数 ('v1' | 'v2'), 默认 'v1' 保持向后兼容
    - forward 内部根据 self.swin_version 走不同分支
    - LayerNorm/Attn/FFN/DropPath 等 module 都不变, 仅顺序不同
    - LoRA adapter 挂在 Linear 层上, 跟 LN 位置无关, 自动兼容两种模式
    
    BiLevelWindowBlock 不受影响:
      它本身就是 Pre-Norm + LayerScale γ=0 设计 (随机初始化, 模块化嵌入),
      不需要切 Post-Norm. 入口的 norm_cross 会归一化掉 SwinEncoderBlock
      输出分布的差异 (Pre-Norm 的 std=1.7 或 Post-Norm 的 std=1.0 都被压回 1).
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数
        window_size: 窗口大小
        mlp_ratio: FFN 隐藏层维度比例
        qkv_bias: QKV 是否使用偏置
        drop: dropout
        attn_drop: 注意力 dropout
        drop_path: DropPath 比例
        swin_version: 'v1' (Pre-Norm) 或 'v2' (Post-Norm)
        
        ★ [2026-04-26 第二轮] pretrained_window_size 参数已删除.
           原因: WindowAttention 已经移除该参数, cpb_mlp 直接用 window_size 归一化.
           调用方 (multimodal.py) 不应再传这个参数.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,               # ★ [2026-06-29] SW-MSA 移位量, 0=纯W-MSA(原行为), ws//2=shifted-window
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        swin_version: str = 'v1',          # ★ [4.26] 新增, 默认 v1 向后兼容
        # ★ [2026-05-09] v19 新增参数 (默认全 False, 完全向后兼容 v18)
        use_layerscale: bool = False,
        layerscale_init: float = 1e-2,    # CaiT 用 1e-4 是给 24+ 层深 ViT, 4 层 Swin 用 1e-2 更合理
        use_swiglu: bool = False,
        use_qk_norm: bool = False,         # 透传到 WindowAttention
        use_rope: bool = False,            # 透传到 WindowAttention
        # ★ [2026-04-26 第二轮] pretrained_window_size 参数已删除
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        # ★ [2026-06-29] SW-MSA 移位量 (标准 Swin shifted-window). 0 时退化为纯 W-MSA (原行为).
        assert 0 <= shift_size < window_size, \
            f"shift_size({shift_size}) 必须满足 0 <= shift_size < window_size({window_size})"
        self.shift_size = shift_size
        self._attn_mask_cache = {}   # {(H_pad, W_pad): attn_mask} 懒构造 + 缓存
        self.mlp_ratio = mlp_ratio
        
        # ★ [4.26] 校验并存储 swin_version
        if swin_version not in ('v1', 'v2'):
            raise ValueError(f"swin_version must be 'v1' or 'v2', got '{swin_version}'")
        self.swin_version = swin_version
        # ★ [4.26 第二轮] self.pretrained_window_size 属性删除 (原代码这里有一行)
        
        # ★ [2026-05-09] v19 新增: 保存 LayerScale/SwiGLU 开关
        self.use_layerscale = use_layerscale
        self.use_swiglu = use_swiglu
        
        # LayerNorm (V1/V2 共享, 仅在 forward 中使用顺序不同)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Window Attention
        # ★ [2026-05-09] v19: 透传 use_qk_norm + use_rope 到 WindowAttention
        self.attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_qk_norm=use_qk_norm,
            use_rope=use_rope,
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # ★ [2026-05-09] v19: FFN 切 SwiGLU 或 GELU
        if use_swiglu:
            # SwiGLU 标准: hidden = 8/3 * dim, 参数量与 4x GELU FFN 相当
            self.ffn = SwiGLUFFN(
                in_features=dim,
                hidden_features=int(dim * 8 / 3),
                drop=drop
            )
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.ffn = FFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                drop=drop
            )
        
        # ★ [2026-05-09] v19: LayerScale 可学习残差缩放 (CaiT, Touvron et al., ICCV 2021)
        #   per-channel 参数, 初始化为 layerscale_init (推荐 1e-2),
        #   让 SwinEncoderBlock 跟 ViTEncoderBlock 保持一致的"残差行为"
        #   注: BiLevelWindowBlock 用的 γ_init=1e-4 是不同语义 (恒等映射保护),
        #       SwinEncoderBlock 这里 γ_init=1e-2 是常规 LayerScale (深层稳定).
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(dim))
        else:
            self.gamma_1 = None
            self.gamma_2 = None
    
    def _get_attn_mask(self, H: int, W: int, device) -> torch.Tensor:
        """
        ★ [2026-06-29] 标准 Swin shifted-window 注意力掩码 (Liu et al., ICCV 2021).
        循环位移后, 同一个窗口里会混入图像不相邻的区块; 用此掩码阻断这些跨边界注意.
        按 (H, W) 懒构造 + 缓存 (stage 分辨率固定, 实际只算一次).
        返回: [num_windows, ws*ws, ws*ws], 跨区为 -100, 同区为 0.
        """
        key = (H, W)
        cached = self._attn_mask_cache.get(key, None)
        if cached is not None and cached.device == device:
            return cached
        ws, ss = self.window_size, self.shift_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, ws)          # [nW, ws, ws, 1]
        mask_windows = mask_windows.view(-1, ws * ws)          # [nW, N]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, N, N]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self._attn_mask_cache[key] = attn_mask
        return attn_mask

    def _windowed_attn(self, x: torch.Tensor, B: int, H: int, W: int, C: int) -> torch.Tensor:
        """
        ★ [4.26] 抽取出来的窗口划分 + WindowAttention + 还原逻辑.
        在 V1/V2 forward 里复用, 避免代码重复.
        ★ [2026-06-29] 支持 shift_size>0 的 SW-MSA (循环位移 + 掩码); shift_size=0 时为原 W-MSA.

        Args:
            x: [B, H*W, C] (已经过 norm1 或没过, 取决于 V1/V2)
        Returns:
            [B, H*W, C] (已还原, 还没加残差也没过 norm1 — V1/V2 自己处理)
        """
        # 转为 4D: [B, H, W, C]
        x = x.view(B, H, W, C)
        
        # Padding (使 H, W 能被 window_size 整除)
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = H + pad_h, W + pad_w
        
        # ★ [2026-06-29] SW-MSA: shift_size>0 时循环位移 + 构造掩码 (注: 要求 ws 整除 H_pad/W_pad)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = self._get_attn_mask(H_pad, W_pad, x.device)
        else:
            shifted_x = x
            attn_mask = None
        
        # 窗口划分
        x_windows = window_partition(shifted_x, self.window_size)  # [B*num_win, ws, ws, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window Attention (shift 时带掩码)
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [B*num_win, ws*ws, C]
        
        # 窗口还原
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)  # [B, H_pad, W_pad, C]
        
        # ★ [2026-06-29] 反向循环位移 (还原回原始空间布局)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # 移除 padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # 转回 3D
        x = x.view(B, H * W, C)
        return x
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]
            H, W: 空间尺寸
            
        Returns:
            out: [B, H*W, C]
        """
        B, N, C = x.shape
        
        # ★ [4.26] 根据 swin_version 走不同的 Norm 顺序
        # ★ [2026-05-09] v19: 在 V1/V2 两条路径上都应用 LayerScale γ
        if self.swin_version == 'v1':
            # ╔═══════════════════════════════════════════════════════════╗
            # ║ V1 Pre-Norm:                                               ║
            # ║   shortcut + drop_path(γ_1 * attn(LN(x)))                  ║
            # ║   shortcut + drop_path(γ_2 * ffn(LN(x)))                   ║
            # ║ 适配: V1 Swin-T 预训练权重                                  ║
            # ╚═══════════════════════════════════════════════════════════╝
            
            # ============ Self-Attention (Pre-Norm) ============
            shortcut = x
            x = self.norm1(x)                              # LN 在 attn 之前
            x = self._windowed_attn(x, B, H, W, C)         # 窗口划分 + WindowAttention + 还原
            # ★ [2026-05-09] v19: 应用 LayerScale γ_1
            if self.gamma_1 is not None:
                x = self.gamma_1 * x
            x = shortcut + self.drop_path(x)               # 残差
            
            # ============ FFN (Pre-Norm) ============
            shortcut = x
            x = self.ffn(self.norm2(x))
            # ★ [2026-05-09] v19: 应用 LayerScale γ_2
            if self.gamma_2 is not None:
                x = self.gamma_2 * x
            x = shortcut + self.drop_path(x)
            
        else:  # self.swin_version == 'v2'
            # ╔═══════════════════════════════════════════════════════════╗
            # ║ V2 Post-Norm:                                              ║
            # ║   x + drop_path(γ_1 * LN(attn(x)))                         ║
            # ║   x + drop_path(γ_2 * LN(ffn(x)))                          ║
            # ║ 适配: SwinV2 预训练权重 (microsoft/swinv2-tiny-*)            ║
            # ╚═══════════════════════════════════════════════════════════╝
            
            # ============ Self-Attention (Post-Norm) ============
            shortcut = x
            x = self._windowed_attn(x, B, H, W, C)              # 直接 attn, 不先 LN
            x = self.norm1(x)                                    # ★ LN 包住 attn 输出
            # ★ [2026-05-09] v19: 应用 LayerScale γ_1
            if self.gamma_1 is not None:
                x = self.gamma_1 * x
            x = shortcut + self.drop_path(x)                     # 残差
            
            # ============ FFN (Post-Norm) ============
            shortcut = x
            x = self.ffn(x)                                      # 直接 ffn, 不先 LN
            x = self.norm2(x)                                    # ★ LN 包住 ffn 输出
            # ★ [2026-05-09] v19: 应用 LayerScale γ_2
            if self.gamma_2 is not None:
                x = self.gamma_2 * x
            x = shortcut + self.drop_path(x)                     # 残差
        
        return x
 
 
# =============================================================================
# Swin Decoder Block (原版，使用 CrossAttentionFast)
# =============================================================================
 
class SwinDecoderBlock(nn.Module):
    """
    Swin Transformer Decoder Block
    
    结构: 
    LayerNorm → Window Attention → 残差 
    → LayerNorm → Cross Attention → 残差 
    → LayerNorm → FFN → 残差
    
    注意: Cross Attention 是全局计算，高分辨率时显存占用大！
    建议高分辨率使用 SwinDeformableDecoderBlock
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        use_cross_pos_embed: bool = True
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Window Attention (Self)
        self.self_attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Cross Attention (全局)
        self.cross_attn = CrossAttentionFast(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_pos_embed=use_cross_pos_embed
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
        B, N, C = x.shape
        
        # ============ Self-Attention (Window) ============
        shortcut = x
        x = self.norm1(x)
        
        # 转为 4D
        x = x.view(B, H, W, C)
        
        # Padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = H + pad_h, W + pad_w
        
        # 窗口划分 → Attention → 窗口还原
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.self_attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)
        
        # 移除 padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # ============ Cross-Attention (全局) ============
        shortcut = x
        x = self.norm2(x)
        
        # Cross Attention: Q=x, K,V=aux
        x = self.cross_attn(
            query=x,
            key=aux,
            value=aux,
            H=H,
            W=W
        )
        
        x = shortcut + self.drop_path(x)
        
        # ============ FFN ============
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        
        return x
 
 
# =============================================================================
# Swin Deformable Decoder Block (新增，显存友好)
# =============================================================================
 
class SwinDeformableDecoderBlock(nn.Module):
    """
    Swin Transformer Decoder Block (使用 Deformable Cross Attention)
    
    结构: 
    LayerNorm → Window Attention → 残差 
    → LayerNorm → Deformable Cross Attention → 残差 (★ 门控)
    → LayerNorm → FFN → 残差
    
    ★ [2026-03-17] 新增 cross_attn_gate 门控:
       cross_attn 输出乘以 gate 系数后再做残差连接
       gate=0.0 → cross-attn 完全无效 (等同纯 Encoder)
       gate=1.0 → 完全融合 (原始行为)
       训练时由外部按 epoch 线性调度 0→1, 防止 Aux 随机噪声搅乱 RGB 主干
    
    与 SwinDecoderBlock 的区别:
    - Cross Attention 使用 Deformable Attention
    - 复杂度从 O(N²) 降到 O(N×K)
    - 高分辨率时显存友好
    
    Args:
        dim: 输入特征维度
        num_heads: 注意力头数
        window_size: 窗口大小
        num_points: Deformable Attention 采样点数
        mlp_ratio: FFN 隐藏层维度比例
        qkv_bias: QKV 是否使用偏置
        drop: dropout
        attn_drop: 注意力 dropout
        drop_path: DropPath 比例
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        num_points: int = 4,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_points = num_points
        
        # ★ [2026-03-17] 门控系数: 外部设置, 控制 cross-attn 融合强度
        self.cross_attn_gate = 1.0
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Window Attention (Self)
        self.self_attn = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        # Deformable Cross Attention (显存友好)
        self.cross_attn = DeformableCrossAttention(
            dim=dim,
            num_heads=num_heads,
            num_points=num_points,
            dropout=attn_drop
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
        B, N, C = x.shape
        
        # ============ Self-Attention (Window) ============
        shortcut = x
        x = self.norm1(x)
        
        # 转为 4D
        x = x.view(B, H, W, C)
        
        # Padding
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        H_pad, W_pad = H + pad_h, W + pad_w
        
        # 窗口划分 → Attention → 窗口还原
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.self_attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H_pad, W_pad)
        
        # 移除 padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # ============ Deformable Cross-Attention ============
        shortcut = x
        x = self.norm2(x)
        
        # Deformable Cross Attention: Q=x, K,V=aux
        x = self.cross_attn(
            query=x,
            key_value=aux,
            H_q=H,
            W_q=W,
            H_kv=H_aux,
            W_kv=W_aux
        )
        
        # ★ [2026-03-17] 门控: gate=0 时 cross-attn 无效, gate=1 时完全融合
        x = shortcut + self.drop_path(x) * self.cross_attn_gate
        
        # ============ FFN ============
        x = x + self.drop_path(self.ffn(self.norm3(x)))
        
        return x
 
 
# =============================================================================
# ViT Encoder Block
# =============================================================================
 
class ViTEncoderBlock(nn.Module):
    """
    ViT Encoder Block
    
    结构: LayerNorm → Global Attention → 残差 → LayerNorm → FFN → 残差
    
    ★ [2026-04-04] V14 升级 (仅影响 Aux ViT 和 RGB Stage3):
      1. LayerScale: 残差分支乘可学习标量 γ, 拯救深层梯度
         (CaiT — Touvron et al., ICCV 2021)
      2. SwiGLU FFN: 门控 FFN 替代 GELU FFN, 更强非线性
         (LLaMA, EVA-02)
      3. RMSNorm: 替代 LayerNorm, 速度更快
         (T5, LLaMA, EVA-02)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        max_relative_position: int = 64,
        # ★ [2026-04-04] V14 新增参数
        use_layerscale: bool = False,
        layerscale_init: float = 1e-4,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        use_flash_attn: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # ★ [2026-04-04] 归一化层: RMSNorm 或 LayerNorm
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # Global Attention
        self.attn = GlobalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            max_relative_position=max_relative_position,
            # ★ [2026-04-04] 传递 FlashAttention 和 RoPE 开关
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # ★ [2026-04-04] FFN: SwiGLU 或传统 GELU
        if use_swiglu:
            self.ffn = SwiGLUFFN(
                in_features=dim,
                # SwiGLU hidden = 8/3 * dim (与 4x FFN 参数量相当)
                hidden_features=int(dim * 8 / 3),
                drop=drop
            )
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.ffn = FFN(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                drop=drop
            )
        
        # ★ [2026-04-04] LayerScale: 可学习残差缩放因子
        #   初始化为 1e-4, 使深层 block 初期几乎不影响主干,
        #   随训练逐步增大, 解决深层 ViT 梯度消失问题
        #   (CaiT — Touvron et al., ICCV 2021)
        self.use_layerscale = use_layerscale
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(dim))
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]
            H, W: 空间尺寸
            
        Returns:
            out: [B, H*W, C]
        """
        if self.use_layerscale:
            # ★ [2026-04-04] LayerScale 路径: γ * block_output
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm2(x)))
        else:
            # 原始路径 (无 LayerScale)
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        
        return x
 
 
# =============================================================================
# ViT Decoder Block
# =============================================================================
 
class ViTDecoderBlock(nn.Module):
    """
    ViT Decoder Block
    
    结构:
    LayerNorm → Global Attention → 残差
    → LayerNorm → Cross Attention → 残差 (★ 门控)
    → LayerNorm → FFN → 残差
    
    ★ [2026-03-17] 新增 cross_attn_gate 门控 (同 SwinDeformableDecoderBlock)
    ★ [2026-04-04] V14: LayerScale + SwiGLU + RMSNorm 支持
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        max_relative_position: int = 64,
        use_cross_pos_embed: bool = False,
        # ★ [2026-04-04] V14 新增参数
        use_layerscale: bool = False,
        layerscale_init: float = 1e-4,
        use_swiglu: bool = False,
        use_rmsnorm: bool = False,
        use_flash_attn: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # ★ [2026-03-17] 门控系数
        self.cross_attn_gate = 1.0
        
        # ★ [2026-04-04] 归一化层: RMSNorm 或 LayerNorm
        norm_layer = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        
        # Global Attention (Self)
        self.self_attn = GlobalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            max_relative_position=max_relative_position,
            use_flash_attn=use_flash_attn,
            use_rope=use_rope,
        )
        
        # Cross Attention
        self.cross_attn = CrossAttentionFast(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_pos_embed=use_cross_pos_embed
        )
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # ★ [2026-04-04] FFN: SwiGLU 或传统 GELU
        if use_swiglu:
            self.ffn = SwiGLUFFN(in_features=dim, hidden_features=int(dim * 8 / 3), drop=drop)
        else:
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.ffn = FFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
        # ★ [2026-04-04] LayerScale (3 个残差分支各一个 γ)
        self.use_layerscale = use_layerscale
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_init * torch.ones(dim))
            self.gamma_2 = nn.Parameter(layerscale_init * torch.ones(dim))
            self.gamma_3 = nn.Parameter(layerscale_init * torch.ones(dim))
    
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
        # ============ Self-Attention (Global) ============
        if self.use_layerscale:
            x = x + self.drop_path(self.gamma_1 * self.self_attn(self.norm1(x), H, W))
        else:
            x = x + self.drop_path(self.self_attn(self.norm1(x), H, W))
        
        # ============ Cross-Attention ============
        shortcut = x
        x = self.norm2(x)
        
        x = self.cross_attn(
            query=x,
            key=aux,
            value=aux,
            H=H,
            W=W
        )
        
        # ★ [2026-03-17] 门控 + ★ [2026-04-04] LayerScale
        if self.use_layerscale:
            x = shortcut + self.drop_path(self.gamma_2 * x) * self.cross_attn_gate
        else:
            x = shortcut + self.drop_path(x) * self.cross_attn_gate
        
        # ============ FFN ============
        if self.use_layerscale:
            x = x + self.drop_path(self.gamma_3 * self.ffn(self.norm3(x)))
        else:
            x = x + self.drop_path(self.ffn(self.norm3(x)))
        
        return x
 
 
# =============================================================================
# 测试代码
# =============================================================================
 
if __name__ == "__main__":
    
    print("=" * 70)
    print("Transformer Block 测试")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 2
    H, W = 64, 64  # 高分辨率测试
    dim = 96
    num_heads = 3
    
    # 输入
    x = torch.randn(batch_size, H * W, dim, device=device)
    aux = torch.randn(batch_size, H * W, dim, device=device)
    
    # 测试 SwinEncoderBlock
    print("\n[1] SwinEncoderBlock")
    encoder = SwinEncoderBlock(
        dim=dim,
        num_heads=num_heads,
        window_size=7
    ).to(device)
    
    params = sum(p.numel() for p in encoder.parameters())
    print(f"参数量: {params / 1e6:.4f}M")
    
    with torch.no_grad():
        out = encoder(x, H, W)
    print(f"输入: {x.shape} -> 输出: {out.shape}")
    
    # 测试 SwinDeformableDecoderBlock (显存友好)
    print("\n[2] SwinDeformableDecoderBlock (显存友好)")
    deform_decoder = SwinDeformableDecoderBlock(
        dim=dim,
        num_heads=num_heads,
        window_size=7,
        num_points=4
    ).to(device)
    
    params = sum(p.numel() for p in deform_decoder.parameters())
    print(f"参数量: {params / 1e6:.4f}M")
    
    with torch.no_grad():
        out = deform_decoder(x, H, W, aux, H, W)
    print(f"输入: x={x.shape}, aux={aux.shape} -> 输出: {out.shape}")
    
    # 显存对比测试
    if device == 'cuda':
        print("\n[3] 显存对比 (256×256)")
        
        H_large, W_large = 256, 256
        x_large = torch.randn(1, H_large * W_large, dim, device=device)
        aux_large = torch.randn(1, H_large * W_large, dim, device=device)
        
        # Deformable 版本
        torch.cuda.reset_peak_memory_stats()
        deform_decoder_large = SwinDeformableDecoderBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=7,
            num_points=4
        ).to(device)
        
        with torch.no_grad():
            out = deform_decoder_large(x_large, H_large, W_large, aux_large, H_large, W_large)
        
        mem_deform = torch.cuda.max_memory_allocated() / 1024**3
        print(f"SwinDeformableDecoderBlock (256×256): {mem_deform:.3f} GB ✓")
        
        # 原版会 OOM，所以跳过
        print("SwinDecoderBlock (256×256): 会 OOM (跳过)")
    
    print("\n" + "=" * 70)
    print("✅ 测试通过!")
    print("=" * 70)
