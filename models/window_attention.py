"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ ★ [2026-06-29] 本次改动 (shift-window 对照实验支持):                          ║
║   WindowAttention.forward 新增 mask 形参 (Optional[Tensor]=None):           ║
║   QK-Norm 与 cosine/FP32 两条路径都在 softmax 前注入 SW-MSA 窗口掩码.        ║
║   mask=None 时与改动前逐字节一致 (默认所有不传 mask 的调用方不受影响).        ║
╚═══════════════════════════════════════════════════════════════════════════╝

Window Attention Module - v19 (2026-05-09 升级)
============================================================================
窗口内多头自注意力, 支持三种位置编码 + 三种 attention 内核.

╔══════════════════════════════════════════════════════════════════════════╗
║ ★ [2026-05-09] v19 重大升级: 引入 QK-Norm + 2D RoPE 双开关                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  历史背景 (v17/v18):                                                     ║
║  ───────────────                                                         ║
║  v17/v18 用 SwinV2 的 Scaled Cosine Attention:                           ║
║      attn = (F.normalize(q) @ F.normalize(k).T) * exp(logit_scale)       ║
║  + Log-CPB (cpb_mlp) 生成相对位置偏置.                                   ║
║  4.26 round-2 把 logit_scale 初值从 log(10) 改为 0 (即 τ=1) 避免梯度爆炸. ║
║  整段用 autocast(enabled=False) 强制 FP32.                               ║
║  这套组合训练 200 epoch 稳定收敛, 但仍有 max_grad ~216 偶发尖峰.         ║
║                                                                          ║
║  v19 新增的两个改动:                                                     ║
║  ───────────────                                                         ║
║                                                                          ║
║  1. QK-Norm 替代 Scaled Cosine Attention (use_qk_norm=True 启用)         ║
║     - 经典 dot-product attention: attn = (q @ k.T) * head_dim^(-0.5)     ║
║     - 但在算 attn 之前对 q, k 各做 LayerNorm:                            ║
║         q_normed = LayerNorm(q),  k_normed = LayerNorm(k)                ║
║     - 优点 vs Cosine Attention:                                          ║
║         * 没有可学习温度 logit_scale 这个怪异超参                        ║
║         * 没有 1/||q|| 的反向梯度爆炸风险                                ║
║         * 跟当代 LLM (LLaMA, Gemini) 实践一致                            ║
║         * 数值稳定性好, 不需要强制 FP32 (autocast 可以正常用)            ║
║     - 参考论文:                                                          ║
║         * Henry et al., EMNLP Findings 2020 (QK-Norm 最早系统研究)        ║
║         * Dehghani et al., ICML 2023 (22B ViT 训练靠 QK-Norm 稳定)        ║
║         * Wortsman et al., ICLR 2024 (系统证明 QK-Norm 防 logit 爆炸)     ║
║                                                                          ║
║  2. 2D RoPE 替代 Log-CPB (use_rope=True 启用)                            ║
║     - 旋转位置编码直接编码 q, k 的方向, 不是加 bias                      ║
║     - 优点 vs Log-CPB:                                                   ║
║         * 零参数 (固定 sin/cos 函数, 不需要 cpb_mlp 那 ~5 万参数/stage)   ║
║         * 平移不变性 (残膜在图像任何位置都同样响应)                      ║
║         * 分辨率外推 (训练 ws=8 推理 ws=16 也能用)                       ║
║     - 在 window 里 RoPE 编码的是窗口内坐标 [0, ws), 不是全图坐标         ║
║     - 参考论文:                                                          ║
║         * Su et al., RoFormer 2021 (RoPE 原始)                           ║
║         * Fang et al., EVA-02 2023 (2D RoPE 在 ViT)                      ║
║         * Ryali et al., ICML 2023 (RoPE 在 hierarchical Transformer)      ║
║                                                                          ║
║  互斥规则:                                                               ║
║  ────────                                                                ║
║   * use_qk_norm 和 cosine attention 互斥 (二选一)                        ║
║   * use_rope 和 Log-CPB 互斥 (二选一)                                    ║
║   * 两个开关可以独立组合 (4 种模式), 用于消融实验                        ║
║   * 默认全 False 时退化到 v18 行为, 完全向后兼容                         ║
║                                                                          ║
║  v19 默认配置 (推荐):                                                    ║
║   use_qk_norm=True, use_rope=True                                        ║
║   → 完全去掉 logit_scale + cpb_mlp, 单 stage 节省 ~5 万参数              ║
║                                                                          ║
║  ────────────────────────────────────────────────────────────            ║
║  原 v18 注释保留在下方 (Scaled Cosine + Log-CPB 那部分).                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import math
from typing import Optional   # ★ [2026-06-29] for SW-MSA mask 形参类型
import torch
import torch.nn as nn
import torch.nn.functional as F

# ★ [2026-05-09] v19: 复用 global_attention 的 RoPE 实现 (单一来源)
from .global_attention import RotaryPositionEncoding2D


class WindowAttention(nn.Module):
    """
    窗口内多头自注意力 — v19 多模式版本
    
    支持四种 (内核 × 位置编码) 组合:
      ┌─────────────────┬─────────────────┬───────────────────────┐
      │                 │ use_rope=False  │ use_rope=True         │
      ├─────────────────┼─────────────────┼───────────────────────┤
      │ qk_norm=False   │ Cosine + LogCPB │ Cosine + RoPE          │
      │  (v18 行为)      │ (v18 默认)      │ (实验组合, 不推荐)     │
      ├─────────────────┼─────────────────┼───────────────────────┤
      │ qk_norm=True    │ QK-Norm+LogCPB  │ QK-Norm+RoPE           │
      │                 │ (实验组合)      │ (v19 推荐, 全 LLM 风格) │
      └─────────────────┴─────────────────┴───────────────────────┘
    
    Args:
        dim: 输入特征维度
        window_size: 窗口大小, 默认 7
        num_heads: 注意力头数
        qkv_bias: QKV 投影是否使用偏置
        attn_drop: 注意力 dropout
        proj_drop: 输出投影 dropout
        use_qk_norm: ★ [2026-05-09] 启用 QK-Norm 替代 Cosine Attention
        use_rope: ★ [2026-05-09] 启用 2D RoPE 替代 Log-CPB
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        # ★ [2026-05-09] v19 新增双开关
        use_qk_norm: bool = False,
        use_rope: bool = False,
    ):
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # ★ [2026-05-09] v19 新增: 保存模式开关
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        
        # ─────────────────────────────────────────────────────────────────
        # ★ [2026-05-09] v19: Attention 内核选择
        # ─────────────────────────────────────────────────────────────────
        if use_qk_norm:
            # === QK-Norm 路径 ===
            #   q_norm = LayerNorm(q),  k_norm = LayerNorm(k)
            #   attn = (q_norm @ k_norm.T) * head_dim^(-0.5)
            #   不需要 logit_scale (经典 scale 即可保证数值稳定)
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
            self.scale = self.head_dim ** -0.5
            # 占位: 不创建 logit_scale
            self.logit_scale = None
        else:
            # === Cosine Attention 路径 (v18 行为) ===
            # ★ [v11/v18] Scaled Cosine Attention: 可学习温度参数 (per-head)
            # ★ [2026-04-26] 初值改为 0 (即 τ = exp(0) = 1)
            self.logit_scale = nn.Parameter(
                torch.zeros((num_heads, 1, 1))   # log(1) = 0 → τ = exp(0) = 1
            )
            # 占位: 不创建 q_norm/k_norm
            self.q_norm = None
            self.k_norm = None
            self.scale = None  # cosine 路径不用 scale
        
        # ─────────────────────────────────────────────────────────────────
        # ★ [2026-05-09] v19: 位置编码选择
        # ─────────────────────────────────────────────────────────────────
        if use_rope:
            # === 2D RoPE 路径 ===
            # window 内编码 [0, ws) 的相对位置, 跟 stage 4 ViT 用同一个 RoPE 类
            # 注: head_dim 必须能被 4 整除 (RoPE 把 head_dim 拆 4 段: y_cos, y_sin, x_cos, x_sin)
            if self.head_dim % 4 != 0:
                raise ValueError(
                    f"[v19] use_rope=True 要求 head_dim % 4 == 0, "
                    f"但 head_dim={self.head_dim} (dim={dim}, num_heads={num_heads})"
                )
            self.rope = RotaryPositionEncoding2D(dim=self.head_dim)
            # 占位: 不创建 cpb_mlp 等 Log-CPB 相关结构
            self.cpb_mlp = None
        else:
            # === Log-CPB 路径 (v18 行为) ===
            self.rope = None
            self.cpb_mlp = nn.Sequential(
                nn.Linear(2, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_heads, bias=False)
            )
            
            # 预计算 log-spaced 相对坐标表 (register_buffer, 不可训练)
            relative_coords_table = self._build_log_spaced_coords(window_size)
            self.register_buffer("relative_coords_table", relative_coords_table)
            
            # 预计算相对位置索引 (用于从坐标表中查找对应的坐标对)
            self.register_buffer(
                "relative_position_index",
                self._build_relative_position_index()
            )
        
        # QKV 投影 (与 V1/V2 一致, 可加载预训练)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影 (与 V1/V2 一致)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Softmax
        self.softmax = nn.Softmax(dim=-1)
    
    def _build_log_spaced_coords(self, window_size: int) -> torch.Tensor:
        """
        构建 Log-spaced 相对坐标表 (仅 use_rope=False 时使用)
        
        SwinV2 论文公式 (4):
          Δ_cx = sign(Δx) · log(1 + |Δx|)
          Δ_cy = sign(Δy) · log(1 + |Δy|)
        
        Returns:
            relative_coords_table: [(2*ws-1)*(2*ws-1), 2]
        """
        ws = window_size
        coords_h = torch.arange(-(ws - 1), ws, dtype=torch.float32)
        coords_w = torch.arange(-(ws - 1), ws, dtype=torch.float32)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij')
        ).permute(1, 2, 0).contiguous()
        coords_log = torch.sign(coords) * torch.log2(1.0 + coords.abs())
        normalizer = math.log2(ws) if ws > 1 else 1.0
        if normalizer > 0:
            coords_log = coords_log / normalizer
        coords_log = coords_log.reshape(-1, 2)
        return coords_log
    
    def _build_relative_position_index(self) -> torch.Tensor:
        """
        预计算窗口内所有 token 对的相对位置索引 (仅 use_rope=False 时使用)
        
        Returns:
            relative_position_index: [ws*ws, ws*ws]
        """
        ws = self.window_size
        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij')
        )
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: 窗口内的 tokens [B*num_windows, ws*ws, C]
            mask: ★ [2026-06-29] SW-MSA 注意力掩码 [num_windows, N, N] 或 None.
                  非 None 时阻断 shift 后跨边界窗口的注意 (标准 Swin shifted-window).
                  仅当上层 SwinEncoderBlock 设了 shift_size>0 才会传入.
            
        Returns:
            out: [B*num_windows, ws*ws, C]
        """
        B_, N, C = x.shape
        ws = self.window_size
        orig_dtype = x.dtype
        
        # QKV 投影: [B_, N, 3C] → [3, B_, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # ╔═══════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-05-09] v19: 根据 use_qk_norm 走两条路径                    ║
        # ║   - QK-Norm 路径: 数值稳定, 可在 autocast 默认精度下运行            ║
        # ║   - Cosine 路径: 强制 FP32 (沿用 v18 防爆方案)                      ║
        # ╚═══════════════════════════════════════════════════════════════════╝
        if self.use_qk_norm:
            # ─────────────────────────────────────────────────────────────
            # ★ [2026-05-09] QK-Norm 路径
            # ─────────────────────────────────────────────────────────────
            # 1) (可选) RoPE 旋转: 在 LayerNorm 之前做更合理 (RoPE 是相位旋转,
            #    LN 后再做相当于改变 LN 的方差归一化前提, 数值上不等价)
            if self.use_rope and self.rope is not None:
                # 注: window 内 H=W=ws
                q, k = self.rope(q, k, ws, ws)
            
            # 2) QK-Norm: 对 q, k 各做 LayerNorm (per head_dim)
            q = self.q_norm(q)
            k = self.k_norm(k)
            
            # 3) 经典 dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, num_heads, N, N]
            
            # 4) 位置偏置 (仅当 use_rope=False 时, 即 QK-Norm + Log-CPB 实验组合)
            if not self.use_rope and self.cpb_mlp is not None:
                relative_coords = self.relative_coords_table[
                    self.relative_position_index.view(-1)
                ]  # [N*N, 2]
                relative_position_bias = self.cpb_mlp(relative_coords)
                relative_position_bias = relative_position_bias.view(N, N, -1)
                relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                attn = attn + relative_position_bias.unsqueeze(0)
            
            # ★ [2026-06-29] SW-MSA 窗口掩码: shift 后阻断跨边界注意 (mask 由 SwinEncoderBlock 计算)
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N, N)
            # 5) Softmax + Dropout (QK-Norm 路径数值稳定, 不需要 FP32)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            
            # 6) 加权求和
            out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        else:
            # ─────────────────────────────────────────────────────────────
            # Cosine Attention 路径 (v18 行为, 强制 FP32)
            # ─────────────────────────────────────────────────────────────
            # ╔════════════════════════════════════════════════════════════╗
            # ║ ★ [2026-04-26] V2 cosine attention 整块 FP32 执行           ║
            # ║   原因: F.normalize 在 FP16 下 eps=1e-12 失效, 反向 1/||q||  ║
            # ║         项极易溢出. 这是 4 月份 rgb_s1=inf 的根因.          ║
            # ║   方案: autocast(enabled=False) 强制 FP32. 速度损失 < 3%.   ║
            # ╚════════════════════════════════════════════════════════════╝
            with torch.amp.autocast('cuda', enabled=False):
                q_fp32 = q.float()
                k_fp32 = k.float()
                v_fp32 = v.float()
                
                # ★ [2026-05-09] v19: Cosine 路径下也支持 RoPE (实验组合)
                if self.use_rope and self.rope is not None:
                    q_fp32, k_fp32 = self.rope(q_fp32, k_fp32, ws, ws)
                
                # L2 normalize → cosine similarity → 乘 τ
                q_fp32 = F.normalize(q_fp32, dim=-1)
                k_fp32 = F.normalize(k_fp32, dim=-1)
                
                logit_scale = torch.clamp(
                    self.logit_scale.float(),
                    min=math.log(0.01),
                    max=math.log(100.0)
                ).exp()
                
                attn = (q_fp32 @ k_fp32.transpose(-2, -1)) * logit_scale
                
                # Log-CPB 位置偏置 (仅当 use_rope=False)
                if not self.use_rope and self.cpb_mlp is not None:
                    relative_coords = self.relative_coords_table[
                        self.relative_position_index.view(-1)
                    ].float()
                    relative_position_bias = self.cpb_mlp(relative_coords)
                    relative_position_bias = relative_position_bias.view(N, N, -1)
                    relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
                    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                    attn = attn + relative_position_bias.unsqueeze(0)
                
                # ★ [2026-06-29] SW-MSA 窗口掩码 (cosine/FP32 路径)
                if mask is not None:
                    nW = mask.shape[0]
                    attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.to(attn.dtype).unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
                attn = self.attn_drop(attn)
                
                out_fp32 = (attn @ v_fp32).transpose(1, 2).reshape(B_, N, C)
            
            # FP32 计算结束, 转回原 dtype
            out = out_fp32.to(orig_dtype)
        
        # 输出投影 (走 autocast 默认精度)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图划分为不重叠的窗口
    
    Args:
        x: [B, H, W, C]
        window_size: 窗口大小
        
    Returns:
        windows: [B * num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows: torch.Tensor, 
    window_size: int, 
    H: int, 
    W: int
) -> torch.Tensor:
    """
    将窗口还原为特征图
    
    Args:
        windows: [B * num_windows, window_size, window_size, C]
        window_size: 窗口大小
        H, W: 原始特征图尺寸
        
    Returns:
        x: [B, H, W, C]
    """
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = windows.shape[0] // (num_windows_h * num_windows_w)
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x
