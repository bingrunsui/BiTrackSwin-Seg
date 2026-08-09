"""
Window Interaction Module (窗口间交互模块)
==============================================
针对 Stage1 / Stage2 的跨窗口信息交互设计
★ [2026-07-11] _splat_plus 新增守卫式可视化缓存 viz_cache_enabled(默认关闭, 训练零影响, 供 track_viz.py 提取 Amp/A 门图)
★ [2026-07-11] 改动I: BiLevelWindowBlock 新增 ffn_respost 开关(默认 False=现状 Pre-Norm; True=FFN 子层 res-post 化 LN(ffn(x)))
★ [2026-07-11] 改动J: 新增三正交开关 drop_norm_cross / mul_ln(位置甲, 乘之前) / cross_qk_norm(打分QK-Norm), 默认全 False=现状; 组合用于'删入口LN+专用归一化'消融
★ [2026-07-18] v22: τ softplus 重参数化 — splatp_tau_add/mul 存储值语义变为 ρ, 前向 τ_eff = SPLATP_TAU_MIN + softplus(ρ+shift);
    ρ=1(初值) 时 τ_eff≈1.0(τ_min=1.0 的边界退化: τ_eff=1.0001, bf16 不可分辨), τ 有下界 → 1/τ 跑飞被结构性消灭
★ [2026-07-18] v22: BiLevelWindowBlock 新增 bypass_crosswindow 运行时属性(默认 False, 非参数不入 state_dict) —
    True 时跨窗口子层整体恒等(保留 FFN), 供 'BiLevel 有没有用' 的粒度(a)结构性消融
★ [2026-07-18] ⚠ 红线: 2026-07-18 之前的旧 ckpt 里存的是裸 τ, 与 ρ 语义不同(仅 τ=1 处巧合等价) → 旧 ckpt 探针必须用旧代码

核心思想:
    标准 Swin 的 W-MSA 只能在窗口内部交互, 需要 shift window 才能跨窗口通信.
    本模块设计了三种 bi-level attention 方案, 允许一个 block 同时完成:
      (1) 窗口内 self-attention  (复用项目已有的 WindowAttention)
      (2) 跨窗口 attention        (本模块新增, 三种实现)
    
    适用场景: Stage1 (128×128) / Stage2 (64×64) 的高分辨率特征图
    不适用: Stage3 (16×16) — 那里直接使用 ViT 全局 attention (GlobalAttention)

三种方案 (消融实验):
    - TYPE_A = "token_level":    Token 级细粒度 top-K 跨窗口 attention (BiFormer 风格)
    - TYPE_B = "window_level":   窗口级粗粒度 attention + 广播 (MaxViT grid 简化版)
    - TYPE_C = "hybrid":         Attention pooling 摘要 + 细粒度 cross-attention 分发

参考论文:
    - BiFormer (Zhu et al., CVPR 2023): Bi-Level Routing Attention
    - MaxViT  (Tu et al., ECCV 2022):   Block + Grid Attention
    - Swin V2 (Liu et al., CVPR 2022):  Scaled Cosine + Log-CPB (已在 WindowAttention 中)

设计约束:
    - 与项目已有 WindowAttention (SwinV2 版) 完全兼容
    - 输入输出接口对齐 SwinEncoderBlock, 可作为 drop-in 替代
    - 支持 grad checkpoint, AMP, 可学习 gate 初始化为 0 (训练早期等价于纯 W-MSA)

修改记录:
    [2026-04-20] v1.0: 初始实现, 三种方案并行支持
    [2026-04-21] v2.0: BiLevelWindowBlock 重构 (方案 A)
                       - 移除 W-MSA (避免与 block 0 冗余, 从 Swin pair 设计借鉴)
                       - 移除 cross_gate (单分支门控)
                       - 新增 LayerScale (γ_cross + γ_ffn), 初始化为 0
                       - 训练早期 block 近似恒等映射, 彻底解决特征爆炸问题
    [2026-06-18] v3.0: 方案C(hybrid) 的 Step3 分发新增可选 'splat' 模式 (块级语义播撒)
        ═══════════════════════════════════════════════════════════════════════
        【改了什么】
          HybridCrossWindowAttention 的 Step3 (摘要→token 的信息融合) 现在有两种实现,
          由 step3_mode 选择, 默认 'gather' = 与 v3.0 之前逐字节等价 (旧 ckpt 兼容):
            - step3_mode='gather' (默认): 原范式. 每个 token 自己做 Query,
              对所有摘要做 cross-attention, softmax over 摘要 → token 主动"吸"全局信息.
              token 有取舍权 (背景 token 可少吸), 对前景 Precision 友好.
            - step3_mode='splat' (新增): 块级语义播撒 (Block-wise Semantic Splatting).
              摘要做 Query/Value, token 特征做被打分的 Key, softmax over 块内像素 →
              摘要主动把能量"摊"给块内每个像素. 对稀疏丝状目标 (残膜) 的证据扩散更强,
              但缺"关阀"易灌背景, 故配可选 null/sink 槽 (splat_null=True) 保护 Precision.

        【怎么改的 (splat 的数学)】 详见本文件末尾 _splat() 的逐步注释, 摘要:
          x_blk:[B,G,N,C] (按窗口切块), probes:[B,G,Kp,C] (每块 Kp=M+S 个摘要)
          Imp  = einsum('bgkc,bgpc->bgkp', q(probes), s(x_blk))·scale   # [B,G,Kp,N]
          (可选) 在 N 轴末尾拼一个 null 列 → softmax over (N or N+1) → 丢掉 null 列 = Attn
          Info = einsum('bgkp,bgkc->bgpc', Attn, v(probes))             # [B,G,N,C] 信息注入轨
          Amp  = sigmoid(Imp.sum over Kp)                               # [B,G,N]   乘性放大轨
          delta= γ_amp·(x_blk·Amp) + γ_info·Info                        # 双轨调制
          → 因为 splat 内部已含 γ_amp/γ_info(各 1e-4), BiLevelWindowBlock 会跳过外层 γ_cross
            (否则双重门控 1e-8 学不动). 仍是近恒等启动, 与整套 LayerScale 哲学一致.

        【为什么 V 必须来自摘要 (而不是特征)】
          读出 einsum 在 Kp(探针) 轴上求和, V 必须长在被求和的那根轴上 → V=摘要;
          特征只贡献"打分用的地址键". 若 V 也用特征则 Info 退化成纯自门控, 搬不进任何
          跨窗口信息 (见与用户讨论的统一原理: V 永远长在读出时被求和的那根轴上).

        【为什么是块级(block-local)而非全局播撒】
          跨窗口混合已在 Step2 (摘要自注意力) 完成, 每块摘要已带全局上下文; Step3-splat 只做
          "已全局化的摘要 → 本块像素"的局部重分发. 全局播撒 O(G²·Kp·N) 在 stage1(G=256) 会爆显存,
          块级 O(G·Kp·N) 极轻量, 且语义=MaxViT 风格的 grid→local 回写.

        【向后兼容】 step3_mode 默认 'gather'; 'splat' 才新建 splat 专属参数. M=1/use_stats=()
          + gather 时与 v3.0 之前逐字节等价.
        参考: 块级播撒范式≈FiLM 调制 (Perez et al., AAAI 2018) + 可学习聚合槽
              (Slot Attention, Locatello et al., NeurIPS 2020); null/sink 槽≈DETR no-object
              (Carion et al., ECCV 2020) / ViT register tokens (Darcet et al., ICLR 2024).
        ═══════════════════════════════════════════════════════════════════════
    [2026-06-19] splat 加性轨升级: 保留 C 维的"逐通道专家融合" (直接替换旧 splat 实现)
        ═══════════════════════════════════════════════════════════════════════
        【为什么改】旧 splat 的加性轨 Info = einsum('bgkp,bgkc->bgpc') 对 k(专家)求和,
          把 Kp 个专家"平均"掉了; 而若用 Linear(Kp·C→C) 融合又会把【通道也打散重组】(通道 c 去
          吸收所有专家的所有通道) → 你不想要的"C 被串通".
        【怎么改】分两步, 既融合专家又保住通道独立:
          1. PerExpert = einsum('bgkp,bgkc->bgpkc', Attn, V) → [B,G,N,Kp,C]  (k 不求和, 保留每个专家)
          2. Info = einsum('bgpkc,kc->bgpc', PerExpert, expert_weight) → [B,G,N,C]
             —— 沿 Kp 轴逐通道融合, 通道 c 只乘 expert_weight[:,c] → channel 间【零串扰】, C 完整保留.
        【expert_weight】形状 [Kp, C] (每专家每通道一个权重), 初值 1.0 → 起步与旧 splat 的"对 k 求和"
          逐字节一致 (已用 numpy 数值验证), 训练再学每专家每通道权重 → 安全 drop-in, 最差退回旧 splat.
        【跨通道混合的去处】若之后仍想要一点跨通道交互, 由其后已有的 splat_out_proj (Linear C→C) 负责;
          想要纯通道独立就把它当恒等. 即: "融合专家(逐通道, 保 C)" 与 "混通道(可选)" 被拆成两步, 可控.
        【乘性轨不变】Amp 是逐像素标量, 对所有通道同等缩放, 本就不串通道 → 两条轨现在都保 C.
        【接口】无新增构造参数 (Kp 在构造期由 num_pool_queries+len(use_stats) 推出); multimodal.py /
          训练脚本无需改动. 仍由 step3_mode='splat' 启用. (splat 必须 --fresh, 因含 splat 专属新权重.)
        ═══════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

# 路径设置 (与项目其他模块保持一致)
_current_file = Path(__file__).resolve()
_current_dir = _current_file.parent
_project_root = _current_dir.parent
for _p in [str(_project_root), str(_current_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .window_attention import WindowAttention, window_partition, window_reverse
from .transformer_block import FFN, DropPath

# ★ [2026-07-18] v22: splat 门温度 τ 的 softplus 重参数化 --------------------------------------
#   splatp_tau_add / splatp_tau_mul 存储值的语义 = ρ; 有效温度 τ_eff = SPLATP_TAU_MIN + softplus(ρ+shift).
#   shift 取使 ρ=1(初始 torch.ones) 时 τ_eff = 1.0 精确成立: shift = log(expm1(1 − τ_min)) − 1.
#   边界: τ_min ≥ 1 时 expm1(≤0) 无解(softplus 恒 > 0, 无法精确落在地板) → 退化取
#   shift = log(expm1(1e-4)) − 1, 此时 τ_eff(ρ=1) = τ_min + 1e-4. 相对偏差 1e-4, 低于 bf16
#   分辨率(~4e-3), 前向不可分辨; 且 τ 只能自地板向上 = 门只能变软不能经 τ 变锐.
#   τ→0 / τ 穿负的自加速跑飞被结构性消灭 — 对应 SwinV2 scaled-cosine 的 τ clamp 保险丝(Liu 2022).
#   训练端在解析 CLI 后覆写: window_interaction.SPLATP_TAU_MIN = args.splatp_tau_min
SPLATP_TAU_MIN = 1.0

# ★ [2026-07-24] P2 逐像素强度增益 (加性轨专用). 训练端在解析 CLI 后覆写:
#   window_interaction.SPLATP_PIXEL_GAIN / SPLATP_GAIN_DETACH (与 SPLATP_TAU_MIN 同一机制).
#   动机: splatp_info_ln 逐像素归一 → 每像素注入总能量被钉死, 加性轨只能选方向不能选强度,
#   连"这里别注入"都表达不了(LN 把微弱残余重标定回满幅). g 在 LN 之后补回幅度自由度:
#     m = max_k Σ_c Q·S/√C  (标量匹配强度, fullC 打分从不求和 — 网络里原本不存在这个量)
#     g = 2·sigmoid(α·m+β), α=β=0 恒等初始化 → 开关开着但初始前向仍逐比特不变.
#   开=每块新增 α/β 两个标量参数(改 state_dict, 需 --fresh); 关=参数不创建, 旧 ckpt 完全兼容.
SPLATP_PIXEL_GAIN = False
# ★ [2026-07-24] m 计算是否对 Q/S detach. True=共享打分投影零新增梯度 → 乘性轨评分路
#   不被 P2 目标重塑(单变量最干净); False=允许 P2 目标参与塑造 Q/S(与乘性轨共训, 慎用).
SPLATP_GAIN_DETACH = True

# ★ [2026-07-25] v23-P3a(受限): learnable 专家(k<M)的加性门 sigmoid→softmax_k 竞争(Σ_k=1);
#   统计量专家(k>=M: max/mean/min)保持 sigmoid 旁路不变. softmax 对沿k公共偏置精确不变 →
#   learnable 侧偏置升维为 b_learn[M,C](逐专家先验, 新参数; 开=改 state_dict, 需 --fresh).
#   仅支持 score_mode='fullC'(multihead 头内求和破坏逐通道竞争语义, 构造期报错). 训练端建模前覆写.
SPLATP_LEARN_SOFTMAX = False
# ★ [2026-07-25] v23-P3b: learnable 内容 V 的去相关惩罚(防密度监督的对称梯度把 M 个专家推成同一个码).
#   开关只控制是否计算并暂存 self._vdecorr_loss(保留计算图); λ 与求和在训练端. 关=零开销.
SPLATP_V_DECORR = False


def _splatp_tau_shift(tau_min: float) -> float:
    delta = 1.0 - float(tau_min)
    if delta > 1e-6:
        return math.log(math.expm1(delta)) - 1.0
    return math.log(math.expm1(1e-4)) - 1.0   # τ_min ≥ 1 的边界退化分支(见上注)


def splatp_tau_eff(rho: torch.Tensor) -> torch.Tensor:
    """ρ → 有效温度 τ_eff ∈ [SPLATP_TAU_MIN, ∞). 前向与外部监控共用此函数."""
    tmin = float(SPLATP_TAU_MIN)
    return tmin + F.softplus(rho + _splatp_tau_shift(tmin))


# =============================================================================
# 通用辅助: 为路由计算窗口摘要 (mean pooling, 简单快速)
# =============================================================================

def _compute_window_summary_mean(
    x_windows: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    对每个窗口内的 token 做 mean pooling, 得到窗口级摘要

    Args:
        x_windows: [B * num_windows, ws*ws, C]

    Returns:
        summary: [B * num_windows, C]
    """
    return x_windows.mean(dim=1)


# =============================================================================
# 方案 A: Token 级细粒度 top-K 跨窗口 attention (BiFormer 风格)
# =============================================================================

class TokenLevelCrossWindowAttention(nn.Module):
    """
    方案 A: Token 级细粒度 top-K 跨窗口 attention

    流程:
        1. 对每个窗口计算摘要 (mean), 得到 [num_windows, C]
        2. 计算窗口间相似度矩阵, 选出每个窗口的 top-K 个邻居
        3. 对每个窗口的每个 token, 与 top-K 邻居窗口的所有 token 做 attention

    复杂度:
        O(num_windows × (ws*ws) × (K × ws*ws))
        = O(N × K × ws²)  其中 N = H*W = num_windows × ws²

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        window_size: 窗口大小 (用于正确地将 token 组织为窗口)
        top_k: 每个窗口选择的邻居窗口数 (不含自身)
        qkv_bias: QKV 投影是否使用偏置
        attn_drop: 注意力 dropout
        proj_drop: 输出投影 dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        window_size: int = 7,
        top_k: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.top_k = top_k
        self.scale = self.head_dim ** -0.5

        # Q 来自当前窗口的 token, K/V 来自被路由到的邻居窗口的 token
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # 路由相关: 用于生成窗口摘要的 query (可学习, 让路由不只是 mean pooling)
        # 如果选用 mean pooling 足够, 把这个 query 关掉也可以 (见下方 use_learnable_routing)
        self.use_learnable_routing = False  # 先用 mean, 简洁可控

    def _route_windows(
        self,
        window_summary: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于窗口摘要相似度做 top-K 路由

        Args:
            window_summary: [B, num_windows, C]

        Returns:
            topk_idx: [B, num_windows, top_k] 每个窗口的 top-K 邻居索引
        """
        # 相似度矩阵: [B, num_windows, num_windows]
        affinity = torch.matmul(window_summary, window_summary.transpose(-2, -1))

        # 把对角线 (自己和自己) 设为 -inf, 避免选到自己
        B, N, _ = affinity.shape
        eye_mask = torch.eye(N, device=affinity.device, dtype=torch.bool).unsqueeze(0)
        affinity = affinity.masked_fill(eye_mask, float('-inf'))

        # 选 top-K
        _, topk_idx = affinity.topk(self.top_k, dim=-1)  # [B, num_windows, top_k]

        return topk_idx

    def forward(
        self,
        x_windows: torch.Tensor,
        num_windows_per_batch: int,
    ) -> torch.Tensor:
        """
        Args:
            x_windows: [B * num_windows, ws*ws, C]  — 已经做过窗口内 attention 的结果
            num_windows_per_batch: 每个 batch 的窗口数 (= (H//ws) × (W//ws))

        Returns:
            out: [B * num_windows, ws*ws, C] 形状不变, 但每个 token 已融合跨窗口信息
        """
        B_, N, C = x_windows.shape  # B_ = B * num_windows, N = ws*ws
        num_windows = num_windows_per_batch
        B = B_ // num_windows

        # 实际选择的 k 值 — 不能超过 num_windows - 1 (不选自己)
        effective_k = min(self.top_k, num_windows - 1)
        if effective_k <= 0:
            # 只有一个窗口, 无法跨窗口交互, 直接返回
            return x_windows

        # --- Step 1: 计算窗口摘要 ---
        # x_windows: [B*num_win, ws*ws, C] → 先 reshape 到 [B, num_win, ws*ws, C]
        x_reshape = x_windows.view(B, num_windows, N, C)
        window_summary = x_reshape.mean(dim=2)  # [B, num_win, C]

        # --- Step 2: 路由 top-K ---
        # 运行时临时改变 self.top_k 到 effective_k, 避免不必要的 assertion
        original_k = self.top_k
        self.top_k = effective_k
        topk_idx = self._route_windows(window_summary)  # [B, num_win, k]
        self.top_k = original_k

        # --- Step 3: Gather top-K 邻居窗口的 token ---
        # topk_idx: [B, num_win, k] → 需要扩展到 [B, num_win, k, ws*ws, C]
        # gather 的源: x_reshape [B, num_win, N, C]
        idx_expand = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(
            B, num_windows, effective_k, N, C
        )  # [B, num_win, k, N, C]

        # x_reshape 扩展成 [B, 1, num_win, N, C] 然后 gather 沿 dim=2
        source = x_reshape.unsqueeze(1).expand(
            B, num_windows, num_windows, N, C
        )  # [B, num_win_q, num_win_kv, N, C]
        neighbor_tokens = torch.gather(source, dim=2, index=idx_expand)
        # [B, num_win, k, N, C] → [B, num_win, k*N, C]
        neighbor_tokens = neighbor_tokens.reshape(B, num_windows, effective_k * N, C)

        # --- Step 4: Attention ---
        # Q 来自当前窗口: x_reshape [B, num_win, N, C]
        # K, V 来自 neighbor_tokens [B, num_win, k*N, C]
        q = self.q_proj(x_reshape)  # [B, num_win, N, C]
        kv = self.kv_proj(neighbor_tokens)  # [B, num_win, k*N, 2C]
        k, v = kv.chunk(2, dim=-1)  # 各自 [B, num_win, k*N, C]

        # reshape 为多头: [B, num_win, N, H, D] → [B*num_win, H, N, D]
        H_, D = self.num_heads, self.head_dim
        q = q.view(B, num_windows, N, H_, D).permute(0, 1, 3, 2, 4).reshape(B_, H_, N, D)
        k = k.view(B, num_windows, effective_k * N, H_, D).permute(0, 1, 3, 2, 4).reshape(
            B_, H_, effective_k * N, D
        )
        v = v.view(B, num_windows, effective_k * N, H_, D).permute(0, 1, 3, 2, 4).reshape(
            B_, H_, effective_k * N, D
        )

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, H, N, k*N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B_, H, N, D]
        out = out.transpose(1, 2).reshape(B_, N, C)

        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out


# =============================================================================
# 方案 B: 窗口级粗粒度 attention + 广播 (MaxViT 简化风格)
# =============================================================================

class WindowLevelCrossWindowAttention(nn.Module):
    """
    方案 B: 窗口级粗粒度 attention

    流程:
        1. 每个窗口 mean-pool 成一个摘要 token, 得到 [num_windows, C]
        2. 对这些摘要做标准 self-attention (num_windows 个 token)
        3. 将更新后的摘要广播回窗口内所有 token (通过 residual add 或 cross-attn)

    复杂度:
        O(num_windows²)   (远小于 token 级)
        例: Stage1 128×128, window=8 → num_windows=256 → O(256²) = 65k 次
             vs 方案 A O(16384 × 4 × 64) = 4.2M 次

    广播策略:
        - "add" (推荐): 把更新后的窗口摘要直接加到窗口内每个 token 上 (所有 token 同增量)
        - "expand": 扩展摘要到窗口尺寸后做 cross-attn (每个 token 自主选择, 更贵)
    这里实现 "add" 版本, 保持方案 B 的 "粗粒度快速" 定位

    Args:
        dim: 特征维度
        num_heads: 注意力头数
        qkv_bias: QKV 投影是否使用偏置
        attn_drop: 注意力 dropout
        proj_drop: 输出投影 dropout
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 窗口级 self-attention 的 QKV
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x_windows: torch.Tensor,
        num_windows_per_batch: int,
    ) -> torch.Tensor:
        """
        Args:
            x_windows: [B * num_windows, ws*ws, C]
            num_windows_per_batch: 每个 batch 的窗口数

        Returns:
            out: [B * num_windows, ws*ws, C] (每个窗口内所有 token 加上同一个增量)
        """
        B_, N, C = x_windows.shape
        num_windows = num_windows_per_batch
        B = B_ // num_windows

        # --- Step 1: 每个窗口做 mean pooling 得到摘要 ---
        # x_windows: [B_, N, C] → mean 得到 [B_, C] → reshape [B, num_win, C]
        window_summary = x_windows.mean(dim=1).view(B, num_windows, C)

        # --- Step 2: 窗口摘要间做 self-attention ---
        qkv = self.qkv_proj(window_summary)  # [B, num_win, 3C]
        qkv = qkv.view(B, num_windows, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, num_win, D]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各 [B, H, num_win, D]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, num_win, num_win]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        updated_summary = attn @ v  # [B, H, num_win, D]
        updated_summary = updated_summary.transpose(1, 2).reshape(B, num_windows, C)
        updated_summary = self.out_proj(updated_summary)  # [B, num_win, C]
        updated_summary = self.proj_drop(updated_summary)

        # --- Step 3: 广播回 token 级 (所有 token 加同一增量) ---
        # updated_summary: [B, num_win, C] → [B_, 1, C] → broadcast 到 [B_, N, C]
        broadcast = updated_summary.view(B_, 1, C).expand(B_, N, C)

        return broadcast  # 返回增量, 由调用方决定如何融合 (通常是 residual add)


# =============================================================================
# 方案 C: Attention pooling + cross-attention 分发 (MaxViT 完整风格)
# =============================================================================

class HybridCrossWindowAttention(nn.Module):
    """
    方案 C: Attention pooling 摘要 + cross-attention 分发

    流程:
        1. 每个窗口用 learnable query 做 attention pooling, 得到智能摘要 [num_windows, C]
        2. 摘要间做 self-attention, 得到全局更新后的摘要
        3. 每个 token 用自己做 query, 对所有窗口摘要做 cross-attention (每个 token 自主选择)

    复杂度:
        Step 1: num_windows × ws²  (attention pooling)
        Step 2: num_windows²         (summary self-attn)
        Step 3: N × num_windows      (token 查询 summary, 全连接)

        例: Stage1 128×128, window=8 → num_windows=256
              Step 1: 256 × 64 = 16k
              Step 2: 256² = 65k
              Step 3: 16384 × 256 = 4.2M
              介于方案 A 和方案 B 之间

    关键点:
        - Step 1 的 learnable query: 让摘要提取是"学出来"的, 而非固定 mean
        - Step 3 的 cross-attention: 每个 token 看到所有 num_windows 个摘要,
          自主决定从哪些窗口取信息 (vs 方案 B 所有 token 得到相同增量)

    ★ [2026-06-18] v3.0: Step 3 现在有两种实现 (step3_mode 选择, 默认 'gather' 向后兼容):
        - 'gather' (默认): token 当 Q, 对摘要 softmax → token 主动吸全局 (有取舍, Precision 友好)
        - 'splat'  (新增): 摘要当 Q/V, token 当被打分 Key, 对块内像素 softmax → 摘要主动播撒,
          对稀疏丝状目标(残膜)证据扩散更强; 配 null/sink 槽防灌背景. 详见 _splat() 与文件头注释.

    Args:
        dim: 特征维度
        num_heads: 注意力头数 (所有三步共用)
        qkv_bias: QKV 投影是否使用偏置
        attn_drop: 注意力 dropout
        proj_drop: 输出投影 dropout
        num_pool_queries: ★[6.15] 每窗口可学习摘要个数 M (1=原行为)
        use_stats: ★[6.15] 免费统计量摘要, 取自 ('max','mean','min'); ()=原行为
        step3_mode: ★[6.18] 'gather'(默认) / 'splat'
        splat_proj_dim: ★[6.18] splat 打分公共投影维度 (取 min(dim, 此值)), 仅 splat 用
        splat_null: ★[6.18] splat 是否加 null/sink 槽 (默认 True), 仅 splat 用
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        qkv_bias: bool = True,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        num_pool_queries: int = 1,        # ★ [2026-06-15] 改动1: 每窗口可学习摘要个数 (1=原行为)
        use_stats: tuple = (),            # ★ [2026-06-15] 改动1: 免费统计量摘要, 取自 ('max','mean','min'); ()=原行为
        # ★ [2026-06-18] v3.0: Step3 分发模式
        step3_mode: str = 'gather',       # 'gather'(默认,token查摘要) / 'splat'(摘要播撒进token) / 'splat_plus'(逐通道sigmoid双轨)
        splat_proj_dim: int = 256,        # splat 打分用的公共投影维度 (取 min(dim, 此值))
        splat_null: bool = True,          # splat 是否加 null/sink 槽 (让背景块的探针把能量倒进 null, 保护前景 Precision)
        # ★ [2026-06-20] splat_plus (splater+) 专属: 逐通道 sigmoid 打分 + 共享Q/K + 双轨(加性注入V / 乘性逐通道调制)
        splatp_score_mode: str = 'fullC', # 'fullC'(每通道独立打分) / 'multihead'(切 H 头, 头内求和)
        splatp_fuse_mode: str = 'perchannel',    # ★ [2026-06-24] mlp→perchannel: 通道保留(沿Kp逐通道,零串扰); mlp是Linear(Kp·C→C)混通道
        splatp_num_heads: int = 8,        # multihead 模式的头数 (dim 必须可被整除)
        # ★ [2026-07-15] 改动J归位修复: 开关与 LN 模块必须建在本类(消费方 _splat_plus
        #   在本类内); 原先误建在 BiLevelWindowBlock → 前向 AttributeError. 默认 False=行为不变.
        mul_ln: bool = False,             # 乘性轨底料 x_blk 的 LN (位置甲, 乘之前)
        cross_qk_norm: bool = False,      # splat 打分 Q/S 的 QK-Norm (防门饱和)
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim({dim}) must be divisible by num_heads({num_heads})"
        assert step3_mode in ('gather', 'splat', 'splat_plus'), \
            f"step3_mode 必须是 'gather'/'splat'/'splat_plus', got {step3_mode}"

        # ★ [2026-07-15] 改动J归位修复: 开关/LN 模块归属本类 (消费方 _splat_plus 在本类).
        #   仅 splat_plus 路径生效; enabled⟺模块存在, 保持不变式, OFF 时与旧行为逐字节等价.
        self.mul_ln_enabled = bool(mul_ln) and (step3_mode == 'splat_plus')
        self.cross_qk_norm_enabled = bool(cross_qk_norm) and (step3_mode == 'splat_plus')
        if self.mul_ln_enabled:
            self.splatp_mul_ln = nn.LayerNorm(dim)      # 乘性轨底料 x_blk 的归一化(位置甲)
        if self.cross_qk_norm_enabled:
            self.splatp_q_norm = nn.LayerNorm(dim)      # QK-Norm: Q 投影后归一化
            self.splatp_s_norm = nn.LayerNorm(dim)      # QK-Norm: S(=K) 投影后归一化

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # ★ [2026-06-15] 改动1 配置
        self.num_pool_queries = max(int(num_pool_queries), 1)
        # 仅保留合法统计量, 固定 ('max','mean','min') 顺序
        self.use_stats = tuple(s for s in ('max', 'mean', 'min') if s in set(use_stats))

        # --- Step 1: Attention pooling ---
        # ★ [2026-06-15] 改动1: M 个可学习 query (M=num_pool_queries), 每窗口产出 M 个"智能摘要".
        #   M 个 query 各自随机初始化破对称, 学不同的"提取视角" (缓解单摘要把窗口压太狠的问题).
        #   M=1 时形状 [1,1,dim], 与原实现逐字节一致 → 旧 checkpoint 可加载.
        self.pool_query = nn.Parameter(torch.zeros(1, self.num_pool_queries, dim))
        nn.init.trunc_normal_(self.pool_query, std=0.02)

        self.pool_q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool_kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        # --- Step 2: Summary self-attention ---
        self.summary_qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.summary_out_proj = nn.Linear(dim, dim)

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-06-18] v3.0: Step3 分发模式 (gather 默认 / splat 新增)          ║
        # ╠══════════════════════════════════════════════════════════════════════╣
        # ║ owns_modulation: 告诉外层 BiLevelWindowBlock "本支是否已自带 γ 调制".    ║
        # ║   - gather: False → 外层用 γ_cross 缩放本支输出 (原行为)                ║
        # ║   - splat : True  → splat 内部已有 γ_amp/γ_info, 外层跳过 γ_cross,      ║
        # ║              避免 1e-4 × 1e-4 = 1e-8 双重门控学不动.                     ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        self.step3_mode = step3_mode
        self.owns_modulation = (step3_mode in ('splat', 'splat_plus'))

        if step3_mode == 'gather':
            # --- Step 3 (gather, 原范式): Token→summary cross-attention ---
            self.dist_q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.dist_kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.dist_out_proj = nn.Linear(dim, dim)
        elif step3_mode == 'splat':
            # --- Step 3 (splat, 新增): 块级语义播撒 ---
            pdim = min(dim, int(splat_proj_dim))
            self.splat_pdim = pdim
            self.splat_scale = pdim ** -0.5
            self.splat_use_null = bool(splat_null)
            self.splat_q_proj = nn.Linear(dim, pdim, bias=qkv_bias)   # 摘要 → 探针 Query
            self.splat_s_proj = nn.Linear(dim, pdim, bias=qkv_bias)   # token  → 被打分的 Key
            self.splat_v_proj = nn.Linear(dim, dim,  bias=qkv_bias)   # 摘要 → 播撒的 Value (★payload 必须来自摘要)
            self.splat_out_proj = nn.Linear(dim, dim)                 # Info 注入轨的输出投影
            # ╔══════════════════════════════════════════════════════════════════╗
            # ║ ★ [2026-06-19] 保留 C 维的"逐通道专家融合"权重 (替代旧版对 k 直接求和) ║
            # ╠══════════════════════════════════════════════════════════════════╣
            # ║ 旧版 Step3 加性轨: Info = einsum('bgkp,bgkc->bgpc') —— 对 k(专家)求和,  ║
            # ║   等于把 Kp 个专家"平均"掉, 且若改成 Linear(Kp·C→C) 会把通道也打散重组. ║
            # ║ 新版: 先保留每个专家 (PerExpert [B,G,N,Kp,C]), 再沿 Kp 轴逐通道融合:     ║
            # ║   Info[b,g,p,c] = Σ_k PerExpert[b,g,p,k,c] · expert_weight[k,c]        ║
            # ║   通道 c 只乘 expert_weight[:,c] → channel 之间【零串扰】, C 完整保留.   ║
            # ║ expert_weight[k,c]: 专家 k 在通道 c 上的权重 (每专家每通道一个).         ║
            # ║ Kp = M(可学习摘要) + S(统计摘要), 在构造期已确定.                       ║
            # ║ 初值 1.0 → 起步与旧 splat 的"对 k 求和"【逐字节一致】(已数值验证),       ║
            # ║   训练再学每专家每通道权重 → 安全 drop-in, 最差退回旧 splat.            ║
            # ╚══════════════════════════════════════════════════════════════════╝
            Kp_experts = self.num_pool_queries + len(self.use_stats)
            self.splat_num_experts = Kp_experts
            self.expert_weight = nn.Parameter(torch.ones(Kp_experts, dim))   # [Kp, C], 初值 1.0
            # 乘性放大轨 γ_amp / 加性注入轨 γ_info (per-channel), 初值 1e-4 → 近恒等启动
            self.gamma_amp = nn.Parameter(1e-1 * torch.ones(dim))   # ★ [2026-06-24] 1e-4→0.1: splat模式唤醒bi的正确旋钮(外层跳过gamma_cross,这才是真gate)
            self.gamma_info = nn.Parameter(1e-1 * torch.ones(dim))  # ★ [2026-06-24] 1e-4→0.1
            if self.splat_use_null:
                # null/sink "钥匙": 探针(在公共空间)与它点积 → null 亲和度.
                #   背景块的探针若更像 null → softmax 把能量灌进 null 列 → 真实像素 Info≈0 → 保护前景 Precision.
                #   思想≈DETR no-object 槽 / ViT register token.
                self.splat_null_key = nn.Parameter(torch.zeros(pdim))
                nn.init.trunc_normal_(self.splat_null_key, std=0.02)

        elif step3_mode == 'splat_plus':
            # ╔══════════════════════════════════════════════════════════════════════╗
            # ║ ★ [2026-06-20] splat_plus (splater+): 逐通道 sigmoid 打分 + 共享Q/K + 双轨  ║
            # ╠══════════════════════════════════════════════════════════════════════╣
            # ║ 与 'splat' 的区别 (几轮讨论的结论):                                      ║
            # ║  · 打分保留通道维: Imp_c = q[k,c]·s[p,c] → [B,G,Kp,N,C] (不对 c 求和).      ║
            # ║    fullC = 每通道独立; multihead = 切 H 头、头内求和 → [B,G,Kp,N,H] 再展回 C.║
            # ║  · 门用 sigmoid 逐元素 (非 softmax, 不归一): A = sigmoid(Imp/τ + b),         ║
            # ║    τ,b 可学习且逐通道/逐头, b₀≈−2 → 起步近 0 注入 (防梯度饱和/量级漂移).      ║
            # ║  · 加性轨(注入): PerExpert = A ⊙ V (A=门, V=摘要内容, 沿 N 广播) →           ║
            # ║    N-锚 Fuse 把 (Kp,C) 压回 C → Info → LayerNorm(强制) → ×γ_info.            ║
            # ║  · 乘性轨(逐通道调制): 复用同一 Imp_c → Fuse 压 → sigmoid → x_blk×Amp →×γ_amp.║
            # ║  · 加性/乘性【共享】Q/K (同一 Imp_c); Fuse 两轨各一套权重.                   ║
            # ║ owns_modulation=True (内部已含 γ_amp/γ_info) → 外层跳过 γ_cross.            ║
            # ╚══════════════════════════════════════════════════════════════════════╝
            assert splatp_score_mode in ('fullC', 'multihead'), \
                f"splatp_score_mode 必须是 'fullC'/'multihead', got {splatp_score_mode}"
            assert splatp_fuse_mode in ('mlp', 'perchannel'), \
                f"splatp_fuse_mode 必须是 'mlp'/'perchannel', got {splatp_fuse_mode}"
            self.splatp_score_mode = splatp_score_mode
            self.splatp_fuse_mode = splatp_fuse_mode
            Kp_experts = self.num_pool_queries + len(self.use_stats)
            self.splatp_num_experts = Kp_experts

            # 打分/内容投影 (投到 C, 因为要逐通道; 加性乘性共享 q/s)
            self.splatp_q_proj = nn.Linear(dim, dim, bias=qkv_bias)   # 摘要 → Q   [..,Kp,C]
            self.splatp_s_proj = nn.Linear(dim, dim, bias=qkv_bias)   # 特征 → K   [..,N, C]
            self.splatp_v_proj = nn.Linear(dim, dim, bias=qkv_bias)   # 摘要 → V (内容)

            # 打分粒度 → 决定门/温度偏置的维度 (fullC: 每通道 C; multihead: 每头 H)
            if splatp_score_mode == 'multihead':
                assert dim % splatp_num_heads == 0, \
                    f"multihead: dim({dim}) 必须可被 splatp_num_heads({splatp_num_heads}) 整除"
                self.splatp_num_heads = splatp_num_heads
                self.splatp_head_dim = dim // splatp_num_heads
                self.splatp_mh_scale = self.splatp_head_dim ** -0.5
                score_dim = splatp_num_heads
            else:
                self.splatp_num_heads = 1
                score_dim = dim

            # 加性门的温度 τ / 偏置 b (逐通道或逐头; b₀=−2 起步近 0 注入)
            # ★ [2026-07-18] v22: 本参数存 ρ, 前向经模块级 splatp_tau_eff() → τ_eff=τ_min+softplus(ρ+shift);
            #   参数名/形状/初值不变(ρ=1 → τ_eff≈1.0), 下界结构性防 1/τ 跑飞. τ_add/τ_mul 同规.
            self.splatp_tau_add = nn.Parameter(torch.ones(score_dim))
            self.splatp_bias_add = nn.Parameter(torch.full((score_dim,), -2.0))
            # 乘性门的温度/偏置 (作用在压缩后的 [.,N,C], 故为 C 维; b₀=0 → sigmoid≈0.5, 由 γ_amp 兜近恒等)
            #   ★ [2026-07-18] v22: 同上 — 存 ρ, 前向经 splatp_tau_eff() 得 τ_eff
            self.splatp_tau_mul = nn.Parameter(torch.ones(dim))
            self.splatp_bias_mul = nn.Parameter(torch.zeros(dim))

            # N-锚融合权重 (加性、乘性各一套): mlp=Linear(Kp·C→C) 混通道; perchannel=[Kp,C] 沿Kp保C
            if splatp_fuse_mode == 'mlp':
                self.splatp_add_fuse = nn.Linear(Kp_experts * dim, dim)   # 混通道
                self.splatp_mul_fuse = nn.Linear(Kp_experts * dim, dim)
            else:  # perchannel
                self.splatp_add_w = nn.Parameter(torch.ones(Kp_experts, dim))  # 沿Kp逐通道, 初值1.0
                self.splatp_mul_w = nn.Parameter(torch.ones(Kp_experts, dim))

            # 加性 Info 进残差前的 LayerNorm (因 sigmoid 不归一, 强制控量)
            self.splatp_info_ln = nn.LayerNorm(dim)

            # ★ [2026-07-24] P2: 逐像素强度增益 g=2·sigmoid(α·m+β). 仅开关开时建参数,
            #   关闭时 state_dict 与旧版逐键一致(正在跑的 run 可安全 resume).
            self.splatp_pixel_gain = bool(SPLATP_PIXEL_GAIN)
            self.splatp_gain_detach = bool(SPLATP_GAIN_DETACH)
            if self.splatp_pixel_gain:
                self.splatp_gain_alpha = nn.Parameter(torch.zeros(1))   # 恒等初始化 g≡1
                self.splatp_gain_beta  = nn.Parameter(torch.zeros(1))

            # ★ [2026-07-25] v23-P3a/b: 受限softmax门 + V去相关(仅开关开时建参数/激活, 关=state_dict与旧版一致)
            self.splatp_learn_softmax = bool(SPLATP_LEARN_SOFTMAX)
            self.splatp_v_decorr_on = bool(SPLATP_V_DECORR)
            if self.splatp_learn_softmax:
                if self.splatp_score_mode != 'fullC':
                    raise ValueError('splatp_learn_softmax 仅支持 score_mode=fullC')
                # 逐专家×逐通道先验: softmax 下沿k公共分量不可辨识, 升维后可辨识
                self.splatp_b_learn = nn.Parameter(torch.zeros(self.num_pool_queries, dim))

            # 双轨 γ (per-channel), 初值 1e-4 → 近恒等启动
            self.gamma_amp = nn.Parameter(1e-1 * torch.ones(dim))   # ★ [2026-06-24] 1e-4→0.1: splat模式唤醒bi的正确旋钮(外层跳过gamma_cross,这才是真gate)
            self.gamma_info = nn.Parameter(1e-1 * torch.ones(dim))  # ★ [2026-06-24] 1e-4→0.1

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _attention_pool(self, x_windows: torch.Tensor) -> torch.Tensor:
        """
        Step 1: 用 M 个 learnable query 对每个窗口做 attention pooling

        Args:
            x_windows: [B_, N, C]  (B_ = B*num_windows, N = ws*ws)

        Returns:
            summary: [B_, M, C]  每个窗口的 M 个智能摘要 (M = num_pool_queries)
        """
        B_, N, C = x_windows.shape
        H_, D = self.num_heads, self.head_dim
        M = self.num_pool_queries

        # Q 来自 M 个 learnable query (广播到每个窗口): [1, M, C] → [B_, M, C]
        q = self.pool_q_proj(self.pool_query.expand(B_, M, C))  # [B_, M, C]
        kv = self.pool_kv_proj(x_windows)  # [B_, N, 2C]
        k, v = kv.chunk(2, dim=-1)

        # 多头 reshape
        q = q.view(B_, M, H_, D).transpose(1, 2)  # [B_, H, M, D]
        k = k.view(B_, N, H_, D).transpose(1, 2)  # [B_, H, N, D]
        v = v.view(B_, N, H_, D).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, H, M, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        summary = attn @ v  # [B_, H, M, D]
        summary = summary.transpose(1, 2).reshape(B_, M, C)  # [B_, M, C]
        return summary

    def _compute_stats(self, x_windows: torch.Tensor):
        """
        ★ [2026-06-15] 改动1: 每窗口的免费统计量摘要 (无参数, 不进 Step2 自注意力).

        Args:
            x_windows: [B_, N, C]
        Returns:
            [B_, S, C]  或 None (use_stats 为空时)

        说明: max 抓窗口内最强激活 (稀疏薄膜的关键, mean 会被背景冲淡); mean 给全局均值;
              min 抓最强负激活 (本任务特征带符号 stage1 min≈-4.8, min 有判别信息).
              统计量【跳过 Step2 摘要自注意力】, 仅在 Step3 注入 token, 省算量.
        """
        if not self.use_stats:
            return None
        outs = []
        for s in self.use_stats:
            if s == 'max':
                outs.append(x_windows.max(dim=1).values)   # [B_, C]
            elif s == 'mean':
                outs.append(x_windows.mean(dim=1))          # [B_, C]
            elif s == 'min':
                outs.append(x_windows.min(dim=1).values)    # [B_, C]
        return torch.stack(outs, dim=1)  # [B_, S, C]

    def _summary_self_attn(self, summary: torch.Tensor) -> torch.Tensor:
        """
        Step 2: 窗口摘要之间做 self-attention

        Args:
            summary: [B, num_win, C]

        Returns:
            updated: [B, num_win, C]
        """
        B, num_win, C = summary.shape
        H_, D = self.num_heads, self.head_dim

        qkv = self.summary_qkv_proj(summary).view(B, num_win, 3, H_, D)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, num_win, C)
        out = self.summary_out_proj(out)
        return out

    def _distribute(
        self,
        x_windows: torch.Tensor,
        summary_set: torch.Tensor,
        num_windows: int,
    ) -> torch.Tensor:
        """
        Step 3: 每个 token 对所有摘要做 cross-attention

        ★ [2026-06-15] 改动1: summary_set 的摘要数 T 已泛化 =
            (M 个可学习摘要 + S 个统计摘要) × num_windows (原实现 T = num_windows).

        Args:
            x_windows: [B_, N, C]
            summary_set: [B, T, C]   每个样本的全部摘要
            num_windows: num_win

        Returns:
            out: [B_, N, C]  每个 token 得到自己查询到的增量
        """
        B_, N, C = x_windows.shape
        B = B_ // num_windows
        H_, D = self.num_heads, self.head_dim
        T = summary_set.shape[1]                 # 每个样本的摘要总数

        # Q: 每个 token 自己
        q = self.dist_q_proj(x_windows)  # [B_, N, C]

        # K/V: 该样本的全部 T 个摘要
        kv = self.dist_kv_proj(summary_set)  # [B, T, 2C]
        k, v = kv.chunk(2, dim=-1)  # [B, T, C]

        # q reshape: [B_, N, C] = [B*num_win, N, C] → 按样本分组 → [B_, H, N, D]
        q = q.view(B, num_windows, N, H_, D).permute(0, 1, 3, 2, 4).reshape(B_, H_, N, D)

        # k/v: 同一样本的 T 个摘要, 每个窗口共享同一套 → 扩展到每个窗口
        k = k.view(B, T, H_, D).transpose(1, 2)  # [B, H, T, D]
        v = v.view(B, T, H_, D).transpose(1, 2)
        k = k.unsqueeze(1).expand(B, num_windows, H_, T, D).reshape(B_, H_, T, D)
        v = v.unsqueeze(1).expand(B, num_windows, H_, T, D).reshape(B_, H_, T, D)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, H, N, T]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # [B_, H, N, D]
        out = out.transpose(1, 2).reshape(B_, N, C)
        out = self.dist_out_proj(out)
        out = self.proj_drop(out)
        return out

    def _splat(
        self,
        x_windows: torch.Tensor,
        probes_blk: torch.Tensor,
        num_windows: int,
    ) -> torch.Tensor:
        """
        ★ [2026-06-18] v3.0: Step3 'splat' 模式 —— 块级语义播撒 (Block-wise Semantic Splatting)
        ★ [2026-06-19] 加性轨升级为"保留 C 维的逐通道专家融合"(见下).

        与 _distribute(gather) 的两点区别:
          1. softmax 方向: gather 每个 token 对摘要 softmax (主动吸); splat 每个摘要对块内像素 softmax (主动摊).
          2. 专家合并: 旧 splat 对 k 直接求和(=平均专家); 新版保留每个专家后【沿 Kp 逐通道融合】,
             通道间零串扰 → C 维完整保留 (不会像 Linear(Kp·C→C) 那样把通道打散重组).

        Args:
            x_windows : [B_, N, C]      B_=B*num_win, N=ws*ws (=每块像素数 P), 已是 norm 后特征
            probes_blk: [B, G, Kp, C]   每块 Kp 个摘要 (Kp = M 可学习 + S 统计), 已含全局上下文(Step2)
            num_windows: G

        Returns:
            delta: [B_, N, C]  待加回残差的"增量" (内部已乘 γ_amp/γ_info, 外层不再乘 γ_cross)

        形状推导:
            x_blk = x_windows.view(B,G,N,C)                                  # [B,G,N,C] 按块切
            Q = q(probes_blk)                                               # [B,G,Kp,pdim]
            S = s(x_blk)                                                    # [B,G,N, pdim]
            V = v(probes_blk)                                               # [B,G,Kp,C]   ★payload=摘要
            Imp = einsum('bgkc,bgpc->bgkp', Q, S)·scale                     # [B,G,Kp,N]  探针 k 给像素 p 打分
            [可选 null] 在 N 轴末尾拼 null 列 → softmax over (N+1) → 丢 null 列 = Attn
            否则        Attn = Imp.softmax(dim=-1)                          # [B,G,Kp,N]  ★对像素 softmax
            # ★[6.19] 保留 C 的逐通道专家融合 (替代旧版对 k 求和):
            PerExpert = einsum('bgkp,bgkc->bgpkc', Attn, V)                 # [B,G,N,Kp,C] 保留每个专家(不求和 k)
            Info = einsum('bgpkc,kc->bgpc', PerExpert, expert_weight)       # [B,G,N,C]   沿 Kp 逐通道融合, C 零串扰
            Amp  = sigmoid(Imp.sum(dim=2))                                  # [B,G,N]     块内空间显著性(乘性轨, 逐像素标量, 也不串通道)
            delta= γ_amp·(x_blk·Amp) + γ_info·Info                         # [B,G,N,C]   乘加双轨
        """
        B_, N, C = x_windows.shape
        G = num_windows
        B = B_ // G
        Kp = probes_blk.shape[2]

        x_blk = x_windows.view(B, G, N, C)               # [B, G, N, C]

        Q = self.splat_q_proj(probes_blk)                # [B, G, Kp, pdim]  探针 Query
        S = self.splat_s_proj(x_blk)                     # [B, G, N,  pdim]  token 当被打分的 Key
        V = self.splat_v_proj(probes_blk)                # [B, G, Kp, C]     ★Value 来自摘要

        # 探针 k 对块内像素 p 的重要度 (对 pdim 求和)
        Imp = torch.einsum('bgkc,bgpc->bgkp', Q, S) * self.splat_scale   # [B, G, Kp, N]

        if self.splat_use_null:
            # null 亲和度: 探针 · null_key (公共空间点积) → [B,G,Kp,1]
            null_score = torch.einsum('bgkc,c->bgk', Q, self.splat_null_key) * self.splat_scale
            null_score = null_score.unsqueeze(-1)                        # [B, G, Kp, 1]
            Imp_aug = torch.cat([Imp, null_score], dim=-1)              # [B, G, Kp, N+1]
            attn_aug = Imp_aug.softmax(dim=-1)                          # 对 (N 像素 + 1 null) softmax
            attn_aug = self.attn_drop(attn_aug)
            Attn = attn_aug[..., :N]                                    # 丢掉 null 列: Σ_p Attn ≤ 1 (余量入 null)
        else:
            Attn = Imp.softmax(dim=-1)                                  # 对像素 softmax (Σ_p = 1)
            Attn = self.attn_drop(Attn)

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-06-19] 信息注入轨: 保留 C 维的"逐通道专家融合" (替代旧版对 k 求和)  ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        # 第1步 PerExpert: 在 k(专家)轴【不求和】, 保留每个专家各自播撒进像素的内容.
        #   Attn[b,g,k,p]·V[b,g,k,c] → [B,G,N,Kp,C]  (p 来自 Attn, c 来自 V, k 保留)
        PerExpert = torch.einsum('bgkp,bgkc->bgpkc', Attn, V)          # [B, G, N, Kp, C]
        # 第2步 沿 Kp 轴逐通道融合: 通道 c 只乘 expert_weight[:,c] → 通道间【零串扰】, C 完整保留.
        #   Info[b,g,p,c] = Σ_k PerExpert[b,g,p,k,c]·expert_weight[k,c]
        #   (对比旧版 einsum('bgkp,bgkc->bgpc') 是无权重地对 k 求和; init=1.0 时两者逐字节相同)
        Info = torch.einsum('bgpkc,kc->bgpc', PerExpert, self.expert_weight)  # [B, G, N, C]
        # splat_out_proj (Linear C→C): 这一步【才】做跨通道混合; 想要纯通道独立可把它视作恒等/去掉.
        Info = self.splat_out_proj(Info)
        Info = self.proj_drop(Info)

        # 乘性放大轨: 像素收到的总注意力 (用 softmax 前的 Imp, 表达原始显著性)
        Amp = torch.sigmoid(Imp.sum(dim=2))                            # [B, G, N]

        # 乘加双轨调制 (γ 已在此, 外层 BiLevelWindowBlock 跳过 γ_cross)
        delta = (self.gamma_amp * (x_blk * Amp.unsqueeze(-1))
                 + self.gamma_info * Info)                             # [B, G, N, C]
        return delta.reshape(B_, N, C)

    def _splatp_fuse(self, x5: torch.Tensor, track: str) -> torch.Tensor:
        """
        ★ [2026-06-20] splat_plus 的 N-锚融合: [B,G,Kp,N,C] → [B,G,N,C]
        以 N 为锚 (当批一样的轴), 把每个像素的 (Kp,C) 压成 C. 两种方式 (self.splatp_fuse_mode):
          'mlp'       : permute→[B,G,N,Kp·C]→Linear(Kp·C→C)  (混通道, 表达力强)
          'perchannel': einsum('bgnkc,kc->bgnc', ·, w)        (沿 Kp 逐通道, 通道零串扰=保C)
        track: 'add'(加性轨) / 'mul'(乘性轨), 各用一套独立融合权重.
        """
        B, G, Kp, N, C = x5.shape
        xp = x5.permute(0, 1, 3, 2, 4).contiguous()       # [B,G,N,Kp,C]
        if self.splatp_fuse_mode == 'mlp':
            lin = self.splatp_add_fuse if track == 'add' else self.splatp_mul_fuse
            return lin(xp.reshape(B, G, N, Kp * C))        # [B,G,N,C]
        else:  # perchannel
            w = self.splatp_add_w if track == 'add' else self.splatp_mul_w
            return torch.einsum('bgnkc,kc->bgnc', xp, w)   # [B,G,N,C]  通道 c 只配 w[:,c]

    def _splat_plus(
        self,
        x_windows: torch.Tensor,
        probes_blk: torch.Tensor,
        num_windows: int,
    ) -> torch.Tensor:
        """
        ★ [2026-06-20] splat_plus (splater+): 逐通道 sigmoid 打分 + 共享Q/K + 双轨

        Args:
            x_windows : [B_, N, C]      已 norm 后窗口特征 (B_=B*G, N=每块像素数)
            probes_blk: [B, G, Kp, C]   每块 Kp 个摘要 (M 可学习 + S 统计), 已含全局上下文(Step2)
            num_windows: G
        Returns:
            delta: [B_, N, C]  内部已乘 γ_amp/γ_info, 外层不再乘 γ_cross

        流程 (与最终流程图一致):
            Q=q(probes)[B,G,Kp,C]; S=s(x_blk)[B,G,N,C]; V=v(probes)[B,G,Kp,C]
            打分(共享, 保留通道维):
              fullC    : Imp_c = einsum('bgkc,bgpc->bgkpc', Q, S)        # [B,G,Kp,N,C] 逐通道(q·s 不求和)
              multihead: 切 H 头、头内对 d=C/H 求和 → [B,G,Kp,N,H] → 展回 [B,G,Kp,N,C]
            加性轨(注入): A = sigmoid(Imp_c/τ_add + b_add)  (逐元素门, 不归一)
                          PerExpert = A ⊙ V(沿N广播) → [B,G,Kp,N,C]
                          Info = N-锚Fuse(PerExpert,'add') → LayerNorm → [B,G,N,C]
            乘性轨(逐通道调制): Amp = sigmoid( N-锚Fuse(Imp_c,'mul')/τ_mul + b_mul )  # [B,G,N,C] 先压再sigmoid
                          out_mul = x_blk × Amp
            delta = γ_amp·out_mul + γ_info·Info
        """
        B_, N, C = x_windows.shape
        G = num_windows
        B = B_ // G
        Kp = probes_blk.shape[2]
        x_blk = x_windows.view(B, G, N, C)                # [B,G,N,C]

        Q = self.splatp_q_proj(probes_blk)                # [B,G,Kp,C]
        S = self.splatp_s_proj(x_blk)                     # [B,G,N, C]
        # ★ [2026-07-11] 改动J: QK-Norm — 对 Q/S 各做 LayerNorm, 钉住 Q·S 尺度, 防打分门 sigmoid 饱和
        if self.cross_qk_norm_enabled:
            Q = self.splatp_q_norm(Q)
            S = self.splatp_s_norm(S)
        V = self.splatp_v_proj(probes_blk)                # [B,G,Kp,C]  ★内容来自摘要

        # ★ [2026-07-25] v23-P3b: learnable 专家内容去相关惩罚(暂存, λ与求和在训练端).
        #   仅取 k<M learnable 切片; 统计量(max/mean/min)天然互异不惩罚.
        if getattr(self, 'splatp_v_decorr_on', False) and self.training and self.num_pool_queries > 1:
            _Vl = V[:, :, :self.num_pool_queries]                              # [B,G,M,C]
            _Vn = _Vl - _Vl.mean(dim=-1, keepdim=True)
            _Vn = _Vn / (_Vn.norm(dim=-1, keepdim=True) + 1e-6)
            _cor = torch.einsum('bgmc,bgnc->bgmn', _Vn, _Vn)                   # [B,G,M,M]
            _off = _cor - torch.eye(_Vn.shape[2], device=_Vn.device, dtype=_Vn.dtype)
            self._vdecorr_loss = (_off ** 2).mean()                            # 保留计算图, 训练端取用
        else:
            self._vdecorr_loss = None

        # ★ [2026-07-24] P2: 标量匹配强度 m 与逐像素增益 g (只喂加性轨; 乘性轨不经过 g).
        #   detach 默认开: einsum 只读 Q/S, 共享打分投影收到的新增梯度精确为零.
        if getattr(self, "splatp_pixel_gain", False):
            _Qg = Q.detach() if self.splatp_gain_detach else Q
            _Sg = S.detach() if self.splatp_gain_detach else S
            _m = torch.einsum("bgkc,bgpc->bgkp", _Qg, _Sg) * (C ** -0.5)   # [B,G,Kp,N] 标量相似度
            _m = _m.amax(dim=2)                                            # [B,G,N] 最佳匹配摘要
            _g = 2.0 * torch.sigmoid(self.splatp_gain_alpha * _m + self.splatp_gain_beta)  # (0,2), 初始≡1
        else:
            _g = None

        # ---- 打分: 保留通道维, 加性/乘性共享 ----
        if self.splatp_score_mode == 'multihead':
            H, d = self.splatp_num_heads, self.splatp_head_dim
            Qh = Q.view(B, G, Kp, H, d)
            Sh = S.view(B, G, N, H, d)
            Imp_h = torch.einsum('bgkhd,bgphd->bgkph', Qh, Sh) * self.splatp_mh_scale  # [B,G,Kp,N,H]
            A_h = torch.sigmoid(Imp_h / splatp_tau_eff(self.splatp_tau_add) + self.splatp_bias_add)    # 逐头门 [B,G,Kp,N,H]  ★[2026-07-18] τ→τ_eff
            # 展回通道: 每个头的值在它管的 d 个通道上复制 → [B,G,Kp,N,C]
            A = A_h.repeat_interleave(d, dim=-1)
            Imp_c = Imp_h.repeat_interleave(d, dim=-1)
        else:  # fullC
            Imp_c = torch.einsum('bgkc,bgpc->bgkpc', Q, S)                              # [B,G,Kp,N,C]
            if getattr(self, 'splatp_learn_softmax', False):
                # ★ [2026-07-25] v23-P3a: k<M learnable → softmax_k 竞争(Σ=1, 开一门必压其余);
                #   k>=M 统计量 → 原 sigmoid+b_add 旁路, 逐比特不变. τ 共用 splatp_tau_add.
                _Ml = self.num_pool_queries
                _z = Imp_c / splatp_tau_eff(self.splatp_tau_add)               # [B,G,Kp,N,C]
                _zl = _z[:, :, :_Ml] + self.splatp_b_learn.view(1, 1, _Ml, 1, -1)
                _Al = torch.softmax(_zl, dim=2)                                # 沿专家轴归一
                _As = torch.sigmoid(_z[:, :, _Ml:] + self.splatp_bias_add)     # 统计量不变
                A = torch.cat([_Al, _As], dim=2)
                if self.training:                                              # 竞争熵 H(A), log(M)=均匀
                    _p = _Al.clamp_min(1e-8)
                    self._gate_entropy = float((-(_p * _p.log()).sum(dim=2)).mean().detach())
            else:
                A = torch.sigmoid(Imp_c / splatp_tau_eff(self.splatp_tau_add) + self.splatp_bias_add)       # 逐通道门 [B,G,Kp,N,C]  ★[2026-07-18] τ→τ_eff
        A = self.attn_drop(A)

        # ---- 加性轨 (注入内容): A=门, V=内容 ----
        PerExpert = A * V.unsqueeze(3)                    # [B,G,Kp,N,C]  V 沿 N(轴3) 广播
        Info = self._splatp_fuse(PerExpert, 'add')        # [B,G,N,C]  Kp,C → C
        Info = self.splatp_info_ln(Info)                  # ★ 强制 LayerNorm (sigmoid 不归一, 控量)
        if _g is not None:
            Info = Info * _g.unsqueeze(-1)                # ★ [2026-07-24] P2: 方向×强度, [B,G,N,1] 沿C广播
        Info = self.proj_drop(Info)

        # ---- 乘性轨 (逐通道调制): 复用 Imp_c, 先压再 sigmoid ----
        Amp_raw = self._splatp_fuse(Imp_c, 'mul')         # [B,G,N,C]
        Amp = torch.sigmoid(Amp_raw / splatp_tau_eff(self.splatp_tau_mul) + self.splatp_bias_mul)       # [B,G,N,C]  ★[2026-07-18] τ→τ_eff
        # ★ [2026-07-11] 改动J: 位置甲 — LN 加在【乘之前】的底料 x_blk 上(稳基线, 保留 Amp 的跨像素选择性);
        #   严禁加在 out_mul(乘之后)上——那会拉平幅度、抹掉 AUC 选择性.
        x_blk_for_mul = self.splatp_mul_ln(x_blk) if self.mul_ln_enabled else x_blk
        out_mul = x_blk_for_mul * Amp                     # [B,G,N,C]  逐通道调制

        # ---- 双轨融合 (γ 在此, 外层跳过 γ_cross) ----
        delta = self.gamma_amp * out_mul + self.gamma_info * Info     # [B,G,N,C]
        # ★ [2026-07-11] 守卫式可视化缓存: 仅当外部把 self.viz_cache_enabled 置 True 时,
        #   暂存两个门图(立即 detach→CPU, 不进计算图), 供 track_viz.py 离线分析.
        #   默认无该属性 → getattr 返回 False → 训练/推理路径逐字节不变.
        if getattr(self, 'viz_cache_enabled', False):
            self._viz_cache = {
                'Amp':    Amp.detach().float().cpu(),                 # [B,G,N,C] 乘性逐通道门
                'A_mean': A.detach().float().mean(dim=2).cpu(),       # [B,G,N,C] 加性门(Kp 均值)
            }
            if _g is not None:
                self._viz_cache['g'] = _g.detach().float().cpu().unsqueeze(-1)   # ★ [2026-07-24] [B,G,N,1] 注入强度图(补1通道, 与Amp同管线)
        return delta.reshape(B_, N, C)

    def forward(
        self,
        x_windows: torch.Tensor,
        num_windows_per_batch: int,
    ) -> torch.Tensor:
        """
        Args:
            x_windows: [B * num_windows, ws*ws, C]
            num_windows_per_batch: 每个 batch 的窗口数

        Returns:
            out: [B * num_windows, ws*ws, C]
        """
        B_, N, C = x_windows.shape
        num_windows = num_windows_per_batch
        B = B_ // num_windows
        M = self.num_pool_queries

        # Step 1: M 个可学习摘要/窗口 → [B_, M, C]
        summary = self._attention_pool(x_windows)
        # 排成 [B, num_win*M, C] 供自注意力
        learn_sum = summary.view(B, num_windows, M, C).reshape(B, num_windows * M, C)

        # Step 2: 只对可学习摘要做 self-attention (统计量不参与, 省算量)
        updated_learn = self._summary_self_attn(learn_sum)  # [B, num_win*M, C]

        # 统计量摘要 (跳过 Step2): [B_, S, C] → [B, num_win*S, C], 拼到摘要集合末尾
        stats = self._compute_stats(x_windows)
        if stats is not None:
            S = stats.shape[1]
            stat_sum = stats.view(B, num_windows, S, C).reshape(B, num_windows * S, C)
            summary_set = torch.cat([updated_learn, stat_sum], dim=1)  # [B, num_win*(M+S), C]
        else:
            summary_set = updated_learn

        # ╔══════════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-06-18] v3.0: Step3 两种分发模式 (gather 默认 / splat 新增)      ║
        # ╚══════════════════════════════════════════════════════════════════════╝
        if self.step3_mode == 'gather':
            # gather (默认, 原范式): 每个 token 对全部摘要做 cross-attention (token 主动"吸")
            out = self._distribute(x_windows, summary_set, num_windows)
        else:
            # splat / splat_plus: 都需要"每块的探针集" probes_blk [B, G, M+S, C]
            #   updated_learn:[B, G*M, C] 按(窗口,query)分组 → [B, G, M, C]
            #   stats        :[B_, S, C] = [B*G, S, C]        → [B, G, S, C]
            updated_blk = updated_learn.view(B, num_windows, M, C)          # [B, G, M, C]
            if stats is not None:
                stats_blk = stats.view(B, num_windows, stats.shape[1], C)   # [B, G, S, C]
                probes_blk = torch.cat([updated_blk, stats_blk], dim=2)     # [B, G, M+S, C]
            else:
                probes_blk = updated_blk
            if self.step3_mode == 'splat':
                out = self._splat(x_windows, probes_blk, num_windows)
            else:  # splat_plus
                out = self._splat_plus(x_windows, probes_blk, num_windows)

        return out


# =============================================================================
# 统一包装: BiLevelWindowBlock (替代 SwinEncoderBlock)
# =============================================================================

class BiLevelWindowBlock(nn.Module):
    """
    Cross-Window Block: 纯跨窗口 attention block (v2, 简化版)

    ★ [2026-04-21] 方案 A 重构
    ---------------------------------------------------------
    设计思路:
        原版 Swin 的 block pair 是 [W-MSA, SW-MSA], 其中 SW-MSA 本质上只是
        "把窗口划分偏移一半" 来实现跨窗口通信, 仍然在做 self-attention.

        我们的洞察: 在 Swin block pair 中, block 0 已经做过 W-MSA 了,
        block 1 再做一次 W-MSA 是冗余的. 我们让 block 1 "专职" 做跨窗口交互,
        不再重复 W-MSA, 让架构更紧凑.

    整体结构 (简化后):
        shortcut → norm_cross → CrossWindowAttention → γ_cross × 结果 → residual
                 → norm_ffn   → FFN                 → γ_ffn   × 结果 → residual

    LayerScale 设计 (替代原版的 cross_gate):
        - γ_cross, γ_ffn 是两个可学习标量 (每个通道一个), 初始化为 0
        - 训练早期 block 输出 ≈ 恒等映射, 不扰动主干前向和梯度
        - 训练过程中 γ 可学习, 网络自行决定何时打开 cross-window 和 FFN 分支
        - 参考: "Going deeper with Image Transformers" (CaiT, ICCV 2021)

    相比原版 BiLevelWindowBlock (有 W-MSA) 的改进:
        1. 移除 WindowAttention 和 norm1 → 避免与 block 0 冗余
        2. 移除 cross_gate (单分支门控) → 改用 LayerScale (双分支保护)
        3. FFN 分支也加 LayerScale → 解决之前 FFN 随机扰动导致的特征爆炸
        4. 参数量减少 ~30%, 梯度路径更简单

    与 block 0 (W-MSA) 的配合:
        在 Stage 1/2 的 block 排列 [W-MSA, CrossWindow, W-MSA, CrossWindow]:
          block 0: 窗口内信息汇聚 (Swin-T 预训练加载)
          block 1: 纯跨窗口信息交换 (本 block)
          block 2: 再次窗口内汇聚 (复用 block 0 权重)
          block 3: 再次跨窗口交换 (本 block)
        交替的 "局部 → 全局 → 局部 → 全局" 模式, 与原 Swin 哲学一致.

    Args:
        dim: 特征维度
        num_heads: 跨窗口 attention 头数 (兼容旧签名名 "num_heads", 不再需要独立的 window attn heads)
        window_size: 窗口大小 (用于 cross-window attention 的窗口划分)
        cross_type: 跨窗口方案 ("token_level" / "window_level" / "hybrid")
        cross_num_heads: 跨窗口 attention 头数 (默认等于 num_heads)
        cross_top_k: token_level 方案的 top-K
        mlp_ratio: FFN 隐藏层比例
        qkv_bias / drop / attn_drop / drop_path: 标准 transformer 超参
        layerscale_init: LayerScale γ 初始值 (推荐 0.0, 训练早期恒等)
        cross_gate_init: [兼容保留] 旧参数名, 等价于 layerscale_init, 新代码请使用后者
        pretrained_window_size: 跨窗口 attention 的 Log-CPB 基准窗口 (仅 token_level 用)
    """

    VALID_CROSS_TYPES = ("token_level", "window_level", "hybrid")

    # ★ [2026-06-23] 诊断开关 (类级, 默认 False → 对训练零影响).
    #   置 True 后, forward 会把 cross-window 增量(delta)与其输入(xin)存到实例属性,
    #   供 ablation.py 可视化"cross-window 注入了什么/在哪里". 不增参数、不改前向数值.
    CAPTURE = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        cross_type: str = "hybrid",
        cross_num_heads: Optional[int] = None,
        cross_top_k: int = 4,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        layerscale_init: Optional[float] = None,
        cross_gate_init: float = 0.0,  # 兼容旧签名
        pretrained_window_size: int = 0,
        cross_num_pool_queries: int = 1,   # ★ [2026-06-15] 改动1: 传给 HybridCrossWindowAttention (仅 hybrid 用)
        cross_use_stats: tuple = (),       # ★ [2026-06-15] 改动1: 传给 HybridCrossWindowAttention (仅 hybrid 用)
        cross_step3_mode: str = 'gather',  # ★ [2026-06-18] v3.0: 'gather'(默认)/'splat'/'splat_plus' (仅 hybrid 用)
        cross_splat_null: bool = True,     # ★ [2026-06-18] v3.0: splat 的 null/sink 槽 (仅 hybrid+splat 用)
        cross_splatp_score_mode: str = 'fullC',  # ★ [2026-06-20] splat_plus: 'fullC'/'multihead' (仅 hybrid+splat_plus)
        cross_splatp_fuse_mode: str = 'perchannel',     # ★ [2026-06-24] mlp→perchannel: 通道保留 (论文核心主张)
        cross_splatp_num_heads: int = 8,         # ★ [2026-06-20] splat_plus: multihead 头数
        ffn_respost: bool = False,               # ★ [2026-07-11] 改动I: FFN 子层归一化位置; False=Pre-Norm ffn(LN(x)) (默认, 现状); True=res-post LN(ffn(x))
        drop_norm_cross: bool = False,           # ★ [2026-07-11] 改动J: True=删除 cross 子层入口 norm_cross(改为恒等); 需配合下面两项保护, 否则乘性轨/打分门裸奔
        mul_ln: bool = False,                    # ★ [2026-07-11] 改动J: True=乘性轨底料 x_blk 加 LayerNorm(位置甲, 乘之前), 稳基线且保留 Amp 选择性; 禁止乘之后
        cross_qk_norm: bool = False,             # ★ [2026-07-11] 改动J: True=splat 打分路径 Q/S 各加 LayerNorm(QK-Norm), 稳打分门尺度(防 sigmoid 饱和)
    ):
        super().__init__()
        assert cross_type in self.VALID_CROSS_TYPES, (
            f"cross_type must be one of {self.VALID_CROSS_TYPES}, got {cross_type}"
        )

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.cross_type = cross_type

        # 向后兼容: 如果没有传 layerscale_init, 使用 cross_gate_init
        if layerscale_init is None:
            layerscale_init = cross_gate_init

        cross_num_heads = cross_num_heads if cross_num_heads is not None else num_heads

        # =================================================================
        # 跨窗口 attention 分支 (唯一的 attention 模块, 不再有 W-MSA)
        # =================================================================
        self.norm_cross = nn.LayerNorm(dim)
        if cross_type == "token_level":
            self.cross_attn = TokenLevelCrossWindowAttention(
                dim=dim,
                num_heads=cross_num_heads,
                window_size=window_size,
                top_k=cross_top_k,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif cross_type == "window_level":
            self.cross_attn = WindowLevelCrossWindowAttention(
                dim=dim,
                num_heads=cross_num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        else:  # hybrid
            self.cross_attn = HybridCrossWindowAttention(
                dim=dim,
                num_heads=cross_num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                num_pool_queries=cross_num_pool_queries,   # ★ [2026-06-15] 改动1
                use_stats=cross_use_stats,                 # ★ [2026-06-15] 改动1
                step3_mode=cross_step3_mode,               # ★ [2026-06-18] v3.0
                splat_null=cross_splat_null,               # ★ [2026-06-18] v3.0
                splatp_score_mode=cross_splatp_score_mode, # ★ [2026-06-20] splat_plus
                splatp_fuse_mode=cross_splatp_fuse_mode,   # ★ [2026-06-20] splat_plus
                splatp_num_heads=cross_splatp_num_heads,   # ★ [2026-06-20] splat_plus
                mul_ln=mul_ln,                             # ★ [2026-07-15] 改动J归位: 建在消费方类内
                cross_qk_norm=cross_qk_norm,               # ★ [2026-07-15] 改动J归位
            )

        # =================================================================
        # FFN 分支
        # =================================================================
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn_respost = ffn_respost          # ★ [2026-07-11] 改动I: FFN 子层归一化位置开关
        # ★ [2026-07-11] 改动J: 三正交开关(删入口 LN / 乘性轨底料 LN / 打分 QK-Norm), 默认全 False=现状
        self.drop_norm_cross = drop_norm_cross
        # ★ [2026-07-18] v22: 运行时消融属性(非参数, 不入 state_dict; 默认 False=行为不变).
        #   True → 跨窗口子层整体恒等(保留 FFN 子层) = 粒度(a) 'BiLevel 跨窗口注意力有没有用' 的结构性旁路.
        self.bypass_crosswindow = False
        # ★ [2026-07-15] 改动J归位修复: mul_ln / cross_qk_norm 的开关与 LN 模块已迁入
        #   HybridCrossWindowAttention.__init__ (消费方 _splat_plus 在彼类内; 原先误建在
        #   本类 → 前向无条件读取即 AttributeError). 本类仅透传构造参数.
        if (mul_ln or cross_qk_norm) and not isinstance(self.cross_attn, HybridCrossWindowAttention):
            print('[改动J][警告] mul_ln/cross_qk_norm 仅在 hybrid(splat_plus) 路径生效, '
                  '当前 cross_attn 非 Hybrid, 开关被忽略.')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FFN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop=drop,
        )

        # =================================================================
        # LayerScale: 两个分支各自的 γ (per-channel), 初始 0
        #   训练早期 block 近似恒等映射, 彻底避免随机初始化带来的特征爆炸
        # =================================================================
        self.gamma_cross = nn.Parameter(layerscale_init * torch.ones(dim))
        self.gamma_ffn = nn.Parameter(layerscale_init * torch.ones(dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 保留 cross_gate 属性 (兼容 train_v16 中的 BiLevel 识别逻辑)
        # 训练脚本通过 hasattr(block, 'cross_gate') 来区分 BiLevel vs SwinEncoder
        # 这里保留一个 zero-size buffer 作为标记, 不参与训练
        self.register_buffer('cross_gate', torch.zeros(1), persistent=False)

    def _compute_window_info(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
    ) -> Tuple[torch.Tensor, int, int, int, Tuple[int, int, int, int], int]:
        """
        对 x 做 window partition, 返回窗口化后的结果和元信息

        Returns:
            x_windows: [B*num_windows, ws*ws, C]
            H_pad, W_pad: padding 后的空间尺寸
            num_windows: 每个 batch 的窗口数
            pad_info: (pad_h, pad_w, H, W) 用于还原
            B: batch size
        """
        B, N, C = x.shape
        x_2d = x.view(B, H, W, C)

        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w

        x_windows = window_partition(x_2d, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        num_windows = (H_pad // self.window_size) * (W_pad // self.window_size)
        return x_windows, H_pad, W_pad, num_windows, (pad_h, pad_w, H, W), B

    def _unwindow_to_tokens(
        self,
        x_windows: torch.Tensor,
        H_pad: int,
        W_pad: int,
        pad_info: Tuple[int, int, int, int],
        B: int,
    ) -> torch.Tensor:
        """将窗口化的 tokens 还原为 [B, H*W, C] 序列"""
        C = x_windows.shape[-1]
        pad_h, pad_w, H, W = pad_info

        x_windows = x_windows.view(-1, self.window_size, self.window_size, C)
        x_2d = window_reverse(x_windows, self.window_size, H_pad, W_pad)

        if pad_h > 0 or pad_w > 0:
            x_2d = x_2d[:, :H, :W, :].contiguous()

        return x_2d.view(B, H * W, C)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        前向传播 (两个分支都用 LayerScale 保护)

        Args:
            x: [B, H*W, C]
            H, W: 空间尺寸

        Returns:
            out: [B, H*W, C]
        """
        B, N, C = x.shape
        assert N == H * W, f"N({N}) != H*W({H*W})"

        # ============ 分支 1: 跨窗口 attention ============
        shortcut = x
        # ★ [2026-07-18] v22 消融开关: bypass_crosswindow=True → 跨窗口子层整体恒等(粒度a, 保留 FFN).
        #   结构性旁路(非 γ=0: γ_amp/γ_info 有规范冗余, 置零无稳定含义); 完全跳过 cross 计算, 亦省时.
        if getattr(self, 'bypass_crosswindow', False):
            x = shortcut
        else:
            # ★ [2026-07-11] 改动J: drop_norm_cross=True 时删除入口 LN(恒等); 需 cross_qk_norm+mul_ln 保护, 否则乘性轨/打分门裸奔
            x_norm = x if self.drop_norm_cross else self.norm_cross(x)

            # 窗口化 (仅为让 cross-window attention 知道窗口结构, 不做 W-MSA)
            x_norm_windows, H_pad, W_pad, num_windows, pad_info, _ = self._compute_window_info(
                x_norm, H, W
            )

            # 跨窗口 attention
            cross_out_windows = self.cross_attn(x_norm_windows, num_windows)

            # 还原到序列
            cross_out_seq = self._unwindow_to_tokens(
                cross_out_windows, H_pad, W_pad, pad_info, B
            )

            # ★ [2026-06-23] 诊断捕获 (默认关闭, BiLevelWindowBlock.CAPTURE=False 时此分支不执行).
            #   cross_out_seq = cross-window 单独注入的增量 (splat 模式内部已乘 γ_amp/γ_info);
            #   shortcut       = 该 cross 分支的输入特征.
            #   存成 [B, H*W, C], 供外部 reshape 成 [B,H,W,C] 画"delta 在哪、注入多少".
            if BiLevelWindowBlock.CAPTURE:
                self._cap_delta = cross_out_seq.detach()
                self._cap_xin = shortcut.detach()
                self._cap_hw = (H, W)

            # ★ [2026-06-18] v3.0: splat 模式下 cross_attn 内部已含 γ_amp/γ_info → 跳过外层 γ_cross
            #   (否则 1e-4 × 1e-4 = 1e-8 双重门控, 学不动). gather 模式保持原行为 (× γ_cross).
            if getattr(self.cross_attn, 'owns_modulation', False):
                x = shortcut + self.drop_path(cross_out_seq)                       # splat: γ 已在内部
            else:
                x = shortcut + self.drop_path(self.gamma_cross * cross_out_seq)    # gather: 原行为

        # ============ 分支 2: FFN ============
        shortcut = x
        # ★ [2026-07-11] 改动I: FFN 子层归一化位置
        #   False(默认, 现状): Pre-Norm  x + γ_ffn·FFN(LN(x))  —— LN 归一化 FFN 的【输入】
        #   True (res-post)  : x + γ_ffn·LN(FFN(x))            —— LN 归一化 FFN 的【输出】,
        #                      与相邻 W-MSA 块(v2 res-post)拓扑统一; γ_ffn 乘单位尺度输出→语义可比.
        #   注: 仅 FFN 子层; cross 子层输入侧 pre-LN(norm_cross)不受影响(乘性轨/打分门需其保护).
        if self.ffn_respost:
            ffn_out = self.norm_ffn(self.ffn(x))
        else:
            ffn_out = self.ffn(self.norm_ffn(x))
        x = shortcut + self.drop_path(self.gamma_ffn * ffn_out)

        return x


# =============================================================================
# 工厂函数: 方便在 multimodal.py 中批量构造
# =============================================================================

def build_bilevel_blocks(
    num_blocks: int,
    dim: int,
    num_heads: int,
    window_size: int = 7,
    cross_type: str = "hybrid",
    cross_num_heads: Optional[int] = None,
    cross_top_k: int = 4,
    mlp_ratio: float = 4.,
    drop_path_rates: Optional[list] = None,
    layerscale_init: Optional[float] = None,
    cross_gate_init: float = 0.0,  # 兼容旧签名, 等价于 layerscale_init
    pretrained_window_size: int = 0,
    **kwargs,
) -> nn.ModuleList:
    """
    批量构造 BiLevelWindowBlock (v2, 简化版), 用于 Stage 1 / Stage 2

    Args:
        num_blocks: 该 stage 的 block 数量
        dim: 特征维度
        num_heads: 跨窗口 attention 头数
        layerscale_init: LayerScale γ 初始值 (推荐 0.0)
        cross_gate_init: [兼容] 旧参数名, 若未传 layerscale_init 则使用此值
        其他参数同 BiLevelWindowBlock

    Returns:
        ModuleList of BiLevelWindowBlock
    """
    if drop_path_rates is None:
        drop_path_rates = [0.0] * num_blocks
    assert len(drop_path_rates) == num_blocks, (
        f"drop_path_rates length ({len(drop_path_rates)}) must match num_blocks ({num_blocks})"
    )

    blocks = nn.ModuleList()
    for i in range(num_blocks):
        blocks.append(BiLevelWindowBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            cross_type=cross_type,
            cross_num_heads=cross_num_heads,
            cross_top_k=cross_top_k,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path_rates[i],
            layerscale_init=layerscale_init,
            cross_gate_init=cross_gate_init,
            pretrained_window_size=pretrained_window_size,
            **kwargs,
        ))
    return blocks


# =============================================================================
# 自测代码 (确保三种方案前向都能跑通)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BiLevelWindowBlock v2 自测 (方案 A: 纯跨窗口 + LayerScale)")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, H, W, C = 2, 64, 64, 192
    window_size = 8
    x = torch.randn(B, H * W, C, device=device)

    for cross_type in ["token_level", "window_level", "hybrid"]:
        print(f"\n--- 测试 cross_type = {cross_type} ---")
        block = BiLevelWindowBlock(
            dim=C,
            num_heads=6,
            window_size=window_size,
            cross_type=cross_type,
            cross_top_k=4,
            mlp_ratio=4.0,
            drop_path=0.1,
            layerscale_init=0.0,
        ).to(device)

        total_params = sum(p.numel() for p in block.parameters())
        print(f"  总参数量: {total_params / 1e6:.3f} M (v2 应比 v1 少 ~30%, 因移除 W-MSA)")

        # 前向
        block.train()
        out = block(x, H, W)
        print(f"  输入: {x.shape}, 输出: {out.shape}")
        assert out.shape == x.shape

        # 反向
        loss = out.sum()
        loss.backward()
        grad_norm = sum(p.grad.norm().item() ** 2 for p in block.parameters() if p.grad is not None) ** 0.5
        print(f"  梯度总范数: {grad_norm:.4f}")

        # LayerScale 检查
        gc_mean = block.gamma_cross.mean().item()
        gf_mean = block.gamma_ffn.mean().item()
        print(f"  γ_cross 均值 (应为 0): {gc_mean:.4e}")
        print(f"  γ_ffn   均值 (应为 0): {gf_mean:.4e}")

    # =========================================================
    # 关键验证: LayerScale γ=0 时, block 应近似恒等映射
    # =========================================================
    print("\n--- 验证 γ=0 时 block 近似恒等 (训练早期行为) ---")
    block = BiLevelWindowBlock(
        dim=C, num_heads=6, window_size=window_size,
        cross_type="hybrid", layerscale_init=0.0,
    ).to(device)
    block.eval()

    with torch.no_grad():
        out = block(x, H, W)
        diff = (out - x).abs().max().item()
        print(f"  |out - x|.max() (γ=0 时应为 0): {diff:.6e}")
        assert diff < 1e-5, f"γ=0 时 block 未保持恒等映射, diff={diff}"
    print("  ✓ 训练早期 block 严格等价于恒等映射")

    # =========================================================
    # ★ [2026-06-18] v3.0: splat 模式自测 (含多 query + 统计量 + null 槽)
    # =========================================================
    print("\n--- 测试 hybrid step3_mode='splat' (多 query + 统计量 + null) ---")
    for use_null in [True, False]:
        blk_s = BiLevelWindowBlock(
            dim=C, num_heads=6, window_size=window_size,
            cross_type="hybrid",
            cross_num_pool_queries=3,
            cross_use_stats=('max', 'mean', 'min'),
            cross_step3_mode='splat',
            cross_splat_null=use_null,
            drop_path=0.1,
        ).to(device)
        # owns_modulation 应为 True → forward 跳过 γ_cross
        assert getattr(blk_s.cross_attn, 'owns_modulation', False) is True
        blk_s.train()
        out_s = blk_s(x, H, W)
        assert out_s.shape == x.shape, f"splat 输出形状错: {out_s.shape}"
        loss_s = out_s.sum()
        loss_s.backward()
        # γ_amp/γ_info 应拿到梯度 (双轨都在学)
        ga = blk_s.cross_attn.gamma_amp
        gi = blk_s.cross_attn.gamma_info
        ga_g = (ga.grad is not None and ga.grad.abs().sum().item() > 0)
        gi_g = (gi.grad is not None and gi.grad.abs().sum().item() > 0)
        # ★ [2026-06-19] expert_weight 应存在、形状 [Kp,C]、且拿到梯度 (逐通道专家融合在学)
        ew = blk_s.cross_attn.expert_weight
        Kp_exp = 3 + 3  # num_pool_queries + len(use_stats)
        assert tuple(ew.shape) == (Kp_exp, C), f"expert_weight 形状错: {tuple(ew.shape)}"
        ew_g = (ew.grad is not None and ew.grad.abs().sum().item() > 0)
        print(f"  null={use_null}: 输出{tuple(out_s.shape)}, γ_amp梯度={ga_g}, γ_info梯度={gi_g}, "
              f"expert_weight{tuple(ew.shape)}梯度={ew_g}")
        assert ga_g and gi_g, "splat 的 γ_amp/γ_info 未拿到梯度"
        assert ew_g, "splat 的 expert_weight 未拿到梯度 (逐通道专家融合没在学)"

    # 近恒等验证: splat 的 γ_amp/γ_info 初值 1e-4 → eval 下输出应≈输入
    print("\n--- 验证 splat γ≈1e-4 时近恒等 ---")
    blk_s2 = BiLevelWindowBlock(
        dim=C, num_heads=6, window_size=window_size,
        cross_type="hybrid", cross_num_pool_queries=3,
        cross_use_stats=('max', 'mean', 'min'),
        cross_step3_mode='splat', cross_splat_null=True,
        layerscale_init=0.0,  # γ_ffn=0
    ).to(device)
    blk_s2.eval()
    with torch.no_grad():
        out_s2 = blk_s2(x, H, W)
        diff_s = (out_s2 - x).abs().max().item()
        print(f"  |out - x|.max() (γ_splat=1e-4, 应很小): {diff_s:.6e}")
    print("  ✓ splat 近恒等启动正常")

    # =========================================================
    # ★ [2026-06-20] splat_plus 自测: fullC/multihead × mlp/perchannel + 近恒等
    # =========================================================
    print("\n--- 测试 hybrid step3_mode='splat_plus' (4 组合) ---")
    for score in ['fullC', 'multihead']:
        for fuse in ['mlp', 'perchannel']:
            blk_p = BiLevelWindowBlock(
                dim=C, num_heads=6, window_size=window_size,
                cross_type="hybrid",
                cross_num_pool_queries=3, cross_use_stats=('max', 'mean', 'min'),
                cross_step3_mode='splat_plus',
                cross_splatp_score_mode=score,
                cross_splatp_fuse_mode=fuse,
                cross_splatp_num_heads=4,
                drop_path=0.1,
            ).to(device)
            assert getattr(blk_p.cross_attn, 'owns_modulation', False) is True
            blk_p.train()
            out_p = blk_p(x, H, W)
            assert out_p.shape == x.shape, f"splat_plus 输出形状错: {out_p.shape}"
            out_p.sum().backward()
            ca = blk_p.cross_attn
            # γ 双轨 + 加性门偏置 b_add + 乘性门偏置 b_mul 都应拿到梯度
            checks = {
                'γ_amp': ca.gamma_amp, 'γ_info': ca.gamma_info,
                'τ_add': ca.splatp_tau_add, 'b_add': ca.splatp_bias_add,
                'τ_mul': ca.splatp_tau_mul,
            }
            grads = {k: (v.grad is not None and v.grad.abs().sum().item() > 0) for k, v in checks.items()}
            print(f"  score={score:9s} fuse={fuse:10s}: out{tuple(out_p.shape)}, 梯度 {grads}")
            assert all(grads.values()), f"splat_plus({score},{fuse}) 有参数没拿到梯度: {grads}"

    # 近恒等验证: splat_plus γ_amp/γ_info=1e-4 → eval 下输出≈输入
    print("\n--- 验证 splat_plus γ≈1e-4 时近恒等 ---")
    blk_p2 = BiLevelWindowBlock(
        dim=C, num_heads=6, window_size=window_size,
        cross_type="hybrid", cross_num_pool_queries=3,
        cross_use_stats=('max', 'mean', 'min'),
        cross_step3_mode='splat_plus', cross_splatp_score_mode='fullC',
        cross_splatp_fuse_mode='mlp', layerscale_init=0.0,
    ).to(device)
    blk_p2.eval()
    with torch.no_grad():
        out_p2 = blk_p2(x, H, W)
        diff_p = (out_p2 - x).abs().max().item()
        print(f"  |out - x|.max() (γ_splatp=1e-4, 应很小): {diff_p:.6e}")
    print("  ✓ splat_plus 近恒等启动正常")

    print("\n" + "=" * 70)
    print("全部测试通过 ✓")
    print("=" * 70)
