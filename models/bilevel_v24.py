"""
BiLevel v24 : 双轨解耦重设计
============================
替代 window_interaction.py 的 HybridCrossWindowAttention._splat_plus.

设计裁决 (2026-07-30, 基于 P3 exp_20260727_201106 的诊断):

  【病因回顾】
  ① fullC 逐通道外积 Imp_c[k,p,c]=Q[k,c]*S[p,c] (无 Σ_c)
     + perchannel fuse (无 Σ_c')  ⇒ 从 Q,S 到 Amp 全程通道对角.
     代数后果: Amp[p,c] = σ(g[c]·S[p,c]/τ + b[c]),  g[c]=Σ_k w[k,c]Q[k,c].
     摘要对乘性轨的全部贡献 = 每通道一个标量.
  ② 加性轨 V[k,c] 无 p 索引 ⇒ 内容分辨率上限 = Kp 个窗口级向量.
  ③ 加性轨 A[k,p,c] 逐通道独立 softmax ⇒ 通道 c 检索摘要 2、通道 c' 检索摘要 5,
     检索出来的不是连贯向量. 实测 cos(μ_膜, μ_背景) = 1.0000.
  ④ info_ln 沿通道归一 ⇒ 每像素注入总能量恒等 = C ⇒ 只能选方向不能选强度.
  ⑤ 三重 LN (s_norm / mul_ln / info_ln) 把逐像素幅度全部抹掉
     ⇒ 只能表达比值型光谱指数, 无法表达差值型/绝对反射率判据.
  ⑥ 两轨共享同一条打分路 (同一 Imp_c), 而两轨对打分的需求相反
     (乘性要逐通道分辨率, 加性要 Σ_c 内积) ⇒ s_proj 收到冲突梯度.

  【本版修正】
  · 乘性轨 = SpectralSpatialGate  (光谱-空间指数门, 承认它是逐点门控, 不是 attention)
      - 保留 fullC+perchannel 的"通道保持"语义 (它对这条轨是正确的)
      - s_proj (Linear C->C) 跨波段混合  + DWConv3x3 逐通道小空间上下文
      - RMSNorm 代替 LayerNorm: 只除范数不减均值 -> 保住幅度符号与相对强度
      - ★ g[c] 直接从 Q 求得, 不经过 N 轴 ⇒ 不再物化 [B,G,Kp,N,C] 五维张量
        (与旧 fullC+perchannel 在数学上逐比特等价, 显存降 ~6x)
      - 去 DC: out = x*(2*Amp-1), 避免 (1+γ·0.5) 的逐块累乘
  · 加性轨 = CrossWindowRetrieval  (跨窗口检索注入)
      - 独立打分路 q_proj_add / s_proj_add (不与乘性轨共享)
      - 头内内积打分 <Qh[k], Sh[p]>/sqrt(d)  ⇒ 真实相似度
      - 单标量 softmax over k  ⇒ 整个摘要向量被一起检索 (修 ③)
      - 输出用 WindowRMSNorm (窗口级标量 RMS) 代替逐像素 LN (修 ④)
      - 打分路保留 QK-RMSNorm: 检索是匹配操作, 归一化在这里是正确的
        (与乘性轨的保幅度取舍相反, 这是有意的不对称)
  · 摘要生成 / 摘要交换加 LayerScale 零初始门控
      P1 (exp_20260728_082354) 在 s2.cross_attn.summary_qkv_proj / pool_query 上
      发散, s2 BiLevel 梯度占比冲到 58%. BiLevel 三步里原本只有第三步被 γ 门住,
      前两步裸奔. 本版给前两步补门.

参考文献:
  Hu et al., Squeeze-and-Excitation Networks, CVPR 2018        (乘性轨的正确家族)
  Yang et al., Gated Channel Transformation, CVPR 2020         (保幅度的通道门)
  Wu et al., CvT, ICCV 2021 / Xie et al., SegFormer, NeurIPS 2021 (投影前后插 DWConv)
  Lee et al., Set Transformer (ISAB), ICML 2019                (inducing points 检索)
  Chu et al., Twins-SVT (GSA), NeurIPS 2021                    (局部窗口+窗口摘要全局注意力)
  Jaegle et al., Perceiver, ICML 2021                          (latent bottleneck)
  Zhang & Sennrich, RMSNorm, NeurIPS 2019                      (只除范数不减均值)
  Touvron et al., CaiT (LayerScale), ICCV 2021
  Bachlechner et al., ReZero, UAI 2021                         (零初始残差门)

显存 (stage1: B=2, G=256, N=64, C=96, Kp=12, H=12, bf16):
  旧 Imp_c [B,G,Kp,N,C] = 37.7 MB/份 x ~3 份(autograd) = 113 MB
  新 alpha [B,G,Kp,N,H] =  9.4 MB/份               ≈  19 MB
  乘性轨已无五维张量.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "RMSNormC",
    "WindowRMSNorm",
    "WindowSummary",
    "SummaryExchange",
    "SpectralSpatialGate",
    "CrossWindowRetrieval",
    "SwiGLUFFN",
    "BiLevelBlockV24",
    "window_partition_tokens",
    "window_reverse_tokens",
]


# ============================================================================
# 归一化原语
# ============================================================================

class RMSNormC(nn.Module):
    """沿通道维的 RMSNorm: 只除范数, 不减均值.

    与 LayerNorm 的关键差别: LN 会减掉逐像素的通道均值, 把"这个像素整体亮"
    这一维信息删掉; RMSNorm 保留它. 对残膜(塑料膜在特定波段绝对反射率高)
    这一维是信号, 不是噪声.

    Zhang & Sennrich, NeurIPS 2019.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C]
        dt = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        out = (xf * rms).to(dt)
        if self.weight is not None:
            out = out * self.weight
        return out


class WindowRMSNorm(nn.Module):
    """窗口级标量 RMS 归一化: 每个窗口除以一个标量.

    对比 splatp_info_ln (沿通道维 LN, 逐像素):
      info_ln  ⇒ 每像素 Σ_c((x_c-b_c)/w_c)^2 ≡ C  ⇒ 逐像素注入总能量被钉死,
                 加性轨只能选方向不能选强度 (病因④).
      本模块   ⇒ 只钉住整个窗口的总尺度, 逐像素/逐通道的相对幅度完整保留.

    输入 [B, G, N, C], 统计在 (N, C) 两轴上做 -> [B, G, 1, 1].
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, G, N, C]
        dt = x.dtype
        xf = x.float()
        rms = xf.pow(2).mean(dim=(-2, -1), keepdim=True).add(self.eps).rsqrt()
        return ((xf * rms).to(dt)) * self.weight


def _tau_eff(raw: torch.Tensor, tau_min: float) -> torch.Tensor:
    """温度重参数化: tau = tau_min + softplus(raw), 恒 > tau_min, 防 tau->0 失控."""
    return tau_min + F.softplus(raw)


def _drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic Depth, 逐样本 (Huang et al., ECCV 2016)."""
    if p <= 0.0 or not training:
        return x
    keep = 1.0 - p
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    return x * x.new_empty(shape).bernoulli_(keep) / keep


def window_partition_tokens(x: torch.Tensor, H: int, W: int, ws: int):
    """[B, H*W, C] -> ([B*G, ws*ws, C], G, Hp, Wp);  H/W 非整除时右下补零."""
    B, N, C = x.shape
    x = x.view(B, H, W, C)
    ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
    if ph or pw:
        x = F.pad(x, (0, 0, 0, pw, 0, ph))
    Hp, Wp = H + ph, W + pw
    nh, nw = Hp // ws, Wp // ws
    x = x.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B * nh * nw, ws * ws, C), nh * nw, Hp, Wp


def window_reverse_tokens(w: torch.Tensor, B: int, C: int, Hp: int, Wp: int,
                          ws: int, H: int, W: int) -> torch.Tensor:
    """([B*G, ws*ws, C]) -> [B, H*W, C];  还原并裁掉 padding."""
    nh, nw = Hp // ws, Wp // ws
    x = w.view(B, nh, nw, ws, ws, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, Hp, Wp, C)
    if Hp != H or Wp != W:
        x = x[:, :H, :W, :].contiguous()
    return x.view(B, H * W, C)


# ============================================================================
# 步骤 1-2: 摘要生成 + 跨窗口摘要交换
# ============================================================================

class WindowSummary(nn.Module):
    """步骤1: 每窗口生成 Kp = M(learnable) + 3(min/max/avg) 个摘要向量.

    P1 事故修正:
      · pool 打分加 RMSNorm (Q/K 都做) -> logits 尺度有界
      · pool softmax 温度有下界 tau_min -> 防注意力过尖导致梯度自加速
      · 统计量 min/max/avg 天然互异, 不参与 learnable 专家的去相关惩罚
    """

    def __init__(
        self,
        dim: int,
        num_learnable: int = 9,
        num_heads: int = 8,
        tau_min: float = 0.9,
        use_stats: Tuple[str, ...] = ("min", "max", "avg"),
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 必须被 num_heads={num_heads} 整除"
        self.dim = dim
        self.M = int(num_learnable)
        self.H = int(num_heads)
        self.d = dim // num_heads
        self.tau_min = float(tau_min)
        self.use_stats = tuple(use_stats)
        self.Kp = self.M + len(self.use_stats)

        self.pool_query = nn.Parameter(torch.randn(self.M, dim) * 0.02)
        # ★ [2026-07-31 归一化审计] 摘要路入口归一化.
        #   理由: 摘要回答"这个窗口里有什么", 是尺度无关的问题; 而 pool_v_proj 与
        #   min/max/avg 统计量原本都直接吃原始残差流 -> probes 幅度 ∝ σ_residual.
        #   P3 实测 s1.std 0.81(ep1) -> 7.00(ep27), probes 会跟着涨 ~9 倍,
        #   下游 g[c] 与 V[k,c] 一起放大 -> 打分门饱和.
        #   ⚠ 只归一化【摘要路】; 双轨路仍吃原始 x (那里的幅度是信号, 见 SpectralSpatialGate).
        self.in_norm = RMSNormC(dim)
        self.pool_k_proj = nn.Linear(dim, dim, bias=False)
        self.pool_v_proj = nn.Linear(dim, dim, bias=False)
        self.pool_q_norm = RMSNormC(self.d)
        self.pool_k_norm = RMSNormC(self.d)
        self.pool_tau = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, G, N, C] -> probes: [B, G, Kp, C]"""
        B, G, N, C = x.shape
        H, d = self.H, self.d
        xn = self.in_norm(x)                                       # ★ 摘要路专用归一化

        q = self.pool_query.view(1, 1, self.M, H, d).expand(B, G, self.M, H, d)
        q = self.pool_q_norm(q)
        k = self.pool_k_proj(xn).view(B, G, N, H, d)
        k = self.pool_k_norm(k)
        v = self.pool_v_proj(xn).view(B, G, N, H, d)

        tau = _tau_eff(self.pool_tau, self.tau_min)
        logits = torch.einsum("bgmhd,bgnhd->bgmnh", q, k) * (d ** -0.5) / tau
        attn = torch.softmax(logits, dim=3)                       # 沿 N 池化
        learn = torch.einsum("bgmnh,bgnhd->bgmhd", attn, v)
        learn = learn.reshape(B, G, self.M, C)

        stats = []
        for s in self.use_stats:
            if s == "min":
                stats.append(xn.amin(dim=2, keepdim=True))
            elif s == "max":
                stats.append(xn.amax(dim=2, keepdim=True))
            elif s == "avg":
                stats.append(xn.mean(dim=2, keepdim=True))
            else:
                raise ValueError(f"未知统计量: {s}")
        if stats:
            return torch.cat([learn] + stats, dim=2)              # [B,G,Kp,C]
        return learn

    def decorr_loss(self, probes: torch.Tensor) -> Optional[torch.Tensor]:
        """learnable 专家之间的内容去相关惩罚 (只惩罚 k < M 的切片).

        完全同码 = (M-1)/M... 这里用 off-diagonal 平方均值, 完全正交 = 0.
        训练端乘 lambda 后加进总 loss.
        """
        if self.M < 2:
            return None
        v = probes[:, :, : self.M]
        v = v - v.mean(dim=-1, keepdim=True)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        cor = torch.einsum("bgmc,bgnc->bgmn", v, v)
        eye = torch.eye(self.M, device=v.device, dtype=v.dtype)
        return ((cor - eye) ** 2).mean()


class SummaryExchange(nn.Module):
    """步骤2: Kp*G 个摘要之间做自注意力, 实现跨窗口信息交换.

    ★ LayerScale 零初始: P1 在 summary_qkv_proj 上发散 (s2 梯度占比 58%).
      原设计里 BiLevel 三步只有第三步有 γ 门, 前两步裸奔. 这里补上.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ls_init: float = 1e-4,
        tau_min: float = 0.9,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.H = int(num_heads)
        self.d = dim // num_heads
        self.tau_min = float(tau_min)
        self.norm = RMSNormC(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNormC(self.d)
        self.k_norm = RMSNormC(self.d)
        self.tau = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.full((dim,), float(ls_init)))

    def forward(self, probes: torch.Tensor) -> torch.Tensor:
        """probes: [B, G, Kp, C] -> [B, G, Kp, C]"""
        B, G, Kp, C = probes.shape
        H, d = self.H, self.d
        S = G * Kp

        h = self.norm(probes).reshape(B, S, C)
        qkv = self.qkv(h).reshape(B, S, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                           # [B,H,S,d]
        q = self.q_norm(q)
        k = self.k_norm(k)
        tau = _tau_eff(self.tau, self.tau_min)
        attn = torch.softmax(torch.einsum("bhsd,bhtd->bhst", q, k) * (d ** -0.5) / tau, dim=-1)
        out = torch.einsum("bhst,bhtd->bhsd", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, S, C)
        out = self.proj(out).reshape(B, G, Kp, C)
        return probes + self.gamma * out


# ============================================================================
# 步骤 3a: 乘性轨 = 光谱-空间指数门
# ============================================================================

class SpectralSpatialGate(nn.Module):
    """乘性轨 v2.

        S      = RMSNorm( DWConv3x3( s_proj(x) ) )         # 学出来的光谱指数 + 小空间上下文
        g[c]   = Σ_k w_mul[k,c] · Q_mul[k,c]               # 摘要 -> 逐通道标量 (无 N 轴!)
        Amp    = sigmoid( g[c]·S[p,c]/τ + b[c] )
        out    = x ⊙ (2·Amp - 1)                           # 去 DC
        x_new  = x + γ_amp ⊙ out = x ⊙ (1 + γ_amp(2Amp-1))

    与旧 fullC+perchannel 的关系:
      旧: Imp_c[k,p,c] = Q[k,c]·S[p,c];  Amp_raw[p,c] = Σ_k w[k,c]·Imp_c[k,p,c]
                                                      = S[p,c]·Σ_k w[k,c]Q[k,c]
      ⇒ 数学上完全等价, 但旧写法先物化了 [B,G,Kp,N,C] 才求和.
        本版直接算 g[c], 五维张量消失.

    保留的设计语义: fullC(逐通道分辨率) + perchannel(通道零串扰).
    这两条对"构造光谱指数"这个目标是正确的, 不改.

    修正的地方: 去掉 mul_ln, 打分路的 LayerNorm 换 RMSNorm.
      理由: 三重 LN 把逐像素幅度全抹掉后, 只能表达比值型指数(NDVI 那类),
            无法表达差值型指数或绝对反射率判据, 而后者是残膜可分性的一部分.
    """

    def __init__(
        self,
        dim: int,
        num_probes: int,
        tau_min: float = 0.9,
        dw_kernel: int = 3,
        gamma_init: float = 0.1,
        use_dwconv: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.Kp = int(num_probes)
        self.tau_min = float(tau_min)
        self.use_dwconv = bool(use_dwconv)

        self.s_proj = nn.Linear(dim, dim, bias=False)              # 跨波段混合
        if self.use_dwconv:
            self.dwconv = nn.Conv2d(
                dim, dim, kernel_size=dw_kernel, padding=dw_kernel // 2,
                groups=dim, bias=True,
            )
            nn.init.zeros_(self.dwconv.bias)
        else:
            self.dwconv = None
        self.dir_weight = nn.Parameter(torch.ones(dim))            # 方向支路的逐通道尺度

        self.q_proj = nn.Linear(dim, dim, bias=False)              # 摘要侧
        # ★ [2026-07-31 归一化审计] 摘要侧 QK-Norm.
        #   没有它 g[c] 随残差流增长 -> sigmoid(g·Ŝ/τ) 饱和.
        #   P3 实测乘性门 logit 跨度 9~18 (标尺 >=2 即算实质调制), 正是饱和的样子.
        self.q_norm = RMSNormC(dim)
        self.w_dir = nn.Parameter(torch.zeros(self.Kp, dim))       # -> g_dir[c] (方向项增益)
        self.w_mag = nn.Parameter(torch.zeros(self.Kp, dim))       # -> g_mag[c] (强度项增益)
        nn.init.normal_(self.w_dir, std=1.0 / math.sqrt(self.Kp))
        nn.init.normal_(self.w_mag, std=1.0 / math.sqrt(self.Kp))

        self.tau = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.full((dim,), float(gamma_init)))

    def forward(
        self,
        x: torch.Tensor,
        probes: torch.Tensor,
        win_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """x: [B,G,N,C]  probes: [B,G,Kp,C]  win_hw: (wh, ww), wh*ww == N"""
        B, G, N, C = x.shape
        wh, ww = win_hw
        assert wh * ww == N, f"win_hw={win_hw} 与 N={N} 不符"

        s = self.s_proj(x)
        if self.dwconv is not None:
            s = s.reshape(B * G, wh, ww, C).permute(0, 3, 1, 2)
            s = self.dwconv(s)
            s = s.permute(0, 2, 3, 1).reshape(B, G, N, C)

        # ── 把"光谱指数"拆成【方向】与【强度】两支 ──────────────────────────
        # ★ [2026-07-31] 上一版只有 RMSNormC(s), 那等于只保留方向.
        #   RMSNorm 与 LayerNorm 只差"减不减均值", 除以逐像素范数这一步两者都做,
        #   所以逐像素强度照样被抹掉 —— 只能表达【比值型】光谱指数(NDVI 那类),
        #   无法表达差值型指数或绝对反射率判据, 而后者是残膜可分性的一部分.
        #
        #   强度支路用【窗口内相对】log 强度而非绝对值:
        #     · 尺度不变 -> 不随残差流膨胀而漂移 (免疫 §3.3 那条 53%->16% 的通路)
        #     · 仍携带"这个像素比邻居亮"这一维 -> 正是 1~2px 细丝的检测线索
        sf = s.float()
        mag = sf.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()      # [B,G,N,1]
        s_dir = (sf / mag).to(s.dtype) * self.dir_weight                 # 方向, 单位 RMS
        lm = torch.log(mag)
        lm_rel = (lm - lm.mean(dim=2, keepdim=True)).to(s.dtype)         # 窗口内相对强度

        q = self.q_norm(self.q_proj(probes))                       # [B,G,Kp,C] ★ 防饱和
        g_dir = torch.einsum("bgkc,kc->bgc", q, self.w_dir)        # [B,G,C]  ★无 N 轴
        g_mag = torch.einsum("bgkc,kc->bgc", q, self.w_mag)        # [B,G,C]

        tau = _tau_eff(self.tau, self.tau_min)
        score = g_dir.unsqueeze(2) * s_dir + g_mag.unsqueeze(2) * lm_rel
        amp = torch.sigmoid(score / tau + self.bias)               # [B,G,N,C]
        return self.gamma * (x * (2.0 * amp - 1.0))                # 去 DC 的增量


# ============================================================================
# 步骤 3b: 加性轨 = 跨窗口检索注入
# ============================================================================

class CrossWindowRetrieval(nn.Module):
    """加性轨 v2.

        Qh, Sh, Vh = 独立投影 (不与乘性轨共享)
        logits[k,p,h] = <Qh[k], Sh[p]> / sqrt(d)          # ★ 头内内积 = 真实相似度
        α[k,p,h]      = softmax_k( logits )                # ★ 单标量竞争, 整向量一起检索
        Info[p]       = Σ_k α[k,p,h] · Vh[k]
        Info          = WindowRMSNorm(Info)                # ★ 窗口级标量, 保逐像素强度
        x_new         = x + γ_info ⊙ Info

    对三条病因的修正:
      ③ 逐通道独立检索 -> 头内单标量竞争. 现在"这个像素检索到哪个摘要"是良定义的.
      ④ 逐像素 LN     -> 窗口级 RMS. 加性轨恢复了"强度"这一维.
      ⑥ 共享打分路     -> 独立 q/s/v 投影. s_proj 不再同时收到两个冲突目标的梯度.

    打分路仍做 QK-RMSNorm: 检索是匹配操作, 归一化在这里是正确的.
    这与乘性轨的"保幅度"取舍相反, 是有意的不对称 —— 两条轨的需求本来就相反.
    """

    def __init__(
        self,
        dim: int,
        num_probes: int,
        num_heads: int = 12,
        tau_min: float = 0.9,
        gamma_init: float = 0.1,
        residual_relative: bool = True,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} 必须被 num_heads={num_heads} 整除"
        self.dim = dim
        self.Kp = int(num_probes)
        self.H = int(num_heads)
        self.d = dim // num_heads
        self.tau_min = float(tau_min)
        # ★ [2026-07-31] True 时 γ_info 直接等于"注入占比"; False 复现旧的绝对幅度语义(消融用)
        self.residual_relative = bool(residual_relative)

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.s_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNormC(self.d)
        self.s_norm = RMSNormC(self.d)
        self.tau = nn.Parameter(torch.zeros(1))
        self.out_norm = WindowRMSNorm(dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(gamma_init)))

    def forward(self, x: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
        """x: [B,G,N,C]  probes: [B,G,Kp,C] -> 增量 [B,G,N,C]"""
        B, G, N, C = x.shape
        H, d, Kp = self.H, self.d, self.Kp

        q = self.q_norm(self.q_proj(probes).view(B, G, Kp, H, d))
        s = self.s_norm(self.s_proj(x).view(B, G, N, H, d))
        v = self.v_proj(probes).view(B, G, Kp, H, d)

        tau = _tau_eff(self.tau, self.tau_min)
        logits = torch.einsum("bgkhd,bgphd->bgkph", q, s) * (d ** -0.5) / tau
        alpha = torch.softmax(logits, dim=2)                       # 沿 k 竞争
        info = torch.einsum("bgkph,bgkhd->bgphd", alpha, v).reshape(B, G, N, C)
        info = self.out_norm(info)                                 # 单位 RMS / 窗口

        # ★ [2026-07-31 归一化审计] 按残差流自身幅度定标注入.
        #   §3.3 的核心量是【注入幅度 / 残差流幅度】:
        #     7.20 残差 1.884, 注入 ~1.0 -> 53%  => 加性轨有效 (8/8, t=-5.44)
        #     P3   残差 6.359, 注入 ~0.9 -> 16%  => 惰性
        #     P1   残差10.089, 注入 ~1.0 -> 10%  => 惰性
        #   上一版仍是 γ_info × (单位 RMS), 幅度被钉死 -> 占比 = γ/σ_residual 仍是【涌现量】,
        #   优化器无法支配它. 乘上残差流自身 RMS 之后:
        #       注入幅度 / 残差流幅度 ≡ γ_info   (恒等, 与 σ_residual 无关)
        #   于是 γ_info 可以【直接读作注入占比】, 设成 0.5 即复刻 7.20 那个工作体制.
        #   (adaLN / FiLM 家族: Perez et al. AAAI 2018; Peebles & Xie ICCV 2023)
        if self.residual_relative:
            rms_x = x.float().pow(2).mean(dim=(-2, -1), keepdim=True).add(1e-6).sqrt()
            info = info * rms_x.to(info.dtype)
        return self.gamma * info

    @torch.no_grad()
    def competition_entropy(self, x: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
        """诊断用: 检索竞争熵 H(α). 均匀 = ln(Kp).

        ★ 判据说明: 旧版 H(A) 判据在 fullC+QK-Norm+tau_min 下不可达,
          因为 logits 被结构性约束在 ~1 量级. 本版用真实内积, logits 可增长,
          所以 H 相对 ln(Kp) 的下降才是可解释的.
        """
        B, G, N, C = x.shape
        H, d, Kp = self.H, self.d, self.Kp
        q = self.q_norm(self.q_proj(probes).view(B, G, Kp, H, d))
        s = self.s_norm(self.s_proj(x).view(B, G, N, H, d))
        tau = _tau_eff(self.tau, self.tau_min)
        logits = torch.einsum("bgkhd,bgphd->bgkph", q, s) * (d ** -0.5) / tau
        a = torch.softmax(logits, dim=2)
        return -(a * (a + 1e-9).log()).sum(dim=2).mean()


# ============================================================================
# 组装
# ============================================================================

class SwiGLUFFN(nn.Module):
    """门控 FFN, 与 transformer_block.SwinEncoderBlock 的 v19 SwiGLU 同构.

    w3( SiLU(w1(x)) * w2(x) ),  hidden = 8/3 * dim (与 Swin block 一致, 参数量可比).
    Shazeer 2020; Touvron et al., LLaMA 2023.
    """

    def __init__(self, dim: int, ratio: float = 8.0 / 3.0):
        super().__init__()
        hidden = int(round(dim * ratio / 8) * 8)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class BiLevelBlockV24(nn.Module):
    """完整的 BiLevel 块: 摘要生成 -> 跨窗口交换 -> 双轨播回.

    forward 输入输出均为 [B*G, N, C] (与现有 SwinEncoderBlock 的窗口张量约定一致).
    需额外传入 num_windows(G) 与 win_hw.

    ★ Kp 不对称说明:
      诊断显示 Kp 的显存代价主要来自旧 fullC 的 [B,G,Kp,N,C]. 本版消掉了它,
      乘性轨的 g[c] 在 N 轴出现之前就算完, 加性轨的 alpha 是 [B,G,Kp,N,H]
      (比旧的小 C/H 倍). 所以 Kp 可以放开: num_learnable 默认 9 -> Kp=12.
      加性轨的内容分辨率上限就是 Kp (病因②), 抬它是直接收益.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_learnable: int = 9,
        pool_heads: int = 8,
        exchange_heads: int = 8,
        retrieval_heads: int = 12,
        tau_min: float = 0.9,
        gamma_init: float = 0.1,
        exchange_ls_init: float = 1e-4,
        use_dwconv: bool = True,
        use_mul: bool = True,
        use_add: bool = True,
        use_ffn: bool = True,
        ffn_ratio: float = 8.0 / 3.0,
        ffn_ls_init: float = 1e-2,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        # ★ [2026-07-31] win_hw 由 window_size 常量推得, 不进 forward 签名,
        #   以保持与现有 _build_stage 的 block 调用契约 forward(x_windows, num_windows) 一致.
        self.window_size = int(window_size)
        self.drop_path_p = float(drop_path)
        self.use_mul = bool(use_mul)
        self.use_add = bool(use_add)

        self.summary = WindowSummary(
            dim, num_learnable=num_learnable, num_heads=pool_heads, tau_min=tau_min
        )
        self.Kp = self.summary.Kp
        self.exchange = SummaryExchange(
            dim, num_heads=exchange_heads, ls_init=exchange_ls_init, tau_min=tau_min
        )
        self.mul_track = (
            SpectralSpatialGate(
                dim, self.Kp, tau_min=tau_min, gamma_init=gamma_init, use_dwconv=use_dwconv
            )
            if self.use_mul
            else None
        )
        self.add_track = (
            CrossWindowRetrieval(
                dim, self.Kp, num_heads=retrieval_heads, tau_min=tau_min, gamma_init=gamma_init
            )
            if self.use_add
            else None
        )
        # ★ [2026-07-31] FFN 子层, res-post-norm 形式, 与 SwinEncoderBlock V2 同构:
        #     shortcut = x; x = ffn(x); x = norm_ffn(x); x = γ_ffn*x; x = shortcut + x
        #   为什么必须有:
        #     ① v23 的 BiLevelWindowBlock 自带 FFN. 去掉它, stage1 的 FFN 数从 6 掉到 3,
        #        v23↔v24 的对比会多一个混淆项.
        #     ② gamma_report 实测: γ_ffn 是 BiLevel 块里【唯一】真在学的 γ
        #        (s2.b3 裸值 1.3883, γ⊙w = 1.9543, 相对 warmup 起点 1.0 长了 39%/95%;
        #         同期 γ_amp/γ_info 均值只动 1~3%). 删掉它等于删掉唯一被证明有效的子层.
        self.ffn = SwiGLUFFN(dim, ratio=ffn_ratio) if use_ffn else None
        self.norm_ffn = RMSNormC(dim) if use_ffn else None
        self.gamma_ffn = nn.Parameter(torch.full((dim,), float(ffn_ls_init))) if use_ffn else None

        # ★ 属性名与 v23 的 HybridCrossWindowAttention 保持一致,
        #   训练端遍历 modules() 取 _vdecorr_loss 的现有逻辑无需改动.
        self.splatp_v_decorr_on: bool = False
        self._vdecorr_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """★ [2026-07-31] 契约与 window_interaction.BiLevelWindowBlock.forward 完全一致:
            输入 [B, H*W, C] 整图 token, 窗口划分在【内部】完成, 输出同形.
        这样 multimodal 的 _run_with_checkpoint(block, rgb_x, H, W, ...) 调用点一行不动.
        """
        B, N_all, C = x.shape
        assert N_all == H * W, f"N({N_all}) != H*W({H * W})"
        ws = self.window_size
        xw, G, Hp, Wp = window_partition_tokens(x, H, W, ws)
        out_w = self._forward_windows(xw, G)
        return window_reverse_tokens(out_w, B, C, Hp, Wp, ws, H, W)

    def _forward_windows(self, x_windows: torch.Tensor, num_windows: int) -> torch.Tensor:
        """x_windows: [B*G, ws*ws, C] -> [B*G, ws*ws, C]"""
        BG, N, C = x_windows.shape
        G = int(num_windows)
        assert BG % G == 0, f"B*G={BG} 不能被 G={G} 整除"
        ws = self.window_size
        assert ws * ws == N, f"window_size={ws} 推出的 N={ws * ws} 与实际 N={N} 不符"
        B = BG // G
        x = x_windows.view(B, G, N, C)

        probes = self.summary(x)
        probes = self.exchange(probes)
        if self.splatp_v_decorr_on and self.training:
            self._vdecorr_loss = self.summary.decorr_loss(probes)
        else:
            self._vdecorr_loss = None

        delta = torch.zeros_like(x)
        if self.mul_track is not None:
            delta = delta + self.mul_track(x, probes, (ws, ws))
        if self.add_track is not None:
            delta = delta + self.add_track(x, probes)
        x = x + _drop_path(delta, self.drop_path_p, self.training)

        if self.ffn is not None:                       # res-post-norm FFN 子层
            f = self.gamma_ffn * self.norm_ffn(self.ffn(x))
            x = x + _drop_path(f, self.drop_path_p, self.training)
        return x.reshape(BG, N, C)

    def pop_decorr_loss(self) -> Optional[torch.Tensor]:
        """可选的显式取用接口; 训练端也可直接遍历 modules() 读 _vdecorr_loss."""
        loss = self._vdecorr_loss
        self._vdecorr_loss = None
        return loss
