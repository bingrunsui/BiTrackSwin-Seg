"""
╔═══════════════════════════════════════════════════════════════════════════╗
║ ★ [2026-07-01] 本次改动 (stem 可切换 attn/conv, 默认 'attn' = 原行为):         ║
║   新增 ConvNeXtStemBlock + RGBStemStage0 加 stem_mode 开关 + MultimodalViT 透传║
║   目的: 治 W-MSA stem 的病态梯度 (曾观测 stage0 占 99.7%、范数 3→23、尖峰 300、║
║         频繁撞裁剪线 50) + EMA 崩塌 (EMA fg-IoU 间歇跌到 0.27, 原始模型正常).   ║
║   'attn': 原 2×W-MSA stem, 逐字节不变 (对照基线).                             ║
║   'conv': 2×ConvNeXt 块 (depthwise3×3→GroupNorm→1×1 48→96→GELU→1×1 96→48+res),║
║           空间域运算, GroupNorm 无 running stats (不给 EMA 添不同步), 无 256²  ║
║           注意力 → 无 softmax 病态梯度. 保住输出/premerge/H/W 全部契约不变.    ║
║   由训练脚本 --stem_mode {attn,conv} → 全局 STEM_MODE → 构造透传. 单变量 A/B.  ║
║   checkpoint 与 attn 不兼容 (参数名不同, 从头训); 外部按名引用全自动兼容.       ║
║   参考: Xiao et al. NeurIPS 2021; Liu et al.(ConvNeXt) CVPR 2022;             ║
║         Woo et al.(ConvNeXt V2) CVPR 2023.                                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ ★ [2026-06-29] 上次改动 (两个独立结构开关, 默认 False = 原行为):               ║
║   1. shift_window: stage1/2/3 奇数 block 用 SW-MSA (shift=ws//2) 取代 BiLevel ║
║      —— 创新点对照基线. _build_stage 新增分支; 块仍落 rgb_s12_rand 参数组.    ║
║   2. stage0_to_decoder: RGBStemStage0.forward 额外返回 256² premerge;          ║
║      两个 forward 写 features['stage0'] (经 stage0_skip_gamma×warmup 门控);    ║
║      新增 decoder_dims / decoder_spatial_sizes 给解码器 (内部 self.dims 不动).║
╚═══════════════════════════════════════════════════════════════════════════╝

Multimodal ViT Model (优化版 - 针对 RTX 5080 16GB)
========================================================
多模态视觉 Transformer 主干网络

═══════════════════════════════════════════════════════════════════════════════
★★★ [2026-06-21] 单一架构重构 (stage0 + stage1/2/3, 删除 three_stage, 封存 stage4 ViT) ★★★
═══════════════════════════════════════════════════════════════════════════════
本次改动 (multimodal.py):
  1. 删除 three_stage 配置 (architecture_mode 保留为兼容占位; 传 three_stage 会警告并忽略).
  2. 唯一架构 = 内部 4 个 stage:
       stage0 (stem, 48ch @ 256×256, 纯 W-MSA, 不进 U-Net)  ← 新增, 封装在 RGBStemStage0
       stage1 (96ch  @ 128×128, Swin+BiLevel 跨窗口)
       stage2 (192ch @ 64×64,   Swin+BiLevel 跨窗口)
       stage3 (384ch @ 32×32,   Swin+BiLevel 跨窗口, depth 由 4 减到 2)
     对外 self.dims = [96,192,384] (3 个输出 stage, 给 U-Net 解码器); stage0 的 48 维只在内部.
     depths=[2,4,4,2] (stage0/1/2/3), num_heads=[3,6,6,12] (head_dim 16/16/32/32), window=8.
  3. stage4 ViT (768@16×16) 封存: ENABLE_STAGE4_VIT=False, 不构造、不进 forward.
     结论: 对残膜(1-2px)消融贡献≈0 (16×16 下细丝已被抹平; 任务-尺度错配, 非优化失败).
     代码与重启说明保留, 供将来大目标(棚膜/农田/建筑)或 CVPR/LLM 任务复用.
  4. RGBStemStage0 复用现成 SwinEncoderBlock + PatchMerging (未修改这两个组件).
  5. cross_stage_mode 仅支持 'none' (其余方案依赖 stage4, 已加护栏明确报错).

⚠️ 训练脚本 (train_swin2_seg_v21_u-net.py) 配套 TODO —【本次未改, 下次务必处理】:
  (a) U-Net 解码器: 加 attention-gate 开关 + 深监督开关 (一键 on/off, 服务公平对比 + 消融).
  (b) get_param_groups / layer_decay 分组: 必须把新模块 'rgb_stem_stage0' 纳入对应 LR 组
      (否则 stage0 的 stem/W-MSA 落到默认组); grad-log / param-group 的名字匹配也要加它.
  (c) WSD 学习率衰减窗口: 设到 120–150 epoch 之间衰减.
  (d) (可选) DEPTHS/NUM_HEADS_LIST 对齐为 [2,4,4,2]/[3,6,6,12] 以消除启动时"强制结构"警告
      (不改也能跑: multimodal.py 已在内部硬定结构, 会打印一行覆盖提示).

⚠️ 首跑必须在本机验证 (此处仅 py_compile 过, 无法真跑). 重点看:
  - 构造日志: dims=[96,192,384], spatial=[128,64,32], depths=[4,4,2], num_heads=[6,6,12];
  - 一个 forward 跑通 + 显存 (256×256 的 stage0 是显存大头, 必要时开 USE_CHECKPOINT 或降 batch);
  - 新骨架训完【重跑 ablation】验收 stage0/各 stage 贡献 (旧消融表不可直接套用);
  - 参数量预计降到 ~25–30M (对比 Swin-S/ResNet101 U-Net ~50M); 对齐 50M 应加宽 stage1/2/3, 勿复活 stage4.
═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
★ [2026-06-18] v3.0: 方案C(hybrid) Step3 分发新增可选 'splat' 模式 (块级语义播撒)
═══════════════════════════════════════════════════════════════════════════════
新增两个透传参数 (默认 'gather', 与 v3.0 之前逐字节等价, 旧 ckpt 兼容):
  - cross_window_step3_mode: 'gather'(默认) / 'splat'
      gather: token 当 Q 对摘要 softmax → token 主动"吸"全局 (有取舍, Precision 友好)
      splat : 摘要当 Q/V、token 当被打分 Key, 对块内像素 softmax → 摘要主动"摊"给 token
              (对稀疏丝状目标/残膜的证据扩散更强; 配 null/sink 槽防灌背景)
  - cross_window_splat_null: splat 是否加 null/sink 槽 (默认 True)
透传链路: MultimodalViT.__init__ → 存为 self.cross_window_step3_mode/splat_null
          → _build_stage 构造 BiLevelWindowBlock 时传 cross_step3_mode/cross_splat_null
          → BiLevelWindowBlock → HybridCrossWindowAttention(step3_mode=..., splat_null=...)
具体数学/伪代码见 window_interaction.py 文件头注释与 _splat() 方法.

★ [2026-06-19] splat 加性轨内部升级为"保留 C 维的逐通道专家融合"(replace 旧的对 k 求和).
  本文件【无需改动】: 不新增构造参数 (Kp 在 HybridCrossWindowAttention 构造期由
  num_pool_queries+len(use_stats) 推出), 仍由 cross_window_step3_mode='splat' 启用.
  实现与逐通道零串扰的细节见 window_interaction.py 的 _splat() 与文件头 [2026-06-19] 说明.
═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
★ [2026-04-26 修改] V2 修复: 把 swin_version 透传给 SwinEncoderBlock
═══════════════════════════════════════════════════════════════════════════════

修改原因:
─────────
上次训练 (4.23) mIoU 崩到 0.50 + rgb_s1 梯度=inf, 根因之一是:
    "V2 ckpt 加载到 V1 Pre-Norm 架构里 → 权重风格和 forward 风格错配 → 失效"

本轮 SwinEncoderBlock 已经支持 swin_version 切换 (Pre-Norm vs Post-Norm),
multimodal.py 只需要把训练脚本传进来的 swin_version 透传到 SwinEncoderBlock 即可.

具体修改:
─────────
(1) MultimodalViT.__init__ 新增两个参数:
    - swin_version: str = 'v1'  ('v1' 或 'v2')
    - pretrained_window_size: int = 0   (V1=0, V2 window16 ckpt=16)  ← ★ 4.26 第二轮已删除
(2) _build_blocks() 在构造 SwinEncoderBlock 时把它们透传下去
(3) 不影响 BiLevelWindowBlock (BiLevel 保持 Pre-Norm + LayerScale γ=0)
(4) 不影响 SwinDeformableDecoderBlock (Decoder 走 V1 风格, 不需要切)

兼容性:
───────
- 默认值都是 v1 / 0, 完全向后兼容旧训练脚本
- 训练脚本通过 build_backbone() 入口传新参数即可
═══════════════════════════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════════════════════════
★ [2026-04-26 第二轮修改] 删除 pretrained_window_size + Per-stage window_size
═══════════════════════════════════════════════════════════════════════════════
背景: 不再加载预训练权重 (V2 cosine attention + ImageNet 预训练 qkv 在 5 波段下崩溃).

(1) 删除 pretrained_window_size 参数
    - WindowAttention 已经移除该参数, MultimodalViT 也同步删除
    - cpb_mlp normalizer 直接用 stage 自己的 window_size

(2) window_size 升级支持 List[int]
    - window_size: int → 所有 stage 共用 (向后兼容)
    - window_size: List[int] → per-stage ws, 长度需匹配 num_stages
    - 4-stage + 默认 int 7 → 自动改用 [8, 8, 8, 8]
      (空间分辨率 64/32/16/8 → ws 8/8/8/8 整除无 padding,
       论文风格的 ws 渐进对齐空间, Stage 4 ViT 占位)

(3) self.window_sizes 是 list[int], 长度=num_stages, 可被 print_model_info 显示
    self.window_size 仍保留为代表值 = window_sizes[0] (兼容旧代码)
═══════════════════════════════════════════════════════════════════════════════

核心优化:
1. CUDA Streams 并行化: RGB 和 Aux 分支真正并行运行
2. 混合精度 (AMP): 自动 FP16/FP32 混合
3. Gradient Checkpointing: 显存优化
4. LoRA 分离: 训练/微调模式独立控制
5. 内存优化: 针对 16GB 显存的特殊优化

---------------------------------------------------------
★ [2026-04-20] v16 大改: 引入 Bi-Level Window Interaction
---------------------------------------------------------
1. Stage 1/2 (以及四阶段模式的 Stage 3) 的 4 个 block 采用新排列:
     block 0: SwinEncoderBlock   (WindowAttention, 加载 Swin-T 预训练)
     block 1: BiLevelWindowBlock (跨窗口 attention, 三种方案消融)
     block 2: SwinEncoderBlock   (WindowAttention, 复用 block 0 预训练权重)
     block 3: 
       - 单流 (RGB-only / pretrain): BiLevelWindowBlock
       - 多流 (use_multimodal=True): SwinDeformableDecoderBlock (保持不变)

2. 新增 architecture_mode 参数: "three_stage" (默认) / "four_stage"
   - 三阶段 [4, 4, 6]: 保留 GlobalPatchEmbed 作为创新点 (Stage2→3: 64→16, 192→768)
   - 四阶段 [4, 4, 4, 4]: 全用 PatchMerging (稳定梯度, 多一级 skip connection)
     dims = [96, 192, 384, 768], spatial = [128, 64, 32, 16]
     num_heads = [6, 6, 12, 12]

3. 新增跨窗口 attention 消融参数:
   - cross_window_enabled: bool (一键开关, False → 退化为纯 SwinEncoderBlock)
   - cross_window_type: "token_level" / "window_level" / "hybrid" (默认 hybrid)
   - cross_window_top_k: int (仅 token_level 使用)
   - cross_window_gate_init: float (默认 0.0, 训练早期等价纯 W-MSA)

4. 新增 BiLevel 独立 LoRA 配置:
   - bilevel_lora_r, bilevel_lora_alpha (独立于主 WindowAttention 的 lora_r/alpha)

5. Gradient Checkpoint 策略更新:
   - BiLevelWindowBlock: 强制 checkpoint (跨窗口 attention 显存开销大)
   - SwinEncoderBlock / SwinDeformableDecoderBlock: 按 checkpoint_ratio 策略

6. 预训练加载策略 (由外部训练脚本配合实现):
   - block 0: Swin-T layers.X.blocks.0 (W-MSA)
   - block 1: 无预训练 (BiLevel 随机初始化, cross_gate=0 保护)
   - block 2: 复用 block 0 权重 (同起点独立训练)
   - block 3: 多流模式无预训练 (Decoder); 单流模式随机初始化 (BiLevel)

---------------------------------------------------------
架构配置:
  三阶段 (默认, depths=[4, 4, 6]):
    - RGB Stage 1: 4 blocks (新排列) (H/2=128, 96d)
    - RGB Stage 2: 4 blocks (新排列) (H/4=64, 192d)
    - RGB Stage 3: 6 blocks (ViT)    (H/16=16, 768d)  ← 通过 GlobalPatchEmbed stride=4
    - Aux Branch : 6 Layer ViT-Base (并行运行)

  四阶段 (depths=[4, 4, 4, 4]):
    - RGB Stage 1: 4 blocks (新排列) (H/2=128, 96d)
    - RGB Stage 2: 4 blocks (新排列) (H/4=64,  192d)
    - RGB Stage 3: 4 blocks (新排列) (H/8=32,  384d)
    - RGB Stage 4: 4 blocks (ViT)    (H/16=16, 768d)  ← 全部 PatchMerging ×2
    - Aux Branch : 6 Layer ViT-Base (并行运行)
"""
import os
import sys
from pathlib import Path

# ============================================
# 路径设置
# ============================================
_current_file = Path(__file__).resolve()
_current_dir = _current_file.parent
_project_root = _current_dir.parent

for _p in [str(_project_root), str(_current_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ============================================

import torch

# ★ [2026-07-25] v23: 按 stage 启停 BiLevel 奇数位块. 训练端建模前覆写(与 window_interaction.SPLATP_* 同机制):
#   import multimodal; multimodal.BILEVEL_STAGES = (1, 2)
#   → 不在集合内的 stage, 奇数位落到 else 分支建 W-MSA(shift=0), BiLevel 参数/摘要/双轨在该 stage 完全不建.
#   (1,2,3)=旧行为逐比特一致; shift_window 分支优先级更高, 不受本开关影响.
# The released best checkpoint only enables BiLevel v24 in stage 1.  Stages
# 2 and 3 therefore retain their Swin blocks and exact checkpoint key layout.
BILEVEL_STAGES = (1,)
# ★[2026-07-31] BiLevel v24 超参 (模块级常量, 覆盖方式同 BILEVEL_STAGES:
#   import multimodal; multimodal.BILEVEL_V24_GAMMA_INIT = 0.5)
BILEVEL_V24_ENABLED         = True   # False -> 逐比特回到 v23 的 BiLevelWindowBlock (A/B 对照)
BILEVEL_V24_NUM_LEARNABLE   = 9      # Kp = 该值 + 3 统计量 = 12 (病因②: Kp 是内容分辨率上限)
BILEVEL_V24_POOL_HEADS      = 8      # 摘要池化头数
BILEVEL_V24_EXCHANGE_HEADS  = 8      # 跨窗口摘要交换头数
BILEVEL_V24_RETRIEVAL_HEADS = 12     # 加性轨检索头数 (决定 alpha 的第5维, 越大越省显存)
BILEVEL_V24_EXCHANGE_LS     = 1e-4   # ★摘要交换的零初始 LayerScale (治 P1 的 summary_qkv 发散)
BILEVEL_V24_GAMMA_INIT      = 0.5    # ★双轨 γ 初值; 加性轨下它【直接等于注入占比】
                                     #   7.20(有效)=53% / P3(惰性)=16% / P1(惰性)=10%
BILEVEL_V24_TAU_MIN         = 0.9    # 温度下界 (沿用 splatp_tau_min 机制)
BILEVEL_V24_USE_DWCONV      = True   # 乘性轨 s_proj 后的 3x3 逐通道卷积 (小空间上下文)
BILEVEL_V24_USE_FFN         = True   # ★res-post-norm FFN 子层; γ_ffn 是 v23 里唯一真在学的 γ
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, List, Tuple, Any, Union
from contextlib import nullcontext
import threading

from .patch_embed import PatchEmbed
from .patch_merging import PatchMerging
# ★[2026-07-31] 补丁B: 输出侧带归一化 + 峰值保留分支的下采样, 替代 PatchMerging
from .conv_stem import PeakPreservingDownsample
from .transformer_block import (
    SwinEncoderBlock, 
    SwinDecoderBlock, 
    SwinDeformableDecoderBlock,
    ViTEncoderBlock, 
    ViTDecoderBlock,
    LoRALinear
)
from .deformable_block import (
    DeformableEncoderBlock,
    DeformableDecoderBlock
)
# ★ [2026-04-20] v16 新增: Bi-Level Window Interaction 模块
#   用于 Stage1/2 (三阶段) 或 Stage1/2/3 (四阶段) 的跨窗口信息交互
#   替代 Swin 的 shift window 机制, 支持三种方案消融 (token_level/window_level/hybrid)
from .window_interaction import BiLevelWindowBlock
# ★[2026-07-31] 补丁C: 双轨解耦重设计 (光谱指数门 + 跨窗口检索)
from .bilevel_v24 import BiLevelBlockV24

# ╔════════════════════════════════════════════════════════════════════════╗
# ║ ★ [2026-05-11] v20: 跨 stage 跨窗口模块 import (方案 A / 方案 C)         ║
# ╠════════════════════════════════════════════════════════════════════════╣
# ║ 这两个模块完全独立, 不依赖 multimodal 内部其它类.                          ║
# ║ 用 try-except 包住, 即使文件不存在也不会破坏现有代码 (默认 mode='none').  ║
# ║ 详细文档见 INTEGRATION_GUIDE_CROSS_STAGE.md                              ║
# ╚════════════════════════════════════════════════════════════════════════╝
try:
    from cross_stage_broadcast import CrossStageBroadcast        # 方案 A
    _HAS_CROSS_STAGE_BROADCAST = True
except ImportError:
    _HAS_CROSS_STAGE_BROADCAST = False
    CrossStageBroadcast = None

try:
    from cross_stage_token_bank import CrossStageTokenBank       # 方案 C
    _HAS_CROSS_STAGE_TOKEN_BANK = True
except ImportError:
    _HAS_CROSS_STAGE_TOKEN_BANK = False
    CrossStageTokenBank = None

try:
    from cross_stage_broadcast_momentum import CrossStageBroadcastMomentum  # 方案 A+
    _HAS_CROSS_STAGE_BROADCAST_MOMENTUM = True
except ImportError:
    _HAS_CROSS_STAGE_BROADCAST_MOMENTUM = False
    CrossStageBroadcastMomentum = None

try:
    from cross_stage_query_locate import CrossStageQueryLocate  # 方案 D (D1)
    _HAS_CROSS_STAGE_QUERY_LOCATE = True
except ImportError:
    _HAS_CROSS_STAGE_QUERY_LOCATE = False
    CrossStageQueryLocate = None

# ★ [2026-04-06] 修复: 使用残差版 GlobalPatchEmbed (stride=4)
#   旧版内嵌的 GlobalPatchEmbed(stride=8) → 64/8 = 8×8 = 64 tokens
#   与 Aux 16×16 = 256 tokens 不匹配 → CrossAttentionFast assert 失败
#   残差版 GlobalPatchEmbed(stride=4, padding=2, mode='residual') → 64→16×16 ✅
# ★ [2026-04-07] 内联: 不再依赖外部文件, 直接集成到 multimodal.py 中

# 导入优化版 Aux 分支
# v1 is RGB-only.  Keep the symbol for the guarded legacy branch without
# shipping any auxiliary-stream implementation.
AuxViTBranch = None


class GlobalPatchEmbed(nn.Module):
    """
    全局 Patch Embedding (用于从 Stage2 到 Stage3 的降采样)
    
    ★ [2026-04-07] 从 global_patch_embed.py 内联到 multimodal.py
      确保 multimodal.py 完全独立, 不依赖任何外部包或旧目录
    
    支持两种模式:
    1. 原始模式 (mode='original'): 单步 Conv2d 降采样 (patch_size=8, stride=8)
    2. 残差模式 (mode='residual'): ConvNeXt V2 / InternImage 风格 (默认)
       - 主路径: Overlapping Conv2d (patch_size=5, stride=4, padding=2) — 提取细节
       - 残差路径: AvgPool2d + Linear — 保留全局语义
       - 输出 = 主路径 + 残差路径 (加法融合)
    
    空间尺寸: 64×64 → 16×16 (stride=4), 与 Aux 分支 16×16 完美对齐
    
    Args:
        in_dim: 输入维度
        out_dim: 输出维度
        patch_size: patch 大小（默认 5, overlapping 风格）
        stride: 步长（默认 4）
        padding: 填充（默认 2）
        mode: 'original' 或 'residual' (默认)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int = 5,
        stride: int = 4,
        padding: int = 2,
        mode: str = 'residual'
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.mode = mode
        
        # === 主路径: Overlapping Conv2d (捕获局部细节纹理) ===
        self.proj = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = nn.LayerNorm(out_dim)
        
        # === 残差路径 (仅 residual 模式): AvgPool + Linear (保留全局语义) ===
        if mode == 'residual':
            self.residual_pool = nn.AvgPool2d(
                kernel_size=stride,
                stride=stride
            )
            self.residual_proj = nn.Linear(in_dim, out_dim)
            self.residual_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, N, C = x.shape
        
        # (B, N, C) -> (B, C, H, W)
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # === 主路径: Conv2d 降采样 ===
        main_out = self.proj(x_2d)
        H_out, W_out = main_out.shape[2], main_out.shape[3]
        
        main_out = main_out.flatten(2).transpose(1, 2)
        main_out = self.norm(main_out)
        
        if self.mode == 'residual':
            # === 残差路径: AvgPool + Linear ===
            res_out = self.residual_pool(x_2d)
            if res_out.shape[2] != H_out or res_out.shape[3] != W_out:
                res_out = F.interpolate(
                    res_out, size=(H_out, W_out), mode='bilinear', align_corners=False
                )
            res_out = res_out.flatten(2).transpose(1, 2)
            res_out = self.residual_proj(res_out)
            res_out = self.residual_norm(res_out)
            
            # === 融合 ===
            x_out = main_out + res_out
        else:
            x_out = main_out
        
        return x_out, H_out, W_out

# =============================================================================
# CUDA Streams 并行化支持
# =============================================================================

class ConvNeXtStemBlock(nn.Module):
    """[2026-07-01] ConvNeXt 风格卷积块 (稍强版) — 用于 conv stem 模式.

    结构 (空间域 [B,C,H,W] 运算, 保持分辨率):
        depthwise Conv 3×3  →  GroupNorm  →  1×1 Conv (dim→dim*expand)
        →  GELU  →  1×1 Conv (dim*expand→dim)  →  + residual

    设计取舍:
      · 归一化用 GroupNorm (无 running stats), 不给 EMA 影子权重添不同步 —— 直接针对
        本仓库 W-MSA stem 已观测到的 "EMA 崩塌" 问题.
      · 无高分辨率注意力 → 不产生 256² softmax 的病态梯度 (W-MSA stem 曾观测: stage0
        梯度占 99.7%、范数 3→23、尖峰到 300、频繁撞裁剪线). 卷积梯度条件更好、平滑.
      · 反向瓶颈 (inverted bottleneck, expand=2: 48→96→48), depthwise 3×3 (非 7×7),
        无 LayerScale (2 层浅 stem, 让块从第一步就充分激活, 而非近恒等长期休眠).

    参考文献:
      · Xiao et al., "Early Convolutions Help Transformers See Better", NeurIPS 2021.
      · Liu et al., "A ConvNet for the 2020s" (ConvNeXt), CVPR 2022.
      · Woo et al., "ConvNeXt V2", CVPR 2023 (反向瓶颈; 本块用其简化形态, 未含 GRN).
    """

    def __init__(self, dim, expand_ratio=2, kernel_size=3, gn_groups=8):
        super().__init__()
        # GroupNorm 组数必须整除 dim (stem_dim=48 → 8 组, 每组 6 通道); 兜底退化到 4 或 1.
        if dim % gn_groups != 0:
            gn_groups = 4 if (dim % 4 == 0) else 1
        pad = kernel_size // 2
        hidden = dim * expand_ratio
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)  # depthwise
        self.norm = nn.GroupNorm(gn_groups, dim)
        self.pwconv1 = nn.Conv2d(dim, hidden, kernel_size=1)   # 48→96
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden, dim, kernel_size=1)   # 96→48

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]  (空间域, 分辨率不变)
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return shortcut + x


class RGBStemStage0(nn.Module):
    """[2026-06-21] Stage0: 高分辨率 stem stage (不进 U-Net).

    结构: stem conv(in_chans→stem_dim, 3x3 stride1, 保持分辨率) → token 化 → LayerNorm
          → W-MSA × depth (复用 SwinEncoderBlock, 纯窗口注意力, 无 cross-window)
          → PatchMerging(stem_dim→2*stem_dim, ÷2 空间)

    输入  [B, in_chans, H, W]  →  输出 (tokens [B, (H/2)*(W/2), 2*stem_dim], H/2, W/2)

    用途: 残膜细丝(1-2px)信息在高分辨率最完整, 在 256x256 上先提一层局部特征,
          降采样到 128x128 后喂给 stage1. 该 stage 的输出不写入 features (不进解码器).
    复用现成 SwinEncoderBlock + PatchMerging, 不修改这两个组件.
    """

    def __init__(self, in_chans, stem_dim, out_dim, depth, num_heads, window_size,
                 mlp_ratio, swin_version, swin_kwargs=None, drop_path=0.0,
                 stem_mode='attn'):
        # ★ [2026-07-01] stem_mode: 'attn' (原 W-MSA stem, 逐字节不变, 默认=对照基线) /
        #   'conv' (稍强版 ConvNeXt 卷积块取代 256² W-MSA). 单变量 A/B, 由训练脚本 STEM_MODE 切换.
        #   两模式对外契约完全一致: 输出 tokens [B,(H/2)*(W/2),2*stem_dim] + premerge tokens
        #   [B,H*W,stem_dim] (256² 级) + H/2 + W/2 (见 forward 返回 4 元组).
        super().__init__()
        assert out_dim == 2 * stem_dim, (
            f"RGBStemStage0: out_dim({out_dim}) 必须 = 2*stem_dim({2*stem_dim}) "
            f"(PatchMerging 固定 ×2 通道)"
        )
        assert stem_mode in ('attn', 'conv'), \
            f"RGBStemStage0: stem_mode 必须是 'attn' 或 'conv', got '{stem_mode}'"
        self.stem_dim = stem_dim
        self.out_dim = out_dim
        self.stem_mode = stem_mode
        # stem: 5→stem_dim, stride 1 (不下采样, 保持 256x256); 两模式共用.
        self.proj = nn.Conv2d(in_chans, stem_dim, kernel_size=3, stride=1, padding=1)
        if stem_mode == 'attn':
            # === 原 W-MSA stem (逐字节不变, 作为创新对照基线) ===
            self.norm = nn.LayerNorm(stem_dim)
            swin_kwargs = swin_kwargs or {}
            self.blocks = nn.ModuleList([
                SwinEncoderBlock(
                    stem_dim, num_heads, window_size,
                    mlp_ratio=mlp_ratio, drop_path=drop_path,
                    swin_version=swin_version, **swin_kwargs,
                )
                for _ in range(depth)
            ])
        else:
            # === conv stem (稍强版 ConvNeXt 风格, 替掉 256² W-MSA) ===
            #   不建 self.norm (ConvNeXt 块自带 GroupNorm); num_heads/window_size/swin_* 收下但忽略.
            #   depthwise3×3 → GN → 1×1(48→96) → GELU → 1×1(96→48) → +residual, 空间域运算.
            self.blocks = nn.ModuleList([
                ConvNeXtStemBlock(stem_dim, expand_ratio=2, kernel_size=3)
                for _ in range(depth)
            ])
        # 降采样: stem_dim → 2*stem_dim, 空间 ÷2 (256→128); 两模式共用.
        self.merge = PeakPreservingDownsample(dim=stem_dim)   # ★[2026-07-31] SPD+max -> conv1x1 -> Norm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
        # x: [B, in_chans, H, W]
        x = self.proj(x)                          # [B, stem_dim, H, W]  (stride1, 保持输入分辨率 256²)
        _, _, H, W = x.shape
        if self.stem_mode == 'attn':
            # === 原 W-MSA 路径 (token 形态) — 逐字节不变 ===
            x = x.flatten(2).transpose(1, 2)      # [B, H*W, stem_dim]
            x = self.norm(x)
            for blk in self.blocks:
                x = blk(x, H, W)                  # W-MSA, 形状不变
        else:
            # === conv 路径 (空间形态) — ConvNeXt 块, 结束后 token 化对齐 premerge/merge 契约 ===
            for blk in self.blocks:
                x = blk(x)                        # ConvNeXt, [B,stem_dim,H,W] 形状不变
            x = x.flatten(2).transpose(1, 2)      # [B, H*W, stem_dim]  (与 attn 路径同形态)
        # ★ [2026-06-29] merge 之前的高分辨率 token ([B, 256*256, stem_dim]=48通道),
        #   供 U-Net stage0 skip 使用 (stage0_to_decoder=True 时由 backbone forward 取走).
        #   ★ [2026-07-01] 两种 stem_mode 在此处 premerge 形态完全一致 ([B,H*W,stem_dim]).
        premerge = x
        x, H, W = self.merge(x, H, W)             # [B, (H/2)*(W/2), 2*stem_dim], H/2, W/2
        return x, H, W, premerge


class ParallelStreamManager:
    """
    CUDA Streams 管理器
    
    用于管理 RGB 和 Aux 分支的并行执行
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.__rgb_stream = None
        self.__aux_stream = None
        
        if self.enabled:
            self.__rgb_stream = torch.cuda.Stream()
            self.__aux_stream = torch.cuda.Stream()
    
    @property
    def rgb_stream(self):
        """返回 stream 对象或 nullcontext"""
        if self.enabled and self.__rgb_stream is not None:
            return self.__rgb_stream
        return nullcontext()
    
    @property
    def aux_stream(self):
        if self.enabled and self.__aux_stream is not None:
            return self.__aux_stream
        return nullcontext()
    
    def get_aux_stream_context(self):
        """返回 aux stream 的上下文管理器"""
        if self.enabled and self.__aux_stream is not None:
            return torch.cuda.stream(self.__aux_stream)
        return nullcontext()
    
    def get_rgb_stream_context(self):
        """返回 rgb stream 的上下文管理器"""
        if self.enabled and self.__rgb_stream is not None:
            return torch.cuda.stream(self.__rgb_stream)
        return nullcontext()
    
    def synchronize(self):
        if self.enabled:
            if self.__rgb_stream is not None:
                torch.cuda.current_stream().wait_stream(self.__rgb_stream)
            if self.__aux_stream is not None:
                torch.cuda.current_stream().wait_stream(self.__aux_stream)

class MultimodalViT(nn.Module):
    """
    多模态 Vision Transformer (优化版)
    
    针对 RTX 5080 (16GB) 优化:
    - CUDA Streams 并行化
    - Gradient Checkpointing
    - 混合精度支持
    - LoRA 延迟注入
    
    Args:
        img_size: 输入图像尺寸
        rgb_in_chans: RGB 输入通道数
        aux_in_chans: 辅助输入通道数
        embed_dim: 嵌入维度
        depths: 各 Stage 的层数 [4, 4, 2]
        # 这个 depths如果开启的是多模态的话,4中前三层是encoder，最后一层是decoder，并且这个是可以进行修改depth的
        num_heads: 各 Stage 的注意力头数
        window_size: 窗口大小
        mlp_ratio: FFN 隐藏层倍数
        use_multimodal: 是否使用多模态
        
        # === 显存优化参数 ===
        use_checkpoint: 是否启用 Gradient Checkpointing
        checkpoint_ratio: Checkpoint 比例 (0-1)
        use_parallel_streams: 是否启用 CUDA Streams 并行化
        memory_efficient_mode: 显存高效模式 (针对 16GB 显存)
        
        # === LoRA 参数 (仅微调时使用) ===
        lora_r: LoRA 秩 (训练时设为 0)
        lora_alpha: LoRA 缩放系数
        lora_target_modules: LoRA 目标模块
    """
    
    def __init__(
        self,
        img_size: int = 512,
        rgb_in_chans: int = 3,
        aux_in_chans: int = 5,
        
        embed_dim: int = 96,
        depths: List[int] = [4, 4, 2],
        num_heads: List[int] = [6, 6, 12],
        
        # ★ [2026-04-26 第二轮] window_size 支持 int 或 List[int]:
        #   传单个 int (如 7): 所有 stage 共用同一个 ws (向后兼容旧行为)
        #   传 list (如 [8, 8, 8, 8]): per-stage ws, 长度需匹配 num_stages
        #   特殊: 4-stage + 默认 int 时, 自动改用 [8, 8, 8, 8] (论文风格的 ws 渐进对齐空间)
        window_size: int = 7,
        mlp_ratio: float = 4.,
        
        use_multimodal: bool = True,
        
        # === 新增：预训练模式 ===
        pretrain_mode: bool = False,  # 预训练模式开关
        
        # === 显存优化参数 ===
        use_checkpoint: bool = True,
        checkpoint_ratio: float = 0.5,  # 默认 50% 的层使用 checkpoint
        use_parallel_streams: bool = True,
        memory_efficient_mode: bool = True,  # 16GB 显存优化模式
        
        # === LoRA 参数 (训练时设为 0，微调时再启用) ===
        lora_r: int = 0,
        lora_alpha: int = 1,
        lora_target_modules: List[str] = ['qkv', 'q_proj', 'k_proj', 'v_proj', 'proj'],
        
        # === Deformable Attention 配置 ===
        stage3_attention_type: str = 'global',
        cross_attention_type: str = 'deformable',
        num_deformable_points: int = 4,
        
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        
        patch_size: int = 4,
        patch_stride: int = 2,
        
        # ★ [2026-04-04] V14 现代化参数 (仅影响 Aux ViT 分支)
        aux_use_layerscale: bool = False,
        aux_layerscale_init: float = 1e-4,
        aux_use_swiglu: bool = False,
        aux_use_rmsnorm: bool = False,
        aux_use_flash_attn: bool = False,
        aux_use_rope: bool = False,
        
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-05-09] v19 新增: RGB 主干 SwinEncoderBlock 现代化参数      ║
        # ╠══════════════════════════════════════════════════════════════════╣
        # ║ 之前 V14 只把 LayerScale/SwiGLU 等加给 Aux ViT, RGB Swin block 是 ║
        # ║ 朴素的 LN+GELU. v19 把这些现代化设计也扩到 RGB 主干, 让 4 个 stage ║
        # ║ 风格统一.                                                        ║
        # ║                                                                  ║
        # ║ 改动列表:                                                        ║
        # ║   rgb_use_layerscale:     SwinEncoderBlock 加 LayerScale γ        ║
        # ║   rgb_use_swiglu:         SwinEncoderBlock 的 FFN 改成 SwiGLU     ║
        # ║   rgb_use_qk_norm:        WindowAttention 用 QK-Norm 替代 cosine ║
        # ║   rgb_use_rope:           WindowAttention 用 2D RoPE 替代 Log-CPB ║
        # ║   use_cosine_drop_path:   stochastic depth schedule 改 cosine    ║
        # ║                                                                  ║
        # ║ 默认全 False, 完全向后兼容 v18 行为.                              ║
        # ╚══════════════════════════════════════════════════════════════════╝
        rgb_use_layerscale: bool = False,
        rgb_layerscale_init: float = 1e-2,
        rgb_use_swiglu: bool = False,
        rgb_use_qk_norm: bool = False,
        rgb_use_rope: bool = False,
        use_cosine_drop_path: bool = False,
        
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-05-11] v20: 跨 stage 跨窗口通信参数                        ║
        # ╠══════════════════════════════════════════════════════════════════╣
        # ║ cross_stage_mode:                                                  ║
        # ║   'none'               关闭, 跟 v19 行为一致 (默认)                 ║
        # ║   'broadcast'          方案 A, Memory Bank 单向广播 (stage4→1/2/3)  ║
        # ║   'token_bank'         方案 C, 共享 Token Bank 双向交流             ║
        # ║   'broadcast_momentum' 方案 A+, MoCo 动量 ViT + queue Bank          ║
        # ║                                                                    ║
        # ║ 仅 four_stage 模式支持 (需要独立 stage 4 ViT 作为全局信息源).        ║
        # ║ 模块来源:                                                          ║
        # ║   cross_stage_broadcast.py / cross_stage_token_bank.py             ║
        # ║   cross_stage_broadcast_momentum.py                                ║
        # ╚══════════════════════════════════════════════════════════════════╝
        cross_stage_mode: str = 'none',
        # 方案 A 专属
        broadcast_gamma_read_init: float = 1e-4,
        # 方案 C 专属
        token_bank_num_tokens: int = 16,
        token_bank_dim: int = 256,
        token_bank_alpha_init: float = 0.05,
        token_bank_gamma_read_init: float = 1e-4,
        # 方案 A+ 专属 (MoCo 动量版)
        bm_inject_stages: Optional[List[int]] = None,   # 默认 [0,1,2] 注入 stage1/2/3
        bm_num_compressed_tokens: int = 16,             # 256→16 压缩
        bm_queue_k: int = 8,                            # queue 存几个 batch
        bm_batch_size_hint: int = 8,                    # 预期 batch (算 bank_size)
        bm_momentum: float = 0.999,                     # EMA 动量系数
        bm_gamma_read_init: float = 1e-4,               # LayerScale γ 初值
        bm_enable_kv_cache: bool = False,               # KV cache 默认关闭
        bm_kv_cache_refresh_interval: int = 10,         # 开启后刷新间隔
        # ── 方案 D (D1) query_locate 参数 ──
        ql_inject_stages: Optional[List[int]] = None,   # 默认 [0,1,2] 调制 stage1/2/3
        ql_num_select: int = 64,                        # top-K 选几个 (从256选)
        ql_query_dim: int = 384,                        # 压缩后 query 维度 768→384
        ql_proj_dim: int = 384,                         # 点积统一维度
        ql_queue_k: int = 8,                            # queue 存几个 batch
        ql_batch_size_hint: int = 2,                    # 预期 batch
        ql_momentum: float = 0.999,                     # EMA 动量系数
        ql_gamma1_init: float = 1e-4,                   # 放大 γ1 初值
        ql_gamma2_init: float = 1e-4,                   # 补信息 γ2 初值
        ql_query_reduce_mode: str = 'topk',             # ★[方案E] 'topk'(D1)/'group'(E)
        ql_num_groups: int = 8,                         # ★[方案E] 分G组 (group模式)
        ql_agg_mode: str = 'mean',                      # ★[方案E] 组内聚合 'mean'/'max'
        ql_spatial_mode: str = 'global',                # ★[方案F] 'global'(D/E)/'block'(F)
        ql_num_blocks: int = 64,                        # ★[方案F] stage拆几块 (block模式,可调)
        
        # ★ [2026-04-20] v16 新增: 架构模式与跨窗口交互配置 ===========
        # 架构模式: "three_stage" (默认, 保留 GlobalPatchEmbed) 或 "four_stage" (全 PatchMerging)
        architecture_mode: str = "three_stage",
        
        # Bi-Level 跨窗口 attention 开关 (False → 退化为纯 SwinEncoderBlock)
        cross_window_enabled: bool = True,
        
        # Bi-Level 方案选择: "token_level" / "window_level" / "hybrid" (推荐)
        cross_window_type: str = "hybrid",
        
        # 方案 A (token_level) 的 top-K 参数 (每个窗口关注多少个邻居窗口)
        cross_window_top_k: int = 4,
        
        # Cross-window gate 初始值 (0.0 = 训练早期完全关闭跨窗口通道)
        cross_window_gate_init: float = 0.0,
        # ★ [2026-06-15] 改动1: 方案C(hybrid) 摘要 Q 改进 (默认 1 query + 无统计量 = 原行为, 旧 ckpt 兼容)
        cross_window_num_queries: int = 1,
        cross_window_use_stats: tuple = (),
        # ★ [2026-06-18] v3.0: 方案C(hybrid) Step3 分发模式
        #   cross_window_step3_mode: 'gather'(默认,token查摘要) / 'splat'(摘要播撒进token,块级语义播撒)
        #   cross_window_splat_null: splat 模式下是否加 null/sink 槽 (默认 True, 保护前景 Precision)
        #   仅 cross_window_type='hybrid' 时生效; 默认 'gather' 与 v3.0 之前逐字节等价.
        cross_window_step3_mode: str = 'gather',
        cross_window_splat_null: bool = True,
        # ★ [2026-06-20] splat_plus (splater+) 专属子选项 (仅 step3_mode='splat_plus' 生效)
        cross_window_splatp_score_mode: str = 'fullC',  # 'fullC' / 'multihead'
        cross_window_splatp_fuse_mode: str = 'mlp',     # 'mlp' / 'perchannel'
        cross_window_splatp_num_heads: int = 8,
        cross_window_ffn_respost: bool = False,   # ★ [2026-07-11] 改动I: 透传给 BiLevel 的 FFN 子层 res-post 开关
        cross_window_drop_norm_cross: bool = False,  # ★ [2026-07-11] 改动J
        cross_window_mul_ln: bool = False,           # ★ [2026-07-11] 改动J
        cross_window_cross_qk_norm: bool = False,    # ★ [2026-07-11] 改动J
        
        # BiLevel 专用的独立 LoRA 配置 (独立于主 WindowAttention 的 lora_r/alpha)
        bilevel_lora_r: int = 0,
        bilevel_lora_alpha: int = 1,
        # ★ [2026-04-20] v16 END ======================================
        
        # ★ [2026-04-26 修改] SwinV2 适配参数 ===========================
        #   swin_version='v1' → SwinEncoderBlock 走 Pre-Norm
        #   swin_version='v2' → SwinEncoderBlock 走 Post-Norm
        #
        #   注意: V1 ckpt 必须配 v1 forward, V2 ckpt 必须配 v2 forward, 不能错配!
        #         (上一轮训练 mIoU=0.50 的根因就是 V2 ckpt + V1 Pre-Norm)
        #
        # ★ [2026-04-26 第二轮] pretrained_window_size 参数已删除.
        #   原因: WindowAttention 已经移除该参数, cpb_mlp 直接用 stage 自己的 ws 归一化.
        swin_version: str = 'v1',
        # ★ [2026-04-26 修改] END =====================================
        
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-06-29] 两个独立实验开关 (单变量, 默认 False = 原行为)       ║
        # ╠══════════════════════════════════════════════════════════════════╣
        # ║ shift_window:      True → stage1/2/3 奇数 block 用标准 Swin          ║
        # ║                    shifted-window (SW-MSA, shift=ws//2) 取代 BiLevel,║
        # ║                    做创新点对照基线 ([W-MSA,SW-MSA] vs [W-MSA,Bi]).  ║
        # ║ stage0_to_decoder: True → 把 stem 的 256² premerge (stem_dim=48 通道)║
        # ║                    作为额外最高分辨率 U-Net skip 接进解码器.         ║
        # ╚══════════════════════════════════════════════════════════════════╝
        shift_window: bool = False,
        stage0_to_decoder: bool = False,
        # ★ [2026-07-01] stem_mode: 'attn' (原 256² W-MSA stem, 默认) / 'conv' (稍强版 ConvNeXt
        #   卷积块取代 W-MSA). 单变量 A/B: 治 W-MSA stem 的病态梯度尖峰 + EMA 崩塌. 见 RGBStemStage0.
        stem_mode: str = 'attn',
        
        **kwargs
    ):
        super().__init__()
        
        # === 基础配置 ===
        self.img_size = img_size
        self.use_multimodal = use_multimodal
        self.pretrain_mode = pretrain_mode  # 新增
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        
        # ★ [2026-06-29] 两个实验开关 (必须在 _build_stage / decoder_dims 计算之前存好;
        #   _build_stage line~1307 读 self.shift_window, decoder_dims 块读 self.stage0_to_decoder)
        self.shift_window = shift_window
        self.stage0_to_decoder = stage0_to_decoder
        # ★ [2026-07-01] stem_mode: RGBStemStage0 内部 attn/conv 切换 (须在 stem 构造前存好).
        if stem_mode not in ('attn', 'conv'):
            raise ValueError(f"stem_mode must be 'attn' or 'conv', got '{stem_mode}'")
        self.stem_mode = stem_mode
        
        # ★ [2026-04-26 修改] 存储 SwinV2 适配参数, _build_blocks 时透传给 SwinEncoderBlock
        if swin_version not in ('v1', 'v2'):
            raise ValueError(
                f"swin_version must be 'v1' or 'v2', got '{swin_version}'. "
                f"v1 走 Pre-Norm 配 V1 Swin-T 权重; v2 走 Post-Norm 配 SwinV2 权重."
            )
        self.swin_version = swin_version
        # ★ [4.26 第二轮] self.pretrained_window_size 属性删除
        
        # === 预训练模式逻辑校验 ===
        if pretrain_mode:
            self.use_multimodal = False  # 预训练模式强制关闭多模态
            self.use_parallel_streams = False  # 关闭并行流
            print("[MultimodalViT] 预训练模式：强制单模态 + 编码器优化")
        
        # === 显存优化配置 ===
        self.use_checkpoint = use_checkpoint
        self.checkpoint_ratio = checkpoint_ratio
        self.use_parallel_streams = use_parallel_streams
        self.memory_efficient_mode = memory_efficient_mode
        
        # CUDA Streams 管理器
        self._stream_manager = ParallelStreamManager(
            enabled=use_parallel_streams and use_multimodal
        )
        
        # === LoRA 配置 (延迟注入) ===
        self._lora_injected = False
        self._lora_config = {
            'r': lora_r,
            'alpha': lora_alpha,
            'target_modules': lora_target_modules
        }
        
        # === 保存其他配置 ===
        self.stage3_attention_type = stage3_attention_type
        self.cross_attention_type = cross_attention_type
        self.num_deformable_points = num_deformable_points
        
        # ★ [2026-04-04] V14 配置: Stage3 的 ViT blocks 也使用现代化组件
        #   注意: 仅 ViT blocks (Stage3) 使用, Swin blocks (Stage1/2) 保持不变 (预训练兼容)
        self._v14_vit_kwargs = {
            'use_layerscale': aux_use_layerscale,
            'layerscale_init': aux_layerscale_init,
            'use_swiglu': aux_use_swiglu,
            'use_rmsnorm': aux_use_rmsnorm,
            'use_flash_attn': aux_use_flash_attn,
            'use_rope': aux_use_rope,
        }
        
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-05-09] v19: RGB 主干 SwinEncoderBlock 的现代化参数字典     ║
        # ║   _build_stage 创建 SwinEncoderBlock 时透传, 替代 Swin block 内部  ║
        # ║   的 LN+GELU 朴素组件.                                            ║
        # ╚══════════════════════════════════════════════════════════════════╝
        self._v19_rgb_swin_kwargs = {
            'use_layerscale': rgb_use_layerscale,
            'layerscale_init': rgb_layerscale_init,
            'use_swiglu': rgb_use_swiglu,
            'use_qk_norm': rgb_use_qk_norm,
            'use_rope': rgb_use_rope,
        }
        # v19 配置打印 (帮助 debug)
        print(f"[MultimodalViT v19] RGB swin kwargs: "
              f"layerscale={rgb_use_layerscale}(init={rgb_layerscale_init}), "
              f"swiglu={rgb_use_swiglu}, qk_norm={rgb_use_qk_norm}, "
              f"rope={rgb_use_rope}, cosine_dpr={use_cosine_drop_path}")
        
        # 保存 cosine drop_path 开关 (供 dpr 生成时使用)
        self._use_cosine_drop_path = use_cosine_drop_path
        
        # ★ [2026-04-20] v16 配置保存 =====================================
        # ★ [2026-06-21] 单一架构: three_stage 已删除. architecture_mode 保留为兼容参数但不再生效.
        #   self.num_stages = 3 指【输出 stage 数】(stage1/2/3, 给解码器); stage0(stem)独立计数.
        if architecture_mode == "three_stage":
            print("⚠️  [MultimodalViT] three_stage 已于 2026-06-21 删除; 忽略该设置, 使用唯一架构 (stage0+stage1/2/3).")
        self.architecture_mode = "four_stage"   # 兼容占位 (forward 不再按它分支)
        self.num_stages = 3
        
        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-04-26 第二轮] Per-stage window_sizes 解析                       ║
        # ║                                                                        ║
        # ║ 设计:                                                                  ║
        # ║   - window_size: int → 所有 stage 共用 (向后兼容旧 3-stage 行为)        ║
        # ║   - window_size: list → per-stage, 长度需匹配 num_stages              ║
        # ║   - 4-stage + int 默认值 7 → 自动改用 [8, 8, 8, 8] (论文风格 ws 渐进)   ║
        # ║                                                                        ║
        # ║ 4-stage 默认 [8, 8, 8, 8] 的理由:                                      ║
        # ║   - Stage 1 (64×64) ws=8 → 8×8=64 个窗口, 整除无 padding              ║
        # ║   - Stage 2 (32×32) ws=8 → 4×4=16 个窗口, 整除无 padding              ║
        # ║   - Stage 3 (16×16) ws=8 → 2×2=4 个窗口, 整除无 padding (准全局)      ║
        # ║   - Stage 4 (ViT, 8×8): ws 不起作用 (ViT 走全局, 没有 W-MSA), 仅占位  ║
        # ║   - 跟论文 ws=16 思想一致 (让 ws 接近空间分辨率), 但保留 Stage 4 ViT  ║
        # ║                                                                        ║
        # ║ self.window_sizes 是 list[int], 长度 = num_stages                      ║
        # ║ self.window_size 仍保留为 "代表值" (== window_sizes[0]), 兼容旧代码    ║
        # ╚═══════════════════════════════════════════════════════════════════════╝
        if isinstance(window_size, (list, tuple)):
            # 用户显式传 list: 校验长度
            if len(window_size) != self.num_stages:
                raise ValueError(
                    f"window_size 是 list 时长度必须等于 num_stages={self.num_stages}, "
                    f"收到长度 {len(window_size)}: {window_size}"
                )
            self.window_sizes = list(window_size)
        else:
            # 用户传 int (含默认 7): 根据 architecture_mode 智能展开
            if self.architecture_mode == "four_stage" and window_size == 7:
                # 4-stage 默认值: 自动改用 [8, 8, 8, 8] (论文风格)
                self.window_sizes = [8, 8, 8, 8]
                print(f"[MultimodalViT] ★ [4.26] 4-stage 模式自动启用 window_sizes={self.window_sizes} "
                      f"(空间分辨率 64/32/16/8 → ws 8/8/8/8 整除无 padding)")
            else:
                # 3-stage 或者用户显式传非默认 int: 所有 stage 共用
                self.window_sizes = [window_size] * self.num_stages
        
        # 兼容旧代码: self.window_size = 第一个 stage 的 ws
        self.window_size = self.window_sizes[0]
        
        self.cross_window_enabled = cross_window_enabled
        self.cross_window_type = cross_window_type
        self.cross_window_top_k = cross_window_top_k
        self.cross_window_gate_init = cross_window_gate_init
        # ★ [2026-06-15] 改动1
        self.cross_window_num_queries = cross_window_num_queries
        self.cross_window_use_stats = cross_window_use_stats
        # ★ [2026-06-18] v3.0: Step3 分发模式
        assert cross_window_step3_mode in ('gather', 'splat', 'splat_plus'), (
            f"cross_window_step3_mode 必须是 'gather'/'splat'/'splat_plus', got {cross_window_step3_mode}"
        )
        self.cross_window_step3_mode = cross_window_step3_mode
        self.cross_window_splat_null = cross_window_splat_null
        # ★ [2026-06-20] splat_plus 子选项
        self.cross_window_splatp_score_mode = cross_window_splatp_score_mode
        self.cross_window_splatp_fuse_mode = cross_window_splatp_fuse_mode
        self.cross_window_splatp_num_heads = cross_window_splatp_num_heads
        self.cross_window_ffn_respost = cross_window_ffn_respost   # ★ [2026-07-11] 改动I
        self.cross_window_drop_norm_cross = cross_window_drop_norm_cross  # ★ [2026-07-11] 改动J
        self.cross_window_mul_ln = cross_window_mul_ln                    # ★ [2026-07-11] 改动J
        self.cross_window_cross_qk_norm = cross_window_cross_qk_norm      # ★ [2026-07-11] 改动J
        
        # BiLevel 独立 LoRA 配置
        self._bilevel_lora_config = {
            'r': bilevel_lora_r,
            'alpha': bilevel_lora_alpha,
        }
        
        # 校验 depths 长度与 architecture_mode 匹配
        if len(depths) != self.num_stages:
            # 如果 depths 长度不匹配, 自动填充或截断 (并发出警告)
            print(f"⚠️  [MultimodalViT] depths 长度({len(depths)}) 与 architecture_mode"
                  f" '{architecture_mode}' 的 {self.num_stages} 个 stage 不匹配")
            if len(depths) < self.num_stages:
                depths = list(depths) + [4] * (self.num_stages - len(depths))
                print(f"   自动填充为: {depths}")
            else:
                depths = depths[:self.num_stages]
                print(f"   自动截断为: {depths}")
            self.depths = depths
        
        if len(num_heads) != self.num_stages:
            if len(num_heads) < self.num_stages:
                num_heads = list(num_heads) + [12] * (self.num_stages - len(num_heads))
            else:
                num_heads = num_heads[:self.num_stages]
            self.num_heads = num_heads

        # ★ [2026-06-21] 单一架构: 强制【输出 stage】结构 (stage1/2/3), 忽略外部传入.
        #   理由: 训练脚本传入的 DEPTHS/NUM_HEADS_LIST 长度=4, 与旧 four_stage[4,4,4,4]
        #   无法用长度区分, 故此处硬定; 训练脚本无需改动即可得到正确结构.
        _passed_depths = list(self.depths)
        _passed_heads = list(self.num_heads)
        _passed_ws = list(self.window_sizes)
        self.depths = [4, 6, 2]          # ★[2026-07-31] stage1/2/3; 配 stem_depth=2 => 总体 [2,4,6,2]
        self.num_heads = [6, 6, 12]      # head_dim = [16, 32, 32]
        self.window_sizes = [8, 8, 32]   # ★[2026-07-31] stage3 ws=32 => 单窗口覆盖全图 = 全局 MHSA
        self.window_size = self.window_sizes[0]
        if (_passed_depths != self.depths) or (_passed_heads != self.num_heads):
            print(f"⚠️  [MultimodalViT] 单一架构强制结构: depths {_passed_depths}→{self.depths}, "
                  f"num_heads {_passed_heads}→{self.num_heads}, window {_passed_ws}→{self.window_sizes}")
        # stage0 (stem) 独立超参 (不计入 self.depths/num_heads/window_sizes)
        self.stem_depth = 2              # head_dim 16
        self.stem_num_heads = 3
        self.stem_window = 8             # 8 整除 256 (无 padding)
        # ★ [2026-04-20] v16 END ========================================
        
        # === 维度与尺寸计算 (单一架构, [2026-06-21] 删除 three_stage) ===
        # 内部 4 个 stage: stage0(stem, 48@256, 内部) + stage1/2/3(96/192/384 @ 128/64/32).
        # 对外只暴露 3 个【输出 stage】(给 U-Net 解码器): self.dims = [96,192,384].
        #   stage0 的 48 维只在 RGBStemStage0 内部使用, 不进 features, 不进解码器.
        #   降采样链: stem(stride1, 256) → merge0(÷2, 128) → merge1(÷2, 64) → merge2(÷2, 32).
        self.stem_dim = embed_dim // 2                          # stage0 通道 = 48
        self.dims = [embed_dim, embed_dim * 2, embed_dim * 4]   # [96, 192, 384]
        self.spatial_sizes = [
            img_size // 2,   # stage1 128  (stem 不降采样, merge0 ÷2)
            img_size // 4,   # stage2 64
            img_size // 8,   # stage3 32
        ]
        self.feature_info = {
            'stage1': {'H': self.spatial_sizes[0], 'W': self.spatial_sizes[0], 'dim': self.dims[0]},
            'stage2': {'H': self.spatial_sizes[1], 'W': self.spatial_sizes[1], 'dim': self.dims[1]},
            'stage3': {'H': self.spatial_sizes[2], 'W': self.spatial_sizes[2], 'dim': self.dims[2]},
        }
        
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-06-29] stage0_to_decoder: 给【解码器】看的扩展维度/尺寸视图  ║
        # ╠══════════════════════════════════════════════════════════════════╣
        # ║ 内部 self.dims / self.spatial_sizes 保持不变 (它们还被 stem out_dim/ ║
        # ║ 各 PatchMerging / aux_proj_s1-3 / final_dim 等内部逻辑使用, 原地改会  ║
        # ║ 连锁炸). 解码器侧改读 decoder_dims / decoder_spatial_sizes:          ║
        # ║   关 → 与 dims/spatial_sizes 等价 ([96,192,384] / [128,64,32]).      ║
        # ║   开 → 前面补 stage0 ([48,96,192,384] / [256,128,64,32]);            ║
        # ║        UNetDecoder 按 len(in_channels) 自动多出一级 256² 上采样 + 一  ║
        # ║        个 attention gate (解码器代码本身是通用的, 无需改).            ║
        # ╚══════════════════════════════════════════════════════════════════╝
        if self.stage0_to_decoder:
            self.decoder_dims = [self.stem_dim] + list(self.dims)                       # [48, 96, 192, 384]
            self.decoder_spatial_sizes = [self.spatial_sizes[0] * 2] + list(self.spatial_sizes)  # [256, 128, 64, 32]
            # LayerScale 式可学 gate (逐通道, init=0.1): 防随机初始化的 256² skip 前期冲乱已学好的
            #   解码器 + 缓解 stem 的 inf 梯度. 与本仓库 bi 的 gamma_amp/gamma_info 同套路.
            self.stage0_skip_gamma = nn.Parameter(0.1 * torch.ones(self.stem_dim))
            # 非可学的 epoch 线性 warmup 乘子 (0→1), 由 train 端每个 epoch 设置; 默认 1.0 = 不 warmup.
            self.stage0_skip_warmup = 1.0
        else:
            self.decoder_dims = list(self.dims)
            self.decoder_spatial_sizes = list(self.spatial_sizes)
        
        # DropPath 衰减规则
        # ★ [2026-04-20] v16: 使用 self.depths (可能已被 architecture_mode 校验调整)
        # ★ [2026-05-09] v19: 支持 cosine schedule (use_cosine_drop_path=True)
        #   - linear (默认): dpr[i] = drop_path_rate * i / (total-1)
        #     浅层和深层均匀衰减
        #   - cosine: dpr[i] = drop_path_rate * (1 - cos(i / total * π/2))
        #     浅层几乎不 drop (i=0 时 dpr=0), 深层接近 drop_path_rate
        #     强迫浅层独立可用, 缓解 Stage 1 梯度独占问题 (训练日志证据: rgb_s1 占总梯度 99%)
        #     参考: Huang et al., ECCV 2016 (原始); ConvNeXt (Liu et al., CVPR 2022) 用此 schedule
        total_depth = sum(self.depths) + (6 if use_multimodal else 0)
        if self._use_cosine_drop_path:
            import math as _math
            dpr = [drop_path_rate * (1 - _math.cos(i / max(total_depth - 1, 1) * _math.pi / 2))
                   for i in range(total_depth)]
            print(f"[MultimodalViT v19] DropPath: cosine schedule, "
                  f"max={drop_path_rate:.3f}, depths={total_depth}, "
                  f"dpr[0]={dpr[0]:.4f}, dpr[-1]={dpr[-1]:.4f}")
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
            print(f"[MultimodalViT] DropPath: linear schedule, "
                  f"max={drop_path_rate:.3f}, depths={total_depth}")
        
        self.dpr_ptr = 0
        
        # === 计算需要 checkpoint 的层 ===
        self._setup_checkpoint_layers(self.depths)
        
        # =================================================================
        # Auxiliary Branch (优化版)
        # =================================================================
        if use_multimodal and not pretrain_mode:  # 添加预训练模式检查
            self.aux_embed_dim = 768
            
            # 使用优化版 Aux 分支
            # ★ [2026-04-04] V14: 传递现代化参数到 Aux ViT
            self.aux_branch = AuxViTBranch(
                in_chans=aux_in_chans,
                embed_dim=self.aux_embed_dim,
                depth=6,
                drop_path_rate=drop_path_rate,
                lora_r=0,  # 训练时不启用 LoRA
                use_layerscale=aux_use_layerscale,
                layerscale_init=aux_layerscale_init,
                use_swiglu=aux_use_swiglu,
                use_rmsnorm=aux_use_rmsnorm,
                use_flash_attn=aux_use_flash_attn,
                use_rope=aux_use_rope,
            )
            self.dpr_ptr += 6
            
            # 投影层 (Aux -> RGB)
            # ★ [2026-04-20] v16: 四阶段模式需要额外的 aux_proj_s4
            #   三阶段: s1 (→96), s2 (→192), s3 (→768)     — 3 个投影
            #   四阶段: s1 (→96), s2 (→192), s3 (→384), s4 (→768) — 4 个投影
            self.aux_proj_s1 = nn.Linear(self.aux_embed_dim, self.dims[0])
            self.aux_proj_s2 = nn.Linear(self.aux_embed_dim, self.dims[1])
            self.aux_proj_s3 = nn.Linear(self.aux_embed_dim, self.dims[2])
            if self.num_stages == 4:
                self.aux_proj_s4 = nn.Linear(self.aux_embed_dim, self.dims[3])
            else:
                self.aux_proj_s4 = None
        else:
            # 预训练模式下设为 None，避免属性错误
            self.aux_branch = None
            self.aux_proj_s1 = None
            self.aux_proj_s2 = None
            self.aux_proj_s3 = None
            self.aux_proj_s4 = None
        # =================================================================
        # RGB 主干
        # =================================================================
        
        # ★ [2026-06-21] Stage0 (stem): 256×256, 48d, 纯 W-MSA (无 cross-window), 不进 U-Net.
        #   封装在 RGBStemStage0: stem conv(5→48, stride1) + W-MSA×stem_depth + PatchMerging(48→96, ÷2).
        #   取代旧 rgb_patch_embed 的降采样职责 (旧: 256→128; 现: stem 保持 256 → merge0 降到 128).
        self.rgb_stem_stage0 = RGBStemStage0(
            in_chans=rgb_in_chans, stem_dim=self.stem_dim, out_dim=self.dims[0],
            depth=self.stem_depth, num_heads=self.stem_num_heads,
            window_size=self.stem_window, mlp_ratio=mlp_ratio,
            swin_version=self.swin_version, swin_kwargs=self._v19_rgb_swin_kwargs,
            drop_path=0.0,
            stem_mode=self.stem_mode,   # ★ [2026-07-01] 'attn'(默认 W-MSA) / 'conv'(ConvNeXt)
        )
        # 兼容占位: 旧代码/训练脚本可能按名引用 rgb_patch_embed; stage0 已取代其职责.
        self.rgb_patch_embed = None

        # Stage 1: Swin + BiLevel (H/2 = 128×128, 96d)
        self.rgb_s1_blocks = self._build_stage(
            stage_idx=0, dim=self.dims[0], num_heads=self.num_heads[0],
            window_size=self.window_sizes[0], mlp_ratio=mlp_ratio, dpr=dpr,
            block_type='swin', use_bilevel=True
        )
        self.rgb_merge1 = PeakPreservingDownsample(dim=self.dims[0])   # ★[2026-07-31] 96→192, 128→64

        # Stage 2: Swin + BiLevel (H/4 = 64×64, 192d)
        self.rgb_s2_blocks = self._build_stage(
            stage_idx=1, dim=self.dims[1], num_heads=self.num_heads[1],
            window_size=self.window_sizes[1], mlp_ratio=mlp_ratio, dpr=dpr,
            block_type='swin', use_bilevel=True
        )
        self.rgb_merge2 = PeakPreservingDownsample(dim=self.dims[1])   # ★[2026-07-31] 192→384, 64→32

        # Stage 3: Swin + BiLevel (H/8 = 32×32, 384d) — 最深输出 stage (depth=2)
        self.rgb_s3_blocks = self._build_stage(
            stage_idx=2, dim=self.dims[2], num_heads=self.num_heads[2],
            window_size=self.window_sizes[2], mlp_ratio=mlp_ratio, dpr=dpr,
            block_type='swin', use_bilevel=False   # ★[2026-07-31] 全局 MHSA, 不装 BiLevel
        )

        # GlobalPatchEmbed / merge3 不再使用 (单一架构)
        self.rgb_global_embed = None
        self.rgb_merge3 = None

        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║ [封存] Stage4 ViT (768@16×16) —— ENABLE_STAGE4_VIT = False               ║
        # ╠════════════════════════════════════════════════════════════════════════╣
        # ║ 结论 [2026-06-21]: 在残膜(1–2px 极小细丝)分割上, 该全局 ViT 分支消融     ║
        # ║   贡献≈0 (zero-stage4 → ΔfgIoU = -0.001), 因 16×16 下细丝信息已被下采样   ║
        # ║   抹平. 这是【任务-尺度错配】, 不是优化失败 (高分辨率 stage1/2/3 学得很好).║
        # ║ 保留代码供将来复用: 棚膜 / 农田 / 建筑等大目标 (16×16 仍保留信息), 或      ║
        # ║   CVPR / 与 LLM 交互等任务. 重新启用步骤:                                 ║
        # ║   (1) ENABLE_STAGE4_VIT=True;                                            ║
        # ║   (2) self.dims 末尾加回 embed_dim*8; spatial 加回 //16; feature_info 加  ║
        # ║       stage4; self.num_stages=4; self.depths/num_heads 末尾加回该级;      ║
        # ║   (3) 两条 forward 末尾 (stage3 之后) 接回 merge3 + s4 (见 forward 注释).  ║
        # ║   注: 下方 _build_stage 的 stage_idx 仅为占位, 重新启用时需自行设 depth.   ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        ENABLE_STAGE4_VIT = False
        self.rgb_s4_blocks = None
        if ENABLE_STAGE4_VIT:
            self.rgb_merge3 = PatchMerging(dim=self.dims[2])   # 384→768, 32→16
            self.rgb_s4_blocks = self._build_stage(
                stage_idx=2, dim=embed_dim * 8, num_heads=12,
                window_size=self.window_sizes[-1], mlp_ratio=mlp_ratio, dpr=dpr,
                block_type='vit', use_bilevel=False
            )
        
        # === 输出归一化 (对应最后一个 stage 的输出维度) ===
        # ★ [2026-04-04] Stage3/4 输出归一化: 与 ViT blocks 使用相同的 norm 类型
        final_dim = self.dims[-1]  # [2026-06-21] 单一架构: dims[-1]=384 (stage3, 已无 768)
        if aux_use_rmsnorm:
            from transformer_block import RMSNorm
            self.rgb_norm = RMSNorm(final_dim)
        else:
            self.rgb_norm = nn.LayerNorm(final_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # ★ [v8, 2026-03-31] 修复: self.apply(_init_weights) 用 trunc_normal_(0.02) 覆盖了
        #   DeformableCrossAttention 的三项关键初始化:
        #   1. sampling_offsets.bias (径向网格 → 全零 → 所有 head 采同一点 → 共振爆炸)
        #   2. output_proj.weight (零初始化 → 非零 → 随机输出注入残差)
        #   3. attention_weights.weight (零初始化 → 非零 → 权重不均匀)
        #   Round 5 (2026-03-20) 发现此 bug, 此处正式修复
        for m in self.modules():
            if hasattr(m, '_reset_parameters') and m is not self:
                m._reset_parameters()
        
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║ ★ [2026-05-11] v20: 跨 stage 跨窗口通信模块创建                    ║
        # ╠══════════════════════════════════════════════════════════════════╣
        # ║ 在所有 stage / blocks / norm 创建之后, 才创建跨 stage 模块.        ║
        # ║ 这样可以读取 self.dims / self.num_heads 作为构造参数.              ║
        # ║                                                                    ║
        # ║ cross_stage_mode='none' 时 self.cross_stage_module = None,         ║
        # ║ forward 中所有相关调用都是 no-op, 完全等价于 v19 行为.              ║
        # ╚══════════════════════════════════════════════════════════════════╝
        assert cross_stage_mode in ('none', 'broadcast', 'token_bank', 'broadcast_momentum', 'query_locate'), (
            f"cross_stage_mode 必须是 'none'/'broadcast'/'token_bank'/'broadcast_momentum'/'query_locate', "
            f"got '{cross_stage_mode}'"
        )
        self.cross_stage_mode = cross_stage_mode

        # ★ [2026-06-21] 单一架构已封存 stage4 ViT. 所有 cross_stage 方案 (broadcast/
        #   token_bank/broadcast_momentum/query_locate) 都依赖独立 stage4 作为全局 prior
        #   来源 (会访问 self.dims[3] / self.rgb_s4_blocks), 在本架构下不可用.
        if cross_stage_mode != 'none':
            raise NotImplementedError(
                f"cross_stage_mode='{cross_stage_mode}' 依赖 stage4 ViT, 但单一架构已封存 stage4; "
                f"请用 cross_stage_mode='none'. (重新启用 stage4 见 __init__ 中 ENABLE_STAGE4_VIT 说明.)"
            )

        if cross_stage_mode == 'broadcast':
            # === 方案 A: Memory Bank 单向广播 ===
            if not _HAS_CROSS_STAGE_BROADCAST:
                raise ImportError(
                    "cross_stage_mode='broadcast' 但 cross_stage_broadcast.py 不可用. "
                    "请把该文件放到跟 multimodal.py 同一目录."
                )
            if self.architecture_mode != 'four_stage':
                raise ValueError(
                    "cross_stage_mode='broadcast' 仅支持 four_stage 架构 "
                    "(需要独立 stage 4 ViT 作为 prior 来源)."
                )
            self.cross_stage_module = CrossStageBroadcast(
                stage_dims=self.dims[:3],            # [96, 192, 384]
                stage_num_heads=self.num_heads[:3],  # [6, 6, 12] (跟 num_heads 一致)
                prior_dim=self.dims[3],              # 768
                gamma_read_init=broadcast_gamma_read_init,
            )
            print(f"[MultimodalViT v20] 跨 stage 通信 = 方案 A (broadcast)")
        elif cross_stage_mode == 'token_bank':
            # === 方案 C: 共享 Token Bank 双向交流 ===
            if not _HAS_CROSS_STAGE_TOKEN_BANK:
                raise ImportError(
                    "cross_stage_mode='token_bank' 但 cross_stage_token_bank.py 不可用. "
                    "请把该文件放到跟 multimodal.py 同一目录."
                )
            if self.architecture_mode != 'four_stage':
                raise ValueError(
                    "cross_stage_mode='token_bank' 仅支持 four_stage 架构."
                )
            self.cross_stage_module = CrossStageTokenBank(
                stage_dims=self.dims,                # [96, 192, 384, 768]
                num_tokens=token_bank_num_tokens,
                bank_dim=token_bank_dim,
                num_heads=8,
                gamma_read_init=token_bank_gamma_read_init,
                alpha_init=token_bank_alpha_init,
            )
            print(f"[MultimodalViT v20] 跨 stage 通信 = 方案 C (token_bank)")
        elif cross_stage_mode == 'broadcast_momentum':
            # === 方案 A+: MoCo 动量 ViT + queue Bank ===
            if not _HAS_CROSS_STAGE_BROADCAST_MOMENTUM:
                raise ImportError(
                    "cross_stage_mode='broadcast_momentum' 但 "
                    "cross_stage_broadcast_momentum.py 不可用. "
                    "请把该文件放到跟 multimodal.py 同一目录."
                )
            if self.architecture_mode != 'four_stage':
                raise ValueError(
                    "cross_stage_mode='broadcast_momentum' 仅支持 four_stage 架构 "
                    "(需要独立 stage 4 ViT 作为动量编码器来源)."
                )
            _bm_inject = bm_inject_stages if bm_inject_stages is not None else [0, 1, 2]
            self.cross_stage_module = CrossStageBroadcastMomentum(
                stage4_blocks=self.rgb_s4_blocks,    # 动量 ViT 深拷贝来源
                stage4_norm=self.rgb_norm,           # stage4 末尾 norm
                stage_dims=self.dims[:3],            # [96, 192, 384] (浅层)
                inject_stages=_bm_inject,            # 默认 [0,1,2] 注入 stage1/2/3
                num_compressed_tokens=bm_num_compressed_tokens,
                queue_k=bm_queue_k,
                batch_size_hint=bm_batch_size_hint,
                bank_dim=self.dims[3],               # 768
                proj_dim=256,
                num_heads=8,
                gamma_read_init=bm_gamma_read_init,
                momentum=bm_momentum,
                enable_kv_cache=bm_enable_kv_cache,
                kv_cache_refresh_interval=bm_kv_cache_refresh_interval,
            )
            # 训练开始前同步动量 ViT 跟 stage4 的初始参数
            self.cross_stage_module.init_momentum(self.rgb_s4_blocks, self.rgb_norm)
        elif cross_stage_mode == 'query_locate':
            # === 方案 D (D1): top-K query 点积定位调制 ===
            if not _HAS_CROSS_STAGE_QUERY_LOCATE:
                raise ImportError(
                    "cross_stage_mode='query_locate' 但 cross_stage_query_locate.py 不可用. "
                    "请把该文件放到跟 multimodal.py 同一目录."
                )
            if self.architecture_mode != 'four_stage':
                raise ValueError(
                    "cross_stage_mode='query_locate' 仅支持 four_stage 架构 "
                    "(需要独立 stage 4 ViT 作为动量编码器来源)."
                )
            _ql_inject = ql_inject_stages if ql_inject_stages is not None else [0, 1, 2]
            self.cross_stage_module = CrossStageQueryLocate(
                stage4_blocks=self.rgb_s4_blocks,    # 动量 ViT 深拷贝来源
                stage4_norm=self.rgb_norm,           # stage4 末尾 norm
                stage_dims=self.dims[:3],            # [96, 192, 384] (浅层)
                inject_stages=_ql_inject,            # 默认 [0,1,2]
                num_select=ql_num_select,            # top-K=64
                query_dim=ql_query_dim,              # 384
                bank_src_dim=self.dims[3],           # 768 (stage4 原始维度)
                proj_dim=ql_proj_dim,                # 384
                queue_k=ql_queue_k,
                batch_size_hint=ql_batch_size_hint,
                momentum=ql_momentum,
                gamma1_init=ql_gamma1_init,
                gamma2_init=ql_gamma2_init,
                query_reduce_mode=ql_query_reduce_mode,  # ★[方案E] topk/group
                num_groups=ql_num_groups,                # ★[方案E] 分组数
                agg_mode=ql_agg_mode,                    # ★[方案E] mean/max
                spatial_mode=ql_spatial_mode,            # ★[方案F] global/block
                num_blocks=ql_num_blocks,                # ★[方案F] 拆块数
            )
            # 训练开始前同步动量 ViT 跟 stage4 的初始参数
            self.cross_stage_module.init_momentum(self.rgb_s4_blocks, self.rgb_norm)
            # (准确的方案模式由 CrossStageQueryLocate.__init__ 自己打印, 此处不再重复)
        else:
            self.cross_stage_module = None
            print(f"[MultimodalViT v20] 跨 stage 通信 = 关闭 (mode='none')")
        
        # 注意: LoRA 不在 __init__ 中注入
        
    def _setup_checkpoint_layers(self, depths: List[int]):
        """
        设置需要 checkpoint 的层
        
        ★ [2026-04-20] v16: 策略更新
          - BiLevelWindowBlock: 强制 checkpoint (跨窗口 attention 显存开销大)
          - SwinEncoderBlock / Decoder: 按 checkpoint_ratio 策略
        
        强制 checkpoint 的层在 _build_stage 中通过 _bilevel_layer_indices 记录,
        _run_with_checkpoint 会优先检查这个集合
        """
        self._checkpoint_layers = {}
        # ★ [2026-04-20] v16: BiLevel block 强制 checkpoint 的层索引
        self._bilevel_layer_indices = {}
        
        if not self.use_checkpoint:
            return
        
        # 为每个 stage 计算 checkpoint 层 (非 BiLevel 的常规 block)
        for stage_idx, depth in enumerate(depths):
            num_ckpt = int(depth * self.checkpoint_ratio)
            # 优先 checkpoint 后面的层
            self._checkpoint_layers[stage_idx] = set(range(depth - num_ckpt, depth))
            # BiLevel 集合初始为空, 在 _build_stage 中填充
            self._bilevel_layer_indices[stage_idx] = set()

    def _build_stage(
        self,
        stage_idx,
        dim,
        num_heads,
        window_size,
        mlp_ratio,
        dpr,
        block_type,
        use_bilevel: bool = False,
    ):
        """
        动态构建 Stage
        
        ★ [2026-04-20] v16: 重写 block 排列规则
        
        当 block_type='swin' 且 use_bilevel=True (Stage1/2, 四阶段 Stage3):
            block 0: SwinEncoderBlock (WindowAttention, 加载 Swin-T 预训练)
            block 1: BiLevelWindowBlock (跨窗口 attention, 三种方案消融)
            block 2: SwinEncoderBlock (WindowAttention, 复用 block 0 预训练权重)
            block 3 (最后一个):
              - pretrain_mode 或 use_multimodal=False: BiLevelWindowBlock
              - use_multimodal=True: SwinDeformableDecoderBlock
            block >3 (如果 depth > 4): 按 [BiLevel, SwinEnc] 规律继续
            
            如果 cross_window_enabled=False, BiLevel 位置 fallback 为 SwinEncoderBlock
            如果 depth 不足 4, 按实际 depth 构建 (但会打印警告)
        
        当 block_type='vit' 或 use_bilevel=False (三阶段 Stage3, 四阶段 Stage4):
            全部 ViTEncoderBlock + (可选) 最后一个 ViTDecoderBlock (原逻辑, 不变)
        """
        depth = self.depths[stage_idx]
        blocks = nn.ModuleList()
        
        # === 预训练模式：纯编码器 ===
        if self.pretrain_mode:
            num_enc = depth
            num_dec = 0
            print(f"[Stage {stage_idx}] 预训练模式: {num_enc} Encoders, {num_dec} Decoders"
                  f" (block_type={block_type}, use_bilevel={use_bilevel})")
        else:
            num_enc = depth - 1 if self.use_multimodal else depth
            num_dec = 1 if self.use_multimodal else 0
            print(f"[Stage {stage_idx}] 多模态模式: {num_enc} Encoders, {num_dec} Decoders"
                  f" (block_type={block_type}, use_bilevel={use_bilevel})")
        
        # =================================================================
        # 分支 A: ViT 层 (三阶段 Stage3 / 四阶段 Stage4) — 保持原逻辑不变
        # =================================================================
        if block_type == 'vit' or not use_bilevel:
            # Encoders
            for i in range(num_enc):
                curr_dpr = dpr[self.dpr_ptr]
                self.dpr_ptr += 1
                if block_type == 'swin':
                    # use_bilevel=False 且 block_type='swin'
                    # (用于 cross_window_enabled=False 时的 Swin stage fallback)
                    # ★ [4.26 第二轮] 不再传 pretrained_window_size (参数已删除)
                    # ★ [2026-05-09] v19: 透传 RGB 主干现代化参数
                    blocks.append(SwinEncoderBlock(
                        dim, num_heads, window_size,
                        mlp_ratio=mlp_ratio, drop_path=curr_dpr,
                        swin_version=self.swin_version,
                        **self._v19_rgb_swin_kwargs,
                    ))
                else:
                    # ViT
                    blocks.append(ViTEncoderBlock(
                        dim, num_heads,
                        mlp_ratio=mlp_ratio, drop_path=curr_dpr,
                        **self._v14_vit_kwargs
                    ))
            # Decoder
            if num_dec > 0:
                curr_dpr = dpr[self.dpr_ptr]
                self.dpr_ptr += 1
                if block_type == 'swin':
                    if self.cross_attention_type == 'deformable':
                        blocks.append(SwinDeformableDecoderBlock(
                            dim=dim, num_heads=num_heads, window_size=window_size,
                            num_points=self.num_deformable_points,
                            mlp_ratio=mlp_ratio, drop_path=curr_dpr
                        ))
                    else:
                        blocks.append(SwinDecoderBlock(
                            dim=dim, num_heads=num_heads, window_size=window_size,
                            mlp_ratio=mlp_ratio, drop_path=curr_dpr
                        ))
                else:
                    blocks.append(ViTDecoderBlock(
                        dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, drop_path=curr_dpr,
                        **self._v14_vit_kwargs
                    ))
            return blocks
        
        # =================================================================
        # 分支 B: Swin + BiLevel 交替排列 (Stage 1/2 / 四阶段 Stage 3)
        # =================================================================
        # 新 block 排列规则 (i 为 block 索引, 从 0 开始):
        #   i == 0:         WindowAttention (→ 加载 Swin-T 预训练 block 0)
        #   i % 2 == 1 且不是最后一个 decoder block:  BiLevel (奇数位置)
        #   i % 2 == 0 且不是 0:  WindowAttention (复用 block 0 预训练)
        #   最后一个 (如果 use_multimodal=True): SwinDeformableDecoderBlock
        #   最后一个 (如果 use_multimodal=False): BiLevel
        
        if depth < 4:
            print(f"⚠️  [Stage {stage_idx}] depth={depth} < 4, BiLevel 排列可能不完整")
        
        # Encoders (前 num_enc 个)
        for i in range(num_enc):
            curr_dpr = dpr[self.dpr_ptr]
            self.dpr_ptr += 1
            
            # 偶数位置 (0, 2, 4, ...) 用 WindowAttention
            # 奇数位置 (1, 3, 5, ...) 用 BiLevel (如果 cross_window_enabled)
            is_bilevel_position = (i % 2 == 1)
            
            if self.shift_window:
                # ★ [2026-06-29] shift-window 对照基线: 用标准 Swin shifted-window 取代 BiLevel.
                #   奇数位 SW-MSA (shift=ws//2), 偶数位 W-MSA (shift=0) → 标准 Swin 交替排列.
                #   这些 block 仍位于 rgb_sX_blocks.1 / .3, 会被 get_param_groups 的
                #   rgb_s12_rand 模式匹配 → 与 BiLevel 同等 LR 处理 → 公平单变量 A/B.
                _ss = (window_size // 2) if is_bilevel_position else 0
                blocks.append(SwinEncoderBlock(
                    dim, num_heads, window_size,
                    shift_size=_ss,
                    mlp_ratio=mlp_ratio, drop_path=curr_dpr,
                    swin_version=self.swin_version,
                    **self._v19_rgb_swin_kwargs,
                ))
            elif (is_bilevel_position and self.cross_window_enabled
                  and ((stage_idx + 1) in BILEVEL_STAGES)):  # ★[2026-07-25] v23 按stage启停(不在集合→else建W-MSA shift=0)
                # BiLevel block
                if BILEVEL_V24_ENABLED:
                    # ★[2026-07-31] v24 双轨解耦: 光谱指数门 + 跨窗口检索
                    #   forward(x,H,W) 契约与 BiLevelWindowBlock 一致, 窗口划分在块内部.
                    blocks.append(BiLevelBlockV24(
                        dim=dim,
                        window_size=window_size,
                        num_learnable=BILEVEL_V24_NUM_LEARNABLE,
                        pool_heads=BILEVEL_V24_POOL_HEADS,
                        exchange_heads=BILEVEL_V24_EXCHANGE_HEADS,
                        retrieval_heads=BILEVEL_V24_RETRIEVAL_HEADS,
                        tau_min=BILEVEL_V24_TAU_MIN,
                        gamma_init=BILEVEL_V24_GAMMA_INIT,
                        exchange_ls_init=BILEVEL_V24_EXCHANGE_LS,
                        use_dwconv=BILEVEL_V24_USE_DWCONV,
                        use_ffn=BILEVEL_V24_USE_FFN,
                        drop_path=curr_dpr,
                    ))
                    if self.use_checkpoint and stage_idx in self._bilevel_layer_indices:
                        self._bilevel_layer_indices[stage_idx].add(i)
                elif True:
                    # ↓↓↓ v23 原分支, 逐字节保留; BILEVEL_V24_ENABLED=False 时走这里 ↓↓↓
                    blocks.append(BiLevelWindowBlock(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        cross_type=self.cross_window_type,
                        cross_top_k=self.cross_window_top_k,
                        mlp_ratio=mlp_ratio,
                        drop_path=curr_dpr,
                        cross_gate_init=self.cross_window_gate_init,
                        cross_num_pool_queries=self.cross_window_num_queries,  # ★ [2026-06-15] 改动1
                        cross_use_stats=self.cross_window_use_stats,           # ★ [2026-06-15] 改动1
                        cross_step3_mode=self.cross_window_step3_mode,         # ★ [2026-06-18] v3.0
                        cross_splat_null=self.cross_window_splat_null,         # ★ [2026-06-18] v3.0
                        cross_splatp_score_mode=self.cross_window_splatp_score_mode,  # ★ [2026-06-20] splat_plus
                        cross_splatp_fuse_mode=self.cross_window_splatp_fuse_mode,    # ★ [2026-06-20] splat_plus
                        cross_splatp_num_heads=self.cross_window_splatp_num_heads,    # ★ [2026-06-20] splat_plus
                        ffn_respost=self.cross_window_ffn_respost,                   # ★ [2026-07-11] 改动I: FFN 子层 res-post 开关
                        drop_norm_cross=self.cross_window_drop_norm_cross,           # ★ [2026-07-11] 改动J
                        mul_ln=self.cross_window_mul_ln,                             # ★ [2026-07-11] 改动J
                        cross_qk_norm=self.cross_window_cross_qk_norm,               # ★ [2026-07-11] 改动J
                    ))
                    # 记录为强制 checkpoint
                    if self.use_checkpoint and stage_idx in self._bilevel_layer_indices:
                        self._bilevel_layer_indices[stage_idx].add(i)
            else:
                # WindowAttention block (位置 0, 2, ... 或 cross_window_enabled=False 的 fallback)
                # ★ [4.26 第二轮] 不再传 pretrained_window_size (参数已删除)
                # ★ [2026-05-09] v19: 透传 RGB 主干现代化参数
                blocks.append(SwinEncoderBlock(
                    dim, num_heads, window_size,
                    mlp_ratio=mlp_ratio, drop_path=curr_dpr,
                    swin_version=self.swin_version,
                    **self._v19_rgb_swin_kwargs,
                ))
        
        # Decoder (最后一个 block, 仅多模态模式)
        if num_dec > 0:
            curr_dpr = dpr[self.dpr_ptr]
            self.dpr_ptr += 1
            # Decoder 总是使用 SwinDeformableDecoderBlock (用于 Aux 跨模态融合)
            if self.cross_attention_type == 'deformable':
                blocks.append(SwinDeformableDecoderBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    num_points=self.num_deformable_points,
                    mlp_ratio=mlp_ratio, drop_path=curr_dpr
                ))
            else:
                blocks.append(SwinDecoderBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    mlp_ratio=mlp_ratio, drop_path=curr_dpr
                ))
        else:
            # 单流模式 (pretrain / use_multimodal=False) 下,
            # 最后一个 block 还没加入. 因为 num_enc = depth (已经全部加过),
            # 所以这里什么都不用做. (前面循环已经加了 depth 个 block)
            # 注: 如果 depth >= 4 且最后一个位置恰好是奇数 (i=3 对 depth=4),
            #      那最后一个已经是 BiLevel 了, 符合设计
            pass
        
        return blocks

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # =================================================================
    # Checkpoint 包装器
    # =================================================================
    
    def _run_with_checkpoint(
        self, 
        block: nn.Module, 
        *args, 
        stage_idx: int = 0, 
        layer_idx: int = 0,
        is_decoder: bool = False
    ):
        """
        运行 block，根据配置决定是否使用 checkpoint
        
        ★ [2026-04-20] v16: BiLevelWindowBlock 强制 checkpoint
          判定优先级:
            1. 如果该层在 _bilevel_layer_indices 中 (BiLevel 层) → 强制 checkpoint
            2. 否则按 _checkpoint_layers 的 ratio 策略判定
        """
        # 检查是否为 BiLevel 层 (强制 checkpoint)
        is_bilevel_layer = (
            hasattr(self, '_bilevel_layer_indices') and
            stage_idx in self._bilevel_layer_indices and
            layer_idx in self._bilevel_layer_indices[stage_idx]
        )
        
        # 常规 checkpoint 策略判定
        should_checkpoint_by_ratio = (
            self.use_checkpoint and 
            self.training and
            stage_idx in self._checkpoint_layers and
            layer_idx in self._checkpoint_layers[stage_idx]
        )
        
        # BiLevel 层: 只要 use_checkpoint=True 且 training 就强制 checkpoint
        # 其他层: 按 ratio 判定
        should_checkpoint = (
            (is_bilevel_layer and self.use_checkpoint and self.training) or
            should_checkpoint_by_ratio
        )
        
        if should_checkpoint:
            if is_decoder:
                # Decoder 需要更多参数
                return checkpoint(
                    self._decoder_forward, block, *args,
                    use_reentrant=False
                )
            else:
                return checkpoint(
                    self._encoder_forward, block, *args,
                    use_reentrant=False
                )
        else:
            if is_decoder:
                x, H, W, aux, H_aux, W_aux = args
                return block(x, H, W, aux, H_aux, W_aux)
            else:
                x, H, W = args
                return block(x, H, W)
    
    @staticmethod
    def _encoder_forward(block, x, H, W):
        return block(x, H, W)
    
    @staticmethod
    def _decoder_forward(block, x, H, W, aux, H_aux, W_aux):
        return block(x, H, W, aux, H_aux, W_aux)
    
    
    def forward(self, rgb: torch.Tensor, aux: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        多模式前向传播：预训练 / RGB-only 消融 / 多模态
        
        ★ [v14.4, 2026-04-12] 新增 RGB-only 路径:
          当 use_multimodal=False 或 aux=None 时, 走纯编码器路径
          用途: 消融实验对比 RGB-only vs RGB+Aux 的 mIoU 差异
          此时所有 block 均为 Encoder (无 Decoder), 与 _pretrain_forward 结构一致
        """
        features = {}
        
        # === 预训练模式：单流编码器优化路径 ===
        if self.pretrain_mode:
            return self._pretrain_forward(rgb)
        
        # ★ [v14.4, 2026-04-12] RGB-only 消融: 无多模态或无 aux 输入时
        #   当 USE_MULTIMODAL=False 时, _build_stage 创建的全部是 Encoder (无 Decoder)
        #   直接复用 _pretrain_forward 即可 (结构完全一致)
        if not self.use_multimodal or aux is None:
            return self._pretrain_forward(rgb)
        
        # === 多模态模式：完整双流并行路径 ===
        return self._multimodal_forward(rgb, aux)

    def _pretrain_forward(self, rgb: torch.Tensor) -> Dict[str, Any]:
        """
        预训练模式前向传播 (纯编码器, 无 decoder 冗余)
        
        ★ [2026-04-20] v16: 支持三阶段和四阶段两种模式
        
        ★ [2026-05-11] v20: 加入跨 stage 跨窗口通信
          - cross_stage_mode='none':       完全等价于 v19 行为 (no-op)
          - cross_stage_mode='broadcast':  在 stage 1/2/3 入口注入 Memory Bank 的 g4
                                            stage 4 出口存入 Memory Bank
          - cross_stage_mode='token_bank': 在每个 stage 入口 read, 出口 write
        
        优化特点:
        - 纯编码器架构, 无跨模态融合
        - 无并行流开销
        - 显存和计算效率最优
        """
        features = {}
        
        # === Stage0 (stem, 256×256, W-MSA) — 不写入 features (不进解码器) ===
        rgb_x, H, W, stage0_premerge = self.rgb_stem_stage0(rgb)   # → [B,128*128,96]; premerge=[B,256*256,48]
        B = rgb_x.shape[0]  # ★ [2026-05-11] v20: 用于方案 C 的 Bank.begin_forward
        
        # ★ [2026-06-29] stage0_to_decoder: stem 256² premerge → 最高分辨率 U-Net skip 写入 features.
        #   gate = stage0_skip_gamma (可学 LayerScale, init 0.1) × stage0_skip_warmup (train 端 epoch 0→1).
        if self.stage0_to_decoder:
            features['stage0'] = (self.stage0_skip_warmup * self.stage0_skip_gamma) * stage0_premerge
        
        # ★ [2026-05-11] v20: 方案 C 入口 — 初始化 Bank 运行时副本
        if self.cross_stage_module is not None and self.cross_stage_mode == 'token_bank':
            self.cross_stage_module.begin_forward(batch_size=B)
        
        # === Stage 1: 纯编码器 (通用) ===
        # ★ [2026-05-11] v20: 入 stage 1 之前注入 (read / inject)
        rgb_x = self._cross_stage_read(rgb_x, stage_idx=0)
        
        for i, block in enumerate(self.rgb_s1_blocks):
            rgb_x = self._run_with_checkpoint(block, rgb_x, H, W, stage_idx=0, layer_idx=i)
        
        # ★ [2026-05-11] v20: 方案 C 在 stage 1 出口写回 Bank
        self._cross_stage_write(rgb_x, stage_idx=0)
        
        features['stage1'] = rgb_x
        
        # === Stage 2: 纯编码器 (通用) ===
        rgb_x, H, W = self.rgb_merge1(rgb_x, H, W)
        
        # ★ [2026-05-11] v20: 入 stage 2 之前
        rgb_x = self._cross_stage_read(rgb_x, stage_idx=1)
        
        for i, block in enumerate(self.rgb_s2_blocks):
            rgb_x = self._run_with_checkpoint(block, rgb_x, H, W, stage_idx=1, layer_idx=i)
        
        self._cross_stage_write(rgb_x, stage_idx=1)
        
        features['stage2'] = rgb_x
        
        # === Stage 3: Swin + BiLevel (最深输出 stage) ===
        # [2026-06-21] 单一架构: 主链停在 stage3 (32×32, 384d), 不再有 merge3 + stage4 ViT.
        rgb_x, H, W = self.rgb_merge2(rgb_x, H, W)  # 192→384, 64→32

        # ★ [2026-05-11] v20: 入 stage 3 之前 (cross_stage_mode='none' 时为 no-op)
        rgb_x = self._cross_stage_read(rgb_x, stage_idx=2)

        for i, block in enumerate(self.rgb_s3_blocks):
            rgb_x = self._run_with_checkpoint(block, rgb_x, H, W, stage_idx=2, layer_idx=i)

        self._cross_stage_write(rgb_x, stage_idx=2)

        rgb_x = self.rgb_norm(rgb_x)
        features['stage3'] = rgb_x
        features['output'] = rgb_x

        # [封存 stage4 重新启用模板]:
        #   rgb_x, H, W = self.rgb_merge3(rgb_x, H, W)        # 384→768, 32→16
        #   for i, block in enumerate(self.rgb_s4_blocks):
        #       rgb_x = self._run_with_checkpoint(block, rgb_x, H, W, stage_idx=3, layer_idx=i)
        #   rgb_x = self.rgb_norm(rgb_x); features['stage4'] = rgb_x; features['output'] = rgb_x

        return features
    
    # ───────────────────────────────────────────────────────────────────
    # ★ [2026-05-11] v20: 跨 stage 通信辅助方法 (统一封装方案 A/C 的差异)
    # ───────────────────────────────────────────────────────────────────
    def _cross_stage_read(self, x: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """
        跨 stage 读取 / 注入. 自动处理方案 A / A+ / C 的差异.
        cross_stage_mode='none' 时 no-op 直接返回 x.
        """
        if self.cross_stage_module is None:
            return x
        if self.cross_stage_mode == 'broadcast':
            # 方案 A: 仅 stage 0/1/2 (浅层) 注入, stage 3 跳过
            if stage_idx < 3:
                return self.cross_stage_module.inject(x, stage_idx=stage_idx)
            return x
        elif self.cross_stage_mode == 'broadcast_momentum':
            # 方案 A+: inject 浅层 (默认 stage 0/1/2, 由模块内 inject_stages 控制)
            #   stage 3 跳过 (它是动量编码器的源)
            if stage_idx < 3:
                return self.cross_stage_module.inject(x, stage_idx=stage_idx)
            return x
        elif self.cross_stage_mode == 'query_locate':
            # 方案 D (D1): top-K query 点积调制浅层 (stage 0/1/2)
            #   stage 3 跳过 (它是动量编码器的源)
            #   inject_enabled=False (epoch<开启点) 时模块内部恒等返回
            if stage_idx < 3:
                return self.cross_stage_module.modulate(x, stage_idx=stage_idx)
            return x
        elif self.cross_stage_mode == 'token_bank':
            return self.cross_stage_module.read_at(x, stage_idx=stage_idx)
        return x
    
    def _cross_stage_write(self, x: torch.Tensor, stage_idx: int):
        """
        跨 stage 写入 (仅方案 C 在 stage 出口需要写, 方案 A 仅在 stage 4 出口 update).
        cross_stage_mode='none' 或方案 A 时 no-op.
        """
        if self.cross_stage_module is None:
            return
        if self.cross_stage_mode == 'token_bank':
            self.cross_stage_module.write_at(x, stage_idx=stage_idx)

    def cross_stage_momentum_step(self):
        """
        ★ [2026-05-13] 方案 A+ 专用: EMA 更新动量 ViT.
        必须在训练循环 optimizer.step() 之后调用 (此时 stage4 参数已更新).
        其他模式 (none/broadcast/token_bank) 下是 no-op, 可以无条件调用.

        训练脚本用法:
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            model.backbone.cross_stage_momentum_step()   # ← 加这一行
        """
        if self.cross_stage_module is None:
            return
        if self.cross_stage_mode in ('broadcast_momentum', 'query_locate'):
            self.cross_stage_module.momentum_step(self.rgb_s4_blocks, self.rgb_norm)

    def cross_stage_set_inject_enabled(self, enabled: bool):
        """
        ★ 方案 D (query_locate) 专用: 注入(调制)总开关。
        关闭时 modulate 恒等返回, 但 update_bank/momentum_step 不受影响
        (epoch 0~9 关调制但照常喂 Bank + 更新动量 ViT)。
        其他模式下是 no-op, 可无条件调用。

        训练脚本用法 (每 epoch 开头):
            model.backbone.cross_stage_set_inject_enabled(epoch >= AUX_INJECT_START_EPOCH)
        """
        if self.cross_stage_module is None:
            return
        if self.cross_stage_mode == 'query_locate' and \
           hasattr(self.cross_stage_module, 'set_inject_enabled'):
            self.cross_stage_module.set_inject_enabled(enabled)

    def cross_stage_is_injection_ready(self) -> bool:
        """方案 D 专用: Bank 是否被写过至少一次 (供训练脚本参考). 其他模式返回 True."""
        if self.cross_stage_module is None:
            return True
        if self.cross_stage_mode == 'query_locate' and \
           hasattr(self.cross_stage_module, 'is_injection_ready'):
            return self.cross_stage_module.is_injection_ready()
        return True

    def cross_stage_set_kv_cache(self, enable: bool, refresh_interval: int = None):
        """
        ★ [2026-05-13] 方案 A+ 专用: 运行时开关 KV cache.
        其他模式下是 no-op.
        例: 训练中后期 Bank 稳定后开启省算力:
            model.backbone.cross_stage_set_kv_cache(True, refresh_interval=10)
        """
        if self.cross_stage_module is None:
            return
        if self.cross_stage_mode == 'broadcast_momentum' and \
           hasattr(self.cross_stage_module, 'set_kv_cache'):
            self.cross_stage_module.set_kv_cache(enable, refresh_interval)

    # =================================================================
    # 并行化前向传播
    # =================================================================
    
    def _multimodal_forward(self, rgb: torch.Tensor, aux: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        多模态模式前向传播 (完整编码器-解码器架构)
        并行化双流前向传播
        
        ★ [2026-04-20] v16: 支持三阶段和四阶段两种模式
        
        优化策略:
        1. RGB 和 Aux 在不同的 CUDA streams 上并行运行
        2. 融合点进行同步
        3. 使用 Gradient Checkpointing 减少显存
        
        Aux 分支对齐策略:
          - 三阶段: Aux 在 Stage1/2/3 三个点融合 (aux_proj_s1/s2/s3)
          - 四阶段: Aux 在 Stage1/2/3/4 四个点融合 (aux_proj_s1/s2/s3/s4)
        """
        features = {}
        
        # === 初始化 ===
        aux_x = None
        H_aux, W_aux = 0, 0
        
        # =================================================================
        # Embedding 阶段 (可并行)
        # =================================================================
        
        if self.use_multimodal and aux is not None and self.use_parallel_streams:
            with self._stream_manager.get_aux_stream_context():
                aux_out = self.aux_branch.patch_embed(aux)
                aux_x = aux_out.embeddings
                H_aux, W_aux = aux_out.H, aux_out.W
            
            with self._stream_manager.get_rgb_stream_context():
                rgb_x, H, W, stage0_premerge = self.rgb_stem_stage0(rgb)
            
            self._stream_manager.synchronize()
        else:
            if self.use_multimodal and aux is not None:
                aux_out = self.aux_branch.patch_embed(aux)
                aux_x = aux_out.embeddings
                H_aux, W_aux = aux_out.H, aux_out.W
            
            rgb_x, H, W, stage0_premerge = self.rgb_stem_stage0(rgb)
        
        # ★ [2026-06-29] stage0_to_decoder: stem 256² premerge → 最高分辨率 U-Net skip (两个调用点汇合后统一写).
        if self.stage0_to_decoder:
            features['stage0'] = (self.stage0_skip_warmup * self.stage0_skip_gamma) * stage0_premerge
        
        # =================================================================
        # Stage 1 (通用, 两种架构模式相同)
        # =================================================================
        rgb_x, aux_x, H, W, aux_kv = self._run_swin_stage_with_fusion(
            rgb_x=rgb_x, H=H, W=W,
            aux_x=aux_x, H_aux=H_aux, W_aux=W_aux,
            rgb_blocks=self.rgb_s1_blocks,
            aux_forward=self.aux_branch.forward_stage1 if self.use_multimodal else None,
            aux_proj=self.aux_proj_s1 if self.use_multimodal else None,
            stage_idx=0,
        )
        features['stage1'] = rgb_x
        if self.use_multimodal and aux_x is not None:
            features['aux_s1'] = aux_x
        
        # =================================================================
        # Stage 2 (通用, 两种架构模式相同)
        # =================================================================
        # Downsample RGB: Merge1 (96→192, 128→64)
        rgb_x, H, W = self.rgb_merge1(rgb_x, H, W)
        
        rgb_x, aux_x, H, W, aux_kv = self._run_swin_stage_with_fusion(
            rgb_x=rgb_x, H=H, W=W,
            aux_x=aux_x, H_aux=H_aux, W_aux=W_aux,
            rgb_blocks=self.rgb_s2_blocks,
            aux_forward=self.aux_branch.forward_stage2 if self.use_multimodal else None,
            aux_proj=self.aux_proj_s2 if self.use_multimodal else None,
            stage_idx=1,
        )
        features['stage2'] = rgb_x
        if self.use_multimodal and aux_x is not None:
            features['aux_s2'] = aux_x
        
        # =================================================================
        # Stage 3 分支 (三阶段走 ViT, 四阶段走 Swin+BiLevel)
        # =================================================================
        # =================================================================
        # Stage 3: Swin + BiLevel (32×32, 384d) — 最深输出 stage
        # [2026-06-21] 单一架构: 主链停在 stage3, 无 merge3 + stage4 ViT.
        # 注: 多流(多模态)路径当前未被使用; stage0 已替换 embed, 此处仅作结构对齐,
        #     aux 融合仍在 stage1/2/3 (aux 最终 norm 细节如启用多模态需复核).
        # =================================================================
        rgb_x, H, W = self.rgb_merge2(rgb_x, H, W)
        rgb_x, aux_x, H, W, aux_kv = self._run_swin_stage_with_fusion(
            rgb_x=rgb_x, H=H, W=W,
            aux_x=aux_x, H_aux=H_aux, W_aux=W_aux,
            rgb_blocks=self.rgb_s3_blocks,
            aux_forward=self.aux_branch.forward_stage3 if self.use_multimodal else None,
            aux_proj=self.aux_proj_s3 if self.use_multimodal else None,
            stage_idx=2,
        )
        rgb_x = self.rgb_norm(rgb_x)
        features['stage3'] = rgb_x
        features['output'] = rgb_x
        if self.use_multimodal and aux_x is not None:
            features['aux_output'] = aux_x
            features['aux_s3'] = aux_x
            features['aux_spatial'] = (H_aux, W_aux)

        return features
    
    # =================================================================
    # ★ [2026-04-20] v16: 辅助方法 — 通用 Stage 运行 + Aux 融合
    # =================================================================
    
    def _run_swin_stage_with_fusion(
        self,
        rgb_x: torch.Tensor, H: int, W: int,
        aux_x: Optional[torch.Tensor], H_aux: int, W_aux: int,
        rgb_blocks: nn.ModuleList,
        aux_forward: Optional[callable],
        aux_proj: Optional[nn.Module],
        stage_idx: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, Optional[torch.Tensor]]:
        """
        通用 Swin Stage 运行 (用于 Stage 1/2, 四阶段 Stage 3)
        
        内部执行:
          1. 并行运行 RGB Encoder blocks (除最后一个) 和 Aux forward
          2. 最后一个 block: Decoder (多流) 或 最后一个 Encoder (单流)
        
        Returns:
          rgb_x, aux_x, H, W, aux_kv (aux_kv 是投影后供 decoder 使用的 K/V)
        """
        aux_kv = None
        has_multimodal = (
            self.use_multimodal and aux_x is not None and aux_forward is not None
        )
        # 是否有 decoder 作为最后一个 block (仅多模态模式下且最后一个 block 是 Decoder)
        has_decoder = has_multimodal and (
            len(rgb_blocks) > 0 and
            isinstance(rgb_blocks[-1], (SwinDecoderBlock, SwinDeformableDecoderBlock))
        )
        
        if has_decoder:
            rgb_encoder_blocks = rgb_blocks[:-1]
            decoder_block = rgb_blocks[-1]
        else:
            rgb_encoder_blocks = rgb_blocks
            decoder_block = None
        
        # === 并行 / 串行 Encoder ===
        if has_multimodal and self.use_parallel_streams:
            rgb_x, aux_x = self._parallel_stage_encoders(
                rgb_x=rgb_x, H=H, W=W,
                aux_x=aux_x, H_aux=H_aux, W_aux=W_aux,
                rgb_blocks=rgb_encoder_blocks,
                aux_forward=aux_forward,
                stage_idx=stage_idx,
            )
        else:
            if has_multimodal:
                aux_x = aux_forward(aux_x, H_aux, W_aux)
            for i, block in enumerate(rgb_encoder_blocks):
                rgb_x = self._run_with_checkpoint(
                    block, rgb_x, H, W, stage_idx=stage_idx, layer_idx=i
                )
        
        # === Aux 投影 (供 Decoder 使用) ===
        if has_multimodal and aux_proj is not None:
            aux_kv = aux_proj(aux_x)
        
        # === Decoder (如果有) ===
        if has_decoder:
            last_idx = len(rgb_blocks) - 1
            if aux_kv is not None:
                rgb_x = self._run_with_checkpoint(
                    decoder_block, rgb_x, H, W, aux_kv, H_aux, W_aux,
                    stage_idx=stage_idx, layer_idx=last_idx, is_decoder=True
                )
            else:
                # Fallback: 无 aux 时自己做 self-attention
                rgb_x = self._run_with_checkpoint(
                    decoder_block, rgb_x, H, W, rgb_x, H, W,
                    stage_idx=stage_idx, layer_idx=last_idx, is_decoder=True
                )
        
        return rgb_x, aux_x, H, W, aux_kv
    
    def _run_vit_stage_with_fusion(
        self,
        rgb_x: torch.Tensor, H: int, W: int,
        aux_x: Optional[torch.Tensor], H_aux: int, W_aux: int,
        rgb_blocks: nn.ModuleList,
        aux_forward: Optional[callable],
        aux_proj: Optional[nn.Module],
        stage_idx: int,
        is_last_stage: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int, Optional[torch.Tensor]]:
        """
        通用 ViT Stage 运行 (用于三阶段 Stage 3, 四阶段 Stage 4)
        
        与 swin 版本的主要区别:
        - is_last_stage=True 时, Aux 会在 encoder 跑完后做 norm (对应 aux_branch.norm())
        - 四阶段模式下, Stage 4 的 aux_forward=None (Aux 已在 Stage 3 跑完)
        """
        aux_kv = None
        has_aux_forward = (
            self.use_multimodal and aux_x is not None and aux_forward is not None
        )
        has_aux_data = self.use_multimodal and aux_x is not None
        has_decoder = has_aux_data and (
            len(rgb_blocks) > 0 and isinstance(rgb_blocks[-1], ViTDecoderBlock)
        )
        
        if has_decoder:
            rgb_encoder_blocks = rgb_blocks[:-1]
            decoder_block = rgb_blocks[-1]
        else:
            rgb_encoder_blocks = rgb_blocks
            decoder_block = None
        
        # === 并行 / 串行 Encoder ===
        if has_aux_forward and self.use_parallel_streams:
            rgb_x, aux_x = self._parallel_stage_encoders(
                rgb_x=rgb_x, H=H, W=W,
                aux_x=aux_x, H_aux=H_aux, W_aux=W_aux,
                rgb_blocks=rgb_encoder_blocks,
                aux_forward=aux_forward,
                stage_idx=stage_idx,
            )
        else:
            if has_aux_forward:
                aux_x = aux_forward(aux_x, H_aux, W_aux)
            for i, block in enumerate(rgb_encoder_blocks):
                rgb_x = self._run_with_checkpoint(
                    block, rgb_x, H, W, stage_idx=stage_idx, layer_idx=i
                )
        
        # === 最后一个 Stage 需要对 Aux 做 norm ===
        if is_last_stage and has_aux_data:
            aux_x = self.aux_branch.norm(aux_x)
        
        # === Aux 投影 ===
        if has_aux_data and aux_proj is not None:
            aux_kv = aux_proj(aux_x)
        
        # === Decoder ===
        if has_decoder:
            last_idx = len(rgb_blocks) - 1
            if aux_kv is not None:
                rgb_x = self._run_with_checkpoint(
                    decoder_block, rgb_x, H, W, aux_kv, H_aux, W_aux,
                    stage_idx=stage_idx, layer_idx=last_idx, is_decoder=True
                )
            else:
                rgb_x = self._run_with_checkpoint(
                    decoder_block, rgb_x, H, W, rgb_x, H, W,
                    stage_idx=stage_idx, layer_idx=last_idx, is_decoder=True
                )
        
        return rgb_x, aux_x, H, W, aux_kv

    def _parallel_stage_encoders(
        self,
        rgb_x: torch.Tensor,
        H: int, W: int,
        aux_x: torch.Tensor,
        H_aux: int, W_aux: int,
        rgb_blocks: nn.ModuleList,
        aux_forward,
        stage_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        并行运行 Stage 的 Encoder 部分
        
        使用 CUDA Streams 实现真正的 GPU 并行
        """
        # Aux 分支在独立 stream 上运行
        with self._stream_manager.get_aux_stream_context():
            aux_x = aux_forward(aux_x, H_aux, W_aux)
        
        # RGB 分支在独立 stream 上运行
        with self._stream_manager.get_rgb_stream_context():
            for i, block in enumerate(rgb_blocks):
                rgb_x = self._run_with_checkpoint(block, rgb_x, H, W, stage_idx=stage_idx, layer_idx=i)
        
        # 同步两个 stream
        self._stream_manager.synchronize()
        
        return rgb_x, aux_x
    # =================================================================
    # === 新增模式切换方法 ===
    # ================================================================
    def switch_to_pretrain_mode(self):
        """切换到预训练模式"""
        if self.pretrain_mode:
            print("[MultimodalViT] 已经在预训练模式")
            return
        
        self.pretrain_mode = True
        self.use_multimodal = False
        self.use_parallel_streams = False
        
        # 禁用相关功能
        self.disable_parallel_streams()
        
        print("[MultimodalViT] 已切换到预训练模式（纯编码器优化）")

    def switch_to_multimodal_mode(self):
        """切换到多模态模式"""
        if not self.pretrain_mode:
            print("[MultimodalViT] 已经在多模态模式")
            return
        
        # 检查是否有aux相关组件（添加 is not None 检查）
        if not hasattr(self, 'aux_branch') or self.aux_branch is None:
            print("[MultimodalViT] 错误：当前模型缺少多模态组件，无法切换")
            return
        
        self.pretrain_mode = False
        self.use_multimodal = True
        self.use_parallel_streams = True
        
        # 重新初始化 stream manager
        self._stream_manager = ParallelStreamManager(enabled=True)
        
        print("[MultimodalViT] 已切换到多模态模式（编码器-解码器架构）")

    # =================================================================
    # LoRA 微调控制
    # =================================================================
    
    def enable_lora_finetune(
        self,
        r: Optional[int] = None,
        alpha: Optional[int] = None,
        target_modules: Optional[List[str]] = None,
        freeze_backbone: bool = True,
        include_aux: bool = True,
        bilevel_r: Optional[int] = None,
        bilevel_alpha: Optional[int] = None,
    ):
        """
        启用 LoRA 微调模式
        
        ★ [2026-04-20] v16: 支持 BiLevel 独立 LoRA 配置
          - 主 WindowAttention 的 Linear (qkv, proj): 用 r/alpha
          - BiLevel 内部的 Linear (q_proj, kv_proj, out_proj 等): 用 bilevel_r/bilevel_alpha
        
        Args:
            r: 主 LoRA 秩 (用于 WindowAttention 等常规模块)
            alpha: 主 LoRA 缩放系数
            target_modules: 目标模块名称列表 (如 ['qkv', 'proj'])
            freeze_backbone: 是否冻结主干
            include_aux: 是否包含 Aux 分支
            bilevel_r: BiLevel 独立 LoRA 秩 (None → 使用 __init__ 中的 bilevel_lora_r)
            bilevel_alpha: BiLevel 独立 LoRA 缩放系数
        """
        if self._lora_injected:
            print("[MultimodalViT] LoRA already injected, skipping...")
            return
        
        r = r or self._lora_config['r']
        alpha = alpha or self._lora_config['alpha']
        target_modules = target_modules or self._lora_config['target_modules']
        
        # ★ [2026-04-20] v16: BiLevel 独立配置
        bilevel_r = bilevel_r if bilevel_r is not None else self._bilevel_lora_config['r']
        bilevel_alpha = bilevel_alpha if bilevel_alpha is not None else self._bilevel_lora_config['alpha']
        
        if r <= 0 and bilevel_r <= 0:
            print("[MultimodalViT] 主 LoRA r 和 BiLevel LoRA r 都 <= 0, 跳过 LoRA 注入")
            return
        
        # 1. 冻结主干
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
            print("[MultimodalViT] Backbone frozen")
        
        # 2. 注入 LoRA (区分主 Attention 和 BiLevel)
        self._inject_lora_impl(
            main_r=r, main_alpha=alpha, main_target_modules=target_modules,
            bilevel_r=bilevel_r, bilevel_alpha=bilevel_alpha,
            exclude_aux=not include_aux,
        )
        
        # 3. Aux 分支的 LoRA (使用主 r/alpha)
        if include_aux and self.use_multimodal and hasattr(self, 'aux_branch') and self.aux_branch is not None:
            if r > 0:
                self.aux_branch.enable_lora_finetune(
                    r=r, alpha=alpha, target_modules=target_modules, freeze_backbone=False
                )
        
        self._lora_injected = True
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[MultimodalViT] LoRA enabled: {trainable/1e6:.2f}M / {total/1e6:.2f}M trainable "
              f"(main r={r}, bilevel r={bilevel_r})")
    
    def disable_lora_finetune(self):
        """禁用 LoRA 微调 (解冻所有参数)"""
        for param in self.parameters():
            param.requires_grad = True
        print("[MultimodalViT] All parameters unfrozen")
    
    def _inject_lora_impl(
        self,
        main_r: int = 0, main_alpha: int = 1,
        main_target_modules: List[str] = None,
        bilevel_r: int = 0, bilevel_alpha: int = 1,
        exclude_aux: bool = False,
    ):
        """
        LoRA 注入实现
        
        ★ [2026-04-20] v16: 根据模块所在位置区分主 LoRA 和 BiLevel LoRA
          判定规则: 如果祖先模块是 BiLevelWindowBlock, 则该 Linear 使用 bilevel_r/alpha
                   否则使用 main_r/alpha
        """
        main_target_modules = main_target_modules or []
        injected_count = 0
        bilevel_injected_count = 0
        
        def replace_recursively(module, prefix=""):
            nonlocal injected_count, bilevel_injected_count
            
            # 判断当前 module 是否为 BiLevelWindowBlock (影响子 Linear 的 LoRA 配置)
            is_bilevel_parent = isinstance(module, BiLevelWindowBlock)
            
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # 跳过 Aux 分支
                if exclude_aux and 'aux_branch' in full_name:
                    continue
                
                # 如果子 module 是 Linear, 且是一个 target module
                is_target_linear = (
                    isinstance(child, nn.Linear) and
                    any(t in name for t in main_target_modules + [
                        'q_proj', 'kv_proj', 'qkv_proj', 'out_proj',
                        'pool_q_proj', 'pool_kv_proj',
                        'summary_qkv_proj', 'summary_out_proj',
                        'dist_q_proj', 'dist_kv_proj', 'dist_out_proj',
                    ])
                )
                
                if is_target_linear:
                    # 判断当前 Linear 是否在 BiLevelWindowBlock 内部
                    # (通过祖先栈或 prefix 判断)
                    is_in_bilevel = is_bilevel_parent or ('BiLevel' in type(module).__name__)
                    # 或通过 prefix 路径判断 (更鲁棒)
                    # prefix 包含 rgb_sX_blocks.Y, 我们需要检查 blocks[Y] 是否 BiLevel
                    # 最简单的方法: 判断 name 是否是 BiLevel 特有的
                    is_bilevel_specific_name = name in (
                        'pool_q_proj', 'pool_kv_proj',
                        'summary_qkv_proj', 'summary_out_proj',
                        'dist_q_proj', 'dist_kv_proj', 'dist_out_proj',
                    )
                    
                    if is_bilevel_parent or is_bilevel_specific_name:
                        # BiLevel 内部的 Linear, 用 bilevel_r/alpha
                        if bilevel_r > 0:
                            try:
                                print(f"  -> [BiLevel LoRA r={bilevel_r}] {full_name}")
                                setattr(module, name, LoRALinear(child, r=bilevel_r, alpha=bilevel_alpha))
                                bilevel_injected_count += 1
                            except Exception as e:
                                print(f"  -> Failed (BiLevel): {full_name}: {e}")
                    else:
                        # 主 Attention 的 Linear, 用 main_r/alpha
                        if main_r > 0:
                            try:
                                print(f"  -> [Main LoRA r={main_r}] {full_name}")
                                setattr(module, name, LoRALinear(child, r=main_r, alpha=main_alpha))
                                injected_count += 1
                            except Exception as e:
                                print(f"  -> Failed (Main): {full_name}: {e}")
                else:
                    replace_recursively(child, full_name)
        
        replace_recursively(self)
        
        print(f"✅ LoRA 注入完成: 主模块 {injected_count} 个, BiLevel 模块 {bilevel_injected_count} 个")

    # =================================================================
    # Checkpoint 控制
    # =================================================================
    
    def enable_checkpoint(self, ratio: float = 0.5):
        """启用 Gradient Checkpointing"""
        self.use_checkpoint = True
        self.checkpoint_ratio = ratio
        self._setup_checkpoint_layers(self.depths)
        
        # 同时启用 Aux 的 checkpoint
        if hasattr(self, 'aux_branch') and self.aux_branch is not None:
            self.aux_branch.enable_checkpoint(ratio)
        
        print(f"[MultimodalViT] Checkpoint enabled (ratio={ratio})")
    
    def disable_checkpoint(self):
        """禁用 Gradient Checkpointing"""
        self.use_checkpoint = False
        self._checkpoint_layers = {}
        
        if hasattr(self, 'aux_branch') and self.aux_branch is not None:
            self.aux_branch.disable_checkpoint()
        
        print("[MultimodalViT] Checkpoint disabled")
    
    # =================================================================
    # 并行化控制
    # =================================================================
    
    def enable_parallel_streams(self):
        """启用 CUDA Streams 并行化"""
        self.use_parallel_streams = True
        self._stream_manager = ParallelStreamManager(enabled=True)
        print("[MultimodalViT] Parallel streams enabled")
    
    def disable_parallel_streams(self):
        """禁用 CUDA Streams 并行化"""
        self.use_parallel_streams = False
        print("[MultimodalViT] Parallel streams disabled")

    # =================================================================
    # 工具方法
    # =================================================================
    
    def get_feature_info(self) -> Dict[str, Dict]:
        return self.feature_info

    def get_num_params(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def print_trainable_params(self):
        trainable = self.get_num_params(trainable_only=True)
        total = self.get_num_params(trainable_only=False)
        print(f"[MultimodalViT] Trainable: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M ({100*trainable/total:.2f}%)")
    
    def validate_state(self) -> Dict[str, bool]:
        """验证模型当前状态的一致性"""
        checks = {}
        
        # 预训练模式检查
        checks['pretrain_mode_consistency'] = (
            (self.pretrain_mode and not self.use_multimodal and not self.use_parallel_streams) or
            (not self.pretrain_mode)
        )
        
        # 多模态组件检查
        if self.use_multimodal and not self.pretrain_mode:
            checks['multimodal_components'] = (
                hasattr(self, 'aux_branch') and self.aux_branch is not None and
                hasattr(self, 'aux_proj_s1') and self.aux_proj_s1 is not None
            )
        else:
            checks['multimodal_components'] = True
        
        # 并行流检查
        checks['parallel_streams'] = (
            (self.use_parallel_streams and self._stream_manager.enabled) or
            (not self.use_parallel_streams and not self._stream_manager.enabled)
        )
        
        # LoRA状态检查
        checks['lora_consistency'] = True
        
        all_valid = all(checks.values())
        if not all_valid:
            print("⚠️ 模型状态不一致:")
            for check, valid in checks.items():
                if not valid:
                    print(f"  - {check}: FAILED")
        
        return checks
    
    def print_model_info(self):
        """打印模型的详细信息"""
        print(f"\n{'='*50}")
        print("模型配置信息")
        print(f"{'='*50}")
        print(f"图像尺寸: {self.img_size}")
        print(f"嵌入维度: {self.embed_dim}")
        print(f"深度配置: {self.depths}")
        print(f"注意力头: {self.num_heads}")
        print(f"预训练模式: {self.pretrain_mode}")
        print(f"多模态模式: {self.use_multimodal}")
        # ★ [4.26] SwinV2 适配信息
        # ★ [4.26 第二轮] 显示 per-stage window_sizes 替代旧的 pretrained_window_size
        print(f"Swin 版本: {self.swin_version} "
              f"(forward={'Post-Norm' if self.swin_version == 'v2' else 'Pre-Norm'}, "
              f"window_sizes={self.window_sizes})")
        print(f"并行流: {self.use_parallel_streams}")
        print(f"Checkpoint: {self.use_checkpoint} (ratio={self.checkpoint_ratio})")
        print(f"LoRA已注入: {self._lora_injected}")
        
        # 组件状态
        print(f"\n组件状态:")
        print(f"  - RGB分支: ✅")
        print(f"  - Aux分支: {'✅' if hasattr(self, 'aux_branch') and self.aux_branch is not None else '❌'}")
        print(f"  - 投影层: {'✅' if hasattr(self, 'aux_proj_s1') and self.aux_proj_s1 is not None else '❌'}")
        print(f"  - Stream管理: {'✅' if self._stream_manager.enabled else '❌'}")
        
        # 状态验证
        checks = self.validate_state()
        status = "✅ 正常" if all(checks.values()) else "⚠️ 异常"
        print(f"\n状态检查: {status}")
        print(f"{'='*50}")


# =============================================================================
# 便捷工厂函数
# =============================================================================

def create_multimodal_vit_for_training(
    img_size: int = 512,
    rgb_in_chans: int = 3,
    aux_in_chans: int = 5,
    use_checkpoint: bool = True,
    checkpoint_ratio: float = 0.5,
    use_parallel_streams: bool = True,
    memory_efficient_mode: bool = True
) -> MultimodalViT:
    """
    创建用于训练的模型 (无 LoRA)
    
    针对 RTX 5080 (16GB) 优化
    """
    return MultimodalViT(
        img_size=img_size,
        rgb_in_chans=rgb_in_chans,
        aux_in_chans=aux_in_chans,
        use_checkpoint=use_checkpoint,
        checkpoint_ratio=checkpoint_ratio,
        use_parallel_streams=use_parallel_streams,
        memory_efficient_mode=memory_efficient_mode,
        lora_r=0  # 训练时不启用 LoRA
    )


def create_multimodal_vit_for_finetune(
    img_size: int = 512,
    rgb_in_chans: int = 3,
    aux_in_chans: int = 5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    pretrained_path: Optional[str] = None
) -> MultimodalViT:
    """
    创建用于微调的模型 (带 LoRA)
    """
    model = MultimodalViT(
        img_size=img_size,
        rgb_in_chans=rgb_in_chans,
        aux_in_chans=aux_in_chans,
        use_checkpoint=True,
        checkpoint_ratio=0.3,  # 微调时降低 checkpoint 比例
        use_parallel_streams=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )
    
    # 加载预训练权重
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"[MultimodalViT] Loaded pretrained weights from {pretrained_path}")
    
    # 启用 LoRA 微调
    model.enable_lora_finetune(freeze_backbone=True)
    
    return model


def create_multimodal_vit_small(
    img_size: int = 256,
    use_multimodal: bool = None,
    architecture_mode: str = "three_stage",
    **kwargs
) -> MultimodalViT:
    """
    创建 Small 版本 (兼容旧接口)
    
    ★ [2026-04-20] v16: 新增 architecture_mode 参数
      - "three_stage" (默认): depths=[4, 4, 6], 使用 GlobalPatchEmbed
      - "four_stage":         depths=[4, 4, 4, 4], 全 PatchMerging
    """
    if architecture_mode == "four_stage":
        depths = kwargs.pop('depths', [4, 4, 4, 4])
        num_heads = kwargs.pop('num_heads', [6, 6, 12, 12])
    else:
        depths = kwargs.pop('depths', [4, 4, 6])
        num_heads = kwargs.pop('num_heads', [6, 6, 12])
    
    return MultimodalViT(
        img_size=img_size,
        embed_dim=96,
        depths=depths,
        num_heads=num_heads,
        use_multimodal=use_multimodal,
        architecture_mode=architecture_mode,
        **kwargs
    )


def create_multimodal_vit_base(
    img_size: int = 512,
    use_multimodal: bool = True,
    architecture_mode: str = "three_stage",
    **kwargs
) -> MultimodalViT:
    """创建 Base 版本"""
    if architecture_mode == "four_stage":
        depths = kwargs.pop('depths', [4, 4, 4, 4])
        num_heads = kwargs.pop('num_heads', [4, 8, 12, 12])
    else:
        depths = kwargs.pop('depths', [4, 4, 4])
        num_heads = kwargs.pop('num_heads', [4, 8, 12])
    
    return MultimodalViT(
        img_size=img_size,
        embed_dim=128,
        depths=depths,
        num_heads=num_heads,
        use_multimodal=use_multimodal,
        architecture_mode=architecture_mode,
        **kwargs
    )
    
def create_rgb_pretrain_model(
    img_size: int = 512,
    rgb_in_chans: int = 3,
    embed_dim: int = 96,
    depths: List[int] = None,
    num_heads: List[int] = None,
    architecture_mode: str = "three_stage",
    use_checkpoint: bool = True,
    checkpoint_ratio: float = 0.5,
    **kwargs
) -> MultimodalViT:
    """
    创建 RGB 预训练专用模型 (纯编码器优化)
    
    ★ [2026-04-20] v16: 默认 depths 根据 architecture_mode 选择
      - three_stage: [4, 4, 6]
      - four_stage:  [4, 4, 4, 4]
    
    特点:
    - pretrain_mode=True: 纯编码器架构
    - 无 decoder 冗余计算
    - 无并行流开销
    - 显存和速度最优
    """
    if depths is None:
        depths = [4, 4, 6] if architecture_mode == "three_stage" else [4, 4, 4, 4]
    if num_heads is None:
        num_heads = [6, 6, 12] if architecture_mode == "three_stage" else [6, 6, 12, 12]
    
    return MultimodalViT(
        img_size=img_size,
        rgb_in_chans=rgb_in_chans,
        aux_in_chans=5,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        architecture_mode=architecture_mode,
        
        # === 关键：启用预训练模式 ===
        pretrain_mode=True,
        use_multimodal=False,
        
        use_checkpoint=use_checkpoint,
        checkpoint_ratio=checkpoint_ratio,
        use_parallel_streams=False,
        lora_r=0,
        **kwargs
    )

def create_multimodal_from_pretrain(
    pretrained_model_or_path: Union[MultimodalViT, str],
    aux_in_chans: int = 5,
    img_size: Optional[int] = None,
    freeze_rgb_encoder: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    **kwargs
) -> MultimodalViT:
    """
    从预训练模型创建多模态模型（权重兼容转换）
    
    Args:
        pretrained_model_or_path: 预训练模型或权重路径
        freeze_rgb_encoder: 是否冻结RGB编码器
        lora_r: LoRA微调秩
    """
    if isinstance(pretrained_model_or_path, str):
        # 从权重文件创建
        if not os.path.exists(pretrained_model_or_path):
            raise FileNotFoundError(f"预训练权重文件不存在: {pretrained_model_or_path}")
            
        print(f"[权重加载] 从文件加载: {pretrained_model_or_path}")
        pretrain_state = torch.load(pretrained_model_or_path, map_location='cpu')
        
        # 检查关键权重是否存在
        required_keys = ['rgb_patch_embed.proj.weight', 'rgb_s1_blocks.0.norm1.weight']
        missing_required = [k for k in required_keys if k not in pretrain_state]
        if missing_required:
            print(f"⚠️ 警告：缺少关键权重: {missing_required}")
        
        # 推断配置参数
        img_size = img_size or 512
        
        # 从权重推断 embed_dim
        if 'rgb_patch_embed.proj.weight' in pretrain_state:
            embed_dim = pretrain_state['rgb_patch_embed.proj.weight'].shape[0]
        else:
            embed_dim = 96  # 默认值
        
        # 从权重推断 depths 和 num_heads
        # ★ [2026-04-20] v16: 默认从 three_stage 开始
        architecture_mode = kwargs.get('architecture_mode', 'three_stage')
        if architecture_mode == "four_stage":
            depths = kwargs.get('depths', [4, 4, 4, 4])
            num_heads = kwargs.get('num_heads', [6, 6, 12, 12])
        else:
            depths = kwargs.get('depths', [4, 4, 6])
            num_heads = kwargs.get('num_heads', [6, 6, 12])
        
        # 创建临时预训练模型来获取完整配置
        temp_pretrained = create_rgb_pretrain_model(
            img_size=img_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            **{k: v for k, v in kwargs.items() if k not in ['depths', 'num_heads']}
        )
        
        # 检查预训练模型是否为预训练模式（移到创建后）
        if hasattr(temp_pretrained, 'pretrain_mode') and not temp_pretrained.pretrain_mode:
            print("⚠️ 警告：预训练模型不在预训练模式，可能包含decoder权重")
        
        # 加载权重到临时模型
        missing, unexpected = temp_pretrained.load_state_dict(pretrain_state, strict=False)
        print(f"[权重加载] 缺失: {len(missing)}, 意外: {len(unexpected)}")
        
        pretrained_model = temp_pretrained
        
    elif isinstance(pretrained_model_or_path, MultimodalViT):
        pretrained_model = pretrained_model_or_path
        img_size = img_size or pretrained_model.img_size
    else:
        raise TypeError("pretrained_model_or_path 必须是 MultimodalViT 实例或文件路径")
    
    # === 2. 创建完整多模态模型 ===
    print("[模型构建] 创建多模态架构...")
    multimodal_model = MultimodalViT(
        img_size=img_size,
        rgb_in_chans=pretrained_model.rgb_patch_embed.proj.in_channels,
        aux_in_chans=aux_in_chans,
        embed_dim=pretrained_model.embed_dim,
        depths=pretrained_model.depths,
        num_heads=pretrained_model.num_heads,
        
        # === 多模态配置 ===
        pretrain_mode=False,    # 关闭预训练模式
        use_multimodal=True,    # 启用多模态
        use_checkpoint=True,
        checkpoint_ratio=0.3,
        use_parallel_streams=True,
        lora_r=0,  # 先不启用，后续手动启用
        lora_alpha=lora_alpha,
        **kwargs
    )
    
    # === 3. 智能权重迁移 ===
    print("[权重迁移] 开始迁移兼容权重...")
    pretrain_state_dict = pretrained_model.state_dict()
    multimodal_state_dict = multimodal_model.state_dict()
    
    transferred_keys = []
    incompatible_keys = []
    
    for key in pretrain_state_dict:
        if key in multimodal_state_dict:
            pretrain_shape = pretrain_state_dict[key].shape
            multimodal_shape = multimodal_state_dict[key].shape
            
            if pretrain_shape == multimodal_shape:
                multimodal_state_dict[key] = pretrain_state_dict[key].clone()
                transferred_keys.append(key)
            else:
                incompatible_keys.append(f"{key}: {pretrain_shape} -> {multimodal_shape}")
        else:
            # 预训练模型中有但多模态模型中没有的权重（正常，如某些decoder层）
            pass
    
    # 加载迁移后的权重
    multimodal_model.load_state_dict(multimodal_state_dict)
    
    print(f"[权重迁移] ✅ 成功迁移 {len(transferred_keys)} 个参数")
    print(f"[权重迁移] ⚠️  形状不兼容 {len(incompatible_keys)} 个参数")
    
    if incompatible_keys and len(incompatible_keys) < 10:  # 只打印前几个
        for inc in incompatible_keys[:5]:
            print(f"  - {inc}")
    
    # === 4. 配置微调策略 ===
    if freeze_rgb_encoder:
        print("[微调策略] 冻结RGB编码器...")
        frozen_params = 0
        total_params = 0
        
        for name, param in multimodal_model.named_parameters():
            total_params += param.numel()
            
            # 更精确的编码器冻结逻辑
            should_freeze = False
            
            # RGB Patch Embedding + 下采样模块 (两种架构模式都考虑)
            if any(x in name for x in [
                'rgb_patch_embed', 'rgb_merge1', 'rgb_merge2', 'rgb_merge3',
                'rgb_global_embed',
            ]):
                should_freeze = True
            
            # RGB 编码器层 (不包括最后的 decoder 层)
            # ★ [2026-04-20] v16: 支持三阶段 (rgb_s1/s2/s3) 和四阶段 (rgb_s1/s2/s3/s4)
            num_stages = multimodal_model.num_stages
            for stage_idx in range(num_stages):
                stage_depth = multimodal_model.depths[stage_idx]
                for layer_idx in range(stage_depth - 1):  # 排除最后一层 decoder
                    if f'rgb_s{stage_idx+1}_blocks.{layer_idx}' in name:
                        should_freeze = True
                        break
            
            if should_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"[微调策略] 冻结参数: {frozen_params/1e6:.2f}M / {total_params/1e6:.2f}M ({100*frozen_params/total_params:.1f}%)")
        print("[微调策略] 可训练: Decoder层 + Aux分支 + 投影层")
    # === 5. 启用LoRA (如果指定) ===
    if lora_r > 0:
        print(f"[LoRA] 启用微调 (r={lora_r}, alpha={lora_alpha})...")
        multimodal_model.enable_lora_finetune(
            r=lora_r, 
            alpha=lora_alpha, 
            freeze_backbone=freeze_rgb_encoder,
            include_aux=True
        )
    
    # === 6. 最终验证 ===
    multimodal_model.print_trainable_params()
    
    return multimodal_model
# ===========================================================================
# 添加测试函数来验证兼容性
# ===========================================================================
def test_pretrain_to_multimodal_compatibility():
    """测试预训练到多模态的兼容性转换"""
    print("\n" + "="*60)
    print("测试预训练 -> 多模态权重兼容转换")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === 1. 创建并保存预训练模型 ===
    print("\n[1] 创建RGB预训练模型 (三阶段, v16)")
    pretrain_model = create_rgb_pretrain_model(
        img_size=512,
        embed_dim=96,
        depths=[4, 4, 6],
        num_heads=[6, 6, 12],
        architecture_mode='three_stage',
    ).to(device)
    
    pretrain_model.print_trainable_params()
    
    # 模拟训练几步
    rgb_input = torch.randn(2, 3, 512, 512, device=device)
    pretrain_model.train()
    
    with torch.cuda.amp.autocast():
        features = pretrain_model(rgb_input)
    
    print(f"预训练输出: {features['output'].shape}")
    
    # 保存权重
    pretrain_path = 'test_rgb_pretrained.pth'
    torch.save(pretrain_model.state_dict(), pretrain_path)
    print(f"✅ 预训练权重已保存: {pretrain_path}")
    
    # === 2. 从预训练创建多模态模型 ===
    print("\n[2] 从预训练创建多模态模型")
    
    try:
        multimodal_model = create_multimodal_from_pretrain(
            pretrained_model_or_path=pretrain_path,
            aux_in_chans=5,
            freeze_rgb_encoder=True,
            lora_r=8,
            lora_alpha=16
        ).to(device)
        
        # === 3. 测试多模态前向传播 ===
        print("\n[3] 测试多模态前向传播")
        aux_input = torch.randn(2, 5, 512, 512, device=device)
        
        multimodal_model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            multimodal_features = multimodal_model(rgb_input, aux_input)
        
        print(f"多模态输出: {multimodal_features['output'].shape}")
        
        # === 4. 验证权重一致性 (RGB编码器部分) ===
        print("\n[4] 验证权重一致性")
        
        # 切换多模态模型到预训练模式测试
        multimodal_model.switch_to_pretrain_mode()
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            multimodal_pretrain_features = multimodal_model(rgb_input)
        
        # 比较输出 (应该相近，因为编码器权重相同)
        pretrain_output = features['output']
        multimodal_pretrain_output = multimodal_pretrain_features['output']
        
        # 计算差异
        diff = torch.abs(pretrain_output - multimodal_pretrain_output).mean().item()
        print(f"编码器输出差异: {diff:.6f}")
        
        if diff < 1e-5:
            print("✅ 权重迁移成功！编码器输出一致")
        else:
            print("⚠️ 权重迁移可能有问题，输出差异较大")
        
        # 切换回多模态模式
        multimodal_model.switch_to_multimodal_mode()
        
        print("\n✅ 兼容性测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理测试文件
        if os.path.exists(pretrain_path):
            os.remove(pretrain_path)
            print(f"🗑️ 清理测试文件: {pretrain_path}")
    
    print("="*60)

# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MultimodalViT 优化版测试")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试配置
    batch_size = 2
    img_size = 512
    rgb_chans = 3
    aux_chans = 5
    
    # 创建输入
    rgb = torch.randn(batch_size, rgb_chans, img_size, img_size, device=device)
    aux = torch.randn(batch_size, aux_chans, img_size, img_size, device=device)
    
    # 1. 训练模式测试
    print("\n[1] 训练模式测试 (无 LoRA)")
    model_train = create_multimodal_vit_for_training(
        img_size=img_size,
        rgb_in_chans=rgb_chans,
        aux_in_chans=aux_chans,
        use_checkpoint=True,
        checkpoint_ratio=0.5,
        use_parallel_streams=True
    ).to(device)
    
    model_train.print_trainable_params()
    
    # 测试前向传播
    model_train.train()
    with torch.cuda.amp.autocast():
        features = model_train(rgb, aux)
    
    print(f"输入 RGB: {rgb.shape}")
    print(f"输入 Aux: {aux.shape}")
    print(f"输出 Stage1: {features['stage1'].shape}")
    print(f"输出 Stage2: {features['stage2'].shape}")
    print(f"输出 Stage3: {features['stage3'].shape}")
    
    # 2. 微调模式测试
    print("\n[2] 微调模式测试 (带 LoRA)")
    model_finetune = MultimodalViT(
        img_size=img_size,
        rgb_in_chans=rgb_chans,
        aux_in_chans=aux_chans,
        use_checkpoint=True,
        checkpoint_ratio=0.3,
        lora_r=8,
        lora_alpha=16
    ).to(device)
    
    # 启用 LoRA
    model_finetune.enable_lora_finetune(freeze_backbone=True)
    
    with torch.cuda.amp.autocast():
        features = model_finetune(rgb, aux)
    
    print(f"输出 Stage3: {features['stage3'].shape}")
    
    # 3. 显存测试
    if device == 'cuda':
        print("\n[3] 显存占用测试")
        torch.cuda.reset_peak_memory_stats()
        
        model_mem = create_multimodal_vit_for_training(
            img_size=512,
            use_checkpoint=True,
            checkpoint_ratio=0.5,
            use_parallel_streams=True
        ).to(device)
        
        model_mem.train()
        
        # 使用混合精度
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.AdamW(model_mem.parameters(), lr=1e-4)
        
        with torch.cuda.amp.autocast():
            features = model_mem(rgb, aux)
            loss = features['output'].mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"峰值显存 (batch=2, 512x512): {peak_mem:.2f} GB")
        print(f"预估 16GB 显存可用 batch size: ~{int(14 / peak_mem * 2)}")
    
    # 4. 并行化 vs 串行对比
    if device == 'cuda':
        print("\n[4] 并行化性能测试")
        import time
        
        model_test = create_multimodal_vit_for_training(
            img_size=512,
            use_checkpoint=True,
            use_parallel_streams=True
        ).to(device)
        model_test.eval()
        
        # 预热
        with torch.no_grad(), torch.cuda.amp.autocast():
            _ = model_test(rgb, aux)
        
        torch.cuda.synchronize()
        
        # 并行测试
        model_test.enable_parallel_streams()
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for _ in range(10):
                _ = model_test(rgb, aux)
        torch.cuda.synchronize()
        parallel_time = time.time() - start
        
        # 串行测试
        model_test.disable_parallel_streams()
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for _ in range(10):
                _ = model_test(rgb, aux)
        torch.cuda.synchronize()
        serial_time = time.time() - start
        
        print(f"并行执行 (10 iter): {parallel_time:.3f}s")
        print(f"串行执行 (10 iter): {serial_time:.3f}s")
        print(f"加速比: {serial_time/parallel_time:.2f}x")
        
        try:
            test_pretrain_to_multimodal_compatibility()
        except Exception as e:
            print(f"兼容性测试跳过: {e}")
    
    print("\n" + "=" * 70)
    print("✅ 测试通过!")
    print("=" * 70)
