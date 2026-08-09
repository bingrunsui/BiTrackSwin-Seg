"""The fixed v1 single-stream segmentation model.

This module deliberately exposes the same top-level state-dict names as the
training run: ``backbone``, ``pixel_decoder`` and ``seg_head``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
from contextlib import redirect_stdout
from io import StringIO

import torch
from torch import nn
from torch.nn import functional as F

from .multimodal import MultimodalViT


class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        def gn(channels: int) -> nn.GroupNorm:
            for groups in (16, 8, 4, 2, 1):
                if channels % groups == 0:
                    return nn.GroupNorm(groups, channels)
            return nn.GroupNorm(1, channels)
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), gn(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False), gn(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self._last_psi = None

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.psi(self.relu(self.W_g(g) + self.W_x(x)))
        if not self.training:
            self._last_psi = psi.detach()
        return x * psi


class UNetDecoder(nn.Module):
    """U-Net decoder used by v1 (stage0/1/2/3 skips, 256 channels)."""
    def __init__(self, in_channels: List[int], dec_dim: int = 256):
        super().__init__()
        self.num_stages = len(in_channels)
        self.output_dim = dec_dim
        self.aux_proj = None
        self.aux_fpa = None
        self.stage3_stage0_gate = None
        self.bottleneck = self._double_conv(in_channels[-1], dec_dim)
        self.attn_gates = nn.ModuleList(
            AttentionGate(dec_dim, ch, max(ch // 4, 16))
            for ch in in_channels[:-1]
        )
        self.up_blocks = nn.ModuleList(
            self._double_conv(dec_dim + in_channels[i], dec_dim)
            for i in range(self.num_stages - 2, -1, -1)
        )
        # The reference checkpoint was trained with gates and skip-dropout off.
        self.skip_drop_idxs: List[int] = []
        self.skip_drop_p = 0.0
        self.skip_drop_joint = True

    @staticmethod
    def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
        groups = min(32, out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, features: Dict[str, torch.Tensor], spatial_sizes: List[int]) -> torch.Tensor:
        keys = sorted(key for key in features if key.startswith("stage"))
        maps = []
        for index, key in enumerate(keys):
            tensor = features[key]
            batch, _, channels = tensor.shape
            side = spatial_sizes[index]
            maps.append(tensor.transpose(1, 2).reshape(batch, channels, side, side))
        x = self.bottleneck(maps[-1])
        for index, up_block in enumerate(self.up_blocks):
            skip_index = self.num_stages - 2 - index
            skip = maps[skip_index]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            # Attention gates exist in the checkpoint but were disabled for v1.
            x = up_block(torch.cat((x, skip), dim=1))
        return x


class SegmentationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, mid_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, mid_dim), mid_dim), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1), nn.Conv2d(mid_dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int] | None = None) -> torch.Tensor:
        logits = self.head(x)
        return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False) if target_size else logits


class BiTrackSwinSegV1(nn.Module):
    """Five-band, single-stream BiTrackSwin segmentation network."""
    def __init__(self, image_size: int = 256, in_channels: int = 5, num_classes: int = 2):
        super().__init__()
        # The original experiment emitted Chinese diagnostic banners during
        # construction.  Suppress those legacy messages so Windows GBK shells
        # can import the public package safely.
        with redirect_stdout(StringIO()):
            self.backbone = MultimodalViT(
            img_size=image_size, rgb_in_chans=in_channels, aux_in_chans=0,
            embed_dim=96, depths=[4, 6, 2], num_heads=[6, 6, 12], window_size=[8, 8, 32],
            use_multimodal=False, pretrain_mode=False, use_checkpoint=False,
            checkpoint_ratio=0.0, use_parallel_streams=False, memory_efficient_mode=True,
            patch_size=4, patch_stride=2, drop_path_rate=0.3, lora_r=0,
            architecture_mode="three_stage", cross_window_enabled=True,
            cross_window_type="hybrid", cross_window_top_k=4, cross_window_gate_init=1e-4,
            cross_window_num_queries=3, cross_window_use_stats=("max", "mean", "min"),
            cross_window_step3_mode="splat_plus", cross_window_splat_null=True,
            cross_window_splatp_score_mode="fullC", cross_window_splatp_fuse_mode="perchannel",
            cross_window_splatp_num_heads=8, cross_window_ffn_respost=True,
            cross_window_drop_norm_cross=True, cross_window_mul_ln=True,
            cross_window_cross_qk_norm=True, bilevel_lora_r=0, bilevel_lora_alpha=1,
            swin_version="v2", shift_window=False, stage0_to_decoder=True, stem_mode="conv",
            cross_stage_mode="none", rgb_use_layerscale=True, rgb_layerscale_init=1e-2,
            rgb_use_swiglu=True, rgb_use_qk_norm=True, rgb_use_rope=True,
            use_cosine_drop_path=True, aux_use_rmsnorm=False,
            )
        self.pixel_decoder = UNetDecoder(self.backbone.decoder_dims, dec_dim=256)
        self.decoder_type = "unet"
        self.spatial_sizes = self.backbone.decoder_spatial_sizes
        self.deep_supervision = False
        self.fpa_gate = 0.0
        self.seg_head = SegmentationHead(256, num_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        features = self.backbone(image)
        for key, value in features.items():
            if key.startswith("stage"):
                features[key] = value.clamp(-10.0, 10.0)
        decoded = self.pixel_decoder(features, self.spatial_sizes).clamp(-10.0, 10.0)
        return self.seg_head(decoded, target_size=(height, width))


def build_model(**kwargs) -> BiTrackSwinSegV1:
    return BiTrackSwinSegV1(**kwargs)
