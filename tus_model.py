# coding=utf-8
"""
Model definition for text-guided thyroid ultrasound segmentation.

This file defines:
  - A lightweight ResNet-style encoder for ultrasound images
  - Multi-head self- and cross-attention blocks
  - A knowledge aggregation module for fusing image and text features
  - Foreground / background segmentation heads
  - A simple GMM-based prototype module
  - A Dempster–Shafer-based fusion (DFS) module
  - TextGuidedSegmentationModel + build_model() factory

The forward() output is designed to be compatible with tus_main.py:
  total_fusion, middle_feature, fore_output, back_output,
  alpha, pre_end, u, gf, gb, att_map

Author: (add your name here)
"""

from __future__ import print_function, division, absolute_import

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -------------------------------------------------------------------------
# 1. Basic ResNet-style encoder
# -------------------------------------------------------------------------


class CommonBlock(nn.Module):
    """
    Basic residual block with same in/out channels and stride = 1.
    """

    def __init__(self, in_channel: int, out_channel: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x = x + identity
        return F.relu(x, inplace=True)


class SpecialBlock(nn.Module):
    """
    Residual block that changes the number of channels and/or spatial resolution.
    """

    def __init__(self, in_channel: int, out_channel: int, stride: Tuple[int, int]):
        super().__init__()
        # projection for the residual branch
        self.change_channel = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=stride[0],
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_channel),
        )
        # main branch
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride[0],
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=stride[1],
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x = x + identity
        return F.relu(x, inplace=True)


class ResNet34(nn.Module):
    """
    A simplified ResNet-34-like encoder for single-channel ultrasound images.

    Note:
      The final feature map resolution is determined by the input size. For
      512×512 inputs, the output spatial size is 32×32 when layer4 is skipped,
      which is assumed by the downstream modules.
    """

    def __init__(self):
        super().__init__()
        self.prepare = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, (2, 1)),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, (2, 1)),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
        )
        # layer4 from ResNet-34 is omitted here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B, 256, H', W']


# -------------------------------------------------------------------------
# 2. GMM-based prototype layer (batch-agnostic)
# -------------------------------------------------------------------------


class GMM_Batch(nn.Module):
    """
    A small GMM-like layer applied on feature maps.

    The module learns a set of prototypes (mu) in feature space and performs
    EM-like updates per batch. The output is a soft assignment per location.

    Args:
        c: number of channels in the feature map.
        num_components: number of prototypes per sample.
    """

    def __init__(self, c: int, num_components: int):
        super().__init__()
        protos = torch.Tensor(1, c, num_components)  # shared across batch
        protos.normal_(0, math.sqrt(2.0 / 30.0))
        protos = self._l2norm(protos, dim=1)
        self.register_buffer("mu", protos)

    @staticmethod
    def _l1norm(inp: torch.Tensor, dim: int) -> torch.Tensor:
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    @staticmethod
    def _l2norm(inp: torch.Tensor, dim: int) -> torch.Tensor:
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    @torch.no_grad()
    def _em_iter(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        One EM procedure (with fixed number of steps) to update prototypes.

        x:  [B, C, H, W]
        mu: [B, C, K]
        """
        R, C, H, W = x.size()
        x = x.view(R, C, H * W)  # R * C * N

        for _ in range(3):
            # E-step: soft assignment
            z = torch.einsum("rcn,rck->rnk", x, mu)  # R * N * K
            z = F.softmax(20.0 * z, dim=2)  # R * N * K
            z = self._l1norm(z, dim=1)  # normalize over N

            # M-step: prototype update
            mu = torch.einsum("rcn,rnk->rck", x, z)  # R * C * K
            mu = self._l2norm(mu, dim=1)  # L2 norm across channels

        return mu

    def _prop(self, feat: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignment of each spatial location to each prototype.

        feat: [B, C, H, W]
        mu:   [B, C, K]
        return: [B, N, K] where N = H * W
        """
        B, C, H, W = feat.size()
        x = feat.view(B, C, -1)  # B * C * N
        z = torch.einsum("bcn,bck->bnk", x, mu)  # B * N * K
        z = F.softmax(z, dim=2)  # B * N * K
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.size()
        # Expand the prototype buffer to batch size
        mu = self.mu.expand(B, -1, -1)  # [B, C, K]
        protos = self._em_iter(x, mu)
        ref_z = self._prop(x, protos)
        # EMA update of shared prototypes
        with torch.no_grad():
            mean_mu = protos.mean(dim=0, keepdim=True)
            self.mu.mul_(0.5).add_(0.5 * mean_mu)
        return ref_z


# -------------------------------------------------------------------------
# 3. Attention & knowledge modules
# -------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention (Q=K=V=x).

    Input:  x  [B, L, D_in]
    Output:    [B, L, D_in]
    """

    def __init__(self, in_dim: int, k_dim: int, v_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(in_dim, k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(in_dim, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        q = (
            self.proj_q(x)
            .view(batch_size, seq_len, self.num_heads, self.k_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.proj_k(x)
            .view(batch_size, seq_len, self.num_heads, self.k_dim)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.proj_v(x)
            .view(batch_size, seq_len, self.num_heads, self.v_dim)
            .permute(0, 2, 1, 3)
        )

        attn = torch.matmul(q, k) / (self.k_dim ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, -1)
        )
        output = self.proj_o(output)
        return output


class CrossAttention(nn.Module):
    """
    Cross-attention from x1 (queries) to x2 (keys/values):

    x1: [B, L1, D1], x2: [B, L2, D2]

    Queried features (x1) are updated using x2.
    """

    def __init__(self, in_dim1: int, in_dim2: int, k_dim: int, v_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len1, _ = x1.size()
        seq_len2 = x2.size(1)

        q1 = (
            self.proj_q1(x1)
            .view(batch_size, seq_len1, self.num_heads, self.k_dim)
            .permute(0, 2, 1, 3)
        )
        k2 = (
            self.proj_k2(x2)
            .view(batch_size, seq_len2, self.num_heads, self.k_dim)
            .permute(0, 2, 3, 1)
        )
        v2 = (
            self.proj_v2(x2)
            .view(batch_size, seq_len2, self.num_heads, self.v_dim)
            .permute(0, 2, 1, 3)
        )

        attn = torch.matmul(q1, k2) / (self.k_dim ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2)
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len1, -1)
        )
        output = self.proj_o(output)
        return output


class KnowledgeAggregator(nn.Module):
    """
    Aggregates image tokens (x1) using text-guided knowledge tokens (x2)
    via cross-attention, followed by a small MLP.

    x1: [B, L1, D1], x2: [B, L2, D2]
    """

    def __init__(self, in_dim1: int, in_dim2: int, k_dim: int, v_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

        self.pro1 = nn.Linear(in_dim1, 128)
        self.pro2 = nn.Linear(128, in_dim1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len1, _ = x1.size()
        seq_len2 = x2.size(1)

        q1 = (
            self.proj_q1(x1)
            .view(batch_size, seq_len1, self.num_heads, self.k_dim)
            .permute(0, 2, 1, 3)
        )
        k2 = (
            self.proj_k2(x2)
            .view(batch_size, seq_len2, self.num_heads, self.k_dim)
            .permute(0, 2, 3, 1)
        )
        v2 = (
            self.proj_v2(x2)
            .view(batch_size, seq_len2, self.num_heads, self.v_dim)
            .permute(0, 2, 1, 3)
        )

        attn = torch.matmul(q1, k2) / (self.k_dim ** 0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v2)
        output = (
            output.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len1, -1)
        )

        output = self.proj_o(output)
        output = self.pro2(self.pro1(output))
        return output


# -------------------------------------------------------------------------
# 4. Activation & segmentation heads
# -------------------------------------------------------------------------


class ActivateHead(nn.Module):
    """
    Simple 1×1 conv + sigmoid to produce an activation map (attention map)
    from feature maps.

    Input:  [B, C, H, W]
    Output: [B, 1, H, W]
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.proj_out = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_out(x)
        x = self.sig(x)
        return x


class SegHead(nn.Module):
    """
    A small upsampling decoder for binary segmentation.

    This head expects feature maps at a relatively low resolution
    and upsamples them several times to output logits at a higher resolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        c = in_channels
        self.conv1 = nn.Conv2d(c, c // 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c // 2)
        self.conv2 = nn.Conv2d(c // 2, c // 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c // 4)
        self.conv3 = nn.Conv2d(c // 4, c // 8, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(c // 8)
        self.conv4 = nn.Conv2d(c // 8, c // 16, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(c // 16)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(c // 16, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.upsample(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.upsample(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.upsample(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.upsample(x)

        x = self.out_conv(x)
        return x  # logits [B, out_channels, H_out, W_out]


# -------------------------------------------------------------------------
# 5. Dempster–Shafer fusion (DFS)
# -------------------------------------------------------------------------


class DFS(nn.Module):
    """
    Dempster–Shafer-based fusion of two evidential sources (foreground and background).

    Inputs:
        fore, back: evidential maps (after a non-negative activation such as tanh)
                    with shape [B, C, H, W], where C = number of classes.

    Outputs:
        alpha_new: Dirichlet parameters
        e_a:       fused evidence
        uncertainty: fused uncertainty
    """

    def __init__(self, class_nums: int):
        super().__init__()
        self.class_nums = class_nums

    def forward(
        self,
        fore: torch.Tensor,
        back: torch.Tensor,
    ):
        e1 = fore
        alpha1 = e1 + 1.0
        S1 = torch.sum(alpha1, dim=1, keepdim=True)
        b1 = e1 / S1
        u1 = self.class_nums / S1

        e2 = back
        alpha2 = e2 + 1.0
        S2 = torch.sum(alpha2, dim=1, keepdim=True)
        b2 = e2 / S2
        u2 = self.class_nums / S2

        # combination rules
        b1b2 = b1 * b2
        b1u2 = b1 * u2
        b2u1 = b2 * u1
        u1u2 = u1 * u2

        C = (1.0 - (b1[:, 0] * b2[:, 1] + b2[:, 0] * b1[:, 1])).unsqueeze(1)

        belief = (b1b2 + b1u2 + b2u1) / C
        uncertainty = u1u2 / C
        S_a = 2.0 / uncertainty
        e_a = belief * S_a
        alpha_new = e_a + 1.0

        return alpha_new, e_a, uncertainty


# -------------------------------------------------------------------------
# 6. Text-guided segmentation model
# -------------------------------------------------------------------------


class TextGuidedSegmentationModel(nn.Module):
    """
    Text-guided foreground/background segmentation model.

    Inputs:
        image: [B, 1, H, W]
        text:  [B, D_text] (e.g., CLIP text embeddings)

    Outputs (in order, to match tus_main.py):
        total_fusion:   combined foreground + background feature maps
        middle_feature: encoder feature maps (for consistency loss)
        fore_output:    foreground segmentation logits
        back_output:    background segmentation logits
        alpha:          Dirichlet parameters from DFS
        pre_end:        fused belief from DFS (used as segmentation logits)
        u:              fused uncertainty (DFS)
        gf:             GMM-based soft assignments for foreground features
        gb:             GMM-based soft assignments for background features
        att_map:        attention map (activation map)
    """

    def __init__(self):
        super().__init__()

        self.encoder = ResNet34()

        # Learnable centers used as query tokens for text-guided attention
        # shape: [num_tokens, dim]
        self.learn_center = nn.Parameter(torch.rand(64, 1024), requires_grad=True)

        # Attention & knowledge aggregation
        self.self_attn = MultiHeadAttention(
            in_dim=1024, k_dim=256, v_dim=256, num_heads=12
        )
        self.cross_attn = CrossAttention(
            in_dim1=1024, in_dim2=512, k_dim=256, v_dim=256, num_heads=12
        )
        self.knowledge_aggregator = KnowledgeAggregator(
            in_dim1=1024, in_dim2=1024, k_dim=256, v_dim=256, num_heads=12
        )

        # Activation & segmentation heads
        self.act_head = ActivateHead(in_channels=256)
        self.seg_head = SegHead(in_channels=256, out_channels=2)

        # Dempster–Shafer fusion
        self.DFS = DFS(class_nums=2)

        # GMM for foreground & background features (channel dim = 256)
        self.GMM = GMM_Batch(c=256, num_components=5)

    def forward(
        self,
        image: torch.Tensor,
        text: torch.Tensor,
    ):
        # ------------------------------------------------------------------
        # 1. Image encoding
        # ------------------------------------------------------------------
        x = self.encoder(image)  # [B, 256, H', W']

        # ------------------------------------------------------------------
        # 2. Text features and learnable query centers
        # ------------------------------------------------------------------
        # Ensure text is float and on the same device as image
        text_fea = text.to(device=image.device, dtype=torch.float32)  # [B, D_text]
        text_fea = text_fea.unsqueeze(1)  # [B, 1, D_text]

        B, _, _ = text_fea.shape
        ceb = self.learn_center  # [64, 1024]
        learn_center = ceb.unsqueeze(0).repeat(B, 1, 1)  # [B, 64, 1024]

        # Self-attention among learnable centers
        learn_center = self.self_attn(learn_center)
        # Cross-attention: queries = learn_center, keys/values = text_fea
        learn_text = self.cross_attn(learn_center, text_fea)

        # ------------------------------------------------------------------
        # 3. Image–text knowledge aggregation
        # ------------------------------------------------------------------
        # Flatten spatial dims: [B, 256, H', W'] -> [B, 256, H'*W']
        # Here H'*W' is assumed to be 1024 for 512×512 inputs
        X = rearrange(x, "b c h w -> b c (h w)")  # [B, 256, N]

        # Knowledge aggregator expects: [B, L, D]
        #   X is treated as 256 tokens with dimension N (e.g., 1024)
        cam = self.knowledge_aggregator(X, learn_text)  # [B, 256, N]
        # Reshape back to 2D feature map. For 512×512 inputs, H'*W' = 32×32.
        C = cam.view(B, 256, 32, 32)

        # ------------------------------------------------------------------
        # 4. Activation map & fore/back features
        # ------------------------------------------------------------------
        att_map = self.act_head(C)  # [B, 1, 32, 32]
        C_f = att_map
        C_b = 1.0 - att_map

        E_f = x * C_f  # foreground features
        E_b = x * C_b  # background features

        fore_output = self.seg_head(E_f)  # [B, 2, H_out, W_out]
        back_output = self.seg_head(E_b)  # [B, 2, H_out, W_out]

        E_total = E_f + E_b  # can be used for ablation

        total_fusion = E_total
        middle_feature = x

        # ------------------------------------------------------------------
        # 5. GMM assignments on fore/back features
        # ------------------------------------------------------------------
        g1 = self.GMM(E_f)  # [B, N, K]
        g2 = self.GMM(E_b)  # [B, N, K]

        # ------------------------------------------------------------------
        # 6. Dempster–Shafer fusion on evidences
        # ------------------------------------------------------------------
        fore_evidence = torch.tanh(fore_output)
        back_evidence = torch.tanh(back_output)
        alpha, b, u = self.DFS(fore_evidence, back_evidence)

        pre_end = b  # fused belief treated as segmentation output

        return total_fusion, middle_feature, fore_output, back_output, alpha, pre_end, u, g1, g2, att_map


# -------------------------------------------------------------------------
# 7. Factory function (for tus_main.py)
# -------------------------------------------------------------------------


def build_model() -> nn.Module:
    """
    Factory function to build the text-guided segmentation model.

    tus_main.py expects to import this symbol:
        from tus_model import build_model
        model = build_model()
    """
    return TextGuidedSegmentationModel()
