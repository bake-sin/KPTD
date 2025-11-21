# coding=utf-8
"""
Text-guided thyroid ultrasound segmentation (inference model).

This file defines:
  - A ResNet34-like encoder for ultrasound images
  - A small GMM-based module (used in training model, not in test wrapper)
  - Multi-head self-attention and cross-attention blocks
  - A "Konwledge_Abstractor" module for image–text fusion
  - Activation and segmentation heads
  - A DFS (Dempster–Shafer) fusion module
  - Two entry points:
      * class build_model(nn.Module): full model with text input (same as training)
      * class build_model_test(nn.Module): inference wrapper that only needs image input

During inference (used in `tus_main.py::test()`), `build_model_test` is
instantiated from a trained `build_model`, and its `forward(image)` returns:

    total_fusion, middle_feature,
    fore_output, back_output,
    alpha, pre_end, u,
    att_map, E_f, E_b

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
# 1. ResNet-style encoder
# -------------------------------------------------------------------------


class CommonBlock(nn.Module):
    """
    Basic residual block: conv -> BN -> ReLU -> conv -> BN + identity.
    """

    def __init__(self, in_channel: int, out_channel: int, stride: int):
        super(CommonBlock, self).__init__()
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
    Residual block that can change channel number and spatial resolution.
    """

    def __init__(self, in_channel: int, out_channel: int, stride: Tuple[int, int]):
        super(SpecialBlock, self).__init__()
        # projection branch
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
    Simplified ResNet-34-like encoder for single-channel ultrasound images.

    For 512×512 input, the output spatial size is 32×32 (layer4 is not used).
    """

    def __init__(self):
        super(ResNet34, self).__init__()
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
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
        )

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prepare(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # [B, 256, H', W']


# -------------------------------------------------------------------------
# 2. GMM-based batch module (used in training model; not used by test wrapper)
# -------------------------------------------------------------------------


class GMM_Batch(nn.Module):
    """
    A small GMM-like module over feature maps.

    Args:
        b:   batch size used during initialization (kept for compatibility)
        c:   number of channels
        num: number of components
    """

    def __init__(self, b: int, c: int, num: int):
        super(GMM_Batch, self).__init__()

        protos = torch.Tensor(b, c, num)
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
        One EM iteration on batch feature maps.

        x:  [R, C, H, W]
        mu: [R, C, K]
        """
        R, C, H, W = x.size()
        x = x.view(R, C, H * W)  # r * c * n
        for _ in range(3):
            z = torch.einsum("rcn,rck->rnk", (x, mu))  # r * n * k
            z = F.softmax(20.0 * z, dim=2)            # r * n * k
            z = self._l1norm(z, dim=1)                # r * n * k

            mu = torch.einsum("rcn,rnk->rck", (x, z))  # r * c * k
            mu = self._l2norm(mu, dim=1)               # r * c * k
        return mu

    def _prop(self, feat: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignments of each spatial location to prototypes.

        feat: [B, C, H, W]
        mu:   [B, C, K]
        return: [B, N, K] where N = H * W
        """
        B, C, H, W = feat.size()
        x = feat.view(B, C, -1)  # B * C * N
        z = torch.einsum("bcn,bck->bnk", (x, mu))  # B * N * K
        z = F.softmax(z, dim=2)  # B * N * K
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mus = self.mu
        protos = self._em_iter(x, self.mus)
        ref_z = self._prop(x, protos)
        self.mus = self.mus * 0.5 + protos * 0.5
        return ref_z


# -------------------------------------------------------------------------
# 3. Attention modules
# -------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention (Q=K=V=x).

    Input:  x  [B, L, D_in]
    Output:    [B, L, D_in]
    """

    def __init__(self, in_dim: int, k_dim: int, v_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
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


class Konwledge_Abstractor(nn.Module):
    """
    Image–text knowledge abstraction module (cross-attention + MLP).

    x1: [B, L1, D1]  (image tokens)
    x2: [B, L2, D2]  (image+text concatenated tokens)
    """

    def __init__(self, in_dim1: int, in_dim2: int, k_dim: int, v_dim: int, num_heads: int):
        super(Konwledge_Abstractor, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(v_dim * num_heads, in_dim1)

        self.pro1 = nn.Linear(in_dim1, 128)
        self.pro2 = nn.Linear(128, in_dim1)
        self.swi_glu = nn.GLU()  # not used explicitly, kept for compatibility

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # debug prints were here originally; removed for cleanliness
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


class CrossAttention(nn.Module):
    """
    Cross-attention from x1 (queries) to x2 (keys/values).

    x1: [B, L1, D1]
    x2: [B, L2, D2]
    """

    def __init__(self, in_dim1: int, in_dim2: int, k_dim: int, v_dim: int, num_heads: int):
        super(CrossAttention, self).__init__()
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


# -------------------------------------------------------------------------
# 4. Activation & segmentation heads
# -------------------------------------------------------------------------


class activate_head(nn.Module):
    """
    Simple 1×1 conv + sigmoid to obtain an activation map
    (used as attention map for foreground/background).
    """

    def __init__(self, in_channels: int):
        super(activate_head, self).__init__()
        self.proj_in = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        # NOTE: BN layers here are kept only for compatibility with training code.
        self.BN = nn.BatchNorm2d(in_channels, in_channels)
        self.RE = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        self.proj_out = nn.Conv2d(
            in_channels, 1, kernel_size=1, stride=1, padding=0
        )
        self.BN_out = nn.BatchNorm2d(1, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In original code, BN and upsampling are not used here.
        x = self.proj_out(x)
        x = self.sig(x)
        return x


class seg_head(nn.Module):
    """
    Small decoder head for binary segmentation.

    Input:  feature maps of shape [B, C_in, H, W]
    Output: logits of shape [B, out_channels, H_out, W_out]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(seg_head, self).__init__()
        self.up1 = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1
        )
        self.BN1 = nn.BatchNorm2d(in_channels // 2, in_channels // 2)
        self.up2 = nn.Conv2d(
            in_channels // 2, in_channels // 4, kernel_size=3, stride=1, padding=1
        )
        self.BN2 = nn.BatchNorm2d(in_channels // 4, in_channels // 4)
        self.up3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, kernel_size=3, stride=1, padding=1
        )
        self.BN3 = nn.BatchNorm2d(in_channels // 8, in_channels // 8)
        self.up4 = nn.Conv2d(
            in_channels // 8, in_channels // 16, kernel_size=3, stride=1, padding=1
        )
        self.BN4 = nn.BatchNorm2d(in_channels // 16, in_channels // 16)
        self.up5 = nn.Conv2d(
            in_channels // 16, in_channels // 32, kernel_size=3, stride=1, padding=1
        )
        self.BN5 = nn.BatchNorm2d(in_channels // 32, in_channels // 32)
        self.Upsample = nn.Upsample(scale_factor=2)
        self.RE = nn.ReLU()
        self.end = nn.Conv2d(
            in_channels // 16, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)
        x = self.BN1(x)
        x = self.RE(x)
        x = self.Upsample(x)

        x = self.up2(x)
        x = self.BN2(x)
        x = self.RE(x)
        x = self.Upsample(x)

        x = self.up3(x)
        x = self.BN3(x)
        x = self.RE(x)
        x = self.Upsample(x)

        x = self.up4(x)
        x = self.BN4(x)
        x = self.RE(x)
        x = self.Upsample(x)

        x = self.end(x)
        return x


class seg_activate(nn.Module):
    """
    Extra conv + ReLU block (not used in current inference wrapper).
    Kept for completeness / compatibility.
    """

    def __init__(self, in_channels: int):
        super(seg_activate, self).__init__()
        self.up4 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.BN4 = nn.BatchNorm2d(in_channels, in_channels)
        self.RE = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up4(x)
        x = self.BN4(x)
        x = self.RE(x)

        x = self.up4(x)
        x = self.BN4(x)
        x = self.RE(x)
        return x


# -------------------------------------------------------------------------
# 5. DFS fusion
# -------------------------------------------------------------------------


class DFS(nn.Module):
    """
    Dempster–Shafer-based fusion of two evidential outputs (fore/back).

    Inputs:
        fore, back: evidence maps with shape [B, C, H, W]

    Outputs:
        alpha_new: Dirichlet parameters after fusion
        e_a:       fused evidence
        uncertainty: fused uncertainty
    """

    def __init__(self, class_nums: int):
        super(DFS, self).__init__()
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

        # combination
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
# 6. Full model (same structure as training-time model)
# -------------------------------------------------------------------------


class build_model(nn.Module):
    """
    Full text-guided segmentation model (training-time architecture).

    Inputs:
        image: [B, 1, H, W]
        text:  [B, D_text]

    Outputs (training-time):
        total_fusion, middle_feature,
        fore_output, back_output,
        alpha, pre_end, u,
        g1, g2, att_map
    """

    def __init__(self):
        super(build_model, self).__init__()

        self.encoder = ResNet34()

        # Learnable query centers, shape [1024, 64]
        self.learn_center = nn.Parameter(torch.rand(1024, 64), requires_grad=True)

        # Text-guided attention modules
        self.MultiheadCA = CrossAttention(
            in_dim1=64, in_dim2=1, k_dim=256, v_dim=256, num_heads=4
        )
        self.attn = MultiHeadAttention(
            in_dim=64, k_dim=256, v_dim=256, num_heads=4
        )
        self.Konwledge_Abstractor = Konwledge_Abstractor(
            in_dim1=256, in_dim2=320, k_dim=256, v_dim=256, num_heads=16
        )

        # Activation + segmentation
        self.act_head = activate_head(256)
        self.seg_head = seg_head(in_channels=256, out_channels=2)

        # DFS fusion
        self.DFS = DFS(class_nums=2)

        # GMM over foreground/background features (b=4 was used originally)
        self.GMM = GMM_Batch(b=4, c=256, num=5)

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        x = self.encoder(image)  # [B, 256, 32, 32] if input is 512×512

        # text branch
        text_fea = text.type(torch.FloatTensor).cuda()
        ceb = self.learn_center

        # Text feature: [B, D_text] -> [B, D_text, 1]
        text_fea = text_fea.unsqueeze(2)
        b, d, _ = text_fea.shape

        # Learnable query centers: [1024, 64] -> [B, 1024, 64]
        learn_center = ceb.repeat(b, 1, 1)

        # Self-attention on learnable centers
        learn_center = self.attn(learn_center)
        # Cross-attention with text features (as keys/values)
        learn_text = self.MultiheadCA(learn_center, text_fea)

        # Flatten image features: [B, C, H, W] -> [B, H*W, C]
        X = rearrange(x, "b c h w -> b (h w) c")

        # Concatenate image tokens and text-guided tokens along feature dim
        X_s = torch.cat([X, learn_text], dim=2)

        # Knowledge abstraction: fuse X with X_s
        cam = self.Konwledge_Abstractor(X, X_s)  # [B, H*W, 256]
        cam = cam.permute(0, 2, 1)              # [B, 256, H*W]
        C = torch.reshape(cam, [b, 256, 32, 32])

        # Activation map (attention)
        att_map = self.act_head(C)
        C_f = att_map
        C_b = 1.0 - att_map

        # Foreground / background features
        E_f = x * C_f
        E_b = x * C_b

        # Segmentation logits
        fore_output = self.seg_head(E_f)
        back_output = self.seg_head(E_b)

        E_total = E_f + E_b
        total_fusion = E_total
        middle_feature = x

        # GMM on fore/back features
        g1 = self.GMM(E_f)
        g2 = self.GMM(E_b)

        # DFS fusion on evidences
        fore_evidence = torch.tanh(fore_output)
        back_evidence = torch.tanh(back_output)
        alpha, b_fused, u = self.DFS(fore_evidence, back_evidence)

        pre_end = b_fused
        return total_fusion, middle_feature, fore_output, back_output, alpha, pre_end, u, g1, g2, att_map


# -------------------------------------------------------------------------
# 7. Inference wrapper (image-only input)
# -------------------------------------------------------------------------


class build_model_test(nn.Module):
    """
    Inference wrapper for text-guided segmentation model.

    This wrapper reuses the encoder + learned centers from a trained `build_model`,
    and allows running inference with image-only input (no text embeddings).

    Inputs:
        image: [B, 1, H, W]

    Outputs (used in tus_main.py::test()):
        total_fusion, middle_feature,
        fore_output, back_output,
        alpha, pre_end, u,
        att_map, E_f, E_b
    """

    def __init__(self, build_model: build_model):
        super(build_model_test, self).__init__()

        # Reuse submodules from a trained build_model
        self.encoder = build_model.encoder
        self.learn_center = build_model.learn_center

        self.attn = build_model.attn
        self.Konwledge_Abstractor = build_model.Konwledge_Abstractor
        self.act_head = build_model.act_head
        self.seg_head = build_model.seg_head
        self.DFS = build_model.DFS

    def forward(self, image: torch.Tensor):
        x = self.encoder(image)  # [B, 256, 32, 32]
        ceb = self.learn_center
        b, d, _, _ = x.shape

        # Use learnable centers only (no explicit text input in test)
        learn_center = ceb.repeat(b, 1, 1)
        learn_center = self.attn(learn_center)

        # Flatten image features
        X = rearrange(x, "b c h w -> b (h w) c")
        # Concatenate flattened image tokens with learnable tokens
        X_s = torch.cat([X, learn_center], dim=2)

        cam = self.Konwledge_Abstractor(X, X_s)
        cam = cam.permute(0, 2, 1)
        C = torch.reshape(cam, [b, 256, 32, 32])

        # Activation map
        att_map = self.act_head(C)
        C_f = att_map
        C_b = 1.0 - att_map

        # Foreground / background features
        E_f = x * C_f
        E_b = x * C_b

        # Segmentation logits
        fore_output = self.seg_head(E_f)
        back_output = self.seg_head(E_b)

        E_total = E_f + E_b
        total_fusion = E_total
        middle_feature = x

        # DFS fusion
        fore_evidence = torch.tanh(fore_output)
        back_evidence = torch.tanh(back_output)
        alpha, b_fused, u = self.DFS(fore_evidence, back_evidence)

        pre_end = b_fused
        return total_fusion, middle_feature, fore_output, back_output, alpha, pre_end, u, att_map, E_f, E_b
