#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for text-guided thyroid ultrasound segmentation.

This script provides:
  - A training loop with a semi-supervised TwoStreamBatchSampler
  - Validation on a separate validation set
  - Final testing with metric computation and mask saving

External dependencies (must be provided in your project):
  - hparam_tus.py      : contains `hparams` (hp) configuration
  - Med_dataset.py     : defines MedData_train / MedData_test
  - tus_model.py       : defines `build_model` (for training)
  - tus_model_test.py  : defines `build_model` and `build_model_test` (for inference)

Author: (add your name here)
"""

import os
import argparse
import random
import itertools
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import StepLR

import clip
import torchio as tio
from torchio.transforms import ZNormalization
from torchio import DATA, AFFINE
from medpy import metric as medpy_metric  # for HD95 / ASD

from hparam_tus import hparams as hp


# -------------------------------------------------------------------------
# 0. Global configuration
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths are taken from hparams so that they can be configured externally
SOURCE_TRAIN_DIR = hp.source_train_dir
LABEL_TRAIN_DIR = hp.label_train_dir

SOURCE_TEST_DIR = hp.source_test_dir
LABEL_TEST_DIR = hp.label_test_dir

# Validation set paths
SOURCE_VAL_DIR = hp.source_val_dir
LABEL_VAL_DIR = hp.label_val_dir

OUTPUT_DIR_TEST = hp.output_dir_test


# -------------------------------------------------------------------------
# 1. Dataloader utilities
# -------------------------------------------------------------------------

def collate_subjects(batch: List[tio.Subject]) -> Dict[str, Any]:
    """
    Custom collate_fn for TorchIO datasets.

    Each element in `batch` is a `tio.Subject`. We convert it into a dict:
      - For image fields (tio.Image): stack 'data' and 'affine' across batch.
      - For non-image fields (e.g., 'name', 'text'): keep as list.
    """
    out: Dict[str, Any] = {}
    first = batch[0]
    for k in first.keys():
        v0 = first[k]
        if isinstance(v0, tio.Image):
            datas = [b[k][DATA] for b in batch]   # [B, C, H, W, (D)]
            affs = [b[k][AFFINE] for b in batch]
            out[k] = {
                "data": torch.stack(datas, dim=0),
                "affine": torch.stack(
                    [torch.as_tensor(a) for a in affs],
                    dim=0,
                ),
            }
        else:
            out[k] = [b[k] for b in batch]
    return out


def parse_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add command-line arguments related to training, validation and testing.
    """

    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=hp.output_dir,
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--latest-checkpoint-file",
        type=str,
        default=hp.latest_checkpoint_file,
        help="Filename for the latest checkpoint (relative to output_dir).",
    )
    parser.add_argument(
        "-k",
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help=(
            "Path to checkpoint for resuming or testing. "
            "If empty, falls back to `latest-checkpoint-file` in `output_dir`."
        ),
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=hp.init_lr,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training (not used by default).",
    )
    parser.add_argument(
        "--text-excel-path",
        type=str,
        default=getattr(hp, "text_excel_path", ""),
        help="Path to Excel file with text descriptions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=getattr(hp, "seed", 1234),
        help="Global random seed.",
    )

    # training setup
    training = parser.add_argument_group("training setup")
    training.add_argument(
        "--epochs",
        type=int,
        default=hp.total_epochs,
        help="Number of total epochs to run.",
    )
    training.add_argument(
        "--epochs-per-checkpoint",
        type=int,
        default=hp.epochs_per_checkpoint,
        help="Number of epochs per checkpoint save.",
    )
    training.add_argument(
        "--batch",
        type=int,
        default=hp.batch_size,
        help="Total batch size (labeled + unlabeled).",
    )
    training.add_argument(
        "--num-labeled",
        type=int,
        default=getattr(hp, "num_labeled", 0),
        help="Number of labeled training samples (must be > 0).",
    )
    training.add_argument(
        "--total-samples",
        type=int,
        default=getattr(hp, "total_samples", 0),
        help="Total number of training samples (labeled + unlabeled, must be > num-labeled).",
    )
    training.add_argument(
        "--labeled-batch-size",
        type=int,
        default=getattr(hp, "labeled_batch_size", 0),
        help="Number of labeled samples per batch (must be > 0 and < batch).",
    )
    training.add_argument(
        "--val-start-epoch",
        type=int,
        default=getattr(hp, "val_start_epoch", 1),
        help="Start running validation from this epoch (inclusive).",
    )

    # misc flags
    training.add_argument(
        "--amp-run",
        action="store_true",
        help="Enable AMP (not used in the current script).",
    )
    training.add_argument(
        "--cudnn-enabled",
        action="store_true",
        default=True,
        help="Enable cudnn.",
    )
    training.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        default=True,
        help="Run cudnn benchmark.",
    )
    training.add_argument(
        "--disable-uniform-initialize-bn-weight",
        action="store_true",
        help="Disable uniform initialization of batchnorm layer weight.",
    )

    return parser


class TwoStreamBatchSampler(Sampler):
    """
    Two-stream batch sampler for semi-supervised learning.

    - primary_indices: indices for labeled data
    - secondary_indices: indices for unlabeled data

    Each batch consists of:
       [primary_batch_size labeled samples] +
       [secondary_batch_size unlabeled samples]
    """

    def __init__(
        self,
        primary_indices,
        secondary_indices,
        batch_size: int,
        secondary_batch_size: int,
    ):
        self.primary_indices = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(indices):
    return np.random.permutation(indices)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks:
    grouper('ABCDEFG', 3) --> ABC DEF
    """
    args = [iter(iterable)] * n
    return zip(*args)


# -------------------------------------------------------------------------
# 2. Metric & loss utilities
# -------------------------------------------------------------------------

def binary_iou(s: np.ndarray, g: np.ndarray) -> float:
    """
    Binary IoU between two masks (numpy arrays of 0/1).
    """
    assert s.shape == g.shape, "Prediction and GT must have the same shape."
    intersection = np.multiply(s, g)
    union = (s + g) > 0
    iou = intersection.sum() / (union.sum() + 1e-10)
    return float(iou)


def dice_coeff(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Dice coefficient between two 0/1 masks (batched).
    Shape: (B, H, W) or (B, 1, H, W)
    """
    smooth = 1.0
    pred_flat = pred.reshape(pred.size(0), -1).float()
    target_flat = target.reshape(target.size(0), -1).float()
    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def cos_simi(embedded_fg: torch.Tensor, embedded_bg: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity matrix between two embedding sets.
    Inputs: [N, C], [M, C]
    Output: [N, M]
    """
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return torch.clamp(sim, min=0.0005, max=0.9995)


class SimMinLoss(nn.Module):
    """
    Similarity-minimizing loss between two embedding sets.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, embedded_bg: torch.Tensor, embedded_fg: torch.Tensor):
        sim = cos_simi(embedded_bg, embedded_fg)
        loss = -torch.log(1.0 - sim)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


class SimMaxLoss(nn.Module):
    """
    Similarity-maximizing loss within a single embedding set.

    The closer two embeddings are, the smaller the loss.
    """

    def __init__(self, metric: str = "cos", alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embeddings: torch.Tensor):
        if self.metric != "cos":
            raise NotImplementedError("Only cosine metric is supported.")

        sim = cos_simi(embeddings, embeddings)      # [N, N]
        loss = -torch.log(sim)
        loss[loss < 0] = 0

        # rank-based weights (closer neighbors are weighted more heavily)
        _, indices = sim.sort(descending=True, dim=1)
        _, rank = indices.sort(dim=1)
        rank = rank - 1
        rank_weights = torch.exp(-rank.float() * self.alpha)
        loss = loss * rank_weights

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            return loss


def KL(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    KL divergence between Dirichlet(α) and uniform Dirichlet(1,...,1).
    Used in evidential / Dirichlet-based uncertainty losses.
    """
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, num_classes), device=alpha.device)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def dce_eviloss(
    p: torch.Tensor,
    alpha: torch.Tensor,
    _e,
    num_classes: int,
    global_step: int,
    annealing_step: int,
) -> torch.Tensor:
    """
    Dirichlet-based cross entropy + KL regularization.

    Args:
        p: one-hot or soft labels, shape [N, C]
        alpha: Dirichlet parameters, shape [N, C, H, W] (will be reshaped)
        num_classes: number of classes
    """
    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2).contiguous()            # [N, HW, C]
    alpha = alpha.view(-1, alpha.size(2))                 # [N*HW, C]

    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1.0

    label = p.view(-1, num_classes)
    # digamma-based cross entropy
    L_ace = torch.sum(
        label * (torch.digamma(S) - torch.digamma(alpha)),
        dim=1,
        keepdim=True,
    )

    annealing_coef = min(1.0, global_step / float(annealing_step))
    alp = E * (1.0 - label) + 1.0
    L_KL = annealing_coef * KL(alp, num_classes)

    return L_ace + 0.2 * L_KL


def dice_loss(score: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Standard soft Dice loss for binary segmentation.
    """
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2.0 * intersect + smooth) / (z_sum + y_sum + smooth)
    return 1.0 - loss


class ContrastiveLoss(nn.Module):
    """
    Simple contrastive loss:
      L = mean(||f1 - f2||^2)
    """

    def __init__(self):
        super().__init__()

    def forward(self, output1: torch.Tensor, output2: torch.Tensor):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive


# -------------------------------------------------------------------------
# 3. Excel text metadata
# -------------------------------------------------------------------------

def data_excel(path: str) -> Dict[str, Any]:
    """
    Read text descriptions from an Excel file in a generic way.

    Typical usage:
      - The first column is treated as a unique key (e.g., case ID or filename).
      - All remaining columns in that row are treated as text-related fields
        (e.g., multiple attributes, phrases, or sentences).

    This function does NOT assume a specific sheet name or a fixed number of columns.
    Users can adapt it according to their own Excel format if needed.

    Returns:
        A dictionary such that:
            data[key] = list_of_text_fields
    """
    df = pd.read_excel(path)  # default: first sheet
    if df.shape[1] < 2:
        raise ValueError(
            f"Excel file {path} must have at least two columns "
            "(first as key, others as text/attributes)."
        )

    key_col = df.columns[0]
    data: Dict[str, Any] = {}

    for _, row in df.iterrows():
        key = str(row[key_col])
        # Collect all remaining columns, drop NaNs, convert to str
        values = [
            str(v) for v in row.iloc[1:].tolist()
            if pd.notna(v)
        ]
        data[key] = values

    return data


# -------------------------------------------------------------------------
# 4. Training
# -------------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser(
        description="PyTorch Medical Segmentation Training"
    )
    parser = parse_training_args(parser)
    args = parser.parse_args()

    # ------------------- basic reproducibility / CUDNN -------------------
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from Med_dataset import MedData_train
    from tus_model import build_model

    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode != "2d":
        raise NotImplementedError("Currently only 2D mode is supported in this script.")

    # ------------------------------------------------------------------
    # 4.1 Build model, optimizer, scheduler
    # ------------------------------------------------------------------
    model = build_model()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = StepLR(
        optimizer,
        step_size=hp.scheduer_step_size,
        gamma=hp.scheduer_gamma,
    )

    # ------------------------------------------------------------------
    # 4.2 (Optional) Load checkpoint
    # ------------------------------------------------------------------
    elapsed_epochs = 0

    # If args.ckpt is defined and exists, use it;
    # otherwise try `latest_checkpoint_file` under output_dir.
    checkpoint_path = None
    if args.ckpt and os.path.isfile(args.ckpt):
        checkpoint_path = args.ckpt
    else:
        candidate = os.path.join(args.output_dir, args.latest_checkpoint_file)
        if os.path.isfile(candidate):
            checkpoint_path = candidate

    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        state_dict = ckpt["model"]
        new_state_dict = {}
        for key, value in state_dict.items():
            # Handle DataParallel "module." prefix if present
            new_key = key.replace("module.", "") if key.startswith("module.") else key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)

        optimizer.load_state_dict(ckpt["optim"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {elapsed_epochs}")
    else:
        print("No checkpoint found. Training from scratch.")

    # ------------------------------------------------------------------
    # 4.3 Loss functions & regularizers
    # ------------------------------------------------------------------
    con_loss = ContrastiveLoss()
    smin_loss = SimMinLoss()
    smax_loss = SimMaxLoss()

    # ------------------------------------------------------------------
    # 4.4 Semi-supervised sampler
    # ------------------------------------------------------------------
    labelnum = args.num_labeled           # number of labeled samples
    max_samples = args.total_samples      # total number of samples
    batch_size = args.batch               # total batch size
    labeled_bs = args.labeled_batch_size  # #labeled samples per batch

    # basic sanity checks
    if labelnum <= 0:
        raise ValueError("`--num-labeled` must be > 0.")
    if max_samples <= labelnum:
        raise ValueError("`--total-samples` must be > `--num-labeled`.")
    if batch_size <= 0:
        raise ValueError("`--batch` must be > 0.")
    if labeled_bs <= 0 or labeled_bs >= batch_size:
        raise ValueError(
            "`--labeled-batch-size` must be > 0 and < `--batch` "
            "(semi-supervised setting requires both labeled and unlabeled data per batch)."
        )

    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, max_samples))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        batch_size,
        batch_size - labeled_bs,
    )

    print(
        f"[DATA] Using {labelnum} labeled and {max_samples - labelnum} unlabeled samples "
        f"({labelnum / max_samples:.2%} labeled). "
        f"Batch size = {batch_size} (labeled per batch = {labeled_bs})."
    )

    def worker_init_fn(worker_id: int):
        # Fix worker seed for reproducibility in each worker
        seed = args.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)

    # load Excel text metadata
    text_excel_path = args.text_excel_path
    if not text_excel_path:
        raise ValueError(
            "text_excel_path is empty. Please set `hp.text_excel_path` "
            "in hparam_tus.py or pass `--text-excel-path`."
        )
    print(f"[TEXT] Loading text metadata from: {text_excel_path}")
    read_data = data_excel(text_excel_path)

    train_dataset = MedData_train(
        SOURCE_TRAIN_DIR,
        LABEL_TRAIN_DIR,
        read_data,
    )
    train_loader = DataLoader(
        train_dataset.training_set,
        batch_sampler=batch_sampler,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_subjects,
    )

    # text encoder (CLIP)
    clip_model, _ = clip.load("ViT-B/32", device)
    clip_model.eval()

    model.train()
    best_dice = 0.0

    # Number of epochs to run in total: args.epochs
    for epoch in range(elapsed_epochs + 1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        for i, batch in enumerate(train_loader):
            if hp.debug and i >= 1:
                # in debug mode, only run one batch per epoch
                break

            print(f"  Batch {i + 1}/{len(train_loader)}")
            optimizer.zero_grad()

            # ------------------------------------------------------------------
            # 4.5 Prepare inputs and labels
            # ------------------------------------------------------------------
            if hp.in_class == 1 and hp.out_class == 1:
                x = batch["source"]["data"].squeeze(4)  # (B, C, H, W)
                y = batch["label"]["data"].squeeze(4)   # (B, 1, H, W)
                # name = batch["name"]  # can be used for debugging

                # binary mask: non-zero -> 1
                y[y != 0] = 1

                # text inputs: list of str
                text_inputs = batch["text"]
                text_inputs = clip.tokenize(text_inputs).to(device)
                text_features = clip_model.encode_text(text_inputs)

                y = y.float().to(device)
                y_back = torch.zeros_like(y)
                y_back[(y == 0)] = 1

                x = x.float().to(device)

                # foreground is '1', background is '0'
                y_fore = torch.cat((y_back, y), dim=1).float().to(device)
                y_back_cat = torch.cat((y, y_back), dim=1).float().to(device)
            else:
                raise NotImplementedError("Only binary (1 in / 1 out) setting is implemented.")

            # ------------------------------------------------------------------
            # 4.6 Forward pass
            # ------------------------------------------------------------------
            (
                total_fusion,
                middle_feature,
                fore_output,
                back_output,
                alpha,
                pre_end,
                _u,
                gf,
                gb,
                att_map,
            ) = model(x, text_features)

            fore_output = torch.softmax(fore_output, dim=1)
            back_output = torch.softmax(back_output, dim=1)
            pre_end = torch.softmax(pre_end, dim=1)

            # discrete predictions for metrics
            total_pred = pre_end.argmax(dim=1)
            fore_pred = fore_output.argmax(dim=1)
            back_pred = back_output.argmax(dim=1)

            # upsample attention map to match segmentation resolution
            att_map_up = F.interpolate(
                att_map,
                scale_factor=16,
                mode="bilinear",
                align_corners=False,
            )
            att_map_up = (att_map_up >= 0.5).float()

            # ------------------------------------------------------------------
            # 4.7 Loss computation
            # ------------------------------------------------------------------
            # Supervised Dice losses on labeled batch (first `labeled_bs` samples)
            loss_fore = dice_loss(
                fore_output[:labeled_bs, 1, :, :],
                y_fore[:labeled_bs, 1, :, :],
            )
            loss_back = dice_loss(
                back_output[:labeled_bs, 1, :, :],
                y_back_cat[:labeled_bs, 1, :, :],
            )
            loss_dice_fusion = dice_loss(
                pre_end[:labeled_bs, 1, :, :],
                y_fore[:labeled_bs, 1, :, :],
            )

            # Attention map supervision for unlabeled data
            loss_map = dice_loss(
                pre_end[labeled_bs:, 1, :, :].unsqueeze(1),
                att_map_up[labeled_bs:],
            )

            # Dirichlet-based evidential loss
            dst_loss = dce_eviloss(
                y_fore,
                alpha,
                pre_end,
                num_classes=2,
                global_step=100,
                annealing_step=50,
            )
            dst_loss = dst_loss.mean()

            # feature-level consistency between two branches
            consis_loss = con_loss(total_fusion, middle_feature) / 50.0

            # foreground & background embedding regularization
            b, n, k = gf.shape
            gf_reshaped = gf.reshape(b * k, n)
            gb_reshaped = gb.reshape(b * k, n)
            loss_sim = (
                smax_loss(gf_reshaped)
                + smax_loss(gb_reshaped)
                + 0.5 * smin_loss(gf_reshaped, gb_reshaped)
            )

            total_loss = (
                0.5 * loss_back
                + 0.5 * loss_fore
                + dst_loss
                + consis_loss
                + loss_sim
                + loss_dice_fusion
                + 0.1 * loss_map
            )

            # ------------------------------------------------------------------
            # 4.8 Backpropagation
            # ------------------------------------------------------------------
            total_loss.backward()
            optimizer.step()

            # ------------------------------------------------------------------
            # 4.9 Logging (Dice on current batch)
            # ------------------------------------------------------------------
            with torch.no_grad():
                dice_total = dice_coeff(
                    total_pred.cpu(),
                    y_fore[:, 1:, :, :].cpu(),
                )
                dice_fore = dice_coeff(
                    fore_pred.cpu(),
                    y_fore[:, 1:, :, :].cpu(),
                )
                dice_back = dice_coeff(
                    back_pred.cpu(),
                    y_back_cat[:, 1:, :, :].cpu(),
                )

            print(
                f"    fusion-loss: {loss_dice_fusion.item():.4f}, "
                f"fore-loss: {loss_fore.item():.4f}, "
                f"back-loss: {loss_back.item():.4f}, "
                f"map-loss: {loss_map.item():.4f}"
            )
            print(
                f"    fusion-dice: {dice_total:.4f}, "
                f"fore-dice: {dice_fore:.4f}, "
                f"back-dice: {dice_back:.4f}"
            )

        # ------------------------------------------------------------------
        # 4.10 Epoch-wise scheduler step & checkpointing
        # ------------------------------------------------------------------
        scheduler.step()

        # Save "latest" checkpoint every epoch
        latest_ckpt_path = os.path.join(args.output_dir, args.latest_checkpoint_file)
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            latest_ckpt_path,
        )

        # (Optional) Save epoch-specific checkpoints
        if epoch % args.epochs_per_checkpoint == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

        # ------------------------------------------------------------------
        # 4.11 Validation & best checkpoint
        # ------------------------------------------------------------------
        if epoch >= args.val_start_epoch:
            print("  Running validation ...")
            val_dice = val(args)
            if val_dice > best_dice:
                best_dice = val_dice
                best_ckpt_path = os.path.join(
                    args.output_dir,
                    f"checkpoint_{epoch:04d}_{best_dice:.4f}.pt",
                )
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    best_ckpt_path,
                )
                print(f"  New best Dice: {best_dice:.4f} (saved: {best_ckpt_path})")


# -------------------------------------------------------------------------
# 5. Testing
# -------------------------------------------------------------------------

def build_model_for_inference(args) -> nn.Module:
    """
    Helper for test() / val(): build inference model and load checkpoint.
    Uses tus_model_test.build_model + build_model_test.
    """
    from tus_model_test import build_model as build_model_train
    from tus_model_test import build_model_test

    base_model = build_model_train()

    # choose which checkpoint to load
    checkpoint_path = None
    if args.ckpt and os.path.isfile(args.ckpt):
        checkpoint_path = args.ckpt
    else:
        candidate = os.path.join(args.output_dir, "checkpoint_best.pt")
        if os.path.isfile(candidate):
            checkpoint_path = candidate
        else:
            # fallback to latest checkpoint file
            latest = os.path.join(args.output_dir, args.latest_checkpoint_file)
            if os.path.isfile(latest):
                checkpoint_path = latest

    if checkpoint_path is None:
        raise FileNotFoundError(
            "No checkpoint found for inference. "
            "Please set --ckpt or ensure checkpoint_best.pt / latest exists."
        )

    print(f"Loading inference checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    state_dict = ckpt["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value

    base_model.load_state_dict(new_state_dict, strict=False)
    model = build_model_test(base_model)
    model.to(device)
    model.eval()
    return model


def test():
    parser = argparse.ArgumentParser(
        description="PyTorch Medical Segmentation Testing"
    )
    parser = parse_training_args(parser)
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from Med_dataset import MedData_test
    from openpyxl import Workbook

    os.makedirs(OUTPUT_DIR_TEST, exist_ok=True)

    model = build_model_for_inference(args)

    test_dataset = MedData_test(SOURCE_TEST_DIR, LABEL_TEST_DIR)
    test_loader = DataLoader(
        test_dataset.training_set,
        batch_size=args.batch,
        shuffle=False,
        pin_memory=True,
        drop_last=False,   # keep all test samples
        collate_fn=collate_subjects,
    )

    znorm = ZNormalization()
    total_dice = 0.0
    total_hd95 = 0.0
    total_iou = 0.0
    total_asd = 0.0

    # prepare Excel logger
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Name", "Dice", "HD95", "IoU", "ASD"])

    # directory for saving predicted masks
    pred_dir = os.path.join(OUTPUT_DIR_TEST, "pred_masks")
    os.makedirs(pred_dir, exist_ok=True)

    for i, batch in enumerate(test_loader):
        x = batch["source"]["data"].squeeze(-1)   # (B, C, H, W)
        x = znorm(x)
        y = batch["label"]["data"]
        names = batch["name"]

        # binary mask
        y[y != 0] = 1

        x = x.float().to(device)
        if hp.mode == "2d":
            # y: (B, 1, H, W, 1) -> (B, 1, H, W) -> (B, H, W)
            y = y.squeeze(-1)
            if y.ndim == 4 and y.size(1) == 1:
                y = y[:, 0, ...]

        with torch.no_grad():
            (
                _total_fusion,
                _middle_feature,
                _fore_output,
                _back_output,
                _alpha,
                pre_end,
                _u,
                _att_map,
                _Ef,
                _Eb,
            ) = model(x)

            pre_end = torch.softmax(pre_end, dim=1)
            total_pred = pre_end.argmax(dim=1).cpu()  # (B, H, W)

        # metrics for each batch (averaged over batch)
        dice = float(dice_coeff(total_pred, y.cpu()).item())
        print(names[0], dice)  # print first name in this batch for quick check

        # convert to numpy for medpy (shapes: (B, H, W))
        total_pred_np = np.array(total_pred)
        y_np = np.array(y.cpu())

        # handle corner cases when there is no foreground
        if np.sum(total_pred_np == 1) == 0 or np.sum(y_np == 1) == 0:
            hd95 = 0.0
            asd = 0.0
        else:
            hd95 = float(
                medpy_metric.binary.hd95(total_pred_np, y_np)
            )
            asd = float(
                medpy_metric.binary.asd(total_pred_np, y_np)
            )
        iou = float(binary_iou(total_pred_np, y_np))

        total_dice += dice
        total_hd95 += hd95
        total_iou += iou
        total_asd += asd

        # log per-case (only the first sample in batch for simplicity)
        sheet.append([names[0], dice, hd95, iou, asd])

        # save predicted mask of the first sample in this batch as PNG
        pred_mask = total_pred[0].numpy().astype(np.uint8) * 255
        im = Image.fromarray(pred_mask)
        if im.mode == "F":
            im = im.convert("RGB")
        save_name = os.path.splitext(str(names[0]))[0] + ".png"
        im.save(os.path.join(pred_dir, save_name))

    num_batches = len(test_loader)
    print("Number of batches:", num_batches)
    print("Mean Dice:", total_dice / num_batches)
    print("Mean HD95:", total_hd95 / num_batches)
    print("Mean IoU:", total_iou / num_batches)
    print("Mean ASD:", total_asd / num_batches)

    # save Excel summary
    excel_path = os.path.join(OUTPUT_DIR_TEST, "test_results.xlsx")
    workbook.save(excel_path)
    print(f"Test results saved to: {excel_path}")


# -------------------------------------------------------------------------
# 6. Validation
# -------------------------------------------------------------------------

def val(args=None) -> float:
    """
    Validation on a separate validation set.

    The validation set paths are specified by:
      - SOURCE_VAL_DIR = hp.source_val_dir
      - LABEL_VAL_DIR  = hp.label_val_dir

    Returns:
        Average Dice over all batches.
    """
    if args is None:
        parser = argparse.ArgumentParser(
            description="PyTorch Medical Segmentation Validation"
        )
        parser = parse_training_args(parser)
        args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from Med_dataset import MedData_test

    model = build_model_for_inference(args)

    val_dataset = MedData_test(SOURCE_VAL_DIR, LABEL_VAL_DIR)
    val_loader = DataLoader(
        val_dataset.training_set,
        batch_size=args.batch,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_subjects,
    )

    znorm = ZNormalization()
    total_dice = 0.0

    for i, batch in enumerate(val_loader):
        x = batch["source"]["data"].squeeze(-1)
        x = znorm(x)
        y = batch["label"]["data"]
        names = batch["name"]

        y[y != 0] = 1

        x = x.float().to(device)
        if hp.mode == "2d":
            # y: (B, 1, H, W, 1) -> (B, 1, H, W) -> (B, H, W)
            y = y.squeeze(-1)
            if y.ndim == 4 and y.size(1) == 1:
                y = y[:, 0, ...]

        with torch.no_grad():
            (
                _total_fusion,
                _middle_feature,
                _fore_output,
                _back_output,
                _alpha,
                pre_end,
                _u,
                _att_map,
                _Ef,
                _Eb,
            ) = model(x)

            pre_end = torch.softmax(pre_end, dim=1)
            total_pred = pre_end.argmax(dim=1).cpu()  # (B, H, W)

        dice = float(dice_coeff(total_pred, y.cpu()).item())
        total_dice += dice

        print(f"[VAL] {names[0]} Dice: {dice:.4f}")

    mean_dice = total_dice / len(val_loader)
    print("Validation mean Dice:", mean_dice)
    return mean_dice


# -------------------------------------------------------------------------
# 7. Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    if hp.train_or_test == "train":
        train()
    elif hp.train_or_test == "test":
        test()
    else:
        raise ValueError(f"Unknown mode hp.train_or_test = {hp.train_or_test}")
