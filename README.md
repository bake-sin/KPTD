# KPTD
Knowledge-Prompted Trustworthy Disentangled Learning for Thyroid Ultrasound Segmentation with Limited Annotations
# Text-Guided Thyroid Ultrasound Segmentation

This repository provides a PyTorch implementation of a text-guided segmentation model for thyroid ultrasound images. The framework combines a CLIP-based text encoder with a medical image encoder and supports **semi-supervised learning** via a two-stream batch sampler.

> Note: This code is research-oriented and assumes familiarity with PyTorch and medical image segmentation workflows.

---

## 1. Features

- Text-guided segmentation with CLIP (`ViT-B/32`) text encoder.
- Semi-supervised training with **TwoStreamBatchSampler**:
  - A fixed subset of samples is treated as labeled.
  - The remaining samples are treated as unlabeled.
- Separate **training**, **validation**, and **testing** pipelines.
- Evaluation metrics on the test set:
  - Dice coefficient
  - IoU
  - 95th percentile Hausdorff distance (HD95)
  - Average surface distance (ASD)
- Automatic saving of predicted masks and an Excel summary for test results.

---

## 2. Project Structure

The main script is:

- `tus_main.py` – entry point for training, validation, and testing.

The following files are expected to be provided by the user:

- `hparam_tus.py`  
  Defines a `hparams` object, e.g.:

  ```python
  class HParams:
      train_or_test = "train"  # or "test"
      mode = "2d"

      source_train_dir = "path/to/train/images"
      label_train_dir  = "path/to/train/labels"

      source_val_dir = "path/to/val/images"
      label_val_dir  = "path/to/val/labels"

      source_test_dir = "path/to/test/images"
      label_test_dir  = "path/to/test/labels"

      output_dir      = "checkpoints"
      output_dir_test = "test_outputs"

      latest_checkpoint_file = "checkpoint_latest.pt"
      ckpt = ""  # optional: path to a specific checkpoint

      init_lr = 1e-4
      total_epochs = 200
      epochs_per_checkpoint = 50
      batch_size = 4

      # semi-supervised settings (can also be overridden by CLI)
      num_labeled = 100
      total_samples = 200
      labeled_batch_size = 2
      val_start_epoch = 1

      text_excel_path = "path/to/text_metadata.xlsx"

      in_class = 1
      out_class = 1
      scheduer_step_size = 50
      scheduer_gamma = 0.1

      debug = False

  hparams = HParams()
