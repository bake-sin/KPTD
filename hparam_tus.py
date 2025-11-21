# -*- coding: utf-8 -*-
"""
Hyper-parameters and path configuration for text-guided thyroid US segmentation.

This file is intentionally kept simple and generic so that it can be used
as a template in public repositories. Please adapt the paths and numbers
to your own dataset.

Typical structure:
  data/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
      masks/
  metadata/
    text_description.xlsx
"""

class hparams:
    # ------------------------------------------------------------------
    # 1. Running mode
    # ------------------------------------------------------------------
    # "train"  : run training loop
    # "test"   : run testing / inference on the test set
    train_or_test = "train"

    # ------------------------------------------------------------------
    # 2. Checkpoints & logging
    # ------------------------------------------------------------------
    # Directory for saving checkpoints and training logs
    output_dir = "./checkpoints"
    # File name (inside output_dir) for the latest checkpoint
    latest_checkpoint_file = "checkpoint_latest.pt"
    # If not None, this checkpoint will be loaded explicitly
    ckpt = None

    # Total number of epochs and how often to save checkpoints
    total_epochs = 300
    epochs_per_checkpoint = 50

    # Batch size used by the DataLoader
    batch_size = 4

    # ------------------------------------------------------------------
    # 3. Optimizer / scheduler
    # ------------------------------------------------------------------
    # Initial learning rate for Adam
    init_lr = 1e-4
    # StepLR: decay LR every `scheduer_step_size` epochs by `scheduer_gamma`
    scheduer_step_size = 20
    scheduer_gamma = 0.8

    # ------------------------------------------------------------------
    # 4. Model / data format settings
    # ------------------------------------------------------------------
    # Whether to enable data augmentation in your dataset class (if supported)
    aug = True

    # "2d" or "3d" (this codebase currently assumes 2D images)
    mode = "2d"

    # Number of input channels (ultrasound is usually 1) and output classes
    in_class = 1
    out_class = 1

    # Debug mode: if True, training loop may run only a few batches per epoch
    debug = False

    # ------------------------------------------------------------------
    # 5. Patch / crop configuration
    # ------------------------------------------------------------------
    # For 2D: (H, W)
    crop_or_pad_size = (512, 512)
    patch_size = (512, 512)

    # Overlap between neighbouring patches when using sliding-window
    # (for pure 2D whole-image training you can simply keep (0, 0))
    patch_overlap = (0, 0)

    # ------------------------------------------------------------------
    # 6. File patterns (if using NIfTI or other formats)
    # ------------------------------------------------------------------
    # Input file pattern used in dataset loader (e.g. "*.nii.gz", "*.png")
    fold_arch = ""

    # Suffix used when saving segmentation results
    save_arch = ""

    # ------------------------------------------------------------------
    # 7. Dataset paths
    # ------------------------------------------------------------------
    # Training set (images & masks)
    source_train_dir = "./data/train/images/"
    label_train_dir  = "./data/train/masks/"

    # Validation set (images & masks)
    # Used by `val()` during training; you can point this to a held-out set.
    source_val_dir = "./data/val/images/"
    label_val_dir  = "./data/val/masks/"

    # Test set (images & masks)
    # Used by `test()` for final evaluation and mask saving.
    source_test_dir = "./data/test/images/"
    label_test_dir  = "./data/test/masks/"

    # Directory where test-time predictions and metrics will be saved
    output_dir_test = "./outputs/test/"

    # ------------------------------------------------------------------
    # 8. Text metadata (Excel)
    # ------------------------------------------------------------------
    # Path to an Excel file containing text descriptions for each case.
    # You can keep it empty ("") if you do not use text guidance.
    # The exact reading logic is implemented in tus_main.py::data_excel().
    text_excel_path = "./metadata/text_description.xlsx"

    # ------------------------------------------------------------------
    # 9. Semi-supervised settings (TwoStreamBatchSampler)
    # ------------------------------------------------------------------
    # Total number of labeled samples in the training set.
    # This is used to split indices into labeled / unlabeled.
    num_labeled = ''

    # Total number of samples in the training set (labeled + unlabeled).
    total_samples = ''

    # Number of labeled samples in each training batch.
    # The remaining (batch_size - labeled_batch_size) are unlabeled.
    labeled_batch_size = ''

    # ------------------------------------------------------------------
    # 10. Validation strategy
    # ------------------------------------------------------------------
    # Start running validation at the end of this epoch (inclusive).
    # For example, set to 1 to validate after every epoch, or to a larger
    # value to delay validation until the model is reasonably trained.
    val_start_epoch = ''
