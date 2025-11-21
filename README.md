KPTD: Knowledge-Prompted Trustworthy Disentangled Learning for Thyroid Ultrasound Segmentation with Limited Annotations

This repository provides the official implementation of the paper:

Knowledge-Prompted Trustworthy Disentangled Learning for Thyroid Ultrasound Segmentation with Limited Annotations

KPTD introduces a text-guided, semi-supervised, and trustworthy segmentation framework tailored for thyroid ultrasound imaging, especially under limited pixel-level annotations.
It integrates three key modules described in the paper:

🚩 1. Knowledge-Prompted Adaptation & Localization (KAPL)

🚩 2. Foreground–Background Disentangled Learning (FBDL)

🚩 3. Foreground–Background Trustworthy Fusion (FBTF)

Together, these modules support high-quality segmentation with very few labeled images.

📁 Repository Structure
KPTD/
│
├── tus_main.py          # Main script: training / validation / testing
├── tus_model.py         # Training-time KPTD network
├── tus_model_test.py    # Inference-time KPTD network
├── hparam_tus.py        # Hyper-parameters & path configuration
├── Med_dataset.py       # User-custom dataset loader (not included)
│
├── simple_tokenizer.py
├── clip-vit-base-patch32/
├── clip_text_weight/
├── bpe_simple_vocab_16e6.txt.gz
│
└── README.md

🧩 Installation & Environment

The original experiments were conducted in a Python 3.9 environment with CUDA-enabled PyTorch.
Below is a minimal environment required to run this repository (extracted from the author’s full environment):
conda create -n kptd python=3.9
conda activate kptd

# Install PyTorch (choose CUDA version appropriately)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install torchio medpy SimpleITK scikit-image opencv-python \
            pandas pillow openpyxl tqdm clip transformers

Used framework highlights

PyTorch ≥ 1.10 / 2.x

TorchIO (image IO / preprocessing)

MedPy (HD95, ASD metrics)

CLIP (OpenAI implementation)

SimpleITK or NiBabel for medical IO (optional)

Your full environment includes many packages (Jupyter, NLP, scikit-learn, PL, etc.).
Only the subset above is needed to run this repository.

📊 Data Structure

You must organize your dataset as follows:
data/
  train/
    images/   *.png / *.jpg / *.nii.gz
    masks/    binary segmentation masks
  val/
    images/
    masks/
  test/
    images/
    masks/

metadata/
  text_description.xlsx   # optional text metadata

Each sample will be loaded as a TorchIO Subject:

"source" → image

"label" → binary mask (only for labeled samples)

"text" → a list of textual descriptions

"name" → case identifier

Your Med_dataset.py must define:
class MedData_train:
    self.training_set = List[tio.Subject]

class MedData_test:
    self.training_set = List[tio.Subject]

📄 Text Metadata (Optional)

The function data_excel() supports flexible free-form Excel/CSV:

First column → case ID

Remaining columns → any textual descriptions, e.g.

CaseID | Texture | Shape | Boundary | BI-RADS | Notes | ...
T001   | solid   | oval  | smooth   | 4a      | hypo
T002   | cystic  | round | regular  | 2       | anechoic


Returned as:

{
  "T001": ["solid", "oval", "smooth", "4a", "hypo", ...,],
  "T002": ["cystic", "round", "regular", "2", "anechoic", ...,]
}

🚀 Training

After editing hparam_tus.py to your dataset paths:

Basic training
python tus_main.py --mode train

Custom semi-supervised configuration
python tus_main.py \
    --num-labeled  \
    --total-samples  \
    --labeled-batch-size  \
    --batch  \
    --epochs  \
    --val-start-epoch  \
    --output_dir checkpoints/

Training features

Foreground branch: lesion segmentation

Background branch: non-lesion segmentation

Fused branch: final prediction (FBTF)

Unlabeled data:

Attention-map pseudo-supervision

Evidential uncertainty regularization

Automatic checkpointing (latest + best)

🧪 Testing / Inference
python tus_main.py --mode test \
    --ckpt checkpoints/checkpoint_latest.pt \
    --batch 4

Output:
outputs/test/
    pred_masks/
    test_results.xlsx   # Dice, HD95, IoU, ASD


Metrics computed:

Dice

95% Hausdorff distance (HD95)

Intersection-over-Union (IoU)

Average Surface Distance (ASD)
