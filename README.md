**KPTD: Knowledge-Prompted Trustworthy Disentangled Learning for Thyroid Ultrasound Segmentation with Limited Annotations

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
**

KPTD/
│── tus_main.py # Main script for training / validation / testing
│── tus_model.py # KPTD network (encoder, KPAL, FBDL, FBTF)
│── tus_model_test.py # Inference pipeline
│── hparam_tus.py # Hyper-parameters & path configuration
│── simple_tokenizer.py # Lightweight tokenizer for CLIP text prompts
│── clip-vit-base-patch32/ # CLIP image encoder weights
│── clip_text_weight/ # CLIP text encoder weights
│── bpe_simple_vocab_16e6.txt.gz # BPE vocabulary
│── README.md


---

# 🖥 Installation & Environment

Experiments were originally conducted in a **Python 3.9** environment with CUDA-enabled PyTorch.

To reproduce our results:

```bash
conda create -n kptd python=3.9 -y
conda activate kptd


🔧 Install PyTorch (choose CUDA version based on your system)

Please follow the official install instruction:

👉 https://download.pytorch.org/whl/cu118

Example (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

📦 Core Dependencies

KPTD relies on the following major libraries:
numpy
pandas
opencv-python
pillow
SimpleITK
scikit-image
einops
tqdm
transformers
open-clip-torch
medpy          # optional (HD95, ASD evaluation)
nibabel        # optional for medical IO

Install them via:
pip install -r requirements.txt

📁 Dataset Structure

You must organize your dataset as follows:
data/
│── train/
│     ├── images/         # *.png / *.jpg ultrasound images
│     ├── masks/          # binary segmentation masks (labeled samples only)
│     ├── text.xlsx       # optional text metadata for prompts
│
│── val/
│     ├── images/
│     ├── masks/
│
│── test/
      ├── images/
      ├── masks/
Each sample corresponds to one image and optional text:

source → ultrasound image

label → segmentation mask (only for labeled subset)

text → several attribute phrases (used to construct CLIP text prompts)

🧪 Semi-Supervised Setting

KPTD uses a labeled + unlabeled training split:

Argument	Meaning
--num-labeled	number of labeled training samples
--total-samples	labeled + unlabeled samples
--labeled-batch-size	labeled samples per batch
--val-start-epoch	start validation from epoch X
