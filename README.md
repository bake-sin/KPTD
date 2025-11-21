# 🩺 KPTD: Knowledge-Prompted Trustworthy Disentangled Learning  
### for Thyroid Ultrasound Segmentation with Limited Annotations

Official PyTorch implementation of:

**Knowledge-Prompted Trustworthy Disentangled Learning for Thyroid Ultrasound Segmentation with Limited Annotations**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9-blue?style=flat-square">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square">
  <img src="https://img.shields.io/badge/License-Academic-green?style=flat-square">
  <img src="https://img.shields.io/badge/Semi--Supervised-Yes-orange?style=flat-square">
</p>

KPTD presents a **text-guided, semi-supervised, disentangled and trustworthy** segmentation framework designed for thyroid ultrasound imaging under limited pixel-level annotations.

---

# 📌 **Highlights**

### 🔺 Knowledge-Prompted Adaptation & Localization (KPAL)  
Uses domain knowledge + CLIP text encoder to guide feature localization.

### 🔺 Foreground–Background Disentangled Learning (FBDL)  
Separates anatomical structures from noise & background.

### 🔺 Foreground–Background Trustworthy Fusion (FBTF)  
Uses uncertainty-aware fusion to improve segmentation reliability.

These modules jointly achieve **high-quality segmentation with very few labeled images**.

---

# 📚 **Table of Contents**
- [Repository Structure](#-repository-structure)
- [Environment & Installation](#-environment--installation)
- [Core Dependencies](#-core-dependencies)
- [Dataset Structure](#-dataset-structure)
- [Semi-Supervised Configuration](#-semi-supervised-configuration)
- [Training](#-training)
- [Testing & Inference](#-testing--inference)
- [Citation](#-citation)
- [Contact](#-contact)

---

# 📂 Repository Structure
KPTD/
│── tus_main.py # Main script (training / validation / testing)
│── tus_model.py # KPTD network (KPAL, FBDL, FBTF modules)
│── tus_model_test.py # Inference pipeline
│── hparam_tus.py # Hyper-parameters & paths
│── simple_tokenizer.py # Lightweight tokenizer for CLIP text prompts
│── clip-vit-base-patch32/ # CLIP image encoder weights
│── clip_text_weight/ # CLIP text encoder weights
│── bpe_simple_vocab_16e6.txt.gz # BPE vocabulary
│── README.md


---

# 🖥 Environment & Installation

Experiments were performed on:

- **Python 3.9**
- **CUDA 11.3 / 11.8**
- **PyTorch 2.1+ / 2.2+ / 2.3+ (all compatible)**  

We provide a minimal environment below that fully covers the KPTD pipeline.

---

## 1️⃣ Create Environment

```bash
conda create -n kptd python=3.9 -y
conda activate kptd

2️⃣ Install PyTorch (choose your CUDA version)

👉 https://download.pytorch.org/whl/cu118

Example (CUDA 11.8): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

🧩 Core Dependencies

These are the essential packages (cleaned from your full environment):
pip install numpy pandas pillow opencv-python SimpleITK scikit-image nibabel
pip install transformers open-clip-torch
pip install tqdm einops
pip install medpy    # optional: HD95 / ASD metrics
Your full Anaconda environment contains many other unrelated packages—
this list is the minimal clean version for using KPTD.

📁 Dataset Structure

Organize your data as:
data/
│── train/
│     ├── images/   # *.png / *.jpg
│     ├── masks/    # segmentation masks (only the labeled subset)
│     ├── text.xlsx # optional text for knowledge prompts
│
│── val/
│     ├── images/
│     ├── masks/
│
│── test/
      ├── images/
      ├── masks/

🔆 Semi-Supervised Configuration

KPTD supports flexible labeled/unlabeled splits:

Argument	Meaning
--num-labeled	number of labeled samples
--total-samples	total training samples
--labeled-batch-size	number of labeled samples per batch
--val-start-epoch	start validation from epoch X

🚀 Training
python tus_main.py \
    --train-root ./data/train \
    --val-root ./data/val \
    --num-labeled  \
    --total-samples  \
    --labeled-batch-size  \
    --epochs 

🧪 Testing & Inference
python tus_model_test.py \
    --test-root ./data/test \
    --model-path ./checkpoints/best_model.pth \
    --save-mask True
