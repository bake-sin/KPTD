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

# 📌 Highlights

### 🔺 Knowledge-Prompted Adaptation & Localization (KPAL)
### 🔺 Foreground–Background Disentangled Learning (FBDL)
### 🔺 Foreground–Background Trustworthy Fusion (FBTF)

These modules jointly achieve **high-quality segmentation with very few labeled images**.

---

# 📚 Table of Contents
- [Repository Structure](#-repository-structure)
- [Environment & Installation](#-environment--installation)
- [Core Dependencies](#-core-dependencies)
- [Dataset Structure](#-dataset-structure)
- [Semi-Supervised Configuration](#-semi-supervised-configuration)
- [Training](#-training)
- [Testing & Inference](#-testing--inference)


---


# 🖥 Environment & Installation

Experiments were conducted with:

Python 3.9

CUDA 11.3 / 11.8

PyTorch 2.1+ / 2.2+ / 2.3+ (any recent 2.x version should work)

Below is a minimal clean environment sufficient to run this repository.
---

## Environment & Installation

Experiments were conducted with:

- Python 3.9
- CUDA 11.3 / 11.8
- PyTorch 2.1+ / 2.2+ / 2.3+ (any recent 2.x version should work)

Below is a minimal clean environment sufficient to run this repository.

---

### 1️⃣ Create Environment

```bash
conda create -n kptd python=3.9 -y
conda activate kptd



# Example (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


# 🧩 Core Dependencies

Install the core packages:

pip install numpy pandas pillow opencv-python SimpleITK scikit-image nibabel
pip install transformers open-clip-torch
pip install tqdm einops
pip install medpy         # optional: HD95 / ASD metrics
pip install torchio       # for I/O & preprocessing (TorchIO)

---

# 🚀 Training

python tus_main.py \
    --train-root ./data/train \
    --val-root ./data/val \
    --num-labeled 200 \
    --total-samples 1200 \
    --labeled-batch-size 2 \
    --epochs 200 \
    --use-text True \
    --text-path ./data/train/text.xlsx

---

# 🧪 Testing & Inference
python tus_model_test.py \
    --test-root ./data/test \
    --model-path ./checkpoints/best_model.pth \
    --save-mask True
