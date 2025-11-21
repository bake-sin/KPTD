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
- [Citation](#-citation)
- [Contact](#-contact)

---


```markdown
# 📂 Repository Structure

```txt
KPTD/
│── tus_main.py              # Main script (training / validation / testing)
│── tus_model.py             # KPTD network (KPAL, FBDL, FBTF modules)
│── tus_model_test.py        # Inference pipeline
│── hparam_tus.py            # Hyper-parameters & paths
│── simple_tokenizer.py      # Lightweight tokenizer for CLIP text prompts
│── clip-vit-base-patch32/   # CLIP image encoder weights
│── clip_text_weight/        # CLIP text encoder weights
│── bpe_simple_vocab_16e6.txt.gz   # BPE vocabulary
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


🧪 Testing & Inference
python tus_model_test.py \
    --test-root ./data/test \
    --model-path ./checkpoints/best_model.pth \
    --save-mask True
