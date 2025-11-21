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

