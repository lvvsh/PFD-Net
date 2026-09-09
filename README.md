# PFD-Net: Physics-Guided Degradation and Frequency-Aware Restoration for Label-Efficient Semi-Supervised Remote Sensing Change Detection

<p align="center">
  <a href="https://github.com/lvvsh/PFD-Net"><img src="https://img.shields.io/badge/Task-Semi--Supervised%20Change%20Detection-blue.svg"></a>
  <a href="https://github.com/lvvsh/PFD-Net"><img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg"></a>
  <a href="https://github.com/lvvsh/PFD-Net"><img src="https://img.shields.io/badge/Python-3.11-green.svg"></a>
  <a href="https://github.com/lvvsh/PFD-Net"><img src="https://img.shields.io/badge/License-MIT-purple.svg"></a>
  <a href="https://github.com/lvvsh/PFD-Net"><img src="https://img.shields.io/badge/Status-Under%20Minor%20Revision-red.svg"></a>
</p>

Official PyTorch implementation of **PFD-Net**, a label-efficient semi-supervised remote sensing change detection framework featuring an offline **Physics-Frequency Data Engine** and an online **Siamese Consistency Architecture**.

---

## 📢 News
- **[2026]**: Code and pretrained weights for LEVIR-CD, WHU-CD, and GZ-CD are released!
- **[2026]**: Paper submitted to *Neurocomputing* (Under Minor Revision).

---

## 🌟 Key Features

- **Physics-Guided Degradation (MRD)**: A training-free, forward-only Ornstein-Uhlenbeck (OU) Stochastic Differential Equation (SDE) that mathematically models continuous environmental shifts (e.g., seasonal transitions, haze, shadows) to synthesize physically meaningful hard negatives without non-rigid spatial distortions.
- **DFT-Based Frequency Restoration**: Counteracts the spectral bias (high-frequency boundary attenuation) inherent in diffusion/stochastic degradation via a 2D Discrete Fourier Transform power-law transformation, preserving sharp topological boundaries of buildings while keeping spatial phase invariant.
- **Null-Change Prior & Consistency Regularization**: Compels the Siamese student network to map physical domain shifts onto absolute zero in the semantic change space, extracting robust domain-invariant representations.
- **Zero Inference Overhead**: The offline data engine takes only **3 minutes 39 seconds** on a single GPU to process the LEVIR-CD training set and introduces **0 additional FLOPs / parameters** during online deployment.

---

## 🏗️ Architecture Overview

<p align="center">
  <img src="Frame.png" width="95%" alt="PFD-Net Overall Architecture">
</p>

*Overview of PFD-Net: (Left) Decoupled Offline Physics-Frequency Data Engine generating physically degraded ($A_{diff}$) and frequency-sharpened ($B_{sharp}$) pairs. (Right) Online Siamese Teacher-Student consistency training framework guided by Null-Change Prior.*

---

## 📊 Benchmark Results

PFD-Net consistently outperforms existing state-of-the-art semi-supervised change detection methods across three challenging public benchmarks under 5%, 10%, 20%, and 30% labeled data regimes.

### 1. Results on LEVIR-CD (Building Change Detection)

| Method | Venue | 5% Labeled (F1 / IoU) | 10% Labeled (F1 / IoU) | 20% Labeled (F1 / IoU) | 30% Labeled (F1 / IoU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AdvNet | BMVC'18 | 81.71 / 69.08 | 84.55 / 73.23 | 86.14 / 75.66 | 86.67 / 76.47 |
| SemiCDNet | TGRS'21 | 82.16 / 69.72 | 84.88 / 73.74 | 86.25 / 75.82 | 86.92 / 76.87 |
| RCL | TGRS'22 | 80.01 / 66.68 | 83.78 / 72.09 | 85.87 / 75.23 | 86.77 / 76.64 |
| UniMatch | CVPR'23 | 87.10 / 77.15 | 88.05 / 78.66 | 88.46 / 79.31 | 88.63 / 79.59 |
| CutMix-CD | TGRS'24 | 87.77 / 78.21 | 88.76 / 79.79 | 89.44 / 80.90 | — / — |
| C2F-SemiCD | TGRS'24 | 89.97 / 81.76 | 90.80 / 83.15 | 91.16 / 83.75 | 91.58 / 84.46 |
| SAM-CR* | TGRS'26 | 88.63 / 79.52 | 89.24 / 80.81 | 90.39 / 82.25 | 90.58 / 82.77 |
| **PFD-Net (Ours)** | — | **90.58 / 82.78** | **90.92 / 83.35** | **91.23 / 83.87** | **91.64 / 83.98** |

*\*Note: SAM-CR utilizes 40% labeled data in the last column, whereas our method utilizes only 30%.*

### 2. Results on WHU-CD & GZ-CD (F1-score / IoU %)

| Dataset | Metric | 5% Labeled | 10% Labeled | 20% Labeled | 30% Labeled |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **WHU-CD** | UniMatch | 88.35 / 79.14 | 88.58 / 79.50 | 89.26 / 80.60 | 90.47 / 82.60 |
| (Aerial & Shadows) | C2F-SemiCD | 85.63 / 74.87 | 86.58 / 76.33 | 90.07 / 81.93 | 92.85 / 86.66 |
| | SAM-CR | 88.79 / 79.91 | 90.88 / 83.21 | 91.09 / 83.54 | 92.89 / 87.08 |
| | **PFD-Net (Ours)** | **87.70 / 78.09** | **89.98 / 81.78** | **92.67 / 86.35** | **92.90 / 86.74** |
| **GZ-CD** | UniMatch | 52.50 / 35.59 | 63.93 / 46.98 | 67.97 / 51.48 | 57.59 / 40.44 |
| (Urban Villages) | C2F-SemiCD | 80.93 / 67.96 | 82.61 / 70.38 | 83.98 / 72.38 | 85.14 / 74.13 |
| | SAM-CR | 64.08 / 47.14 | 69.63 / 53.41 | 71.16 / 55.23 | 72.71 / 57.12 |
| | **PFD-Net (Ours)** | **82.62 / 70.39** | **84.07 / 72.51** | **85.54 / 74.73** | **86.13 / 75.64** |

---

## ⚙️ Computational Complexity

| Method | Venue | Params (M) | FLOPs (G) | Inference Overhead |
| :--- | :---: | :---: | :---: | :---: |
| SemiCDNet | TGRS'21 | 46.85 | 585.85 | Heavy |
| ChangeFormer | IGARSS'22 | 41.02 | 238.21 | Moderate |
| UniMatch | CVPR'23 | 32.55 | 110.45 | Moderate |
| C2F-SemiCD | TGRS'24 | 16.10 | 62.10 | Fast |
| **PFD-Net (Ours)** | — | **16.10** | **62.10** | **Real-time (0 Extra Inference FLOPs)** |

---

## 🛠️ Installation

```bash
# 1. Clone this repository
git clone https://github.com/lvvsh/PFD-Net.git
cd PFD-Net

# 2. Create conda environment
conda create -n pfdnet python=3.11 -y
conda activate pfdnet

# 3. Install PyTorch & dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
