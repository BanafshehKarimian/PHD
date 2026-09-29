# Single-Exam Mammography Risk Prediction with Privileged History Distillation (Code)

This repository contains training code for:
- **Base longitudinal models** and **horizon-specific teachers** (trained with full history as privileged information).
- A **student model** that learns from teachers and is designed to operate with limited or no prior exams at inference.

The code relies on two external codebases:
- **VMRA-MaR**: https://github.com/Mortal-Suen/VMRA-MaR/
- **Mirai**: https://github.com/yala/Mirai
## ABSTRACT:


Longitudinal mammography screening has become an important source of information for improving future breast cancer risk prediction. However, the performance of current longitudinal mammography models degrades when prior examinations are unavailable at inference, creating a structured privileged-information setting in which temporal context is available during training but absent at deployment. We propose Single-Exam Mammography risk prediction with privileged History Distillation (SEM-HD), a framework that uses longitudinal history as privileged information available only during training to preserve the predictive benefits of longitudinal modeling while requiring only the current screening examination at deployment. During training, the student relies on the current examination to predict latent representations of prior visits, while horizon-specific teachers provide additional supervision from the observed longitudinal history. Together, latent history prediction and teacher distillation preserve the temporal modeling structure of longitudinal predictors under current-exam-only inference. We validate SEM-HD on three longitudinal mammography cohorts, the CSAW-CC, EMBED, and OMI-DB, using the transformer-based Longitudinal Mammography Risk (LoMaR) and recurrent Visual Memory Recurrent Attention (VMRA) backbones. Under current-exam-only inference, SEM-HD consistently improves long-horizon AUC and pAUC over longitudinal models evaluated without history, particularly in the clinically relevant low false-positive-rate region. It also recovers much of the performance gap with respect to full-history inference across datasets and backbones. Ablations further show that these gains are not reproduced by masking or heuristic history imputation. The strongest performance is achieved by combining patient-specific latent history prediction with distilled temporal risk supervision. These results suggest that temporal context can be effectively exploited as privileged supervision while preserving the underlying longitudinal modeling structure.

<img width="1636" height="1000" alt="image" src="https://github.com/user-attachments/assets/b895733c-3690-440d-a890-c21f400c1ea7" />

## 1) Requirements

You must install / set up both dependencies:

### Install VMRA-MaR
Follow the installation instructions in the VMRA-MaR repository:
https://github.com/Mortal-Suen/VMRA-MaR/

### Install Mirai
Follow the installation instructions in the Mirai repository:
https://github.com/yala/Mirai

> Note: These projects have their own environment and dataset setup requirements. Please make sure each dependency runs correctly on your machine before using this code.

---

## 2) Repository structure (high level)

- `main.py`: trains **base model(s)** and/or **teacher model(s)** (including horizon-specific teachers).
- `main_student.py`: trains the **student** using knowledge distillation from the trained teacher(s).

(Other scripts/modules are used internally for data loading, evaluation, losses, and model definitions.)

---

## 3) Training

### A) Train base model / teachers (privileged full history)
Use `main.py` to train:
- A standard baseline model, and/or
- Horizon-specific teacher branches that use **full screening history** during training.

Example:
```bash
python main.py --config <YOUR_CONFIG> [other args...]
