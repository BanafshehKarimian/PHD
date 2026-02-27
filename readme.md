# Privileged History Distillation for Mammography Risk Prediction (Code)

This repository contains training code for:
- **Base longitudinal models** and **horizon-specific teachers** (trained with full history as privileged information).
- A **student model** that learns from teachers and is designed to operate with limited or no prior exams at inference.

The code relies on two external codebases:
- **VMRA-MaR**: https://github.com/Mortal-Suen/VMRA-MaR/
- **Mirai**: https://github.com/yala/Mirai

---

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