# NTIRE 2026 Remote Sensing Infrared Image Super-Resolution (x4)

##  Frequency-Assisted Mamba Super-Resolution Network (FreMamba)

---

##  Introduction

This repository presents our solution for the **NTIRE 2026 Challenge on Remote Sensing Infrared Image Super-Resolution (×4)**.

The goal of this challenge is to reconstruct **high-resolution infrared images** from **low-resolution inputs** with a scaling factor of **4×**, while preserving both structural and thermal information.

Our approach is based on a **Frequency-Assisted Vision State Space Model (FreMamba)**, which combines:
- **Mamba-based state-space modeling** for global context learning
- **Frequency-domain feature enhancement** for fine details
- **Hybrid gating mechanisms** for adaptive feature fusion

---

##  Model Overview

The model consists of:
- **Shallow feature extraction (Conv)**
- **Frequency-assisted Mamba Groups (FMG)**
- **Frequency-assisted Mamba Blocks (FMB)**
- **Vision State Space Module (VSSM)**
- **Frequency Selection Module (FSM)**
- **Hybrid Gate Module (HGM)**
- **Upsampling via PixelShuffle**

---

##  Architecture

<p align="center">
  <img src="architecture.png" width="100%">
</p>

---

##  Environment Setup

```bash
conda create -n FreMamba python=3.9.13
conda activate FreMamba

pip install torch==1.9.1 torchvision==0.10.1
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1

pip install -r requirements.txt
 Dataset Structure
/NTIRE2026/C14_RemSenseISR/
├── train/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── train_LR_X4/
│   └── X4/
│       ├── 0.png
│       ├── 1.png
│       └── ...
├── val_LR_X4/
│   └── X4/
 Training

Training is performed using train_4x.py.

Command
CUDA_VISIBLE_DEVICES=0 python train_4x.py \
--data_dir /NTIRE2026/C14_RemSenseISR \
--upscale_factor 4 \
--batchSize 4 \
--nEpochs 300 \
--patch_size 64 \
--lr 1e-4
Resume Training
CUDA_VISIBLE_DEVICES=0 python train_4x.py \
--start_epoch 200 \
--nEpochs 300 \
--pretrained True \
--pretrained_sr path_to_checkpoint.pth
 Testing / Inference

Testing is performed using test.py.

CUDA_VISIBLE_DEVICES=0 python test.py \
--valid_dir /path/to/val \
--test_dir /path/to/test \
--save_dir ./results \
--model_id 0
 Evaluation

Evaluation is done using eval_4x.py or official eval.py.

python eval_4x.py
Official Evaluation
python eval.py \
--output_folder "./results" \
--target_folder "/path/to/test/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0
 Evaluation Metric
Score = PSNR + 20 × SSIM

Computed on infrared intensity channel

4-pixel border shaved

Higher score = better performance

 Loss Function

We use a combination of:

Loss = Charbonnier Loss + 0.1 × SSIM Loss

Charbonnier Loss → robust pixel-level reconstruction

SSIM Loss → structural similarity preservation

 Repository Structure
.
├── dataload/
├── model_archs/
├── models/
│   └── team06_DAT/
├── eval_4x.py
├── test.py
├── train_4x.py
├── requirements.txt
└── README.md
Important Notes

Follow official NTIRE submission format

Add your model inside:

./models/[TeamID_ModelName]

Add checkpoint inside:

./model_zoo/[TeamID_ModelName]

Remove unnecessary images before submission

 Key Highlights

✔ Captures long-range dependencies (Mamba)
✔ Enhances high-frequency infrared details
✔ Efficient (linear complexity vs transformers)
✔ Designed specifically for infrared remote sensing

 Citation
@inproceedings{ntire2026rsirsrx4,
  title={NTIRE 2026 Challenge on Remote Sensing Infrared Image Super-Resolution (x4): Methods and Results},
  booktitle={CVPRW},
  year={2026}
}

