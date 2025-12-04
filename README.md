# 🧠 SicTTA: Single Image Continual Test-Time Adaptation for Medical Image Segmentation

This repository contains the official PyTorch implementation of:

**SicTTA: Single Image Continual Test Time Adaptation for Medical Image Segmentation**  
*Accepted by Medical Image Analysis (MedIA)*  
🔗 https://linkinghub.elsevier.com/retrieve/pii/S1361841525004050

---

## 🔍 Overview

SicTTA is a **continual test-time adaptation (TTA)** framework designed for robust medical image segmentation under distribution shifts.  
It adapts models **using only a single test image at a time**, without access to the source data or large target batches—making it suitable for real-world deployment where memory and data availability are limited.

This repository also provides **re-implementations of several state-of-the-art (SOTA) TTA methods** under a unified segmentation framework for fair comparison.

### ✨ Key Features

- 🔁 **Single-image** test-time adaptation
- 🌊 Continual adaptation for non-stationary test streams
- ❌ **Source-free** (no source images or labels required)
- 🧪 Comprehensive evaluation on the **M&MS cardiac MRI dataset**
- 🧩 Includes multiple SOTA TTA methods: **TENT, MEANT, SAR, SiTTA**, etc.

---

## 📦 Installation

We recommend using a conda environment:

```bash
conda create -n sictta python=3.10
conda activate sictta
pip install -r requirements.txt
```

---

## 📂 Dataset: M&MS

We adopt the **M&MS dataset** for cardiac MRI segmentation.  
Official website: https://www.ub.edu/mnms/

### Steps:

1. Apply for and obtain **official permission** to use the dataset.  
2. Download the dataset through the official portal.  
3. (Optional) Use our **processed M&MS 2D version**:

   👉 Google Drive:  
   https://drive.google.com/file/d/1jaT2nsbF1-rPoWs6fnF9DsFxTuaYXqW2/view?usp=sharing

4. Extract the processed data into: SicTTA/data/mms2d/


---

## 🚀 Source Model Training

To train a UNet segmentation model on the source domain:

```bash
python train_source.py --cfg cfgs/mms/source.yaml
```

- Checkpoints are saved to: `save_model/`

---

## 🧪 Test-Time Adaptation

We provide evaluation scripts for SicTTA and other methods. All experiments are configured via YAML files.

```bash
# Baseline: No adaptation
python test_time_adaptation.py --cfg cfgs/mms/norm.yaml

# Source test
python test_time_adaptation.py --cfg cfgs/mms/source_test.yaml

# TENT
python test_time_adaptation.py --cfg cfgs/mms/tent.yaml

# MEANT (Mean Teacher)
python test_time_adaptation.py --cfg cfgs/mms/meant.yaml

# SAR
python test_time_adaptation.py --cfg cfgs/mms/sar.yaml

# SicTTA (our method)
python test_time_adaptation.py --cfg cfgs/mms/sictta.yaml
```

---

## 📚 Citation

If you find SicTTA useful, please consider citing:

```bibtex
@article{wu2026sictta,
  title   = {SicTTA: Single Image Continual Test Time Adaptation for Medical Image Segmentation},
  author  = {Wu, Jianghao and Liu, Xinya and Wang, Guotai and Zhang, Shaoting},
  journal = {Medical Image Analysis},
  volume  = {108},
  pages   = {103859},
  year    = {2026},
  doi     = {10.1016/j.media.2025.103859}
}
```



## 🙋 Acknowledgements

This repo builds upon ideas from:

- [TENT](https://github.com/DequanWang/tent)
- and others, re-implemented for medical segmentation tasks.

---

## 📮 Contact

If you have questions, feel free to open an issue or reach out.

Happy Adapting! 🎯

