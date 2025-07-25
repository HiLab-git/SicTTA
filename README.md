# 🧠 SicTTA: Single Image Continual Test-Time Adaptation for Medical Image Segmentation

This repository contains the official PyTorch implementation of:

**SicTTA: Single Image Continual Test-Time Adaptation for Medical Image Segmentation**

> 📝 *Under Review / To appear*

SicTTA is a continual test-time adaptation (TTA) framework that adapts segmentation models to distribution shifts using only **a single test image at a time**, without access to the source data or large target batches. It is specifically designed for medical image segmentation where robust deployment is critical.

We also provide re-implementations of several state-of-the-art (SOTA) TTA methods for fair comparison.

---

## 📌 Highlights

- 🔁 **Single Image** Test-Time Adaptation (SicTTA)
- 🌊 Continual adaptation across non-stationary test streams
- ❌ Source-Free: no source data or labels required
- 🧪 Comprehensive benchmark on M&MS dataset
- 🧩 Includes several SOTA methods: TENT, MEANT, SAR, SITTA, etc.

---

## 📦 Installation

```bash
# Create a conda environment
conda create -n sictta python=3.10
conda activate sictta

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Dataset: M&MS

We use the [M&MS dataset](https://www.ub.edu/mnms/) for cardiac MRI segmentation.

📥 Download the dataset and place it in the expected folder.  

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

# TENT
python test_time_adaptation.py --cfg cfgs/mms/tent.yaml

# MEANT (Mean Teacher)
python test_time_adaptation.py --cfg cfgs/mms/meant.yaml

# SAR
python test_time_adaptation.py --cfg cfgs/mms/sar.yaml

# SITTA
python test_time_adaptation.py --cfg cfgs/mms/sitta.yaml

# SicTTA (our method)
python test_time_adaptation.py --cfg cfgs/mms/sictta.yaml
```

---

## 🙋 Acknowledgements

This repo builds upon ideas from:

- [TENT](https://github.com/DequanWang/tent)
- and others, re-implemented for medical segmentation tasks.

---

## 📮 Contact

If you have questions, feel free to open an issue or reach out.

Happy Adapting! 🎯
