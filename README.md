# 2D Brain Tumor Diagnostic Workspace

An interactive Streamlit web application for AI-based brain tumor segmentation on 2D MRI slices. The app performs U-Net inference on uploaded MRI images, calculates tumor area, and generates downloadable PDF clinical reports.

> **Disclaimer**: This is an academic prototype only. It is not a certified medical diagnostic tool. All outputs are for research/educational purposes.

---

## Project Structure

```
brain-tumor-diagnostic-workspace/
├── app.py                          # Streamlit application (entry point)
├── requirements.txt
├── README.md
├── data/
│   └── prepare_dataset.py          # Downloads/converts .tif pairs to .png, splits train/val/test
├── model/
│   ├── unet.py                     # U-Net model definition (ResNet34 encoder via SMP)
│   ├── losses.py                   # Dice Loss implementation
│   ├── train.py                    # Training loop with Albumentations augmentation
│   ├── evaluate.py                 # Computes IoU, Precision, Recall, Dice on test set
│   └── checkpoints/                # Saved .pth weights (created at runtime, gitignored)
├── core/
│   ├── preprocessing.py            # Resize, normalize, tensor conversion
│   ├── inference.py                # Loads model, runs forward pass, thresholds mask
│   ├── postprocessing.py           # Alpha-blend overlay, pixel-to-area calculation
│   └── report.py                   # ReportLab PDF generation
└── tests/
    └── test_pipeline.py            # Unit tests for preprocessing/postprocessing math
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: If you have a specific CUDA version, install PyTorch first from https://pytorch.org/get-started/locally/ before running `pip install -r requirements.txt`.

### 2. Prepare the dataset

Download the [LGG MRI Segmentation dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) from Kaggle and place the zip file in the project root (or specify its path with `--zip-path`).

```bash
python data/prepare_dataset.py --zip-path ./lgg-mri-segmentation.zip --extract-dir ./raw_data/
```

This will produce:
```
data/processed/
├── train/{images/, masks/}
├── val/{images/, masks/}
└── test/{images/, masks/}
```

### 3. Train the model

```bash
python model/train.py --epochs 30 --batch-size 16
```

Options:
| Argument | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--data-dir` | `./data/processed` | Path to processed data |
| `--checkpoint-dir` | `./model/checkpoints` | Where to save checkpoints |

The best model is saved to `model/checkpoints/best_model.pth` and a loss curve is saved to `model/checkpoints/loss_curve.png`.

### 4. Evaluate the model

```bash
python model/evaluate.py
```

Results are printed to console and saved to `model/checkpoints/eval_results.md`.

### 5. Launch the app

```bash
streamlit run app.py
```

The app works immediately in **mock mode** (before training) — a placeholder circular mask is shown with a warning banner. Once `model/checkpoints/best_model.pth` exists, it automatically switches to real model inference.

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Pipeline Overview

```
Upload MRI (.png/.jpg) 
        │
        ▼
 core/preprocessing.py   →  normalize [0,255]→[0,1], resize 256×256, HWC→CHW tensor (1,3,256,256)
        │
        ▼
  core/inference.py      →  U-Net forward pass → sigmoid → probability map [0,1]
        │
        ▼
  core/inference.py      →  threshold @ 0.5 → binary mask
        │
        ▼
core/postprocessing.py   →  area = N_pixels × 0.013225 cm², alpha-blend overlay
        │
        ▼
   core/report.py        →  ReportLab PDF (in-memory BytesIO, never written to disk)
```

---

## Mathematical Constants

| Constant | Value | Meaning |
|---|---|---|
| `PIXEL_AREA_CM2` | 0.013225 cm² | Area per pixel (0.115 cm edge length, squared) |
| Threshold | 0.5 | Sigmoid probability cutoff for tumor classification |

---

## Dataset

**LGG MRI Segmentation** by Mateusz Buda  
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

Contains paired `.tif` MRI slices and binary ground-truth masks for lower-grade glioma patients from TCGA.
