# 🚁 Ariel Project — Low-Altitude Aerial Object Detection (3–9m)

> Fine-tuned YOLOv8n model for robust object detection in drone footage captured at 3–9 meter altitude — a range critically underrepresented in existing datasets like VisDrone and UAVDT.

<br>

## 🤝 Collaboration

<table>
  <tr>
    <td align="center" width="33%">
      <br/>
      <a href="https://www.thi.de"><img src="assets/thi_logo.png" height="60" alt="THI"/></a>
      <br/><br/>
      <b>Technische Hochschule Ingolstadt</b><br/>
      <sub>Research Guidance & Supervision</sub>
      <br/>
    </td>
    <td align="center" width="33%">
      <br/>
      <a href="https://www.ivi.fraunhofer.de"><img src="assets/fraunhofer_ivi_logo.png" height="60" alt="Fraunhofer IVI"/></a>
      <br/><br/>
      <b>Fraunhofer IVI</b><br/>
      <sub>Industry Mentorship & Data Provision</sub>
      <br/>
    </td>
    <td align="center" width="33%">
      <br/>
      <a href="https://www.th-deg.de"><img src="assets/thd_logo.png" height="60" alt="THD"/></a>
      <br/><br/>
      <b>Technische Hochschule Deggendorf</b><br/>
      <sub>Student Project Team</sub>
      <br/>
    </td>
  </tr>
</table>

<br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-00FFFF?style=flat)
![mAP](https://img.shields.io/badge/mAP%400.5-91.8%25-28a745?style=flat)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-Academic-blue?style=flat)

---

## 📌 Overview

Existing aerial datasets (VisDrone, UAVDT) are designed for flights above 30 meters, leading to missed detections, poor localization accuracy, and high false positives at low altitudes. This project addresses that gap by:

- Building a **custom annotated dataset** from real drone footage at 3–9m across 4 distinct scenes
- Fine-tuning **YOLOv8n** on this dataset using aerial-specific hyperparameter tuning
- Achieving **91.8% mAP@0.5** on a held-out 10% test split
- Running at **2.3 ms inference** per image — suitable for real-time deployment

<br>







https://github.com/user-attachments/assets/6bf30f82-0f2c-4328-b598-dad79f8f0d81
> *YOLOv8n · 2x speed · conf=0.50*

---

## 📊 Results

### Overall Performance

| Metric | Value |
|---|---|
| **mAP@0.5** | **0.918** |
| **mAP@0.5:0.95** | **0.850** |
| **Precision** | **0.956** |
| **Recall** | **0.926** |
| **F1 Score** | **0.94** |
| Inference Speed | 2.3 ms / image |

### Per-Class Breakdown

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| **All** | 231 | 501 | 0.956 | 0.926 | **0.918** | 0.850 |
| Human | 188 | 385 | 0.949 | 0.958 | 0.951 | 0.845 |
| Bicycle | 34 | 55 | 0.918 | 0.818 | 0.807 | 0.713 |
| Vehicle | 61 | 61 | 1.000 | 1.000 | 0.995 | 0.993 |

> **Note:** Bicycle scores lower due to class imbalance (597 instances vs 3,642 Human). Vehicle achieves perfect precision and recall — visually distinct from top-down perspective.

---

## 🗂️ Dataset

### Statistics

| Metric | Value |
|---|---|
| Total Annotations | 4,776 |
| Frames Labeled | 2,227 |
| Video Sequences | 10 |
| Distinct Scenes | 4 |
| Negative Samples | ~15% |
| Format | YOLO (.txt) |
| Dataset Split | 80% train / 10% val / 10% test |

### Classes

| Class ID | Name | Count | Share |
|---|---|---|---|
| 0 | Human | 3,642 | 76.2% |
| 1 | Bicycle | 597 | 12.5% |
| 2 | Vehicle | 537 | 11.2% |

### Scenes

| Scene | Videos | Annotations |
|---|---|---|
| DK_backyard | v1, v3, v5 | 1,342 |
| DK_parking | v3, v4 | 1,757 |
| THI_Bikepark | v1, v3 | 1,248 |
| THI_Grass | v1, v2, v3 | 429 |

### Annotation Tool

Annotations were created using **[Roboflow](https://roboflow.com/)** — a web-based collaborative labeling platform with built-in quality review, version control, and direct YOLO export.

**Labeling rules enforced:**
- Tight bounding boxes — zero padding, edge-to-edge
- Physical silhouette only — shadows excluded
- Occlusion threshold — label if ≥20% of object visible
- Negative samples — ~15% empty frames to reduce false positives

---

## 🛠️ Model & Training

### Architecture

- **Model:** YOLOv8n (Nano) — optimized for edge/drone devices
- **Base weights:** COCO pretrained (`yolov8n.pt`)
- **Platform:** Google Colab (T4 GPU)
- **Export:** ONNX

### Training Configuration

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="./unified_dataset_80_10_10/data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    project="Ariel_Project",
    name="drone_model_final",
    save=True,
    plots=True,

    # Early stopping
    patience=15,          # Stops if mAP doesn't improve for 15 epochs

    # Aerial-specific optimizations
    box=7.5,              # Enforces strict edge-to-edge bounding constraints
    cls=1.5,              # Higher weight on exact object classifications
    cos_lr=True,          # Smooth cosine learning rate decay
)
```

### Evaluation

```python
from ultralytics import YOLO

model = YOLO("runs/detect/Ariel_Project/drone_model_final-3/weights/best.pt")

metrics = model.val(
    data="./unified_dataset_80_10_10/data.yaml",
    split="test",         # Evaluates strictly on locked 10% test split
    conf=0.50,
    plots=True,
)
```

---

## 🗂️ Repository Structure

```
ariel-project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── assets/                       # Logos, demo images, GIFs
│
├── data/
│   ├── raw/                      # Raw drone footage (not tracked)
│   └── unified_dataset_80_10_10/ # Train/val/test split with data.yaml
│
├── train.py                      # Training script
├── test.py                       # Evaluation on test split
│
├── runs/
│   └── detect/
│       └── Ariel_Project/
│           └── drone_model_final/
│               └── weights/
│                   └── best.pt   # Best model weights (not tracked)
│
└── results/
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── PR_curve.png
    ├── F1_curve.png
    └── R_curve.png
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or Google Colab

### Installation

```bash
git clone https://github.com/BUAksakal/ariel-project.git
cd ariel-project
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

### Evaluate on Test Split

```bash
python test.py
```

---

---

## 📄 License

This project is developed for academic research purposes as part of the THI × Fraunhofer IVI × THD collaboration.

---

*Part of the Student Case-Study Project Program — THI / Fraunhofer IVI / THD, SS26*
