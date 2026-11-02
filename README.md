# AI Liver and Hepatic Tumor Segmentation for Surgical Planning

> SwinUNETR-based automatic 3D segmentation of liver and hepatic tumors on CT, with Couinaud segment labeling, Future Liver Remnant (FLR) estimation, 3D Slicer integration via MONAILabel, and MediaPipe-based hand gesture control for touchless manipulation during surgical planning.

Master's thesis project — 

---

## Overview

This project provides an end-to-end pipeline that helps clinicians explore liver anatomy and plan surgical resection on CT scans:

1. **Automatic segmentation** of the liver and hepatic tumors with a 3D SwinUNETR model
2. **Couinaud segment labeling** of the liver using TotalSegmentator
3. **FLR estimation** (simple and segment-based) with Makuuchi criteria thresholds
4. **Interactive refinement** inside 3D Slicer through a MONAILabel server
5. **Touchless control** of the 3D Slicer view with hand gestures (MediaPipe), designed for sterile environments

---

## Features

- 3D SwinUNETR trained on the LiTS Challenge dataset (labels: background / liver / tumor)
- Sliding-window inference with MONAI
- Automatic Couinaud segments via TotalSegmentator `liver_segments` task
- Future Liver Remnant (FLR) calculation:
  - **Simple FLR** = (liver − tumor) / liver × 100
  - **Segment-based FLR** = (liver − tumor-bearing segment) / liver × 100
  - Clinical thresholds: > 25% (normal liver), > 40% (cirrhotic liver) based on Makuuchi (1989), Abdalla (2004), EASL (2018)
- MONAILabel app for 3D Slicer integration
- Hand gesture control (file-based JSON polling, no socket):
  - Index finger → rotate view
  - Pinch → zoom in
  - Victory sign → toggle liver + tumor overlay
  - Four fingers → zoom out
  - Thumb up → reset to calibrated view
- DICOM → NIfTI conversion script (SimpleITK)

---

## Repository structure

```
LiverProject/
├── monailabel_app/              # MONAILabel app for 3D Slicer
│   └── radiology/
│       ├── lib/
│       │   ├── configs/         # Model config (switch v4 ↔ v5)
│       │   ├── infers/          # Inference pipeline + KeepLargest fix
│       │   └── trainers/
│       └── main.py
├── pipeline/                    # Full case analysis
│   ├── full_pipeline.py         # End-to-end case processing
│   ├── fusion_analysis.py       # Segment × tumor fusion + FLR
│   └── run_full_analysis.py
├── gesture_control/             # MediaPipe + 3D Slicer listener
│   ├── gesture_control.py       # Hand tracking + command emission
│   ├── slicer_listener.py       # QTimer-based listener for 3D Slicer
│   └── slicer_startup.py
├── preprocessing/
│   ├── dicom_to_nifti.py        # DICOM series → NIfTI (SimpleITK)
│   ├── prepare_lits_batch1.py
│   └── rename_lits_batch1.py
├── examples/                    # Example scripts
├── docs/                        # Additional documentation
│   └── images/
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## Results

Validation on **27 held-out LiTS Challenge cases** (same split across all model versions).

| Model | Dice (Liver) | Dice (Tumor, aggregated) | Dice (Tumor, per-case mean ± std) | Detection (tumor Dice > 0) |
|-------|-------------|-------------------------|-----------------------------------|----------------------------|
| V3    | 0.924       | 0.492                   | 0.481 ± 0.351                     | 19 / 27                    |
| V4    | 0.916       | 0.561                   | 0.503 ± 0.344                     | 20 / 27                    |
| V5    | 0.922       | **0.679**               | **0.579 ± 0.328** (median 0.717) | **22 / 27**                |

**V4 is the currently deployed model** in the clinical workflow. V5 showed higher per-case Dice but produced liver fragments on real cases prior to the `KeepLargestConnectedComponent` fix (now applied to label 1 only). V5 re-validation after the fix is planned.

Metrics definitions:
- **Aggregated Dice**: voxel-level, computed over the whole concatenated validation set
- **Per-case Dice**: Dice computed per case, then averaged (sensitive to small lesions)
- **Detection**: proportion of cases where tumor Dice > 0

---

## Training setup

| Setting | V3 | V4 | V5 |
|---------|-----|-----|-----|
| Loss | DiceFocalLoss | DiceFocalLoss | DiceFocalLoss |
| Focal gamma | — | 2.0 | 3.0 |
| Class weights (bg / liver / tumor) | — | 0.02 / 0.08 / 0.90 | 0.03 / 0.12 / 0.85 |
| pos / neg sampling | — | 4 / 1 | 6 / 1 |
| Optimizer | AdamW (wd 1e-5) | AdamW (wd 1e-5) | AdamW (wd 1e-5) |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| T_max | 200 | 80 | 100 |
| LR | 3e-5 | 3e-5 | 2e-5 |
| Best epoch | 150 / 193 | 65 / 75 | 45 / 100 |
| Inference overlap | 0.25 | 0.25 | 0.50 |

Model: **SwinUNETR** (MONAI 1.5.2), `in_channels=1`, `out_channels=3`, `feature_size=48`, 63.1 M parameters, initialized from SwinUNETR self-supervised pretrained weights (MONAI Model Zoo, ~5050 public CT scans).

Pre-processing: Spacing (1.5, 1.5, 2.0) mm · ScaleIntensityRange HU [−200, 250] → [0, 1] · CropForeground.

Inference: sliding window roi (96, 96, 64), gaussian weighting, mixed precision.

Post-processing: argmax + `KeepLargestConnectedComponent(applied_labels=[1])` (liver only, NEVER on tumor label — this preserves multiple lesions).

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/wafamidouni1/liver-tumor-segmentation.git
cd liver-tumor-segmentation
```

### 2. Core environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# or
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

**Note on NumPy**: this project pins `numpy==1.26.4`. Do NOT upgrade to NumPy 2.x — it breaks `numpymaxflow` and MONAILabel.

### 3. Hand gesture control (separate env recommended)

MediaPipe requires NumPy 2.x, which conflicts with the main env. Install it in its own environment:

```bash
python -m venv gesture_env
# activate it, then:
pip install mediapipe==0.10.13 opencv-python
```

Download the MediaPipe hand landmarker model manually (not included in this repo):
```bash
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 4. 3D Slicer integration

- Install [3D Slicer](https://download.slicer.org/) (≥ 5.6)
- Install the MONAILabel extension from the Extensions Manager
- Start the MONAILabel server (see Usage below)
- Open the MONAILabel panel in 3D Slicer and connect to `http://localhost:8000`

---

## Model weights

The pretrained weights are **not included in this repository** (file size exceeds GitHub limits).

Download from Hugging Face:
```
https://huggingface.co/Wafaamidouni/swinunetr-liver-tumor/blob/main/swin_best_model_v4.pt

Place the `.pt` file in:
```
monailabel_app/radiology/model/swin_best_model_v4.pt
```

> *Weights hosting URL to be updated after upload.*

---

## Usage

### Running the MONAILabel server

```bash
python -m monailabel start_server \
  --app ./monailabel_app \
  --studies ./data \
  --conf models segmentation_liver
```

The server exposes a REST API at `http://localhost:8000`. Open `http://localhost:8000/docs` for the full OpenAPI spec.

### Running the full pipeline on a case

```bash
python pipeline/run_full_analysis.py --case_name case_001
```

This runs: inference → Couinaud segmentation → FLR calculation → report generation.

### Running the hand gesture control

In one terminal (gesture env):
```bash
python gesture_control/gesture_control.py
```

In 3D Slicer, open the Python console (Ctrl+3) and run:
```python
exec(open("gesture_control/slicer_listener.py").read())
```

---

## Dataset

Training and validation used the **LiTS Challenge** dataset (Liver Tumor Segmentation Benchmark). The dataset itself is **not included** in this repo. Download it from the official challenge page:
- https://competitions.codalab.org/competitions/17094

Our data split:
- Training: Task 03 train set
- Validation: 27 cases held out from Task 03

LiTS tumor labels aggregate all hepatic lesion types (HCC, cholangiocarcinoma, metastases); no sub-type distinction is made.

---

## Known limitations

- Model trained only on portal-venous phase CT; other phases may degrade performance
- Small lesions (< 5 mm) are often missed (reflected in per-case Dice variance)
- V5 has higher Dice but caused liver fragmentation on real cases before the `KeepLargest` fix — V5 is kept archived until re-validation
- No test set independent from LiTS has been used yet (planned: LiTS Batch 1 as independent test set)
- Hand gesture control requires good lighting and a webcam at eye level

---

## Disclaimer

> This project is a research prototype developed as part of a Master's thesis. **It has not been clinically validated and must not be used for clinical diagnosis, treatment planning, or any medical decision-making on real patients.** All results are intended for research, education, and development purposes only.

---

## Citation

If you use this code or the trained models in your research, please cite:

```bibtex
@mastersthesis{midouni2026liver,
  author       = {Wafa Midouni},
  title        = {AI-Assisted Liver and Hepatic Tumor Segmentation for Preoperative Surgical Planning},
  year         = 2026,
  address      = {tunis}
}
```

See also `CITATION.cff` for a machine-readable citation entry.

---

## Acknowledgments

- **LiTS Challenge** organizers for the dataset — Bilic et al., 2023
- **MONAI** framework and **MONAILabel** — Project MONAI
- **SwinUNETR** — Tang et al., *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis*, CVPR 2022
- **TotalSegmentator** — Wasserthal et al., *Robust Segmentation of 104 Anatomical Structures in CT Images*, Radiology AI 2023
- **3D Slicer** community
- **MediaPipe** — Google Research
- Master's supervisors in ENSIT tunisia

---

## License

MIT License — see [LICENSE](LICENSE) for full terms, including the medical research-use disclaimer.
