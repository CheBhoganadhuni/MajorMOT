# 👣 People Attendance System – YOLOv9 + StrongSORT + OSNet

This repository contains my B.Tech major work on **multi-object person tracking** using **YOLOv9** for detection and **StrongSORT + OSNet** for re-identification.

The core idea is:

> Detect people → Track each person consistently across frames →  
> Use those stable track IDs as a foundation for an **entry/exit attendance system**.

The current version is a **Motion Object Tracking (MOT) pipeline** that produces tracking videos and ID overlays. Attendance logic (IN/OUT counting, logs, dashboards) is being built on top of this.

---

## 🚀 Features

- **Person detection with YOLOv9**
  - Uses converted YOLOv9 weights (`yolov9-c-converted.pt`) for fast, accurate detection.
- **StrongSORT-based multi-object tracking**
  - Kalman filter for motion prediction.  
  - IOU + appearance matching to keep IDs stable.
- **OSNet Re-ID backbone**
  - Uses `osnet_x1_0_market1501` weights for robust appearance embeddings.
- **Multiple demo videos**
  - Sample walking clips (`walking1.mp4` … `walking5.mp4`).  
  - Saved tracking runs under `runs/track/exp*`.
- **Jupyter notebook workflow**
  - `major.ipynb` walks through the full pipeline (easy to demo in college / viva).
- **Attendance-ready design (WIP)**
  - Code and structure prepared to plug in:
    - Line-crossing logic (gate/door).
    - IN/OUT event generation per ID.
    - CSV/DB logging for future web dashboards.

---

## 🧱 Repository Structure

```text
Major/
├── major.ipynb              # Main notebook to explore and run the pipeline
├── trackv9.py               # Script entry point for YOLOv9 + StrongSORT tracking
├── requirements.txt         # Root requirements (tracking + core libs)
├── dataloaders.py           # Data loading utilities (if used by notebook/script)
├── datasets.py              # Dataset utilities (if used)
├── general.py               # Shared helpers/utilities
├── osnet_x1_0_market1501.pt # OSNet Re-ID weights
├── osnet_x1_0_market1501.pth# (alternative format of OSNet weights)
├── yolov9-c-converted.pt    # YOLOv9 detection weights
├── walking1.mp4             # Sample input videos
├── walking2.mp4
├── walking3.mp4
├── walking4.mp4
├── walking5.mp4
│
├── runs/
│   └── track/
│       ├── exp/             # Different experiment outputs (auto-created)
│       ├── exp2/
│       ├── exp3/            # Each exp contains the tracked video
│       │   └── test_video.mp4 / walkingX.mp4
│       └── ...
│
├── strong_sort/             # StrongSORT tracker implementation
│   ├── strong_sort.py
│   ├── configs/
│   │   └── strong_sort.yaml # Tracker hyperparameters
│   ├── deep/
│   │   └── reid_model_factory.py
│   ├── sort/                # Kalman filter, IOU, matching, tracker logic
│   └── utils/               # Drawing, evaluation, IO helpers
│
└── yolov9/                  # YOLOv9 detection codebase
    ├── models/
    ├── utils/
    ├── detect.py
    ├── export.py
    ├── data/
    └── requirements.txt
```
---
## ⚙️ Setup & Installation

Tested with Python 3.11 (adjust if needed).


> As Of Now We Ran Our Project In **Google Colab** For GPU Requirement, File: **major.ipynb**, For Local System You Can Just Run Tracker ( Make Changes Accordingly )

### 1️⃣ Clone the repo
```
git clone https://github.com/CheBhoganadhuni/MajorMOT.git
cd MajorMOT
```
### 2️⃣ Create and activate a virtual environment
```
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```
### 3️⃣ Install dependencies

Root requirements:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```
> If you face CUDA/torch issues, start with CPU-only PyTorch and upgrade later.

---
## ▶️ Running the Tracker

**via trackv9.py script**

A typical command (adjust arguments, file paths as per your script):
```
python trackv9.py \
  --source walking1.mp4 \
  --yolo-weights yolov9-c-converted.pt \
  --reid-weights osnet_x1_0_market1501.pt \
  --img 1280 \
  --classes 0 \
  --config-strongsort /strong_sort/configs/strong_sort.yaml
```
> --source : input video path

> --yolo-weights : YOLOv9 detection weights

> --reid-weights : OSNet Re-ID weights

> --classes : to only identify humans

The output will appear under:
> runs/track/exp*/<video-name>.mp4
(Each run increments the exp folder number.)

---

## 🎯 Future Work – Attendance System

This MOT project is the core engine for a full attendance solution. Planned features:
- IN/OUT event logging per track ID
- Simple dashboard showing:
  - Who entered at what time
  - Total count
  - Day-wise logs
Once these parts are stable, the same engine can be used for:
- Classroom entry monitoring
- Lab/office attendance
- Crowd flow analysis

---
### 🚀 Built as a major project to explore real-world multi-object tracking and attendance automation.