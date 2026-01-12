# Project Safe State - MajorMOT
**Date:** 2026-01-12
**Status:** Stable / Face Recognition Integrated

## Overview
This document records the current "Safe State" of the project after cleaning up temporary scripts (`runtime_gallery.py`, `identity_manager.py`) and consolidating logic into `trackv9.py`.

## 1. Core Logic (`trackv9.py`)
The main tracking script is fully functional with the following features:

### A. Hybrid Identity System
-   **Primary Authority**: Face Recognition (InsightFace/ArcFace).
-   **Secondary/Continuity**: StrongSORT (Appearance ReID) + OSNet.
-   **Fusion Logic**:
    -   **Voting**: Accumulates confidence scores from face matches over multiple frames.
    -   **Identity Locking**: Once a track crosses a confidence threshold (1.0 accumulated score), the identity is **LOCKED**.
    -   **Locking Effect**: Prevents "flickering" or ReID overrides due to clothing changes.

### B. Attendance Summary
-   Includes logic to generate `attendance_summary.csv`.
-   Handles session splitting (gap detection) and filters out 'Unknowns' and short noise.

### C. Logging
-   Writes frame-level data to `attendance_log.csv`.
-   Includes `Confidence` and `Dist` metrics for debugging.

## 2. Key Files
| File | Purpose | Status |
| :--- | :--- | :--- |
| `trackv9.py` | Main execution script. Contains Tracking + Face Rec + Logging logic. | **Verified** (Contains Locking Logic) |
| `generate_face_gallery.py`| Standalone script to scan `dataset/` and create `face_gallery.pkl`. | **Verified** |
| `requirements.txt` | Lists dependencies including `insightface` and `onnxruntime`. | **Verified** |
| `gallery.pkl` | Legacy appearance embeddings (OSNet). | Optional (System falls back if missing) |
| `face_gallery.pkl` | Face recognition embeddings (ArcFace). | **Required** for Face Rec |

## 3. How to Run
### Step 1: Generate Face Gallery
```bash
python generate_face_gallery.py --dataset dataset --output face_gallery.pkl
```

### Step 2: Run Tracking
```bash
python trackv9.py --source assets/walking4.mp4 --yolo-weights yolov9-c-converted.pt --reid-weights osnet_x1_0_market1501.pt --img 640 --vid-stride 2 --save-log --face-gallery face_gallery.pkl
```

## 4. Dependencies
-   **Python 3.11** Compatible.
-   Requires: `insightface`, `onnxruntime` (Note: `insightface` needs C++ Build Tools on Windows, recommended to run on Colab).

## 5. Recent Changes
-   **Removed**: `identity_manager.py`, `runtime_gallery.py` (Logic moved/simplified).
-   **Added**: "Track-Level Voting & Locking" inside `trackv9.py` to fix clothing change issues.
