# Roadmap: AI-Powered Object Tracking & Recognition System

## 1. Project Overview
Transitioning a high-compute DL project (Motion Tracking + Face Recognition) from a research environment (Colab) to a production-grade cloud architecture.

## 2. The High-Level Architecture
To handle the 100-second processing time, we use an **Asynchronous Task Queue**.

* **Django Portal:** The user-facing interface for uploads.
* **GCS (Google Cloud Storage):** The "Bucket" that holds raw and processed video files.
* **Redis:** The message broker that tracks "Jobs" to be done.
* **Celery Worker:** The background process on your **GPU VM** that runs the DL model.

## 3. Implementation Steps

### Phase 1: Local Preparation (The "Logic")
* Refactor your Colab code into a clean Python script: `process_video.py`.
* **Dockerize:** Create a `Dockerfile` using `nvidia/cuda:11.8.0-base-ubuntu22.04` as the base to ensure GPU compatibility.

### Phase 2: Google Cloud Setup (The "Infrastructure")
* **GCS:** Create a bucket named `project-video-storage`.
* **Compute Engine:** Launch a VM with an **NVIDIA T4 GPU**.
* **Network:** Open Port 80 for the web portal and Port 22 for SSH.

### Phase 3: The Django Development (The "Portal")
* Integrate `django-storages` to handle GCS uploads.
* Set up a simple Dashboard to show "Processing..." and "Download Result" states.

---

## 4. Resume & Portfolio Summary
**Project: Enterprise-Scale Motion Tracking & Face Recognition Pipeline**
* **Role:** Lead Developer / Cloud Architect
* **Tech:** Python, Django, PyTorch (YOLO), StrongSORT, GCP (GCE/GCS), Docker, Redis, Celery.
* **Impact:** * Successfully migrated a DL heavy-inference model to a scalable IaaS environment.
    * Reduced user-perceived latency by implementing an asynchronous task queue.
    * Containerized the pipeline for environment-agnostic deployment.
