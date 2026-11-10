# ai-vision-industrial-inspection
INDUSTRIAL VISION SYSTEM: Real-time quality control application using OpenCV and YOLOv8 to inspect a simulated assembly line. Features customizable region-of-interest (ROI) filtering and confidence threshold tuning for component validation (OK/NG states).

# 👁️ AI-Powered Computer Vision Industrial Inspection System

This repository contains a high-speed **Computer Vision Quality Control System** designed for automated manufacturing and assembly lines. Utilizing **OpenCV** for low-level image processing and **YOLOv8 (You Only Look Once)** for real-time deep learning inference, the application isolates an active scanning window to classify components into **OK (Conforming)** or **NG (Non-Conforming/Defective)** states based on geometric object parameters.

---

## 🛠️ Technical Features & Optimization

*   **Region of Interest (ROI) Filtering:** Rather than executing deep learning inference across the entire camera frame, the pipeline crops a specific matrix slice ("Virtual Belt Zone"). This limits CPU/GPU cycles, reduces computational latency, and optimizes frames-per-second (FPS) output.
*   **Confidence Threshold Tuning:** Integrated boundary controls (`conf=0.65`) force the neural network to discard weak predictions (under 65% certainty). This structural filtration mitigates environmental noise, lighting shifts, and false positives like edge-bleeding artifact detections from operator hands.
*   **Decoupled Edge Design:** Engineered using an lightweight architecture (`yolov8n.pt`) making it completely portable to be deployed directly on industrial edge-computing nodes such as NVIDIA Jetson or Raspberry Pi single-board hardware setups.

---

## 📂 Project Layout

```text
├── .venv/                   # Isolated environment containing Ultralytics core
├── inspeccion_vision.py     # Main AI Computer Vision Script
└── README.md                # System Documentation & Logs
```

---

## 🚀 Setup & Execution Guide

### 1. Initialize dependencies
Inside your active virtual environment, install the AI vision package stack:
```bash
.\.venv\Scripts\pip install opencv-python ultralytics
```

### 2. Execute the Inspection Pipeline
```bash
python "inspeccion_vision.py"
```

### 3. Operator Controls
*   Place your open hand or flat surface in the inspection square: Yields **`COMPONENT VALID [OK]`** (Green indicator).
*   Introduce a closed object (like a mobile phone or cup): Yields **`DEFECTIVE COMPONENT [NG]`** (Red alert trigger).
*   Press **`q`** inside the video display window to terminate the hardware capture stream cleanly.

---

## 🧠 Engineering Insights: Critical Tuning Logs

### Problem: False Positives & Stray Object Detections
*   **Symptom:** When the inspection area was empty or when an operator's hand entered the border space, the bounding box would flicker chaotically, misclassifying skin folds or shadows as random tools or parts.
*   **Diagnosis:** The default base configuration of the YOLO weights processes any inference with a loose confidence barrier of 25% (`conf=0.25`). While useful for wide-area surveillance cameras, this settings is too unstable for precision tolerances required on close-up manufacturing lines.
*   **Resolution:** Tightened system tolerances by introducing strict filtering at the ingestion function: `model(frame, conf=0.4)`. This raised the activation energy of the weights, forcing the network to only trigger a state change when a high-density probability matrix matches the targets, completely eliminating edge noise from stray fingers.
