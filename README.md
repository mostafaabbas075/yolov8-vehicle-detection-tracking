# 🚗 YOLOv8 Vehicle Detection & Tracking

A real-time vehicle detection and tracking system built using **YOLOv8** and **DeepSORT**, trained on a custom dataset to detect multiple vehicle types including **car**, **truck**, **bus**, and **motorcycle**.

---

## 📌 Features

- 🔍 **YOLOv8 Detection** — Detects cars, buses, trucks, and motorcycles
- 🔁 **DeepSORT Tracking** — Tracks each vehicle across frames with unique IDs
- 🧠 **Two-Phase Training**:
  - Phase 1: Trained from scratch for 33 epochs using `yolov8l.pt`
  - Phase 2: Resumed training from `best.pt` for 17 additional epochs
- 💾 **Checkpoint Management** — Keeps only `best.pt` and `last.pt`
- 🟢 **Real-time Inference** — Fast, efficient frame-by-frame tracking
- 📊 **Evaluation on val/test splits** — mAP, precision, recall, and more

---

## 🛠️ Tech Stack

- `ultralytics` (YOLOv8)
- `torch` (PyTorch)
- `opencv-python`
- `deep_sort_realtime`
- `albumentations`
- `numpy`, `PyYAML`, `matplotlib`, `pandas`

Trained on **Kaggle Notebooks** using **Tesla P100 GPU**.

---

## 📂 Dataset

Custom dataset containing labeled images for:
- `car`, `truck`, `bus`, `motorcycle`

> 📁 Dataset: *https://www.kaggle.com/datasets/mostafaelmelegy/license-detection/data*  
> *https://www.kaggle.com/datasets/mostafaabbas075/dataset-cars*

📌 **Disclaimer**  
This dataset is a compilation of images collected from various open-source Kaggle datasets.  
The original datasets were publicly available at the time of collection and published under licenses that allowed reuse.  

I do not claim ownership of the original images.  
This dataset is shared for **educational and non-commercial purposes only**.  
If you are the owner of any content and would like it removed or credited, please contact me.

🔗 Unfortunately, I no longer have the direct links to the original sources.

---

## 🧠 Training Details

Training was done in two phases:

```text
Phase 1: 33 epochs from scratch using yolov8l.pt  
Phase 2: 17 epochs resumed from saved checkpoint (best.pt)  
Total: 50 epochs
```

Checkpoint saving is handled automatically, with cleanup for intermediate files.

---

## 🧾 Notebooks

The project is organized into three Jupyter notebooks:

- `01_train_vehicles.ipynb` – Initial training for vehicles & Resume from checkpoint 
- `02_train_license_plates.ipynb` Initial training for plate Licenses 
- `03_inference.ipynb` – Real-time inference and DeepSORT tracking

---

## 🔍 Example Inference (Simple)

```python
from ultralytics import YOLO

model = YOLO("runs/yolov8l_vehicles/weights/best.pt")
results = model("video.mp4", save=True, conf=0.5)
```

To run with DeepSORT tracking, use the code inside `03_inference_and_tracking.ipynb`.

---

## 📦 Installation

```bash
pip install ultralytics opencv-python torch deep_sort_realtime albumentations pyyaml pandas matplotlib
```

---

## 📸 Demo Video
*https://www.linkedin.com/posts/activity-7326374763663171585-gfM_?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAAExppuYBLUqjnF-K3rXxALthfcmZWvcr_S8*

---

## ✅ Future Work

- 🔤 Add OCR to detect license plate numbers  
- 🕓 Estimate vehicle speed  
- 🌐 Deploy with a real-time dashboard (e.g. Streamlit or Flask)

---

## 👨‍💻 Author

**Mostafa Abbas**  
AI Student @ Arab Open University  
Gmail: mostafa.a.s075@gmail.com
LinkedIn: https://www.linkedin.com/in/mostafa-abbas-91a4052b9/
```
