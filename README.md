# Real-Time Object Detection and Inference Pipeline (YOLOv8) on AWS/GCP

Train and deploy a real-time object detection API using YOLOv8 and serve it using Flask (locally) or on AWS/GCP cloud. Optionally, connect a simple web UI.

## 📦 Structure

```
.
├── src/
│   ├── yolo_infer.py          # Inference and visualization
│   ├── train.py               # Training script (for custom data)
│   └── api.py                 # REST API (Flask)
├── webapp/                    # (Optional) Simple frontend
├── data/                      # Place images/videos here
├── results/                   # Output images, logs
├── requirements.txt
├── README.md
└── .gitignore
```

## 🏁 Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run local inference:
   ```
   python src/yolo_infer.py --image data/sample.jpg
   ```
3. Start API server:
   ```
   python src/api.py
   ```
4. (Optional) Deploy to AWS/GCP—see deployment instructions in repo.

## 🔬 Recommended Datasets

- [COCO](https://cocodataset.org/) (Standard benchmark)
- [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
- Use Ultralytics YOLOv8 pretrained models for demo

## 🌐 API

- POST `/predict` with image file, returns bounding boxes and class labels

---
