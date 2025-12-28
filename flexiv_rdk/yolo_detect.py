# yolo_detect.py
import torch
import cv2
import numpy as np
import os
from pathlib import Path

# 模型路径（训练后得到的 best.pt）
MODEL = "yolov5/runs/train/rod_yolov5s/weights/best.pt"
DATA_ROOT = "data/rgbd"   # 你的测试10组所在目录
OUT_DIR = "yolo_output"
os.makedirs(OUT_DIR, exist_ok=True)

# load model (yolov5's torch hub interface)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL, force_reload=False).to(device)
model.conf = 0.25  # confidence threshold
model.iou = 0.45

colors = [
    (255, 0, 0),   # class 0 -> blue? (OpenCV BGR) (we'll map to BGR)
    (0, 255, 0),   # class 1 -> green
    (0, 0, 255),   # class 2 -> red
]

scenes = sorted([d for d in os.listdir(DATA_ROOT) if d.isdigit()])
for s in scenes:
    img_path = os.path.join(DATA_ROOT, s, "rgb.png")
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)
    orig = img.copy()
    results = model(img[..., ::-1])  # yolov5 expects RGB
    # results: boxes in xyxy, confidences, class
    df = results.pandas().xyxy[0]  # pandas DataFrame
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = float(row['confidence'])
        cls = int(row['class'])
        color = colors[cls] if cls < len(colors) else (0,255,255)
        label = f"{cls}:{conf:.2f}"
        cv2.rectangle(orig, (x1,y1), (x2,y2), color, 2)
        cv2.putText(orig, label, (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = os.path.join(OUT_DIR, f"{s}_det.png")
    cv2.imwrite(out_path, orig)
    print("Saved", out_path)
