# convert_labels.py
import os
import json
import cv2
from pathlib import Path
import shutil
import random

ROOT = "/home/wangbingquan/flexiv_rdk/ropewrap/data/rgbd_dataset"   # 原始文件夹
OUT = "/home/wangbingquan/flexiv_rdk/ropewrap/data/yolo_dataset"    # 输出 yolo 格式数据集
TRAIN_RATIO = 0.8

os.makedirs(OUT, exist_ok=True)
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

scenes = sorted([d for d in os.listdir(ROOT) if d.isdigit()])

# collect scenes and split
random.seed(42)
random.shuffle(scenes)
n_train = int(len(scenes) * TRAIN_RATIO)
train_scenes = scenes[:n_train]
val_scenes = scenes[n_train:]

def process_scene(scene, split="train"):
    folder = os.path.join(ROOT, scene)
    rgb_path = os.path.join(folder, "rgb.png")
    if not os.path.exists(rgb_path):
        return
    img = cv2.imread(rgb_path)
    h, w = img.shape[:2]

    json_path = os.path.join(folder, "label.json")
    labels = []
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        rods = data.get("rods", [])
        # rods expected length up to 3
        for cls_id, rod in enumerate(rods):
            cx = float(rod["cx"])
            cy = float(rod["cy"])
            bw = float(rod["w"])
            bh = float(rod["h"])
            # convert to yolo (cx_norm, cy_norm, w_norm, h_norm)
            cxn = cx / w
            cyn = cy / h
            bwn = bw / w
            bhn = bh / h
            labels.append(f"{cls_id} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f}")

    # save image to OUT/images/<split>/
    dst_img = os.path.join(OUT, "images", split, f"{scene}.png")
    shutil.copyfile(rgb_path, dst_img)

    # save label file
    dst_lbl = os.path.join(OUT, "labels", split, f"{scene}.txt")
    with open(dst_lbl, "w") as f:
        if labels:
            f.write("\n".join(labels))
        else:
            # write empty file (no objects)
            f.write("")

for s in train_scenes:
    process_scene(s, "train")
for s in val_scenes:
    process_scene(s, "val")

# create dataset yaml
yaml_text = f"""train: {os.path.abspath(os.path.join(OUT, 'images/train'))}
val: {os.path.abspath(os.path.join(OUT, 'images/val'))}

nc: 3
names: ['rod0','rod1','rod2']
"""
with open(os.path.join(OUT, "dataset.yaml"), "w") as f:
    f.write(yaml_text)

print("转换完成，输出到:", OUT)
