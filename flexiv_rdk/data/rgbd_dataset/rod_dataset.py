import os
import json
import numpy as np
import cv2
from torch.utils.data import Dataset

class RodDataset(Dataset):
    def __init__(self, root):
        """
        root/
            01/
                rgb.png
                depth.npy
                label.json (可选)
            02/
                ...
        """
        self.root = root
        self.scenes = sorted([d for d in os.listdir(root) if d.isdigit()])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        folder = os.path.join(self.root, scene)

        # --- load rgb ---
        rgb_path = os.path.join(folder, "rgb.png")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise RuntimeError(f"无法读取 {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # --- load depth ---
        depth_path = os.path.join(folder, "depth.npy")
        depth = np.load(depth_path)

        # --- load label.json ---
        json_path = os.path.join(folder, "label.json")
        masks = np.zeros((3, depth.shape[0], depth.shape[1]), dtype=np.uint8)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                label = json.load(f)

            rods = label.get("rods", [])

            # 按 cx（x 坐标）从小到大排序
            rods_sorted = sorted(rods, key=lambda r: r["cx"])

            for i, rod in enumerate(rods_sorted):
                if i >= 3:
                    break  # 只考虑三根杆
                cx, cy = rod["cx"], rod["cy"]
                w, h = rod["w"], rod["h"]
                angle = rod["angle"]

                rect = ((cx, cy), (w, h), angle)
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.fillConvexPoly(masks[i], box, 1)

        return rgb, depth, masks
