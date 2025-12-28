#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from ropewrap.unet import UNet

# ===================== 配置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "/home/wangbingquan/flexiv_rdk/ropewrap/checkpoints1126/unet_epoch_60.pt"
DATA_DIR = "/home/wangbingquan/flexiv_rdk/ropewrap/data/rgbd_dataset"
OUT_DIR = "/home/wangbingquan/flexiv_rdk/ropewrap/test_output"
os.makedirs(OUT_DIR, exist_ok=True)
# ==================================================

# -------------------- 测试数据集（无标签） --------------------
class RodDatasetTest:
    def __init__(self, root):
        self.root = root
        self.scenes = sorted([d for d in os.listdir(root) if d.isdigit()])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        folder = os.path.join(self.root, scene)

        # load RGB
        rgb_path = os.path.join(folder, "rgb.png")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            raise RuntimeError(f"无法读取 {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # load depth (可选，用于后续扩展)
        depth_path = os.path.join(folder, "depth.npy")
        depth = np.load(depth_path)

        return rgb, depth

# -------------------- 载入模型 --------------------
def load_model():
    print("🔍 加载模型中:", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    state_dict = ckpt["model"]
    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("✅ 模型加载完成")
    return model

# -------------------- 推理函数 --------------------
def predict_mask(model, rgb):
    """ 输入 RGB 图像，输出模型预测 mask """
    img = rgb.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > 0.5).cpu().numpy()[0]  # shape (3,H,W)
    return pred_mask

# -------------------- 可视化 --------------------
def visualize_mask(rgb, pred_mask, save_path):
    """根据预测的 mask 用红绿蓝标出三根杆"""
    rgb_vis = rgb.copy()

    colors = [
        (0, 0, 255),   # 红
        (0, 255, 0),   # 绿
        (255, 0, 0)    # 蓝
    ]

    for i in range(3):
        mask = pred_mask[i]
        mask = mask.astype(np.uint8)
        color_layer = np.zeros_like(rgb)
        color_layer[:, :, 0] = colors[i][2]
        color_layer[:, :, 1] = colors[i][1]
        color_layer[:, :, 2] = colors[i][0]

        rgb_vis = np.where(mask[..., None] > 0, 
                           0.7 * rgb_vis + 0.3 * color_layer,
                           rgb_vis)
    cv2.imwrite(save_path, rgb_vis)
    print(f"✔ 已保存可视化: {save_path}")

# -------------------- 主函数 --------------------
def main():
    model = load_model()

    dataset = RodDatasetTest(DATA_DIR)
    print(f"数据集大小: {len(dataset)} 组")

    for idx in range(len(dataset)):
        rgb, depth = dataset[idx]

        pred_mask = predict_mask(model, rgb)

        save_path = os.path.join(OUT_DIR, f"vis_{idx:02d}.png")
        visualize_mask(rgb, pred_mask, save_path)

    print("\n🎉 全部测试完成，结果保存在：", OUT_DIR)

if __name__ == "__main__":
    main()
