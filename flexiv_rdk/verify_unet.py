import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from ropewrap.unet import UNet
from ropewrap.data.rgbd_dataset.rod_dataset import RodDataset

MODEL_PATH = "checkpoints/unet_epoch_30.pt"
DATA_ROOT = "data/rgbd_dataset"
SAVE_DIR = "output_vis"

os.makedirs(SAVE_DIR, exist_ok=True)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    model = UNet(n_channels=3, n_classes=3).to(device)

    ckpt = torch.load(MODEL_PATH, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model, device


def visualize_mask(rgb, pred_mask, save_path):
    """把 3 通道 mask 渲染到 RGB 上"""

    rgb_vis = rgb.copy()

    # 三根杆分别用三种颜色
    colors = [
        (0, 0, 255),   # 红
        (0, 255, 0),   # 绿
        (255, 0, 0)    # 蓝
    ]

    h, w = rgb.shape[:2]
    for i in range(3):
        mask = pred_mask[i]
        mask = cv2.resize(mask, (w, h))

        color_layer = np.zeros_like(rgb)
        color_layer[:, :, 0] = colors[i][2]
        color_layer[:, :, 1] = colors[i][1]
        color_layer[:, :, 2] = colors[i][0]

        rgb_vis = np.where(mask[..., None] > 0.5, 
                           0.7 * rgb_vis + 0.3 * color_layer,
                           rgb_vis)

    cv2.imwrite(save_path, rgb_vis)
    print(f"✔ 已保存可视化: {save_path}")


def main():
    model, device = load_model()

    # 只取一组数据测试，比如 01
    dataset = RodDataset(DATA_ROOT)
    print(f"数据集大小: {len(dataset)} 组")

    # 测试多张，比如每一组
    for idx in range(len(dataset)):
        rgb, depth, mask_gt = dataset[idx]

        rgb_input = torch.tensor(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        rgb_input = rgb_input.to(device)

        # 推理
        with torch.no_grad():
            pred = model(rgb_input)
            pred = torch.sigmoid(pred)
            pred = pred.squeeze(0).cpu().numpy()  # (3,H,W)

        # 可视化
        save_path = os.path.join(SAVE_DIR, f"vis_{idx:02d}.png")
        visualize_mask(rgb, pred, save_path)


if __name__ == "__main__":
    main()