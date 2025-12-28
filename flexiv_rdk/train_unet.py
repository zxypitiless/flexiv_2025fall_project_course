import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ropewrap.data.rgbd_dataset.rod_dataset import RodDataset
from ropewrap.unet import UNet
import os
import re

# ============================
#   训练参数
# ============================
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 100
SAVE_DIR = "checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================
#   自动检测最新模型（如 unet_epoch_40.pt）
# ============================================
def find_latest_checkpoint():
    files = os.listdir(SAVE_DIR)

    # 匹配 unet_epoch_xx.pt
    ckpts = []
    for f in files:
        m = re.match(r"unet_epoch_(\d+)\.pt", f)
        if m:
            epoch = int(m.group(1))
            ckpts.append((epoch, f))

    if not ckpts:
        return None, 0

    # 根据 epoch 最大值排序
    ckpts.sort(key=lambda x: x[0], reverse=True)

    latest_epoch, latest_file = ckpts[0]
    return os.path.join(SAVE_DIR, latest_file), latest_epoch


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    # 数据集
    dataset = RodDataset("data/rgbd_dataset")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = UNet(n_channels=3, n_classes=3).to(device)

    # ============================================
    #   检查并加载最近的 checkpoint
    # ============================================
    ckpt_path, start_epoch = find_latest_checkpoint()

    if ckpt_path:
        print(f"🔍 检测到最近模型：{ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"👉 从 epoch {start_epoch} 继续训练\n")
    else:
        print("未发现 checkpoint，将从头开始训练。\n")
        start_epoch = 0

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ============================================
    #   正式训练
    # ============================================
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        total_loss = 0

        for rgb, depth, masks in loader:
            # if epoch == start_epoch :
                # print("Mask min:", masks.min().item())
                # print("Mask max:", masks.max().item())
                # print("Mask unique:", torch.unique(masks))
                # print("Mask shape:", masks.shape)  # 应是 B,3,H,W
                # print("Mask sum:", masks.sum(dim=(1,2)))  # 三个 channel 的像素总数


            rgb = rgb.float().permute(0, 3, 1, 2).to(device) / 255.0
            masks = masks.float().to(device)

            optimizer.zero_grad()
            pred = model(rgb)
            loss = criterion(pred, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss = {total_loss / len(loader):.4f}")

        # ============================================
        #   🔥 每 10 个 epoch 保存一次，不覆盖旧模型
        # ============================================
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(SAVE_DIR, f"unet_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
            }, save_path)
            print(f"✔ 已保存模型：{save_path}")
    



    print("\n🎉 训练完成！")


if __name__ == "__main__":
    train()
