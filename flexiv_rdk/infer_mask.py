import torch
import cv2
from ropewrap.unet_tiny import UNet
import numpy as np

def infer(rgb_path, depth_path):
    net = UNet().cuda()
    net.load_state_dict(torch.load("rod_seg.pth"))
    net.eval()

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, -1)

    rgb_t = torch.from_numpy(rgb).permute(2,0,1).float()/255.
    depth_t = torch.from_numpy(depth).unsqueeze(0).float()/5000.

    rgb_t, depth_t = rgb_t.cuda()[None], depth_t.cuda()[None]

    with torch.no_grad():
        pred = net(rgb_t, depth_t)[0].cpu().numpy()

    pred = (pred*255).astype(np.uint8)

    for i in range(3):
        cv2.imwrite(f"mask_rod_{i+1}.png", pred[i])

    return pred