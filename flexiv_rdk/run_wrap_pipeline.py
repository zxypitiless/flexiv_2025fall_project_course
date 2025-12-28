from infer_mask import infer
from fit_axis import mask_to_pointcloud, fit_cylinder_axis
import json
import cv2
import numpy as np

def main():
    pred = infer("rgb.png", "depth.png")

    rgb = cv2.imread("rgb.png")
    depth = cv2.imread("depth.png", -1)
    depth = depth.astype(np.float32) / 1000.0

    intr = json.load(open("intrinsics.json"))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]

    results = []

    for i in range(3):
        pc = mask_to_pointcloud(pred[i], depth, (fx,fy,cx,cy))
        center, axis = fit_cylinder_axis(pc)

        results.append({
            "center": center.tolist(),
            "axis": axis.tolist()
        })

    json.dump(results, open("rod_axes.json","w"), indent=2)
    print("完成：rod_axes.json 已生成")

if __name__ == "__main__":
    main()