#!/usr/bin/env python3
import cv2
import json
import os
import numpy as np
from pathlib import Path

DATA_ROOT = "/home/wangbingquan/flexiv_rdk/ropewrap/data/rgbd_dataset"

current_points = []      # 当前杆的 4 个点
all_rectangles = []      # 所有杆拟合出的旋转矩形
img = None
img_disp = None


def fit_rotated_rect(pts):
    pts_np = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(pts_np)   # ((cx,cy),(w,h),angle)
    return rect


def draw_rotated_rect(image, rect, color=(0, 255, 0)):
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)   # ← 修复 np.int0 报错
    cv2.polylines(image, [box], True, color, 2)
    return image


def on_mouse(event, x, y, flags, param):
    global current_points, img_disp, img, all_rectangles

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])
        print(f"👉 点击点 {len(current_points)}: ({x}, {y})")

        cv2.circle(img_disp, (x, y), 3, (0, 255, 0), -1)

        if len(current_points) == 4:
            rect = fit_rotated_rect(current_points)
            all_rectangles.append(rect)

            print(f"✔ 已拟合第 {len(all_rectangles)} 根杆:")
            print(rect)

            draw_rotated_rect(img_disp, rect, (0, 0, 255))

            current_points = []  # 重置


def save_labels(folder, rects, img):
    out = {"rods": []}

    for rect in rects:
        ((cx, cy), (w, h), angle) = rect
        out["rods"].append({
            "cx": float(cx),
            "cy": float(cy),
            "w": float(w),
            "h": float(h),
            "angle": float(angle)
        })

    json_path = os.path.join(folder, "label.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"💾 已保存标注 JSON: {json_path}")

    vis = img.copy()
    for rect in rects:
        draw_rotated_rect(vis, rect, (255, 0, 0))
    vis_path = os.path.join(folder, "label_vis.png")
    cv2.imwrite(vis_path, vis)
    print(f"🖼️ 已保存可视化标签: {vis_path}")


def label_folder(folder):
    global img, img_disp, current_points, all_rectangles

    rgb_path = os.path.join(folder, "rgb.png")
    if not os.path.exists(rgb_path):
        print(f"跳过（没有 rgb.png）: {folder}")
        return

    print(f"\n========== 标注文件夹：{folder} ==========\n")
    print("操作说明：")
    print("  - 点击 4 个点 → 自动生成旋转矩形（拟合杆）")
    print("  - 一共要标注 3 根杆（12 个点）")
    print("  - 按 N 保存并进入下一组")
    print("  - 按 R 重置当前标注")
    print("  - 按 ESC 退出\n")

    img = cv2.imread(rgb_path)
    img_disp = img.copy()

    current_points = []
    all_rectangles = []

    cv2.namedWindow("Label")
    cv2.setMouseCallback("Label", on_mouse)

    while True:
        cv2.imshow("Label", img_disp)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('r'):
            print("↩ 重置所有标注")
            img_disp = img.copy()
            current_points = []
            all_rectangles = []

        elif key == ord('n'):
            if len(all_rectangles) != 3:
                print(f"❌ 当前只有 {len(all_rectangles)} 根杆，必须是 3 根")
                continue
            save_labels(folder, all_rectangles, img)
            break

        elif key == 27:
            print("ESC 退出标注")
            break

    cv2.destroyWindow("Label")


def main():
    folders = sorted([f for f in os.listdir(DATA_ROOT) if f.isdigit()])
    print(f"发现 {len(folders)} 个数据文件夹，将逐个标注。\n")

    for f in folders:
        label_folder(os.path.join(DATA_ROOT, f))

    print("\n🎉 全部标注完成！")


if __name__ == "__main__":
    main()