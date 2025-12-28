import os
import cv2

def visualize_yolo_labels(img_dir, label_dir, save_dir="vis"):
    os.makedirs(save_dir, exist_ok=True)

    img_files = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if not os.path.exists(label_path):
            print(f"⚠️ 没有标签文件: {label_path}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            cls, xc, yc, ww, hh = map(float, line.strip().split())

            # 转回像素坐标
            xc *= w
            yc *= h
            ww *= w
            hh *= h

            x1 = int(xc - ww / 2)
            y1 = int(yc - hh / 2)
            x2 = int(xc + ww / 2)
            y2 = int(yc + hh / 2)

            # 每个类别用不同颜色
            color = [
                (255, 0, 0),   # class 0 → blue
                (0, 255, 0),   # class 1 → green
                (0, 0, 255),   # class 2 → red
                (0, 255, 255), # class 3 → yellow
                (255, 255, 0), # class 4 → cyan
            ][int(cls)]

            # 画框
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"cls {int(cls)}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)
        print(f"已保存可视化: {save_path}")


if __name__ == "__main__":
    visualize_yolo_labels(
        img_dir="/home/wangbingquan/flexiv_rdk/ropewrap/data/yolo_dataset/images/train",   # 图片目录
        label_dir="/home/wangbingquan/flexiv_rdk/ropewrap/data/yolo_dataset//labels/train", # 标签目录
        save_dir="/home/wangbingquan/flexiv_rdk/ropewrap/data/yolo_dataset/rod/vis"            # 输出目录
    )
