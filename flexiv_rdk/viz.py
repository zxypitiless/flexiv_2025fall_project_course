# ropewrap/viz.py  —— 无 GUI Headless 渲染版
import os
os.environ["OPEN3D_CPU_ONLY"] = "True"
os.environ["LIBGL_ALWAYS_INDIRECT"] = "1"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import open3d as o3d
import numpy as np

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_pointcloud_as_file(pcd, filename="debug_pointcloud.ply"):
    """保存点云到文件"""
    full_path = f"{OUTPUT_DIR}/{filename}"
    o3d.io.write_point_cloud(full_path, pcd)
    print(f"✅ 点云已保存：{full_path}")


def save_cylinder_fit_result(pcd, cylinders, filename="cylinder_fit.png"):
    """
    保存杆子拟合结果，可能有多个杆
    cylinders: [(center, axis, radius), ...]
    """
    import matplotlib.pyplot as plt

    pts = np.asarray(pcd.points)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)

    for idx, (center, axis, radius) in enumerate(cylinders):
        ax.text(center[0], center[1], center[2], f"Rod{idx}", color='r')

    full_path = f"{OUTPUT_DIR}/{filename}"
    plt.savefig(full_path)
    plt.close()
    print(f"✅ 杆识别结果保存：{full_path}")