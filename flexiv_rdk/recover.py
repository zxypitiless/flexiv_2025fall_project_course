#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import open3d as o3d
import os

# ===================== 配置 =====================
RGB_FILE = "/home/wangbingquan/flexiv_rdk/ropewrap/data/rgbd/01/rgb.png"
DEPTH_FILE = "/home/wangbingquan/flexiv_rdk/ropewrap/data/rgbd/01/depth.npy"

# D435i 内参
fx = 606.9369506835938
fy = 606.93359375
cx = 316.3077392578125
cy = 244.40113830566406

# ===================== 读取 RGB+D =====================
if RGB_FILE.endswith('.npy'):
    rgb = np.load(RGB_FILE)  # shape: H x W x 3
else:
    rgb = cv2.imread(RGB_FILE)
    if rgb is None:
        raise FileNotFoundError(f"无法读取图片文件: {RGB_FILE}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

depth = np.load(DEPTH_FILE)  # 假设单位为米，如果 uint16，需要 /1000.0
if depth.dtype == np.uint16:
    depth = depth.astype(np.float32) / 1000.0

H, W = depth.shape

# ===================== 像素 -> 相机坐标系 3D =====================
u = np.arange(W)
v = np.arange(H)
uu, vv = np.meshgrid(u, v)

Z = depth
X = (uu - cx) * Z / fx
Y = (vv - cy) * Z / fy
points_3d = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
colors = rgb.reshape(-1, 3) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors)

# ===================== 平移点云到相机原点附近 =====================
center = np.mean(points_3d, axis=0)
pcd.translate(-center)

# ===================== Open3D 可视化（翻转180度相机视角） =====================
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Camera View", width=800, height=600)
vis.add_geometry(pcd)

ctr = vis.get_view_control()
# 相机参数：从前看点云，沿Z轴负方向
camera_position = np.array([0.0, 0.0, 0.0])
camera_target = np.array([0.0, 0.0, 1.0])
camera_up = np.array([0.0, -1.0, 0.0])

# 翻转 180°：front 改为负方向
ctr.set_front(np.array([0.0, 0.0, -1.0]))
ctr.set_lookat(camera_target)
ctr.set_up(camera_up)
ctr.set_zoom(0.8)

vis.run()
vis.destroy_window()
