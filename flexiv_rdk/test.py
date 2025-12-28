#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import cv2
import torch
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import pyrealsense2 as rs
import time
import argparse

from unet import UNet

# ===================== 配置 =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "/home/wangbingquan/flexiv_rdk/ropewrap/checkpoints1126/unet_epoch_90.pt"
OUT_DIR = "/home/wangbingquan/flexiv_rdk/ropewrap/test_output"
os.makedirs(OUT_DIR, exist_ok=True)

# D435i 内参
fx = 606.9369506835938
fy = 606.93359375
cx = 316.3077392578125
cy = 244.40113830566406

# 基座到相机的变换矩阵
CALIB_FILE = "/home/wangbingquan/flexiv_rdk/ropewrap/data/calib/T_base_cam.json"
with open(CALIB_FILE, "r") as f:
    T_base_cam = np.array(json.load(f)["T_base_cam"], dtype=np.float32)  # 4x4

# ===================== Flexiv RDK 相关 =====================
# Import Flexiv RDK Python library
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk

# ===================== RealSense 捕捉 =====================
def capture_rgbd_aligned(num_warm_frames=10):
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    frames = None
    for _ in range(num_warm_frames):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color = np.asanyarray(color_frame.get_data())
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth = depth_raw.astype(np.float32) * depth_frame.get_units()
    pipeline.stop()
    return color, depth

# ===================== 模型加载 =====================
def load_model():
    print("加载模型:", CHECKPOINT)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model = UNet(n_channels=3, n_classes=3)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE)
    model.eval()
    print("模型加载完成")
    return model

# ===================== mask 预测 =====================
def predict_mask(model, rgb):
    img = rgb.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    t = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(t)
        pred = torch.sigmoid(pred).squeeze(0).cpu().numpy()
    ch_max = np.argmax([pred[c].max() for c in range(pred.shape[0])])
    mask = pred[ch_max]
    return mask

# ===================== 热力图 =====================
def visualize_prob_map(mask, save_path):
    norm_mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-6)
    prob_map = (norm_mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_map, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, heatmap)
    print("保存热力图:", save_path)

# ===================== mask+depth -> 3D 点云 =====================
def mask_to_pointcloud(mask, depth, fx, fy, cx, cy,
                       prob_thresh=0.3, min_depth=0.4, max_depth=0.8):

    ys, xs = np.where(mask > prob_thresh)
    if len(xs) == 0:
        return np.zeros((0, 3))

    z_vals = depth[ys, xs]

    valid = (z_vals >= min_depth) & (z_vals <= max_depth)
    if not np.any(valid):
        return np.zeros((0, 3))

    xs = xs[valid]
    ys = ys[valid]
    z_vals = z_vals[valid]

    X = (xs - cx) * z_vals / fx
    Y = (ys - cy) * z_vals / fy
    Z = z_vals
    return np.stack([X, Y, Z], axis=1)

# ===================== DBSCAN + PCA 拟合圆柱 =====================
def cluster_rods(points):
    clustering = DBSCAN(eps=0.015, min_samples=20).fit(points)
    labels = clustering.labels_
    clusters = [points[labels==l] for l in set(labels) if l != -1 and points[labels==l].shape[0]>80]
    clusters = sorted(clusters, key=lambda x: -x.shape[0])[:3]
    return clusters

def fit_cylinder(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    axis = pca.components_[0]/np.linalg.norm(pca.components_[0])
    pts_proj = points - np.dot(points - points.mean(axis=0), axis)[:, None]*axis
    center = pts_proj.mean(axis=0)
    v = points - center
    radial = v - np.dot(v, axis)[:, None]*axis
    radius = np.mean(np.linalg.norm(radial, axis=1))
    t = np.dot(points - center, axis)
    t_min, t_max = t.min(), t.max()
    line_end = center + t_min*axis
    line_start = center + t_max*axis
    return dict(center=center, axis=axis, radius=radius,
                length=np.linalg.norm(line_end-line_start),
                line_start=line_start, line_end=line_end)

def fit_three_rods(points_3d):
    clusters = cluster_rods(points_3d)
    rods = []
    for i, pts in enumerate(clusters):
        rod = fit_cylinder(pts)
        rods.append(rod)
    return rods

# ===================== 可视化 =====================
def visualize_pointcloud_with_rods(points_3d, rods):
    pts_3d_vis = points_3d.copy()
    pts_3d_vis[:, 1] *= -1  # 翻转Y轴

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d_vis)
    colors = np.tile([0.8, 0.2, 0.2], (pts_3d_vis.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    lines = []
    line_colors = []
    pts = []

    for r in rods:
        line_start = r["line_start"].copy()
        line_end = r["line_end"].copy()
        line_start[1] *= -1
        line_end[1] *= -1

        pts.append(line_start)
        pts.append(line_end)
        lines.append([len(pts)-2, len(pts)-1])
        line_colors.append([0, 1, 0])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(pts)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    o3d.visualization.draw_geometries([pcd, line_set])

# ===================== 基座坐标系 =====================
def transform_rods_to_base(rods, T_base_cam):
    rods_base = []
    for i, rod in enumerate(rods):
        center_hom = np.append(rod["center"], 1.0)
        center_base = (T_base_cam @ center_hom)[:3]
        axis_base = T_base_cam[:3,:3] @ rod["axis"]

        rod_base = rod.copy()
        rod_base["center_base"] = center_base
        rod_base["axis_base"] = axis_base
        rods_base.append(rod_base)

        print(f"\n==== Rod {i} 基座坐标系下结果 ====")
        print("中心 (base frame):", center_base)
        print("轴向 (base frame):", axis_base)
        print("半径: %.4f m" % rod["radius"])
        print("长度: %.4f m" % rod["length"])
    return rods_base

# ===================== 计算三角形的质心 =====================
def compute_triangle_center(rods_base):
    centers = [rod["center_base"] for rod in rods_base]
    center_triangle = np.mean(centers, axis=0)
    return center_triangle

# ===================== 计算目标位置 =====================
def compute_target_position(center_triangle, offset_z=0.05):
    target_position = center_triangle + np.array([0, 0, offset_z])
    return target_position

# ===================== Flexiv RDK 控制机械臂 =====================
def move_arm_to_target_position(robot, target_position, timeout=15.0):
    mode = flexivrdk.Mode
    robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)

    target_tcp = f"{target_position[0]} {target_position[1]} {target_position[2]} 180 0 180 WORLD WORLD_ORIGIN"
    print(f"执行 MoveL: {target_tcp}")

    robot.executePrimitive(f"MoveL(target={target_tcp})")

    # 等待机械臂到达目标（检测 reachedTarget）
    start_time = time.time()
    while True:
        prim_states = robot.getPrimitiveStates()
        reached = [s for s in prim_states if "reachedTarget=1" in s]
        if reached or (time.time() - start_time) > timeout:
            break
        time.sleep(0.1)

    if reached:
        print("目标位置已到达")
    else:
        print("⚠ 等待超时，机械臂可能未完全到达目标")

    # 停止机械臂，并切换到 IDLE 模式
    try:
        robot.stop()
        robot.setMode(mode.IDLE)
        print("机械臂控制已停止，切换到 IDLE 模式")
    except Exception as e:
        print("停止机械臂时出错:", e)



# ===================== 主函数 =====================
def main():
    model = load_model()
    rgb_bgr, depth = capture_rgbd_aligned()
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    print("捕捉到相机图像")

    mask = predict_mask(model, rgb)
    visualize_prob_map(mask, os.path.join(OUT_DIR, "camera_prob.png"))

    pts_3d = mask_to_pointcloud(mask, depth, fx, fy, cx, cy,
                                prob_thresh=0.3, min_depth=0.4, max_depth=0.8)
    print("点云数量:", pts_3d.shape[0])

    if pts_3d.shape[0] >= 50:
        rods = fit_three_rods(pts_3d)
        rods_base = transform_rods_to_base(rods, T_base_cam)
        
        center_triangle = compute_triangle_center(rods_base)
        target_position = compute_target_position(center_triangle, offset_z=0.20)

        print("目标位置 (基座坐标系):", target_position)

        # Initialize robot and move arm to target position
        robot_ip = "192.168.2.100"  # 替换为你的机器人IP
        local_ip = "192.168.2.102"  # 替换为你的本地IP
        robot = flexivrdk.Robot(robot_ip, local_ip)

        robot.enable()
        while not robot.isOperational():
            time.sleep(1)

        # move_arm_to_target_position(robot, target_position)
        

        visualize_pointcloud_with_rods(pts_3d, rods)
    else:
        print("⚠ 点太少，无法拟合")

    print("完成识别，结果保存到:", OUT_DIR)

if __name__ == "__main__":
    main()
