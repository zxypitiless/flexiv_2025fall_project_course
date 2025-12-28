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
    line_start = center + t_min*axis
    line_end = center + t_max*axis
    return dict(center=center, axis=axis, radius=radius,
                length=np.linalg.norm(line_end-line_start),
                line_start=line_start, line_end=line_end)

def fit_three_rods(points_3d):
    clusters = cluster_rods(points_3d)
    rods = []
    for pts in clusters:
        rod = fit_cylinder(pts)
        rods.append(rod)
    return rods

# ===================== 基座坐标系 =====================
def transform_rods_to_base(rods, T_base_cam):
    rods_base = []
    for i, rod in enumerate(rods):
        center_hom = np.append(rod["center"], 1.0)
        line_start_hom = np.append(rod["line_start"], 1.0)
        line_end_hom = np.append(rod["line_end"], 1.0)

        rod_base = rod.copy()
        rod_base["center_base"] = (T_base_cam @ center_hom)[:3]
        rod_base["axis_base"] = T_base_cam[:3,:3] @ rod["axis"]
        rod_base["line_start_base"] = (T_base_cam @ line_start_hom)[:3]
        rod_base["line_end_base"] = (T_base_cam @ line_end_hom)[:3]

        rods_base.append(rod_base)
        print(f"\n==== Rod {i} 基座坐标系下结果 ====")
        print("中心 (base frame):", rod_base["center_base"])
        print("轴向 (base frame):", rod_base["axis_base"])
        print("半径: %.4f m" % rod["radius"])
        print("长度: %.4f m" % rod["length"])
    return rods_base

# ===================== 计算三杆中心轴 =====================
def compute_center_axis(rods_base):
    centers = [rod["center_base"] for rod in rods_base]
    axes = [rod["axis_base"] for rod in rods_base]
    axis_mean = np.mean(axes, axis=0)
    axis_mean /= np.linalg.norm(axis_mean)
    center_point = np.mean(centers, axis=0)
    return center_point, axis_mean

# ===================== 计算最终目标点 =====================
def compute_final_target_v2(center_point, center_axis, rods_base):
    """
    新版最终目标点计算方式：
    1. 找距离基座原点最近的杆的几何中心 Rmin
    2. 计算从 center_point 指向 Rmin 的方向向量 D
    3. C20 = center_point + 0.20 * center_axis
    4. final = C20 + 0.15 * D
    """
    # 找距离基座原点最近的杆
    dists = [np.linalg.norm(rod["center_base"]) for rod in rods_base]
    idx = np.argmin(dists)
    Rmin = rods_base[idx]["center_base"]

    # C0 = center_point
    C0 = center_point

    # 单位方向向量 D
    direction = Rmin - C0
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    # C20 = C0 + 20cm
    C20 = C0 + 0.20 * center_axis

    # 最终目标 = C20 + 15cm * D
    final_target = C20 + 0.15 * direction

    return final_target


# ===================== 根据三杆中心轴法向量计算 R、P =====================
def compute_RP_from_axis(axis):
    ax, ay, az = axis

    # 保证 z > 0
    if az < 0:
        ax, ay, az = -ax, -ay, -az

    # ---------- 计算 R ----------
    # R：法向量与 X-O-Z 平面的夹锐角
    r = np.degrees(np.arcsin(abs(ay) / np.linalg.norm([ax, ay, az])))

    if ay >= 0:
        R = 180 - r
    else:
        R = 180 + r

    # ---------- 计算 P ----------
    # P：法向量与 Y-O-Z 平面的夹锐角
    p = np.degrees(np.arcsin(abs(ax) / np.linalg.norm([ax, ay, az])))

    if ax < 0:
        P = -p
    else:
        P = p

    # Y 角不需要确定，默认保持 0
    Y = 0

    return R, Y, P

# ===================== 控制机械臂移动 =====================
def move_arm_to_target_position(robot, target_position, R, P, Y, timeout):
    """
    机械臂移动到指定目标点，使用给定的欧拉角方向
    """
    target_tcp = f"{target_position[0]} {target_position[1]} {target_position[2]} {R} {P} {Y} WORLD WORLD_ORIGIN"
    cmd = f"MoveL(target={target_tcp})"
    print(f"MoveL to: {target_tcp}")
    robot.executePrimitive(cmd)

    start_time = time.time()
    while True:
        states = robot.getPrimitiveStates()
        if "reachedTarget=1" in "".join(states) or (time.time() - start_time) > timeout:
            break
        time.sleep(0.01)

    if "reachedTarget=1" in "".join(states):
        print("目标位置已到达")
    else:
        print("⚠ 等待超时，机械臂可能未完全到达目标")

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

# ===================== 生成螺旋轨迹 =====================
def generate_spiral_along_axis(center_point, center_axis, final_target, 
                               pitch=0.05, turns=2.0, num_points=361):
    """
    生成沿三杆中心轴的螺旋线轨迹。
    半径 R 自动计算为 final_target 到 center_axis 所在直线的垂直距离。
    
    参数:
    final_target (np.ndarray): 螺旋线的起点 P0 (基座坐标系)。
    center_point (np.ndarray): 轴线上经过的一个点 C (基座坐标系)。
    center_axis (np.ndarray): 螺旋线轴线的方向向量 A (已归一化或未归一化均可)。
    pitch (float): 螺距 H (m)。
    turns (float): 圈数 (默认 2.0)。
    num_points (int): 采样点数量 (默认 361)。
    
    返回:
    np.ndarray: N x 3 的路径点坐标数组。
    """
    
    # 归一化轴线向量 u
    u = center_axis / np.linalg.norm(center_axis)
    u = -u

    # 1. 确定起始径向向量 R0 和投影点 C_proj
    # 计算起点 final_target 到轴线的垂直投影点 C_proj
    CP0 = final_target - center_point
    C_proj = center_point + np.dot(CP0, u) * u
    
    # 确定起始径向向量 R0 (从投影点 C_proj 指向起点 P0)
    R0 = final_target - C_proj
    
    # 自动计算半径 R
    R = np.linalg.norm(R0)
    if R < 1e-6:
        print("⚠ 警告: 自动计算半径过小，可能导致几何错误。")
        return np.array([final_target]) # 返回起点，避免错误

    # 2. 计算正交基 R90
    R90_unnormalized = np.cross(u, R0)
    # R90 必须与 R0 垂直，且长度与 R0 (半径 R) 相等
    R99_norm = np.linalg.norm(R90_unnormalized)
    if R99_norm < 1e-6:
         # 如果叉积接近零，说明 R0 和 u 共线 (不应发生)，临时构造一个
         # 这种情况通常发生在 final_target 已经接近轴线时，前面 R < 1e-6 已经处理
         R90 = np.cross(np.array([1, 0, 0]), u)
         if np.linalg.norm(R90) < 1e-6: R90 = np.cross(np.array([0, 1, 0]), u)
         R90 = (R90 / np.linalg.norm(R90)) * R
    else:
        R90 = (R90_unnormalized / R99_norm) * R # 归一化再乘半径 R
    
    # 3. 参数化螺旋线
    max_angle = turns * 2.0 * np.pi 
    thetas = np.linspace(0, max_angle, num_points)
    pitch_factor = pitch / (2.0 * np.pi)

    points = []
    for theta in thetas:
        # 径向分量 (旋转)
        R_theta = np.cos(theta) * R0 + np.sin(theta) * R90
        
        # 轴向分量 (平移)
        Z_theta = pitch_factor * theta * u
        
        # 螺旋线上的点 P(theta)
        P_theta = C_proj + R_theta + Z_theta
        points.append(P_theta)
        
    return np.array(points)



# ===================== 执行螺旋轨迹 =====================
def execute_spiral_trajectory(robot, spiral_points, R, P, Y, max_vel=0.15, timeout=0.2):
    """
    依次控制机械臂沿螺旋线轨迹移动
    """
    for i, pt in enumerate(spiral_points):
        print(f"[螺旋线] Move to point {i+1}/{len(spiral_points)}: {pt}")
        move_arm_to_target_position(robot, pt, R, P, Y, timeout=timeout)
    print("螺旋线轨迹执行完成")


# ===================== 可视化 (修正 u, v 向量逻辑) =====================
def visualize_spiral_with_plane(center_point, center_axis, final_target, spiral_points, radius=None):
    """
    可视化三杆中心轴、最终目标点、螺旋线，以及圆周平面和两个垂直向量 R0, R90
    R0: 起始径向向量 (半径)
    R90: 垂直于 R0 和 center_axis 的向量 (半径)
    """
    
    # 1. 绘制螺旋线点 (已有的逻辑)
    spiral_pcd = o3d.geometry.PointCloud()
    spiral_pcd.points = o3d.utility.Vector3dVector(spiral_points)
    spiral_pcd.paint_uniform_color([1, 0, 0])  # 红色

    # 2. 绘制中心轴线 (已有的逻辑)
    axis = center_axis / np.linalg.norm(center_axis)
    axis_len = 0.5
    axis_line_start = center_point - axis*axis_len/2
    axis_line_end = center_point + axis*axis_len/2
    axis_lines = [[0,1]]
    axis_pts = np.vstack([axis_line_start, axis_line_end])
    axis_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axis_pts),
        lines=o3d.utility.Vector2iVector(axis_lines)
    )
    axis_line_set.colors = o3d.utility.Vector3dVector([[0,1,0]])  # 绿色

    # 3. 绘制起点（最终目标点）(已有的逻辑)
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    start_sphere.paint_uniform_color([0,0,1])
    start_sphere.translate(final_target)

    # 4. **核心修改：重建起始径向向量 R0 和 R90**
    u = axis # 轴线单位向量
    CP0 = final_target - center_point
    C_proj = center_point + np.dot(CP0, u) * u
    R0 = final_target - C_proj # 起始径向向量

    R = np.linalg.norm(R0) # 实际半径
    if radius is None: # 如果没有显式半径，使用计算出的半径
        radius = R 

    R90_unnormalized = np.cross(u, R0)
    R99_norm = np.linalg.norm(R90_unnormalized)
    if R99_norm < 1e-6: # 避免除零
         R90 = np.array([0, 0, 0])
    else:
        R90 = (R90_unnormalized / R99_norm) * R # 归一化再乘半径 R
    
    # 5. 可视化 R0, R90 向量
    # R0 向量 (指向起点)
    R0_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([C_proj, C_proj + R0])),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    R0_line.colors = o3d.utility.Vector3dVector([[1,1,0]]) # 黄色

    # R90 向量
    R90_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.vstack([C_proj, C_proj + R90])),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    R90_line.colors = o3d.utility.Vector3dVector([[1,0,1]]) # 紫色

    # 6. 绘制圆周平面 (以 C_proj 为中心)
    num_circle = 50
    theta = np.linspace(0, 2*np.pi, num_circle)
    # 这里的 circle_pts 应该使用 R0 和 R90 作为基底，而不是用 center_point
    circle_pts = C_proj + R * np.outer(np.cos(theta), R0/R) + R * np.outer(np.sin(theta), R90/R)
    circle_pcd = o3d.geometry.PointCloud()
    circle_pcd.points = o3d.utility.Vector3dVector(circle_pts)
    circle_pcd.paint_uniform_color([0,1,1])  # 青色

    # 7. 可视化所有
    o3d.visualization.draw_geometries([
        spiral_pcd, 
        axis_line_set, 
        start_sphere, 
        circle_pcd,
        R0_line, 
        R90_line
    ])


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

        # 计算三杆中心轴
        center_point, center_axis = compute_center_axis(rods_base)
        if center_axis[2] < 0:
            center_axis = -center_axis
        first_point = center_point + center_axis * 0.20  # 沿中心轴正方向20cm
        first_point[0] = first_point[0] + 0.01
        print("三杆中心轴点 (基座坐标系):", first_point)
        # 计算 R、P
        R, Y, P = compute_RP_from_axis(center_axis)
        print("根据法向量计算出的 R,Y,P =", R, Y, P)


        # 最终目标点
        final_target = compute_final_target_v2(center_point, center_axis, rods_base)
        print("最终目标点 (基座坐标系):", final_target)


        # 初始化机械臂
        robot_ip = "192.168.2.100"  # 替换为你的机器人IP
        local_ip = "192.168.2.102"  # 替换为你的本地IP
        robot = flexivrdk.Robot(robot_ip, local_ip)

    # 1. 清除故障
    if robot.isFault():
        print("⚠ Robot is in fault state, trying to clear...")
        robot.clearFault()
        time.sleep(2)
        if robot.isFault():
            raise RuntimeError("Robot fault cannot be cleared. Please check the robot.")
        print("Robot fault cleared.")

    # 2. Enable 并等待 operational
    robot.enable()
    while not robot.isOperational():
        time.sleep(0.5)

    # 3. 设置 NRT_PRIMITIVE_EXECUTION 模式（全局一次即可）
    robot.setMode(flexivrdk.Mode.NRT_PRIMITIVE_EXECUTION)

    # 假设 R, P, Y 已经根据中心轴计算好
    # 机械臂先到三杆中心轴点，再到最终目标点
    move_arm_to_target_position(robot, first_point, R, P, Y, timeout=2)
    move_arm_to_target_position(robot, final_target, R, P, Y, timeout=2)

    # 生成螺旋轨迹
    spiral_points = generate_spiral_along_axis(center_point, center_axis, final_target,
                                                pitch=0.05, turns=3, num_points=50)
    # 执行螺旋轨迹
    execute_spiral_trajectory(robot, spiral_points, R, P, Y)

    # 可视化
    visualize_spiral_with_plane(center_point, center_axis, final_target, spiral_points)

    visualize_pointcloud_with_rods(pts_3d, rods)

if __name__ == "__main__":
    main()