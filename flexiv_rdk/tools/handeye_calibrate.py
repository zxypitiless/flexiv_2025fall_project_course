#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
handeye_calibrate.py
从手动采集的多组数据计算 T_base_cam（相机->机械臂基座）。

数据结构（每组必须含有这两个文件）：
  /home/wangbingquan/flexiv_rdk/ropewrap/data/calib/01/T_cam_board.json
  /home/wangbingquan/flexiv_rdk/ropewrap/data/calib/01/ee_pose.json
  ...
说明：
- T_cam_board.json 来自相机PnP（相机->棋盘）
- 本脚本会自动取逆：T_board_cam = inv(T_cam_board)
- ee_pose.json 的 quat 默认顺序为 [qx, qy, qz, qw]
- 兼容 OpenCV 不同版本（有的返回 (R,t)，有的返回 (retval,R,t)）
输出：
  /home/wangbingquan/flexiv_rdk/ropewrap/data/calib/T_base_cam.json
"""

import os, json, glob
import numpy as np
import cv2

DATASET_DIR = "/home/wangbingquan/flexiv_rdk/ropewrap/data/calib"
SAVE_PATH   = "/home/wangbingquan/flexiv_rdk/ropewrap/data/calib/T_base_cam.json"

# ---------- 工具函数 ----------
def quat_xyzw_to_R(qx, qy, qz, qw):
    """四元数 [qx,qy,qz,qw] -> 旋转矩阵"""
    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Invalid quaternion norm.")
    qx, qy, qz, qw = q / n
    R = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)]
    ])
    return R

def solve_handeye(R_g2b, t_g2b, R_t2c, t_t2c):
    """兼容不同OpenCV版本的 calibrateHandEye 返回值"""
    ret = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_TSAI)
    if isinstance(ret, tuple) and len(ret) == 3:
        _, R, t = ret
    else:
        R, t = ret
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t.squeeze()
    return T

# ---------- 主流程 ----------
def main():
    folders = sorted([p for p in glob.glob(os.path.join(DATASET_DIR, "*")) if os.path.isdir(p)])
    if not folders:
        print("❌ 没找到任何数据组：", DATASET_DIR)
        return

    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []

    used, skipped = 0, 0
    for folder in folders:
        ee_path  = os.path.join(folder, "ee_pose.json")
        cam_path = os.path.join(folder, "T_cam_board.json")
        if not (os.path.exists(ee_path) and os.path.exists(cam_path)):
            skipped += 1
            print(f"⚠️ 跳过 {os.path.basename(folder)}（缺少 ee_pose 或 T_cam_board）")
            continue

        ee = json.load(open(ee_path))
        qx, qy, qz, qw = [float(v) for v in ee["quat"]]  # 默认顺序 [qx,qy,qz,qw]
        R_be = quat_xyzw_to_R(qx, qy, qz, qw)
        t_be = np.array(ee["xyz"], dtype=float).reshape(3,1)

        T_cam_board = np.array(json.load(open(cam_path))["T_cam_board"], dtype=float)
        # hand-eye需求：target(棋盘)->camera
        T_board_cam = np.linalg.inv(T_cam_board)
        R_bc = T_board_cam[:3, :3]
        t_bc = T_board_cam[:3, 3].reshape(3,1)

        R_g2b.append(R_be);  t_g2b.append(t_be)
        R_t2c.append(R_bc);  t_t2c.append(t_bc)
        used += 1

    if used < 5:
        print(f"❌ 有效数据太少：仅 {used} 组，至少 5 组以上")
        return

    print(f"📂 有效数据 {used} 组（跳过 {skipped} 组），开始标定 ...")

    T_base_cam = solve_handeye(R_g2b, t_g2b, R_t2c, t_t2c)

    # 如果高度为负，尝试翻转一次（部分数据集会出现对称解）
    if T_base_cam[2,3] < 0:
        R_t2c_flip = [R @ np.diag([1,1,-1]) for R in R_t2c]
        T_try = solve_handeye(R_g2b, t_g2b, R_t2c_flip, t_t2c)
        if T_try[2,3] > 0 or abs(T_try[2,3]) > abs(T_base_cam[2,3]):
            T_base_cam = T_try

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    json.dump({"T_base_cam": T_base_cam.tolist()}, open(SAVE_PATH, "w"), indent=2)

    print("\n✅ 标定完成，结果保存：", SAVE_PATH)
    print("T_base_cam =\n", np.array2string(T_base_cam, precision=5, suppress_small=True))
    print("相机高度(约)：%.3f m" % T_base_cam[2,3])

if __name__ == "__main__":
    main()