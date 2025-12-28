#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_alignment.py

用来验证手工标定的 T_base_cam 是否准确。

做法：
    1. 载入手工标定外参 T_base_cam.json
    2. 载入相机采集的 T_cam_board.json
    3. 计算 T_base_board = T_base_cam @ T_cam_board
    4. 打印棋盘格在机械臂基座下的位置（xyz）、欧拉角
"""

import json
import numpy as np
import os

# 路径（你可以根据需要修改）
CALIB_DIR = "/home/wangbingquan/flexiv_rdk/ropewrap/data/calib/"
T_BASE_CAM_FILE = os.path.join(CALIB_DIR, "T_base_cam.json")
T_CAM_BOARD_FILE = os.path.join(CALIB_DIR, "T_cam_board.json")


def rot_to_euler_xyz(R):
    """ 将旋转矩阵转换为XYZ欧拉角，角度制（度） """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.degrees([x, y, z])


def main():
    if not os.path.exists(T_BASE_CAM_FILE):
        print("❌ 未找到 T_base_cam.json：", T_BASE_CAM_FILE)
        return

    if not os.path.exists(T_CAM_BOARD_FILE):
        print("❌ 未找到 T_cam_board.json：", T_CAM_BOARD_FILE)
        return

    # 读取矩阵
    T_base_cam = np.array(json.load(open(T_BASE_CAM_FILE))["T_base_cam"], dtype=float)
    T_cam_board = np.array(json.load(open(T_CAM_BOARD_FILE))["T_cam_board"], dtype=float)

    # 计算棋盘格在机械臂基座下的位姿
    T_base_board = T_base_cam @ T_cam_board

    R = T_base_board[:3, :3]
    t = T_base_board[:3, 3]
    euler = rot_to_euler_xyz(R)

    print("\n====================== 验证结果 ======================")
    print("T_base_cam：")
    print(np.array2string(T_base_cam, precision=6, suppress_small=True))

    print("\nT_cam_board：")
    print(np.array2string(T_cam_board, precision=6, suppress_small=True))

    print("\n➡️ 计算得到：棋盘在机械臂基座下的位姿 T_base_board：")
    print(np.array2string(T_base_board, precision=6, suppress_small=True))

    print("\n📍 棋盘在基座坐标系下的位置 (m)：", np.round(t, 4))
    print("   x=前后, y=左右, z=上下")

    print("\n🎯 棋盘 Euler XYZ (deg)：", np.round(euler, 2))

    print("\n=======================================================\n")

    # 实际高度 sanity check
    z = t[2]
    if 0.10 < z < 1.20:
        print(f"✔ 棋盘高度 {z:.3f} m，在合理范围。标定大概率正确。\n")
    else:
        print(f"❌ 高度 {z:.3f} m 不合理，请检查标定。\n")


if __name__ == "__main__":
    main()