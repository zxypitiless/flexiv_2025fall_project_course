#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_fixed_T_base_cam.py

根据“已知的几何关系”直接生成相机到机械臂基座的外参 T_base_cam：
- 单位：你给的是毫米 mm，这里自动转成米 m
- 坐标轴：相机坐标轴与机械臂基座坐标轴完全平行
  * 相机看向 = 机械臂 X 正方向
  * 相机上方 = 机械臂 Z 正方向
  * 相机侧向 = 机械臂 Y 正方向
=> 旋转矩阵 R = I
"""

import os
import json
import numpy as np

# ========= 你给的相机位置（单位：mm）=========
px_mm = 40.0    # 相机相对基座原点 X 方向（前后）
py_mm = -160.0  # Y 方向（左右）
pz_mm = 65.0    # Z 方向（上下）
# ============================================

# 转换为米
t = np.array([px_mm / 1000.0,
              py_mm / 1000.0,
              pz_mm / 1000.0], dtype=float)

# 旋转矩阵：坐标轴完全平行 => 单位阵
R = np.eye(3, dtype=float)

# 组装 4x4 齐次矩阵
T = np.eye(4, dtype=float)
T[:3, :3] = R
T[:3, 3]  = t

SAVE_PATH = "/home/wangbingquan/flexiv_rdk/ropewrap/data/calib/T_base_cam.json"

def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "w") as f:
        json.dump({"T_base_cam": T.tolist()}, f, indent=2)

    print("✅ 已生成固定外参文件：", SAVE_PATH)
    print("T_base_cam =")
    print(np.array2string(T, precision=6, suppress_small=True))
    print("\n说明：")
    print("  - 位置 (m)：", np.round(t, 4))
    print("  - R = I（相机坐标轴与机械臂基座坐标轴平行）")
    print("  - X 轴：机械臂正前方；Y 轴：平行机械臂 Y；Z 轴：竖直向上")

if __name__ == "__main__":
    main()