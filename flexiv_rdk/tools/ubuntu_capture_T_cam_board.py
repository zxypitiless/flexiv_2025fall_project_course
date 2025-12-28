#!/usr/bin/env python3
"""
ubuntu_capture_T_cam_board.py
从 RealSense 相机采集棋盘格图像，计算相机->棋盘的外参矩阵 T_cam_board，
并保存为 JSON 文件到 /home/wangbingquan/flexiv_rdk/ropewrap/data/calib/T_cam_board.json
"""

import pyrealsense2 as rs
import numpy as np
import cv2, json, os

# ==================== 参数设置 ====================
cols, rows = 8, 8           # 棋盘格角点数（内角点）
square = 0.018              # 方格边长（米）
save_dir = "/home/wangbingquan/flexiv_rdk/ropewrap/data/calib"
save_path = os.path.join(save_dir, "T_cam_board.json")
# ==================================================

os.makedirs(save_dir, exist_ok=True)

# 启动 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

print("\n✅ 实时画面开启，按下 [S] 采集棋盘图并计算外参")
print("若检测失败，会提示“未检测到棋盘格”请重新调整角度\n")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("RealSense RGB", color_image)

        k = cv2.waitKey(1)
        if k == ord('s'):
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows))

            if not ret:
                print("❌ 未检测到棋盘格，请调整角度后重试。")
                continue

            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

            # 构造棋盘格世界坐标（原点在棋盘角点）
            objp = np.zeros((rows * cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

            # 相机内参
            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx],
                          [0, intr.fy, intr.ppy],
                          [0, 0, 1]], dtype=np.float64)

            # PnP 求解相机→棋盘变换
            _, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
            R, _ = cv2.Rodrigues(rvec)

            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = R
            T_cam_board[:3, 3] = tvec.squeeze()

            # 保存 JSON
            out = {
                "T_cam_board": T_cam_board.tolist(),
                "intrinsics": {
                    "fx": intr.fx, "fy": intr.fy,
                    "ppx": intr.ppx, "ppy": intr.ppy,
                    "width": intr.width, "height": intr.height
                }
            }

            json.dump(out, open(save_path, "w"), indent=2)
            print(f"\n✅ 已检测到棋盘格并保存：{save_path}")
            print("矩阵 (T_cam_board):\n", np.array2string(T_cam_board, precision=4, suppress_small=True))
            break

        elif k == 27:  # ESC 退出
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("相机已停止。")