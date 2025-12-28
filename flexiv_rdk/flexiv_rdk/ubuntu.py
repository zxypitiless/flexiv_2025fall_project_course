import pyrealsense2 as rs
import numpy as np
import cv2, json, os

# 参数设置
cols, rows, square = 8, 8, 0.025  # 棋盘格参数（与 config.yaml 一致）
save_path = "T_base_cam.json"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

align = rs.align(rs.stream.color)

print("按下 [S] 采集标定图像")

while True:
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())

    cv2.imshow('RGB', color_image)
    k = cv2.waitKey(1)
    if k == ord('s'):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not ret:
            print("未检测到棋盘格")
            continue

        objp = np.zeros((rows*cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]])

        _, rvec, tvec = cv2.solvePnP(objp, corners, K, None)
        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.squeeze()

        os.makedirs("data", exist_ok=True)
        json.dump({"T": T.tolist()}, open(save_path, "w"), indent=2)
        print(f"保存成功：{save_path}")
        break

pipeline.stop()
cv2.destroyAllWindows()
