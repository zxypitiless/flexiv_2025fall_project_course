#!/usr/bin/env python3
"""
get_ee_pose.py

用于获取当前机械臂末端 (TCP) 位姿，并保存为 ee_pose.json，
以便后续与相机数据配对进行手眼标定。
"""

import flexivrdk
import json
import time

# ===================== 用户配置 =====================
ROBOT_IP = "192.168.2.100"      # 机械臂控制箱 IP
LOCAL_IP = "192.168.2.102"      # 本机（Ubuntu）IP，与你的控制箱同网段
OUTPUT_FILE = "ee_pose.json"    # 保存路径，可修改为 dataset/x/ee_pose.json
# ====================================================

def main():
    log = flexivrdk.Log()
    log.info("正在连接机械臂 ...")

    # 建立连接
    robot = flexivrdk.Robot(ROBOT_IP, LOCAL_IP)

    # 检查并清除故障
    if robot.isFault():
        log.warn("检测到故障，尝试清除 ...")
        robot.clearFault()
        time.sleep(2)
        if robot.isFault():
            log.error("无法清除故障，请检查机械臂状态。")
            return
        log.info("故障已清除 ✅")

    # 使能机械臂
    log.info("使能机械臂 ...")
    robot.enable()
    while not robot.isOperational():
        time.sleep(0.5)
    log.info("机械臂已进入可操作状态 ✅")

    # 获取当前状态
    state = flexivrdk.RobotStates()
    robot.getRobotStates(state)

    # 提取末端位姿（tcpPose）
    tcp = list(state.tcpPose)  # [x, y, z, qx, qy, qz, qw]
    xyz = tcp[:3]
    quat = tcp[3:]

    ee = {"xyz": xyz, "quat": quat}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(ee, f, indent=2)

    print("✅ 末端位姿已保存：", OUTPUT_FILE)
    print("位置 (m):", [round(v, 4) for v in xyz])
    print("四元数:", [round(v, 4) for v in quat])

if __name__ == "__main__":
    main()