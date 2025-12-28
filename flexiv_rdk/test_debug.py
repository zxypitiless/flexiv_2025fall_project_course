#!/usr/bin/env python

"""
debug_move_tcp_fixed_ip.py

调试用：机械臂回到初始位姿，然后移动到指定 TCP 位置。
机器人 IP 和本地 IP 已固定在代码中，无需命令行参数。
"""

import time
import sys
sys.path.insert(0, "../lib_py")
import flexivrdk

# -------------------------
# 固定 IP 配置
# -------------------------
robot_ip = "192.168.2.100"  # 替换为你的机器人 IP
local_ip = "192.168.2.102"  # 替换为你的本地 PC IP

def primitive_reached_target(robot):
    """检查 primitive 是否完成"""
    pt_states = robot.getPrimitiveStates()
    for s in pt_states.split(","):
        if "reachedTarget" in s:
            return s.split("=")[1].strip() == "1"
    return False

def main():
    log = flexivrdk.Log()
    mode = flexivrdk.Mode

    try:
        # 初始化机器人
        robot = flexivrdk.Robot(robot_ip, local_ip)

        # 清除故障
        if robot.isFault():
            log.warn("检测到故障，正在清除...")
            robot.clearFault()
            time.sleep(2)
            if robot.isFault():
                log.error("故障无法清除，退出")
                return
            log.info("故障已清除")

        # 启用机器人
        log.info("启用机器人...")
        robot.enable()
        while not robot.isOperational():
            time.sleep(1)
        log.info("机器人已就绪")

        # 设置 Primitive 执行模式
        robot.setMode(mode.NRT_PRIMITIVE_EXECUTION)

        # -------------------------
        # 1. 回到 Home 位姿
        # -------------------------
        log.info("执行 Home 动作")
        robot.executePrimitive("Home()")
        while robot.isBusy():
            time.sleep(0.5)
        log.info("到达 Home 位姿")

        # -------------------------
        # 2. MoveL 到指定 TCP 位置
        # -------------------------
        target_tcp = "0.5 0 0.6 180 0 180 WORLD WORLD_ORIGIN"
        log.info(f"移动到 TCP 位置: {target_tcp}")
        robot.executePrimitive(f"MoveL(target={target_tcp}, maxVel=0.2)")
        while not primitive_reached_target(robot):
            time.sleep(0.5)
        log.info("已到达目标 TCP 位置")

        # -------------------------
        # 停止机器人
        # -------------------------
        robot.stop()
        log.info("调试完成，机器人已停止")

    except Exception as e:
        log.error(f"程序异常: {e}")

if __name__ == "__main__":
    main()
