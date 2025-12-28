import flexivrdk as frdk
import time
import json
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FlexivRobot:
    def __init__(self, cfg):
        ip = cfg["ip"]
        try:
            self.robot = frdk.Robot(ip)
            print(f"✅ 已连接 Flexiv RDK: {ip}")
        except Exception as e:
            raise RuntimeError(f"无法连接 Flexiv：{e}")

    def execute_traj(self, traj):
        """
        traj: list of joint arrays
        {"points": [...]}
        """
        print("🤖 正在执行缠绕轨迹 ...")
        for point in traj["points"]:
            self.robot.set_joint_positions(point)
            time.sleep(0.03)  # 控制刷新率

        print("✅ 缠绕完成，轨迹执行结束")

    def save_traj(self, traj, filename="spiral_traj.json"):
        path = f"{OUTPUT_DIR}/{filename}"
        json.dump(traj, open(path, "w"), indent=2)
        print(f"✅ 轨迹保存：{path}")