import numpy as np

def generate_spiral_traj(center, axis, radius, turns=3, points_per_turn=200):
    """
    单根杆螺旋路径
    center: 杆中心 (3,)
    axis:   杆方向 (3,)
    radius: 缠绕半径
    """
    axis = axis / np.linalg.norm(axis)
    z_vec = axis

    # 找到垂直方向 x_vec
    tmp = np.array([1, 0, 0])
    if abs(np.dot(tmp, z_vec)) > 0.9:
        tmp = np.array([0, 1, 0])

    x_vec = tmp - np.dot(tmp, z_vec) * z_vec
    x_vec /= np.linalg.norm(x_vec)

    y_vec = np.cross(z_vec, x_vec)

    # 螺旋参数
    total_pts = turns * points_per_turn
    theta = np.linspace(0, turns * 2*np.pi, total_pts)
    height = np.linspace(0, 0.6, total_pts)  # 60cm 高度，可根据需要改

    traj = []
    for t, h in zip(theta, height):
        pos = (
            center
            + radius * np.cos(t) * x_vec
            + radius * np.sin(t) * y_vec
            + h * z_vec
        )

        # 固定工具朝向，朝向圆心（简单处理）
        quat = np.array([1, 0, 0, 0])
        traj.append(np.concatenate([pos, quat]))

    return np.array(traj)


def generate_spiral_traj_multi(cylinders, wrap_radius, turns, points_per_turn):
    """
    对 3 根杆生成连续缠绕路径
    """
    traj_all = []

    for cyl in cylinders:
        center = np.array(cyl["center"])
        axis   = np.array(cyl["axis"])

        traj = generate_spiral_traj(
            center,
            axis,
            wrap_radius,
            turns=turns,
            points_per_turn=points_per_turn
        )

        traj_all.append(traj)

    # 拼接三段轨迹
    return np.vstack(traj_all)