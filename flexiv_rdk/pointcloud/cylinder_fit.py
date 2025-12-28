# pointcloud/cylinder_fit.py
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

def color_mask(rgb):
    """提取蓝色 & 紫色点区域（根据你实拍图调好的范围）"""
    B = rgb[:, 2]
    G = rgb[:, 1]
    R = rgb[:, 0]

    mask = (
        (B > 150) & 
        (R < 120)
    )
    return mask

def line_ransac(points):
    """对点云拟合直线（axis）"""
    if len(points) < 50:
        return None, None

    X = points[:, :2]
    Z = points[:, 2]

    ransac = RANSACRegressor()
    ransac.fit(X, Z)

    coef = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_

    # 轴线方向向量（近似）
    direction = np.array([coef[0], coef[1], 1.0])
    direction = direction / np.linalg.norm(direction)

    center = points.mean(axis=0)

    return center, direction


def detect_three_rods(xyz, rgb):
    mask = color_mask(rgb)
    rod_pts = xyz[mask]

    if len(rod_pts) < 300:
        print("❌ 没检测到明显杆区域")
        return []

    # 聚类分三堆点
    db = DBSCAN(eps=0.02, min_samples=20).fit(rod_pts)
    labels = db.labels_

    rods = []
    for lab in set(labels):
        if lab == -1:
            continue
        pts = rod_pts[labels == lab]
        center, direction = line_ransac(pts)
        if center is not None:
            rods.append((center, direction))

    return rods