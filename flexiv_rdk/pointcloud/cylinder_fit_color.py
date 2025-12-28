import numpy as np
import open3d as o3d

def detect_cylinders_color(xyz, rgb,
                           max_depth=1.2,
                           min_points=2000,
                           debug=False):
    """
    输入：
        xyz: (N,3)
        rgb: (N,3) uint8, 顺序为 [R, G, B]
    输出：
        cylinders: list of dict {center, axis, radius}
    """

    # ========= 1. 深度过滤 =========
    mask = xyz[:, 2] < max_depth
    xyz = xyz[mask]
    rgb = rgb[mask]

    if debug: print(f"[debug] 深度过滤后点数: {len(xyz)}")

    # ========= 2. 直接在 RGB 空间筛选蓝/紫色 =========
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    # 蓝色：B 大，R 和 G 较小
    blue_mask = (b > 100) & (b > r + 30) & (b > g + 30)

    # 紫色：R 和 B 都大，G 小
    purple_mask = (r > 80) & (b > 80) & (g < 80) & (r + b > 200)

    mask_color = blue_mask | purple_mask

    xyz = xyz[mask_color]
    rgb = rgb[mask_color]

    if debug: print(f"[debug] 颜色过滤后点数: {len(xyz)}")

    if len(xyz) < min_points:
        raise RuntimeError("颜色点太少，可能没看到杆")

    # ========= 3. 聚类 - 三根杆 =========
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    labels = np.array(pcd.cluster_dbscan(eps=0.03, min_points=500))
    cluster_count = labels.max() + 1

    if debug: print(f"[debug] 聚类得到 {cluster_count} 个团块")

    cylinders = []

    for cid in range(cluster_count):
        pts = xyz[labels == cid]
        if len(pts) < 1500:
            continue

        # ========= 4. RANSAC 圆柱拟合 =========
        pcd_sub = o3d.geometry.PointCloud()
        pcd_sub.points = o3d.utility.Vector3dVector(pts)

        try:
            model, inliers = pcd_sub.segment_cylinder(
                distance_threshold=0.01,
                radius=0.02,
                ransac_n=10,
                num_iterations=2000
            )
        except Exception as e:
            if debug: print(f"[debug] 圆柱拟合失败: {e}")
            continue

        if model is None:
            continue

        center, axis, radius = model

        cylinders.append({
            "center": center.tolist(),
            "axis": axis.tolist(),
            "radius": float(radius)
        })

        if debug:
            print(f"[debug] 圆柱 {cid}: center={center}, axis={axis}, radius={radius:.4f}")

    return cylinders