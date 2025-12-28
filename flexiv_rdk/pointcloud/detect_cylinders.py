import numpy as np
import open3d as o3d


def detect_cylinders_open3d(
    xyz,
    voxel_size=0.01,        # 降采样从 5mm → 10mm（保持更多点）
    max_depth=1.5,
    min_cluster_points=800, # 降低阈值，让杆能被聚出来
    eps=0.05,               # 聚类半径从 0.03 → 0.05（扩大搜索）
    debug=False
):
    # ====== 1. 深度过滤 ======
    mask = xyz[:, 2] < max_depth
    xyz = xyz[mask]

    if len(xyz) < 3000:
        raise RuntimeError("点云太少，可能视野里没杆或深度失真")

    if debug:
        print(f"[debug] 深度过滤后点数: {len(xyz)}")

    # ====== 2. 构造点云 ======
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # ====== 3. 降采样（保守） ======
    pcd = pcd.voxel_down_sample(voxel_size)

    if debug:
        print(f"[debug] 降采样后点数: {len(pcd.points)}")

    # ====== 4. 估计法向量，圆柱拟合必须要 ======
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_nn=30
        )
    )

    # ====== 5. 聚类 ======
    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,         # 5cm 内算同一簇
            min_points=200,  # 降低门槛
            print_progress=False
        )
    )

    clusters = labels.max() + 1

    if debug:
        print(f"[debug] 聚类得到 {clusters} 个簇")

    if clusters < 1:
        raise RuntimeError("聚类失败：点云不连续或噪声太大")

    cylinders = []

    # ====== 6. 对每个 cluster 做圆柱拟合 ======
    for cid in range(clusters):
        idx = (labels == cid)
        pts = np.asarray(pcd.points)[idx]

        if len(pts) < min_cluster_points:
            continue

        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts)
        sub.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd.normals)[idx]
        )

        try:
            model, inliers = sub.segment_cylinder(
                distance_threshold=0.015,  # 1.5cm 的拟合容差
                radius=0.02,               # 2cm 半径
                ransac_n=20,
                num_iterations=3000
            )
        except Exception as e:
            if debug:
                print(f"[debug] cluster {cid} cylinder fit error: {e}")
            continue

        if model is None:
            continue

        center, axis, radius = model

        cylinders.append({
            "center": center.tolist(),
            "axis": axis.tolist(),
            "radius": float(radius),
            "points": len(pts)
        })

        if debug:
            print(f"[debug] cluster {cid}: center={center}, axis={axis}, r={radius}")

    if len(cylinders) == 0:
        raise RuntimeError("没有圆柱被成功拟合")

    return cylinders