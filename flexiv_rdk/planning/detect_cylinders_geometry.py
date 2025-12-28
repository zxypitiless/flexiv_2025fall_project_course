import open3d as o3d
import numpy as np

def detect_cylinders_geometry(
        xyz,
        max_depth=1.5,
        min_cluster_points=800,
        voxel=0.008,
        eps=0.05,
        debug=False):

    # ===== 1. 粗深度过滤 =====
    mask = (xyz[:,2] > 0.05) & (xyz[:,2] < max_depth)
    xyz = xyz[mask]

    # ===== 2. 空间裁剪（根据你现场） =====
    # 你的杆大约在 x = 0.2~0.8, y = -0.2~0.2, z = 0.2~1.0
    mask2 = (xyz[:,0] > 0.2) & (xyz[:,0] < 0.8) & \
            (xyz[:,1] > -0.2) & (xyz[:,1] < 0.2)
    xyz = xyz[mask2]

    if debug:
        print(f"[debug] 过滤后点数: {len(xyz)}")

    if len(xyz) < 5000:
        raise RuntimeError("杆区域点太少")

    # ===== 3. voxel 下采样（核心提速） =====
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd = pcd.voxel_down_sample(voxel_size=voxel)

    xyz = np.asarray(pcd.points)

    if debug:
        print(f"[debug] 下采样后点数: {len(xyz)}")

    # ===== 4. DBSCAN 聚类 =====
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_cluster_points)
    )
    n_clusters = labels.max() + 1

    if debug:
        print(f"[debug] 聚类得到 {n_clusters} 个团块")

    cylinders = []

    # ===== 5. 拟合每根杆 =====
    for cid in range(n_clusters):
        pts = xyz[labels == cid]
        if len(pts) < min_cluster_points:
            continue

        sub = o3d.geometry.PointCloud()
        sub.points = o3d.utility.Vector3dVector(pts)

        try:
            model, inliers = sub.segment_cylinder(
                distance_threshold=0.01,
                radius=0.02,      # 杆半径
                ransac_n=8,
                num_iterations=500
            )
        except:
            continue

        center, axis, radius = model

        if debug:
            print(f"[debug] 杆: center={center}, axis={axis}, radius={radius}")

        cylinders.append({
            "center": center.tolist(),
            "axis": axis.tolist(),
            "radius": float(radius)
        })

    return cylinders