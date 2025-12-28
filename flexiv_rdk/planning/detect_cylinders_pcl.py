import pcl
import numpy as np

def detect_cylinders_pcl(xyz,
                         distance_threshold=0.01,
                         radius_limits=(0.015, 0.03),
                         normal_radius=0.03,
                         max_iterations=5000,
                         debug=False):

    # ===== 1. 转成 PCL 格式 =====
    cloud = pcl.PointCloud()
    cloud.from_array(xyz.astype(np.float32))

    # ===== 2. 法向量估计（关键！Open3D 做不到） =====
    ne = cloud.make_NormalEstimation()
    tree = cloud.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(normal_radius)
    normals = ne.compute()

    # ===== 3. Cylinder Segmentation（工业级算法） =====
    seg = cloud.make_segment_cylinder()
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_normal_distance_weight(0.1)
    seg.set_MaxIterations(max_iterations)
    seg.set_distance_threshold(distance_threshold)
    seg.set_radius_limits(radius_limits[0], radius_limits[1])
    seg.set_input_normals(normals)

    # ===== 4. 拟合 =====
    inliers, model = seg.segment()

    if len(inliers) == 0:
        raise RuntimeError("未检测到圆柱（点太少或表面太薄）")

    # 模型格式：
    # [center_x, center_y, center_z, axis_x, axis_y, axis_z, radius]
    center = model[:3]
    axis = model[3:6] / np.linalg.norm(model[3:6])
    radius = model[6]

    if debug:
        print("Cylinder center:", center)
        print("Cylinder axis:", axis)
        print("Cylinder radius:", radius)

    return {
        "center": center.tolist(),
        "axis": axis.tolist(),
        "radius": radius
    }