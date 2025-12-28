import open3d as o3d
import numpy as np
from viz import save_pointcloud_as_file


def depth_to_pointcloud(rgb, depth, intr):
    """ RealSense depth → 点云 """
    h, w = depth.shape
    fx, fy, cx, cy, depth_scale = intr.fx, intr.fy, intr.cx, intr.cy, intr.depth_scale

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    z = depth * depth_scale
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy

    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    save_pointcloud_as_file(pcd)   # ✅ 无 GUI

    return pcd