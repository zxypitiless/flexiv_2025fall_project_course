import numpy as np

def mask_to_pointcloud(mask, depth, intrinsic):
    fx, fy, cx, cy = intrinsic
    h, w = depth.shape

    ys, xs = np.where(mask > 127)
    ds = depth[ys, xs]

    X = (xs - cx) * ds / fx
    Y = (ys - cy) * ds / fy
    Z = ds

    return np.vstack([X, Y, Z]).T


def fit_cylinder_axis(points):
    # 用 PCA 拟合主轴
    centroid = np.mean(points, axis=0)
    cov = np.cov(points - centroid, rowvar=False)
    eigvals, eigvecs = np.linalg.eig(cov)

    axis = eigvecs[:, np.argmax(eigvals)]
    return centroid, axis