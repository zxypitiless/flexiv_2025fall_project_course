import json, numpy as np

def load_extrinsic(path):
    T = np.array(json.load(open(path))["T"], dtype=np.float64)
    return T

def transform_points(T, pts):
    pts_h = np.hstack([pts, np.ones((len(pts),1))])
    return (T @ pts_h.T).T[:, :3]

def rotm_batch_cam2base(T, R_cam_batch):
    R_cb = T[:3,:3]
    return np.einsum("ij,njk->nik", R_cb, R_cam_batch)
