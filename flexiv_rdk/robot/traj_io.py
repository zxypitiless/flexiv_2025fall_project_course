import numpy as np, os

def save_traj_csv(path, pos, rot):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Rf = rot.reshape(len(rot), 9)
    data = np.hstack([pos, Rf])
    header = "x,y,z,r11,r12,r13,r21,r22,r23,r31,r32,r33"
    np.savetxt(path, data, delimiter=",", header=header, comments="")
