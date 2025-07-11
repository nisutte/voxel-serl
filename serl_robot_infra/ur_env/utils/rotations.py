import numpy as np
from scipy.spatial.transform import Rotation as R

"""
UR5 represents the orientation in axis angle representation
"""


def rotvec_2_quat(rotvec):
    return R.from_rotvec(rotvec).as_quat()

def rotvec_2_mrp(rotvec):
    return R.from_rotvec(rotvec).as_mrp()

def quat_2_rotvec(quat):
    return R.from_quat(quat).as_rotvec()

def quat_2_euler(quat):
    return R.from_quat(quat).as_euler('xyz')

def quat_2_mrp(quat):
    return R.from_quat(quat).as_mrp()

def pose_2_quat(rotvec_pose) -> np.ndarray:
    return np.concatenate((rotvec_pose[:3], rotvec_2_quat(rotvec_pose[3:])))

def pose_2_rotvec(quat_pose) -> np.ndarray:
    return np.concatenate((quat_pose[:3], quat_2_rotvec(quat_pose[3:])))

def rotvec_frame_transform(rotvec, rot_matrix):
    return (R.from_matrix(rot_matrix).inv() * R.from_rotvec(rotvec) * R.from_matrix(rot_matrix)).as_rotvec()
