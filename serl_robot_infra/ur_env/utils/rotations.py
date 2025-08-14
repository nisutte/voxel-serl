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

def omega_to_mrp_dot(sigma, omega):
    sigma = np.asarray(sigma)
    omega = np.asarray(omega)
    s2 = np.dot(sigma, sigma)
    sigma_skew = np.array([
        [0, -sigma[2], sigma[1]],
        [sigma[2], 0, -sigma[0]],
        [-sigma[1], sigma[0], 0]
    ])
    B = (1 - s2) * np.eye(3) + 2 * sigma_skew + 2 * np.outer(sigma, sigma)
    return 0.25 * (B @ omega)
