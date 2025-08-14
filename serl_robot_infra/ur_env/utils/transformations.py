import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_to_T(pose):
    """
    pose can be either quat or rot_vec (no euler here!)
    """
    pose = np.asarray(pose)
    tmp = np.eye(4)
    tmp[:3, 3] = pose[:3]
    if pose.shape == (7,):
        tmp[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    elif pose.shape == (6,):
        tmp[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    else:
        raise ValueError(f"pose has shape {pose.shape}")
    return tmp

def T_to_pose(T, quat=True):
    T = np.asarray(T)
    assert T.shape == (4, 4)
    pos = T[:3, 3].transpose()
    rot = R.from_matrix(T[:3, :3])
    rotation = rot.as_quat() if quat else rot.as_rotvec()
    return np.concatenate((pos, rotation))

def apply_rotation(pose, T, quat=True):
    pose_ = np.zeros_like(pose)
    rot = R.from_matrix(T[:3, :3])
    pose_[:3] = T[:3, :3] @ pose[:3]
    if quat:
        pose_[3:] = (rot.inv() * R.from_quat(pose[3:]) * rot).as_quat()
    else:
        pose_[3:] = (rot.inv() * R.from_rotvec(pose[3:]) * rot).as_rotvec()
    return pose_

def transform_twist(twist, T):
    """
    Rotate a 6D spatial velocity (twist) [v; w] into another frame using only rotation.
    Assumes both twists are measured at their own TCPs but expressed in their local base frames.
    """
    Rm = T[:3, :3]
    twist = np.asarray(twist)
    out = np.zeros_like(twist)
    out[:3] = Rm @ twist[:3]
    out[3:] = Rm @ twist[3:]
    return out

