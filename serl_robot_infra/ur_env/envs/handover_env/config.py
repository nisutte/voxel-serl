from ur_env.envs import DefaultEnvConfig
import numpy as np


class UR5DualCameraConfigRight(DefaultEnvConfig):
    RESET_Q = np.array([[-1.771, -1.943, 2.005, -3.244, -1.597, -1.5412]])
    RANDOM_RESET = False
    RANDOM_POSITION_RANGE = (0.05, 0.05, 0.05)
    RANDOM_ROT_RANGE = (0.03,)
    ABS_POSE_LIMIT_HIGH = np.array([0.1, 0.68, 0.75, 0.15, 0.2, 0.1])
    ABS_POSE_LIMIT_LOW = np.array([-0.1, 0.4, 0.55, -0.15, -0.2, -0.1])
    ABS_POSE_RANGE_LIMITS = np.array([0.4, 0.8])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP = "192.168.1.66"  # for the vacuum pump
    CONTROLLER_HZ: int = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ZEROMQ_PUBLISHER_PORT: int = 5558
    ZEROMQ_SUBSCRIBER_PORT: int = 5559

    REALSENSE_CAMERAS = {
        "wrist_right": "218622277164"
    }
    VOXEL_PARAMS = {
        "voxel_box_size": [0.15, 0.12, 0.18],  # in m
        "voxel_grid_shape": [40, 32, 48]
    }
    CAMERA_PARAMS = {
        "wrist_right": {
            "angle": [0., -30., 90.],     # new orientation for the wrist camera
            "center_offset": [0.0, 0.02, -0.15 + 0.07],
        }
    }
    CALIBRATION_PATH = "/home/nico/robot_ipc_control/configs/b2r_pose_right_adam.npy"


class UR5DualCameraConfigLeft(UR5DualCameraConfigRight):
    ROBOT_IP = "192.168.1.33"
    ZEROMQ_PUBLISHER_PORT: int = 5555
    ZEROMQ_SUBSCRIBER_PORT: int = 5556

    RESET_Q = np.array([[-1.8715, -1.9142, 1.888, -3.100, -1.5387, -1.6140]])
    REALSENSE_CAMERAS = {
        "wrist_left": "218622270808",
    }
    CAMERA_PARAMS = {
        "wrist_left": {
            "angle": [0., -30., 90.],
            "center_offset": [0.0, 0.02, -0.15 + 0.07],
        }
    }
    CALIBRATION_PATH = "/home/nico/robot_ipc_control/configs/b2r_pose_left_adam.npy"


class UR5DualCameraConfig90DegreesRight(UR5DualCameraConfigRight):
    ABS_POSE_LIMIT_HIGH = np.array([0.1, 0.7, 0.65, 0.1, 0.15, 0.1])
    ABS_POSE_LIMIT_LOW = np.array([-0.1, 0.4, 0.45, -0.1, -0.15, -0.1])
    RESET_Q = np.array([[-1.8542, -1.7199, 1.4690, -2.1082, -1.6014, 0.0]])


class UR5DualCameraConfig90DegreesLeft(UR5DualCameraConfigLeft):
    ABS_POSE_LIMIT_HIGH = np.array([0.1, 0.7, 0.65, 0.1, 0.15, 0.1])
    ABS_POSE_LIMIT_LOW = np.array([-0.1, 0.4, 0.45, -0.1, -0.15, -0.1])
    RESET_Q = np.array([[-1.7322, -1.9437, 1.7611, -2.2971, -1.6246, -3.1415]])
    


def get_box_handover_state_noise_assignments(total_dim: int):
    """
    Default per-index std assignment for state Gaussian noise (normalized space)
    for the box handover task, matching SERLObsWrapper's flattened layout.

    Returns a dict suitable for build_std_vec_from_slices or None if unknown.
    """
    pos_std = 0.01
    mrp_std = 0.01
    lin_vel_std = 0.02
    ang_vel_std = 0.01
    force_std = 0.02
    torque_std = 0.01

    if total_dim == 104:  # with EMA features
        return {
            (0, 3): pos_std, (3, 6): mrp_std,                 # l2r/tcp_pose
            (6, 9): lin_vel_std, (9, 12): ang_vel_std,        # l2r/tcp_vel
            # left/action -> no noise
            (19, 22): force_std, (22, 25): torque_std,        # left/ema_force
            (25, 28): lin_vel_std, (28, 31): ang_vel_std,     # left/ema_tcp_vel
            # left/gripper_state -> 0
            (33, 36): force_std,                              # left/tcp_force
            (36, 39): pos_std, (39, 42): mrp_std,             # left/tcp_pose
            (42, 45): torque_std,                             # left/tcp_torque
            (45, 48): lin_vel_std, (48, 51): ang_vel_std,     # left/tcp_vel
            # left/time_diff -> 0
            (52, 55): pos_std, (55, 58): mrp_std,             # r2l/tcp_pose
            (58, 61): lin_vel_std, (61, 64): ang_vel_std,     # r2l/tcp_vel
            # right/action -> no noise
            (71, 74): force_std, (74, 77): torque_std,        # right/ema_force
            (77, 80): lin_vel_std, (80, 83): ang_vel_std,     # right/ema_tcp_vel
            # right/gripper_state -> 0
            (85, 88): force_std,                              # right/tcp_force
            (88, 91): pos_std, (91, 94): mrp_std,             # right/tcp_pose
            (94, 97): torque_std,                             # right/tcp_torque
            (97, 100): lin_vel_std, (100, 103): ang_vel_std,  # right/tcp_vel
            # right/time_diff -> 0
        }
    return None
