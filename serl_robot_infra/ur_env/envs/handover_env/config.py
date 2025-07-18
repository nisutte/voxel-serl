from ur_env.envs import DefaultEnvConfig
import numpy as np


class UR5DualCameraConfigRight(DefaultEnvConfig):
    RESET_Q = np.array([[-1.771, -1.943, 2.005, -3.244, -1.597, -1.5412]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.15, 0.65, 0.7, 0.15, 0.15, 0.3])
    ABS_POSE_LIMIT_LOW = np.array([-0.15, 0.4, 0.4, -0.15, -0.15, -0.3])
    ABS_POSE_RANGE_LIMITS = np.array([0.4, 0.8])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP = "192.168.1.66"  # for the vacuum pump
    CONTROLLER_HZ: int = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ZEROMQ_PUBLISHER_PORT: int = 5558
    ZEROMQ_SUBSCRIBER_PORT: int = 5559

    REALSENSE_CAMERAS = {
        "wrist": "218622279756",
    }
    VOXEL_PARAMS = {
        "voxel_box_size": [0.20, 0.20, 0.16],  # in m
        "voxel_grid_shape": [50, 50, 40]
    }
    CAMERA_PARAMS = {
        "wrist": {
            "angle": [30.5, 0., 0.],
            "center_offset": [-0.008, 0.05, -0.05 - 0.2 / 2.],
        }
    }
    CALIBRATION_PATH = "/home/nico/robot_ipc_control/configs/b2r_pose_left_adam.npy"


class UR5DualCameraConfigLeft(UR5DualCameraConfigRight):
    ROBOT_IP = "192.168.1.33"
    ZEROMQ_PUBLISHER_PORT: int = 5555
    ZEROMQ_SUBSCRIBER_PORT: int = 5556

    RESET_Q = np.array([[-1.8715, -1.9142, 1.888, -3.100, -1.5387, -1.6140]])
    REALSENSE_CAMERAS = {
        "wrist": "218622277164"
    }
    CALIBRATION_PATH = "/home/nico/robot_ipc_control/configs/b2r_pose_right_adam.npy"
