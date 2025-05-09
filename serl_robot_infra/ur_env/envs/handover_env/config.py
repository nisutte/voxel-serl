from ur_env.envs import DefaultEnvConfig
import numpy as np


class UR5DualCameraConfigRight(DefaultEnvConfig):
    p = np.pi / 2.
    RESET_Q = np.array([[-p, -p, p, -p, -p, 0.]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.2, 0.6, 0.6, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.2, 0.4, -0.006, -0.05, -0.05, -0.2])
    ABS_POSE_RANGE_LIMITS = np.array([0.2, 0.9])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP = "192.168.1.66"  # fot the vacuum pump
    CONTROLLER_HZ: int = 100
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ZEROMQ_PUBLISHER_PORT: int = 5557
    ZEROMQ_SUBSCRIBER_PORT: int = 5558

    REALSENSE_CAMERAS = {
        "wrist": "218622279756",
    }
    VOXEL_PARAMS = {
        "voxel_box_size": [0.15, 0.15, 0.12],  # in m
        "voxel_grid_shape": [50, 50, 40]
    }
    CAMERA_PARAMS = {
        "wrist": {
            "angle": [30.5, 0., 0.],
            "center_offset": [-0.008, 0.1, -0.085 - 0.06],
        }
    }


class UR5DualCameraConfigLeft(UR5DualCameraConfigRight):
    ROBOT_IP = "192.168.1.33"
    ZEROMQ_PUBLISHER_PORT: int = 5555
    ZEROMQ_SUBSCRIBER_PORT: int = 5556

    REALSENSE_CAMERAS = {
        "wrist": "218622277164"
    }