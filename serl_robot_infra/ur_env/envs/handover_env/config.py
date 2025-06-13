from ur_env.envs import DefaultEnvConfig
import numpy as np


class UR5DualCameraConfigRight(DefaultEnvConfig):
    RESET_Q = np.array([[-1.7415, -1.8875,  1.9666, -3.2068, -1.5384,  1.4565]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.2, 0.7, 0.8, 0.1, 0.1, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.2, 0.4, 0.1, -0.1, -0.1, -0.2])
    ABS_POSE_RANGE_LIMITS = np.array([0.4, 1.0])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP = "192.168.1.66"  # for the vacuum pump
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
    CALIBRATION_PATH = "/home/nico/robot_ipc_control/calibration/T_cam_to_robot_base_left_20250523_153325.npy"


class UR5DualCameraConfigLeft(UR5DualCameraConfigRight):
    ROBOT_IP = "192.168.1.33"
    ZEROMQ_PUBLISHER_PORT: int = 5555
    ZEROMQ_SUBSCRIBER_PORT: int = 5556

    RESET_Q = np.array([[-1.6342, -2.0612,  2.1379, -3.2342, -1.5971,  1.5422]])
    REALSENSE_CAMERAS = {
        "wrist": "218622277164"
    }
    CALIBRATION_PATH = "/home/nico/robot_ipc_control/calibration/T_cam_to_robot_base_right_20250523_152927.npy"
