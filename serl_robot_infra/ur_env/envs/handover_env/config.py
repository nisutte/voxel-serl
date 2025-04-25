from ur_env.envs import DefaultEnvConfig
import numpy as np

class UR5DualCameraConfigRight(DefaultEnvConfig):
    RESET_Q = np.array([[1.3502, -1.2897, 1.9304, -2.2098, -1.5661, 1.4027]])
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.00,)
    RANDOM_ROT_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.array([0.2, -0.4, 0.22, 0.05, 0.05, 0.2])
    ABS_POSE_LIMIT_LOW = np.array([-0.2, -0.7, - 0.006, -0.05, -0.18, -0.2])
    ABS_POSE_RANGE_LIMITS = np.array([0.36, 0.83])
    ACTION_SCALE = np.array([0.02, 0.1, 1.], dtype=np.float32)

    ROBOT_IP = "172.22.22.2"            # fot the vacuum pump
    GRIPPER_TIMEOUT = 2000  # in milliseconds
    ZEROMQ_PUBLISHER_PORT: int = 5555
    ZEROMQ_SUBSCRIBER_PORT: int = 5556

    REALSENSE_CAMERAS = {
        # "wrist": "218622277164",
    }


class UR5DualCameraConfigLeft(UR5DualCameraConfigRight):
    ROBOT_IP = "172.17.0.2"
    ZEROMQ_PUBLISHER_PORT: int = 5565
    ZEROMQ_SUBSCRIBER_PORT: int = 5566

    REALSENSE_CAMERAS = {
        # "wrist": "218622279756"
    }