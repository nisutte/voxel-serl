import numpy as np
from typing import Tuple

from ur_env.envs.dual_ur5_env import DualUR5Env
from ur_env.envs.handover_env import UR5DualCameraConfigLeft, UR5DualCameraConfigRight


class UR5HandoverEnv(DualUR5Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # TODO not done yet
    def compute_reward(self, obs, action) -> float:
        return False

    def reached_goal_state(self, obs) -> bool:
        # obs[0] == gripper pressure, obs[4] == force in Z-axis
        return False

    def close(self):
        super().close()