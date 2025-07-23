import gymnasium as gym
import numpy as np
import time

from gym import spaces, Env

from ur_env.envs.dual_ur5_env import DualUR5Env
from ur_env.envs.wrappers import SpacemouseIntervention
from ur_env.utils.rotations import quat_2_mrp, rotvec_2_mrp


class DualToMrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to mrp angles
    """
    def __init__(self, dual_env: Env, transform_obs=True):
        super().__init__(dual_env)
        self.transform_obs = transform_obs
        # from xyz + quat to xyz + mrp
        self.observation_space["state"]["left/tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))
        self.observation_space["state"]["right/tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))
        self.observation_space["state"]["l2r/tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))
        self.observation_space["state"]["r2l/tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))

    def observation(self, obs):
        # convert tcp pose from quat to mrp
        obs["state"]["left/tcp_pose"] = np.concatenate(
            (obs["state"]["left/tcp_pose"][:3], quat_2_mrp(obs["state"]["left/tcp_pose"][3:]))
        )
        obs["state"]["right/tcp_pose"] = np.concatenate(
            (obs["state"]["right/tcp_pose"][:3], quat_2_mrp(obs["state"]["right/tcp_pose"][3:]))
        )
        obs["state"]["l2r/tcp_pose"] = np.concatenate(
            (obs["state"]["l2r/tcp_pose"][:3], quat_2_mrp(obs["state"]["l2r/tcp_pose"][3:]))
        )
        obs["state"]["r2l/tcp_pose"] = np.concatenate(
            (obs["state"]["r2l/tcp_pose"][:3], quat_2_mrp(obs["state"]["r2l/tcp_pose"][3:]))
        )

        if self.transform_obs:
            obs["state"]["left/tcp_vel"][3:6] = rotvec_2_mrp(obs["state"]["left/tcp_vel"][3:6])
            obs["state"]["right/tcp_vel"][3:6] = rotvec_2_mrp(obs["state"]["right/tcp_vel"][3:6])
            obs["state"]["left/tcp_torque"] = rotvec_2_mrp(obs["state"]["left/tcp_torque"])
            obs["state"]["right/tcp_torque"] = rotvec_2_mrp(obs["state"]["right/tcp_torque"])

        return obs

class DualNormalizationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper scales the observations with the provided hyperparams
    """

    """
    from analyzing data: 
        action: -
        pose pos: 0., 0.1
        pose rot: 0., 0.02
        vel pos: 0., 0.05
        vel rot: 0., 0.01
        force: 0., 0.005
        torque: 0., 0.002
        t_diff: 0.16, 0.5
    """

    def __init__(self, env):
        super().__init__(env)
        self.pose_scale = [1. / 0.1, 1e-1 / 0.02 ]
        self.vel_scale = [1. / 0.05, 1e-1 / 0.01]
        self.force_scale = [1e-3 / 0.005, 1e-2 / 0.002]
        self.t_norm = [0.16, 1. / 0.5]

    def scale_wrapper_get_scales(self):
        return dict(
            pose_scale={"pos": self.pose_scale[0], "rot": self.pose_scale[1]},
            vel_scale={"pos": self.vel_scale[0], "rot": self.vel_scale[1]},
            force_scale={"force": self.force_scale[0], "torque": self.force_scale[1] },
            t_norm={"mean": self.t_norm[0], "std": self.t_norm[1]}
        )

    def observation(self, obs):
        for both in ["left/", "right/"]:
            obs["state"][f"{both}tcp_pose"][:3] *= self.pose_scale[0]
            obs["state"][f"{both}tcp_pose"][3:] *= self.pose_scale[1]
            obs["state"][f"{both}tcp_vel"][:3] *= self.vel_scale[0]
            obs["state"][f"{both}tcp_vel"][3:] *= self.vel_scale[1]
            obs["state"][f"{both}tcp_force"] *= self.force_scale[0]
            obs["state"][f"{both}tcp_torque"] *= self.force_scale[1]
            obs["state"][f"{both}time_diff"] -= self.t_norm[0]
            obs["state"][f"{both}time_diff"] *= self.t_norm[1]

        obs["state"]["l2r/tcp_pose"][:3] *= self.pose_scale[0]
        obs["state"]["l2r/tcp_pose"][3:] *= self.pose_scale[1]
        obs["state"]["l2r/tcp_vel"][:3] *= self.vel_scale[0]
        obs["state"]["l2r/tcp_vel"][3:] *= self.vel_scale[1]
        obs["state"]["r2l/tcp_pose"][:3] *= self.pose_scale[0]
        obs["state"]["r2l/tcp_pose"][3:] *= self.pose_scale[1]
        obs["state"]["r2l/tcp_vel"][:3] *= self.vel_scale[0]
        obs["state"]["r2l/tcp_vel"][3:] *= self.vel_scale[1]
        return obs

