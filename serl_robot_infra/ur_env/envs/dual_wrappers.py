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

class DualScaleObservationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper scales the observations with the provided hyperparams
    (to somewhat normalize the observations space)
    """

    """
    from analyzing data: 
        action: -
        pose pos: 0., 0.1
        pose rot: 0., 0.05
        vel pos: 0., 0.06
        vel rot: 0., 0.02
        force: 0., 0.003
        torque: 0., 0.001
        t_diff: 0.16, 0.5
    """

    def __init__(self,
                 env,
                 pose_scale=[1. / 0.1, 1./0.05 * 1e-1],
                 vel_scale = [1. / 0.06, 1. / 0.02 * 1e-1],
                 force_scale = [1. / 0.003 * 1e-3, 1. / 0.001 * 1e-2],
                 t_norm = [0.16, 0.5]
                 ):
        super().__init__(env)
        self.pose_scale = pose_scale
        self.vel_scale = vel_scale
        self.force_scale = force_scale
        self.t_norm = t_norm

    def scale_wrapper_get_scales(self):
        return dict(
            pose_scale=self.pose_scale,
            vel_scale=self.vel_scale,
            force_scale=self.force_scale,
            t_norm=self.t_norm
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
            obs["state"][f"{both}time_diff"] /= self.t_norm[1]

        obs["state"]["l2r/tcp_pose"][:3] *= self.pose_scale[0]
        obs["state"]["l2r/tcp_pose"][3:] *= self.pose_scale[1]
        obs["state"]["l2r/tcp_vel"][:3] *= self.vel_scale[0]
        obs["state"]["l2r/tcp_vel"][3:] *= self.vel_scale[1]
        obs["state"]["r2l/tcp_pose"][:3] *= self.pose_scale[0]
        obs["state"]["r2l/tcp_pose"][3:] *= self.pose_scale[1]
        obs["state"]["r2l/tcp_vel"][:3] *= self.vel_scale[0]
        obs["state"]["r2l/tcp_vel"][3:] *= self.vel_scale[1]
        return obs

