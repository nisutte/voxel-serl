import gymnasium as gym
import numpy as np

from gymnasium import Env

from ur_env.utils.rotations import quat_2_mrp, omega_to_mrp_dot


class DualToMrpWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to mrp angles
    """

    def __init__(self, dual_env: Env, transform_obs=True):
        import ur_env
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
            # Map angular velocity to MRP rate using current MRPs
            sigma_left = obs["state"]["left/tcp_pose"][3:6]
            sigma_right = obs["state"]["right/tcp_pose"][3:6]
            omega_left = obs["state"]["left/tcp_vel"][3:6]
            omega_right = obs["state"]["right/tcp_vel"][3:6]
            obs["state"]["left/tcp_vel"][3:6] = omega_to_mrp_dot(sigma_left, omega_left)
            obs["state"]["right/tcp_vel"][3:6] = omega_to_mrp_dot(sigma_right, omega_right)
            # Convert EMA velocities if present
            if "left/ema_tcp_vel" in obs["state"]:
                obs["state"]["left/ema_tcp_vel"][3:6] = omega_to_mrp_dot(
                    sigma_left, obs["state"]["left/ema_tcp_vel"][3:6]
                )
            if "right/ema_tcp_vel" in obs["state"]:
                obs["state"]["right/ema_tcp_vel"][3:6] = omega_to_mrp_dot(
                    sigma_right, obs["state"]["right/ema_tcp_vel"][3:6]
                )

            # Convert relative angular velocities if present
            if "l2r/tcp_vel" in obs["state"]:
                sigma_l2r = obs["state"]["l2r/tcp_pose"][3:6]
                omega_l2r = obs["state"]["l2r/tcp_vel"][3:6]
                obs["state"]["l2r/tcp_vel"][3:6] = omega_to_mrp_dot(sigma_l2r, omega_l2r)
            if "r2l/tcp_vel" in obs["state"]:
                sigma_r2l = obs["state"]["r2l/tcp_pose"][3:6]
                omega_r2l = obs["state"]["r2l/tcp_vel"][3:6]
                obs["state"]["r2l/tcp_vel"][3:6] = omega_to_mrp_dot(sigma_r2l, omega_r2l)

            # make Rx rotation positive and canonicalize (avoid +/- pi flips)
            for key in ["l2r/tcp_pose", "r2l/tcp_pose"]:
                mrp_rx = obs["state"][key][3]
                obs["state"][key][3] = mrp_rx - np.sign(mrp_rx) * 1.0   # 0.9 -> -0.1 and -0.9 -> 0.1

        return obs

class DualNormalizationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper scales the observations with the provided hyperparams
    """

    """
    from analyzing data: 
        action: -
        pose pos: x10 (10cm is std 1)
        pose rot: x10 (22° is std 1)
        vel pos: x10 (10cm/s is std 1, max possible is 20cm/s)
        vel rot: x10 (max is 0.1/s)
        force: leave as tested in the dataset, x0.2
        torque: leave as tested in the dataset, x0.2
        t_diff: off by 0.16, x2
    """

    def __init__(self, env):
        super().__init__(env)
        self.pose_scale = [10., 10.]
        self.vel_scale = [10., 10.]
        self.force_scale = [0.2, 10.]
        self.t_norm = [0.16, 2]

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

