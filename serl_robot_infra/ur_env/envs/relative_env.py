from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gymnasium import Env
from franka_env.utils.transformations import (
    construct_homogeneous_matrix,
    construct_rotation_matrix,
)

class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base UR5 environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)),
                "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "gripper_state": spaces.Box(-np.inf, np.inf, shape=(2,)),
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.rotation_matrix_reset = np.eye((3))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"]

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # obs['state']['tcp_pose'][:2] -= info['reset_shift']  # set rel pose to original reset pose (no random)

        self.rotation_matrix_reset = construct_rotation_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the rotation and homogeneous matrix
        """
        obs["state"]["tcp_vel"][:3] = self.rotation_matrix_reset.T @ obs["state"]["tcp_vel"][:3]
        obs["state"]["tcp_vel"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["tcp_vel"][3:6]
        obs["state"]["tcp_force"] = self.rotation_matrix_reset.T @ obs["state"]["tcp_force"]
        obs["state"]["tcp_torque"] = self.rotation_matrix_reset.T @ obs["state"]["tcp_torque"]
        obs["state"]["action"] = self.transform_action_inv(obs["state"]["action"])

        if "ema_tcp_vel" in obs["state"]:
            obs["state"]["ema_tcp_vel"][:3] = self.rotation_matrix_reset.T @ obs["state"]["ema_tcp_vel"][:3]
            obs["state"]["ema_tcp_vel"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["ema_tcp_vel"][3:6]
        if "ema_force" in obs["state"]:
            obs["state"]["ema_force"][:3] = self.rotation_matrix_reset.T @ obs["state"]["ema_force"][:3]
            obs["state"]["ema_force"][3:6] = self.rotation_matrix_reset.T @ obs["state"]["ema_force"][3:6]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into spatial(base) frame
        using the rotation matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:3] = self.rotation_matrix_reset @ action[:3]
        action[3:6] = self.rotation_matrix_reset @ action[3:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the rotation matrix.
        """
        action = np.array(action)
        action[:3] = self.rotation_matrix_reset.T @ action[:3]
        action[3:6] = self.rotation_matrix_reset.T @ action[3:6]
        return action


class DualRelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the DualUR5Env, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ...
            }
        ),
        ......
    }, and at least 12 DoF action space
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.rot_mat_left = np.eye((3))
        self.rot_mat_right = np.eye((3))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.zeros((4, 4))
            self.right_T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"]

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update rotation matrices
        self.rot_mat_left = construct_rotation_matrix(obs["state"]["left/tcp_pose"])
        self.rot_mat_right = construct_rotation_matrix(obs["state"]["right/tcp_pose"])

        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.left_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            )
            self.right_T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            )
        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the rotation and homogeneous matrix
        """
        for both, rot_mat in zip(("left/", "right/"), (self.rot_mat_left, self.rot_mat_right)):
            # velocities (twist) rotate as vectors
            obs["state"][f"{both}tcp_vel"][:3] = rot_mat.T @ obs["state"][f"{both}tcp_vel"][:3]
            obs["state"][f"{both}tcp_vel"][3:6] = rot_mat.T @ obs["state"][f"{both}tcp_vel"][3:6]
            # forces/torques are vectors/pseudovectors
            obs["state"][f"{both}tcp_force"] = rot_mat.T @ obs["state"][f"{both}tcp_force"]
            obs["state"][f"{both}tcp_torque"] = rot_mat.T @ obs["state"][f"{both}tcp_torque"]
            # action in observation assumed to be a twist; rotate like velocities
            obs["state"][f"{both}action"][:3] = rot_mat.T @ obs["state"][f"{both}action"][:3]
            obs["state"][f"{both}action"][3:6] = rot_mat.T @ obs["state"][f"{both}action"][3:6]

            key_v = f"{both}ema_tcp_vel"
            key_f = f"{both}ema_force"
            if key_v in obs["state"]:
                obs["state"][key_v][:3] = rot_mat.T @ obs["state"][key_v][:3]
                obs["state"][key_v][3:6] = rot_mat.T @ obs["state"][key_v][3:6]
            if key_f in obs["state"]:
                obs["state"][key_f][:3] = rot_mat.T @ obs["state"][key_f][:3]
                obs["state"][key_f][3:6] = rot_mat.T @ obs["state"][key_f][3:6]


        if self.include_relative_pose:
            left_T_b_o = construct_homogeneous_matrix(obs["state"]["left/tcp_pose"])
            left_T_b_r = self.left_T_r_o_inv @ left_T_b_o

            left_p_b_r = left_T_b_r[:3, 3]
            left_theta_b_r = R.from_matrix(left_T_b_r[:3, :3]).as_quat()
            obs["state"]["left/tcp_pose"] = np.concatenate((left_p_b_r, left_theta_b_r))

            right_T_b_o = construct_homogeneous_matrix(obs["state"]["right/tcp_pose"])
            right_T_b_r = self.right_T_r_o_inv @ right_T_b_o

            right_p_b_r = right_T_b_r[:3, 3]
            right_theta_b_r = R.from_matrix(right_T_b_r[:3, :3]).as_quat()
            obs["state"]["right/tcp_pose"] = np.concatenate((right_p_b_r, right_theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action (12d) from body(end-effector) frame into spatial(base) frame
        using the rotation matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:3] = self.rot_mat_left @ action[:3]
        action[3:6] = self.rot_mat_left @ action[3:6]
        action[7:10] = self.rot_mat_right @ action[7:10]
        action[10:13] = self.rot_mat_right @ action[10:13]
        return action

class BaseFrameRotation(gym.Wrapper):
    """
    Watch out, is legacy code, not used anywhere.
    """
    def __init__(self, env: Env, rx=0., ry=0., rz=0.):
        super().__init__(env)
        self.base_frame_rotation = R.from_euler("xyz", [rx, ry, rz]).as_matrix()

    def step(self, action: np.ndarray):
        transformed_action = self.base_transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"]

        transformed_obs = self.base_transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.base_transform_observation(obs), info

    def base_transform_observation(self, obs):
        """
        Transform observations from base frame to the rotated frame
        """
        obs["state"]["tcp_pose"][:3] = self.base_frame_rotation @ obs["state"]["tcp_pose"][:3]
        obs["state"]["tcp_pose"][3:] = (R.from_quat(obs["state"]["tcp_pose"][3:6]) * R.from_matrix(self.base_frame_rotation)).as_quat()
        obs["state"]["tcp_vel"][:3] = self.base_frame_rotation.T @ obs["state"]["tcp_vel"][:3]
        obs["state"]["tcp_vel"][3:6] = self.base_frame_rotation.T @ obs["state"]["tcp_vel"][3:6]
        obs["state"]["tcp_force"] = self.base_frame_rotation.T @ obs["state"]["tcp_force"]
        obs["state"]["tcp_torque"] = self.base_frame_rotation.T @ obs["state"]["tcp_torque"]
        return obs

    def base_transform_action(self, action: np.ndarray):
        action = np.array(action)  # in case action is a jax read-only array
        action[:3] = self.base_frame_rotation @ action[:3]
        action[3:6] = (R.from_mrp(action[3:6]) * R.from_matrix(self.base_frame_rotation)).as_mrp()
        return action