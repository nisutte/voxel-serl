import copy

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from serl_robot_infra.ur_env.envs.relative_env import DualRelativeFrame
from scipy.spatial.transform import Rotation as R

zero_obs = {
    "state": {
        "left/tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1]),
        "left/tcp_vel": np.zeros(6),
        "left/tcp_force": np.zeros(3),
        "left/tcp_torque": np.zeros(3),
        "left/action": np.zeros(6),
        "right/tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1]),
        "right/tcp_vel": np.zeros(6),
        "right/tcp_force": np.zeros(3),
        "right/tcp_torque": np.zeros(3),
        "right/action": np.zeros(6),
        "l2r/tcp_pose": np.zeros(7),
        "l2r/tcp_vel": np.zeros(6),
        "r2l/tcp_pose": np.zeros(7),
        "r2l/tcp_vel": np.zeros(7)
    }
}


def apply_action(pose, action):
    next_pos = pose.copy()
    next_pos[:3] += action[:3] * 0.02
    next_pos[3:] = (R.from_mrp(action[3:6] * 0.1 / 4.) * R.from_quat(next_pos[3:])).as_quat()
    return next_pos


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict({
            "state": spaces.Dict({
                "left/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "left/tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "left/tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "left/tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "left/action": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "right/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "right/tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "right/tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "right/tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "right/action": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "rl2/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "r2l/tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
                "l2r/tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32),
                "l2r/tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            })
        })
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.curr_obs = None

        self.left_next_pose = None
        self.right_next_pose = None

    def reset(self, reset_pose=None, **kwargs):
        obs = zero_obs.copy()
        if reset_pose is not None:
            obs["state"]["left/tcp_pose"] = reset_pose[:7]
            obs["state"]["right/tcp_pose"] = reset_pose[7:]
        info = {}
        self.curr_obs = copy.deepcopy(obs)
        return obs, info

    def step(self, action):
        if self.left_next_pose is not None and self.right_next_pose is not None:
            self.curr_obs["state"]["left/tcp_pose"] = self.left_next_pose
            self.curr_obs["state"]["right/tcp_pose"] = self.right_next_pose

        obs = copy.deepcopy(self.curr_obs)
        for both, a in zip(["left/", "right/"], [action[:7], action[7:]]):
            next_pos = self.curr_obs["state"][both + "tcp_pose"]
            next_pos[:3] += a[:3] * 0.02
            next_pos[3:] = (R.from_mrp(a[3:6] * 0.1 / 4.) * R.from_quat(next_pos[3:])).as_quat()
            if "left" in both:
                self.left_next_pose = next_pos
            else:
                self.right_next_pose = next_pos

        return obs, 0.0, False, False, {}


def test_dual_relative_frame_identity():
    env = DummyEnv()
    env = DualRelativeFrame(env)  # type: ignore
    obs, info = env.reset()
    left_pose = obs["state"]["left/tcp_pose"]
    right_pose = obs["state"]["right/tcp_pose"]
    np.testing.assert_allclose(left_pose[:3], 0, atol=1e-6)
    np.testing.assert_allclose(left_pose[3:], [0, 0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(right_pose[:3], 0, atol=1e-6)
    np.testing.assert_allclose(right_pose[3:], [0, 0, 0, 1], atol=1e-6)


def test_dual_relative_frame_step():
    env = DummyEnv()
    env = DualRelativeFrame(env)  # type: ignore

    # random reset pose
    reset_pose = np.random.uniform(low=-1.0, high=1.0, size=(14,))
    reset_pose[3:7] = R.random().as_quat()
    reset_pose[10:14] = R.random().as_quat()

    obs, _ = env.reset(reset_pose=reset_pose)
    last_action = None

    for i in range(10):
        action = np.random.uniform(low=-1.0, high=1.0, size=(14,))
        new_obs, *_ = env.step(action)

        if last_action is not None:
            np.testing.assert_allclose(apply_action(obs["state"]["left/tcp_pose"], last_action[:7]), new_obs["state"]["left/tcp_pose"], atol=1e-5)
            np.testing.assert_allclose(apply_action(obs["state"]["right/tcp_pose"], last_action[7:]), new_obs["state"]["right/tcp_pose"], atol=1e-6)

        last_action = action
        obs = copy.deepcopy(new_obs)
