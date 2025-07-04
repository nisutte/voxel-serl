import numpy as np
import gymnasium as gym
from gymnasium import spaces
from serl_robot_infra.ur_env.envs.relative_env import DualRelativeFrame


class DummyEnv(gym.Env):
    def __init__(self, offset=0.0, step_offset=0.0):
        super().__init__()
        self.offset = offset
        self.step_offset = step_offset
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
            })
        })
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self._stepped = False
    def reset(self, **kwargs):
        self._stepped = False
        obs = {
            "state": {
                "left/tcp_pose": np.array([0 + self.offset, 0, 0, 1, 0, 0, 0]),
                "left/tcp_vel": np.zeros(6),
                "left/tcp_force": np.zeros(3),
                "left/tcp_torque": np.zeros(3),
                "left/action": np.zeros(6),
                "right/tcp_pose": np.array([1 + self.offset, 0, 0, 1, 0, 0, 0]),
                "right/tcp_vel": np.zeros(6),
                "right/tcp_force": np.zeros(3),
                "right/tcp_torque": np.zeros(3),
                "right/action": np.zeros(6),
            }
        }
        info = {}
        return obs, info
    def step(self, action):
        self._stepped = True
        obs = {
            "state": {
                "left/tcp_pose": np.array([0 + self.offset + self.step_offset, 0, 0, 1, 0, 0, 0]),
                "left/tcp_vel": np.zeros(6),
                "left/tcp_force": np.zeros(3),
                "left/tcp_torque": np.zeros(3),
                "left/action": np.zeros(6),
                "right/tcp_pose": np.array([1 + self.offset + self.step_offset, 0, 0, 1, 0, 0, 0]),
                "right/tcp_vel": np.zeros(6),
                "right/tcp_force": np.zeros(3),
                "right/tcp_torque": np.zeros(3),
                "right/action": np.zeros(6),
            }
        }
        return obs, 0.0, False, False, {}

def test_dual_relative_frame_identity():
    env = DummyEnv()
    wrapper = DualRelativeFrame(env)  # type: ignore
    obs, info = wrapper.reset()
    left_pose = obs["state"]["left/tcp_pose"]
    right_pose = obs["state"]["right/tcp_pose"]
    np.testing.assert_allclose(left_pose[:3], 0, atol=1e-6)
    np.testing.assert_allclose(left_pose[3:], [1,0,0,0], atol=1e-6)
    np.testing.assert_allclose(right_pose[:3], 0, atol=1e-6)
    np.testing.assert_allclose(right_pose[3:], [1,0,0,0], atol=1e-6)

def test_dual_relative_frame_offset():
    env = DummyEnv(step_offset=0.5)
    wrapper = DualRelativeFrame(env)  # type: ignore
    wrapper.reset()
    obs, *_ = wrapper.step(np.zeros(14))
    left_pose = obs["state"]["left/tcp_pose"]
    right_pose = obs["state"]["right/tcp_pose"]
    np.testing.assert_allclose(left_pose[:3], [0.5,0,0], atol=1e-6)
    np.testing.assert_allclose(left_pose[3:], [1,0,0,0], atol=1e-6)
    np.testing.assert_allclose(right_pose[:3], [0.5,0,0], atol=1e-6)
    np.testing.assert_allclose(right_pose[3:], [1,0,0,0], atol=1e-6) 