import numpy as np
from collections import deque
import gymnasium as gym

from ur_env.envs.dual_ur5_env import DualUR5Env
from ur_env.envs.ur5_env import UR5Env


class ObservationStatisticsWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """
    This wrapper will keep track of the observation statistics.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``obsStat``.
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        gym.utils.RecordConstructorArgs.__init__(self, deque_size=deque_size)
        gym.Wrapper.__init__(self, env)

        self.buffer = {}
        if isinstance(env.unwrapped, DualUR5Env):
            max_episode_length = self.env_left.max_episode_length
            self.curr_path_length = self.env_left.curr_path_length
        elif isinstance(env.unwrapped, UR5Env):
            max_episode_length = self.max_episode_length
        else:
            raise NotImplementedError(f"Observation statistics wrapper does not support {type(env)}")

        # make buffer
        for name, space in self.env.observation_space["state"].items():
            self.buffer[name] = np.zeros(shape=(max_episode_length, space.shape[0]))

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."

        for name, obs in observations["state"].items():
            self.buffer[name][self.curr_path_length - 1, :] = obs

        dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)
        if num_dones and self.curr_path_length:
            calc_buffs = {}
            calc_buffs.update({
                name + "_mean": np.mean(obs[:self.curr_path_length], axis=0) for name, obs in self.buffer.items()
            })
            calc_buffs.update({
                name + "_std": np.std(obs[:self.curr_path_length], axis=0) for name, obs in self.buffer.items()
            })
            buff = {}
            for name, value in calc_buffs.items():
                for i in range(value.shape[0]):
                    buff[name + f"_{['x', 'y', 'z', 'rx', 'ry', 'rz', 'grip'][i]}"] = value[i]
            infos["obsStat"] = buff
            # print(buff)

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(**kwargs)

        # reset buffer to zero
        for name, value in self.buffer.items():
            value[...] = 0

        return obs, info
