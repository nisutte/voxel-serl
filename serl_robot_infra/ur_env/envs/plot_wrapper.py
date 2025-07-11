import os
import gymnasium as gym

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class PlotWrapper(gym.Wrapper):
    def __init__(self, env, save_dir="trajectories"):
        super().__init__(env)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.trajectory = []
        self._keys = ("left/tcp_pose", "right/tcp_pose", "left/tcp_vel", "right/tcp_vel", "left/action", "right/action")

    def reset(self, **kwargs):
        self.trajectory = []
        obs, info = self.env.reset(**kwargs)
        self.trajectory.append({k:v for k, v in obs["state"].items() if k in self._keys})
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.trajectory.append({k:v for k, v in obs["state"].items() if k in self._keys})
        if done and len(self.trajectory) > 25:
            self._plot_trajectory()
        return obs, reward, done, truncated, info

    def _plot_trajectory(self):
        trajectory = {k: np.array([d[k] for d in self.trajectory]) for k in self._keys}
        plt.figure(figsize=(10, 8))

        num_keys = len(self._keys)
        fig, axes = plt.subplots(num_keys, 1, figsize=(10, 5 * num_keys))
        t = np.arange(len(self.trajectory))
        for i, key in enumerate(self._keys):
            data = np.array(trajectory[key])
            axes[i].plot(t, data[:, 0], marker='.', markersize=4, label="X")
            axes[i].plot(t, data[:, 1], marker='.', markersize=4, label="Y")
            axes[i].plot(t, data[:, 2], marker='.', markersize=4, label="Z")
            axes[i].set_title(f"Trajectory for {key}")
            axes[i].set_xlabel("step")
            axes[i].set_ylabel("position / velocity")
            axes[i].legend(loc="upper left")

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.save_dir, f"trajectory_{timestamp}.svg")
        plt.savefig(filepath)
        plt.close()
