import numpy as np
import gymnasium as gym
import queue
import time
import threading
from typing import Dict, Tuple


class DualUR5Env(gym.Env):
    def __init__(
            self,
            env_left,
            env_right,
    ):
        self.env_left = env_left
        self.env_right = env_right

        action_dim = len(self.env_left.action_space.low) + len(self.env_right.action_space.low)
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )
        if env_left.camera_mode is not None:
            image_dict = ({f"left/{key}": self.env_left.observation_space["images"][key] for key in
                           self.env_left.observation_space["images"].keys()} |
                          {f"right/{key}": self.env_right.observation_space["images"][key] for key in
                           self.env_right.observation_space["images"].keys()})

        state_dict = ({f"left/{key}": self.env_left.observation_space["state"][key] for key in
                       self.env_left.observation_space["state"].keys()} |
                      {f"right/{key}": self.env_right.observation_space["state"][key] for key in
                       self.env_right.observation_space["state"].keys()})

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(state_dict),
                # "images": gym.spaces.Dict(image_dict)         # TODO only temporarly
            }
        )

    def step(self, action: np.ndarray) -> tuple:
        action_left = action[:len(action) // 2]
        action_right = action[len(action) // 2:]

        def step_env_left():
            global ob_left, reward_left, done_left
            ob_left, reward_left, done_left, _, _ = self.env_left.step(action_left)

        def step_env_right():
            global ob_right, reward_right, done_right
            ob_right, reward_right, done_right, _, _ = self.env_right.step(action_right)

        # Create threads for each function
        thread_left = threading.Thread(target=step_env_left)
        thread_right = threading.Thread(target=step_env_right)

        # Start the threads
        thread_left.start()
        thread_right.start()

        # Wait for both threads to complete
        thread_left.join()
        thread_right.join()
        ob = self.combine_obs(ob_left, ob_right)
        # TODO check if int(left and right) is right!
        return ob, int(reward_left and reward_right), done_left or done_right, False, {}

    def reset(self, **kwargs):
        def reset_env_left():
            global ob_left
            ob_left, _ = self.env_left.reset(**kwargs)

        def reset_env_right():
            global ob_right
            ob_right, _ = self.env_right.reset(**kwargs)

        thread_left = threading.Thread(target=reset_env_left)
        thread_right = threading.Thread(target=reset_env_right)
        thread_left.start()
        thread_right.start()
        thread_left.join()
        thread_right.join()

        ob = self.combine_obs(ob_left, ob_right)
        return ob, {}

    def combine_obs(self, ob_left, ob_right):
        # left_images = {f"left/{key}": ob_left["images"][key] for key in ob_left["images"].keys()}
        # right_images = {f"right/{key}": ob_right["images"][key] for key in ob_right["images"].keys()}
        left_state = {f"left/{key}": ob_left["state"][key] for key in ob_left["state"].keys()}
        right_state = {f"right/{key}": ob_right["state"][key] for key in ob_right["state"].keys()}
        ob = {
            "state": left_state | right_state,
            # "images": left_images | right_images
        }
        return ob
