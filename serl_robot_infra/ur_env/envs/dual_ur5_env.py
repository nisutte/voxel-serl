import numpy as np
import gymnasium as gym
import threading
from typing import Dict, Tuple


from ur_env.envs.ur5_env import ImageDisplayer, PointCloudDisplayer


class CombinedQueue:
    def __init__(self, queue_left, queue_right):
        self.queue_left = queue_left
        self.queue_right = queue_right

    def get(self):
        retval = {}
        for key, value in self.queue_left.get().items():
            retval[key +"_left"] = value
        for key, value in self.queue_right.get().items():
            retval[key +"_right"] = value
        return retval


class DualPointCloudDisplayer:
    def __init__(self, left=100, top=100):
        self.left = PointCloudDisplayer(left=left, top=top)
        self.right = PointCloudDisplayer(left=left + 520, top=top)

    def display_left(self, points):
        self.left.display(points)

    def display_right(self, points):
        self.right.display(points)


class DualUR5Env(gym.Env):
    def __init__(
            self,
            env_left,
            env_right,
    ):
        self.env_left = env_left
        self.env_right = env_right

        assert self.env_left.camera_mode == self.env_right.camera_mode
        self.camera_mode = self.env_left.camera_mode

        action_dim = len(self.env_left.action_space.low) + len(self.env_right.action_space.low)
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )
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
                "images": gym.spaces.Dict(image_dict)
            }
        )

        if self.camera_mode is not None:
            combined_queue = CombinedQueue(self.env_left.img_queue, self.env_right.img_queue)
            if self.camera_mode in ["pointcloud"]:
                self.pc_displayer = DualPointCloudDisplayer()
            else:
                self.displayer = ImageDisplayer(combined_queue)
                self.displayer.start()


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

        # visualize pointcloud (has to be in the main thread)
        if self.camera_mode in ["pointcloud"]:
            self.pc_displayer.display_left(self.env_left.displayer.get())
            self.pc_displayer.display_right(self.env_right.displayer.get())

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
        left_images = {f"left/{key}": ob_left["images"][key] for key in ob_left["images"].keys()}
        right_images = {f"right/{key}": ob_right["images"][key] for key in ob_right["images"].keys()}
        left_state = {f"left/{key}": ob_left["state"][key] for key in ob_left["state"].keys()}
        right_state = {f"right/{key}": ob_right["state"][key] for key in ob_right["state"].keys()}
        ob = {
            "state": left_state | right_state,
            "images": left_images | right_images
        }
        return ob
