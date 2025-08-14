import numpy as np
import gymnasium as gym
import threading

from ur_env.envs.ur5_env import ImageDisplayer, PointCloudDisplayer, UR5Env
from ur_env.utils.threaded_collision_detection import ThreadedCollisionDetector
from ur_env.utils.transformations import T_to_pose, pose_to_T, transform_twist


class CombinedQueue:
    def __init__(self, queue_left, queue_right):
        self.queue_left = queue_left
        self.queue_right = queue_right

    def get(self):
        retval = {}
        for key, value in self.queue_left.get().items():
            retval[key + "_left"] = value
        for key, value in self.queue_right.get().items():
            retval[key + "_right"] = value
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
            env_left: UR5Env,
            env_right: UR5Env,
            fake_env=False,
    ):
        self.env_left = env_left
        self.env_right = env_right
        self.fake_env = fake_env

        self.cost_infos = {}

        assert self.env_left.camera_mode == self.env_right.camera_mode
        self.camera_mode = self.env_left.camera_mode

        action_dim = len(self.env_left.action_space.low) + len(self.env_right.action_space.low)
        self.action_space = gym.spaces.Box(
            np.ones((action_dim,), dtype=np.float32) * -1,
            np.ones((action_dim,), dtype=np.float32),
        )
        self.last_action = np.zeros(self.action_space.shape)

        state_dict = ({f"left/{key}": self.env_left.observation_space["state"][key] for key in
                       self.env_left.observation_space["state"].keys()} |
                      {f"right/{key}": self.env_right.observation_space["state"][key] for key in
                       self.env_right.observation_space["state"].keys()})

        state_dict["l2r/tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(7,))
        state_dict["r2l/tcp_pose"] = gym.spaces.Box(-np.inf, np.inf, shape=(7,))
        state_dict["l2r/tcp_vel"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))
        state_dict["r2l/tcp_vel"] = gym.spaces.Box(-np.inf, np.inf, shape=(6,))

        self.observation_space = gym.spaces.Dict({"state": gym.spaces.Dict(state_dict)})

        if self.camera_mode is not None:
            image_dict = ({f"left/{key}": self.env_left.observation_space["images"][key] for key in
                           self.env_left.observation_space["images"].keys()} |
                          {f"right/{key}": self.env_right.observation_space["images"][key] for key in
                           self.env_right.observation_space["images"].keys()})
            self.observation_space["images"] = gym.spaces.Dict(image_dict)

        if fake_env:
            print("[DualUR5Env] is fake!")
            return

        # collision detection
        T_base2left = np.load(env_left.config.CALIBRATION_PATH)
        T_base2right = np.load(env_right.config.CALIBRATION_PATH)
        self.T_left2right = np.linalg.inv(T_base2left) @ T_base2right
        self.T_right2left = np.linalg.inv(self.T_left2right)
        self.collision_detector = ThreadedCollisionDetector(np.eye(4), self.T_left2right, headless=False,
                                                            distance_margin=0.02)
        self.collision_detector.start()

        if self.camera_mode is not None:
            combined_queue = CombinedQueue(self.env_left.img_queue, self.env_right.img_queue)
            if self.camera_mode in ["pointcloud"]:
                self.pc_displayer = DualPointCloudDisplayer()
            else:
                self.displayer = ImageDisplayer(combined_queue)
                self.displayer.start()

    def compute_reward(self, obs, action) -> float:
        raise NotImplementedError  # overwrite for each task

    def reached_goal_state(self, obs, **kwargs) -> bool:
        raise NotImplementedError  # overwrite for each task

    def _is_truncated(self, obs) -> bool:
        raise NotImplementedError  # overwrite for each task

    def get_cost_infos(self, done):
        if not done:
            return {}
        cost_infos = self.cost_infos.copy()
        self.cost_infos = {}
        return cost_infos

    def step(self, action: np.ndarray) -> tuple:
        action_left = action[:len(action) // 2]
        action_right = action[len(action) // 2:]

        def step_env_left():
            global ob_left, truncated_left, infos_left
            ob_left, _, _, truncated_left, infos_left = self.env_left.step(action_left)

        def step_env_right():
            global ob_right, truncated_right, infos_right
            ob_right, _, _, truncated_right, infos_right = self.env_right.step(action_right)

        # Create threads for each function
        thread_left = threading.Thread(target=step_env_left)
        thread_right = threading.Thread(target=step_env_right)

        # Start the threads
        thread_left.start()
        thread_right.start()

        # Wait for both threads to complete
        thread_left.join()
        thread_right.join()
        obs = self.combine_obs(ob_left, ob_right)

        truncated = truncated_left or truncated_right or self._is_truncated(obs)
        if truncated:
            print(f"is truncated on step {self.env_right.curr_path_length}   {(truncated_left, truncated_right, self._is_truncated(obs))}")
        done = (self.env_left.curr_path_length >= self.env_left.max_episode_length or truncated or
                self.reached_goal_state(obs))

        reward = self.compute_reward(obs, action)
        reward = reward if (not truncated or self.env_left.curr_path_length < 2) else reward - 25.     # cost for truncation

        # visualize pointcloud (has to be in the main thread)
        if self.camera_mode in ["pointcloud"]:
            self.pc_displayer.display_left(self.env_left.displayer.get())
            self.pc_displayer.display_right(self.env_right.displayer.get())

        return obs, reward, done, truncated, self.get_cost_infos(done)

    def reset(self, **kwargs):
        raise NotImplementedError

    def combine_obs(self, ob_left, ob_right):
        left_state = {f"left/{key}": ob_left["state"][key] for key in ob_left["state"].keys()}
        right_state = {f"right/{key}": ob_right["state"][key] for key in ob_right["state"].keys()}

        # T_l2r = T_eeLeft2baseLeft @ T_baseLeft2baseRight @ T_baseRight2eeRight
        T_l2r = np.linalg.inv(pose_to_T(ob_left["state"]["tcp_pose"])) @ self.T_left2right @ pose_to_T(
            ob_right["state"]["tcp_pose"])

        # Both tcp velocities are in base frame, but on the ee (weird i know...)
        vel_diff_right_base = ob_right["state"]["tcp_vel"] - transform_twist(ob_left["state"]["tcp_vel"], self.T_left2right)
        vel_diff_left_base = ob_left["state"]["tcp_vel"] - transform_twist(ob_right["state"]["tcp_vel"], self.T_right2left)
        vel_diff_left = transform_twist(vel_diff_left_base, pose_to_T(ob_left["state"]["tcp_pose"]).T)
        vel_diff_right = transform_twist(vel_diff_right_base, pose_to_T(ob_right["state"]["tcp_pose"]).T)

        diff = {
            "l2r/tcp_pose": T_to_pose(T_l2r),
            "l2r/tcp_vel": vel_diff_left,
            "r2l/tcp_pose": T_to_pose(np.linalg.inv(T_l2r)),
            "r2l/tcp_vel": vel_diff_right,
        }
        ob = {"state": left_state | right_state | diff}

        if self.camera_mode:
            left_images = {f"left/{key}": ob_left["images"][key] for key in ob_left["images"].keys()}
            right_images = {f"right/{key}": ob_right["images"][key] for key in ob_right["images"].keys()}
            ob["images"] = left_images | right_images

        if not self.fake_env:
            self.collision_detector.update_joint_state("robot_left", self.env_left.curr_Q)
            self.collision_detector.update_joint_state("robot_right", self.env_right.curr_Q)

        return ob

    def close(self):
        self.env_left.close()
        self.env_right.close()
        super().close()

