import threading
import numpy as np
import time
from ur_env.envs.dual_ur5_env import DualUR5Env


class SimpleBehaviorTree:
    def __init__(self, env):
        self.env = env
        self.pickup_done = threading.Event()

    def move_to_joints(self, target_Q):
        self.env.send_reset_command(target_Q)

    def retreat(self):
        self.env.update_currpos()
        back = self.env.curr_pos.copy()
        back[1] = 0.5
        self.move_to_pose(back)

    def move_to_pose(self, pose, velocity=0.001):
        self.env.update_currpos()
        old = self.env.curr_pos
        max_pos_diff = np.max(np.abs(old - pose)[:3])
        N = int(max_pos_diff/velocity)
        print(N)
        for i in range(N):
            alpha = (1. - np.cos(i/N * np.pi)) / 2.
            self.env.send_pos_command(alpha * pose + (1. - alpha) * old)
            time.sleep(0.02)
        self.env.send_pos_command(pose)
        print(f"moved to pose {pose}")

    def pickup(self) -> bool:
        pickup_Q = [-0.7027, -0.8565, 1.1014, -1.8162, -1.5657, -0.7058]
        self.env.send_reset_command(np.asarray(pickup_Q))
        i = 0
        for _ in range(200):
            self.env.update_currpos()
            DOWN = np.asarray([0, 0, -0.01, 0, 0, 0, 0])
            if self.env.gripper_state[1] == 1:
                DOWN = -DOWN
                i += 1
            if i > 10:
                self.pickup_done.set()
                return True

            new_pose = self.env.curr_pos + DOWN
            self.env.send_pos_command(new_pose)
            time.sleep(0.05)

            if self.env.curr_vel[2] >= 0. and self.env.gripper_state[1] == 0:
                self.env._send_gripper_command(np.array(1))
        return False


class UR5HandoverEnv(DualUR5Env):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        BTleft, BTright = SimpleBehaviorTree(self.env_left), SimpleBehaviorTree(self.env_right)

        def reset_env_left():
            global ob_left
            BTleft.retreat()
            while not BTright.pickup_done.is_set():
                time.sleep(0.1)
            ob_left, _ = self.env_left.reset(**kwargs)

        def reset_env_right():
            global ob_right
            BTright.retreat()
            time.sleep(0.5)
            self.env_right.send_reset_command(np.asarray([-0.7027, -0.8565,  1.1014, -1.8162, -1.5657, -0.7058]))
            while not BTright.pickup():
                time.sleep(0.5)
            ob_right, _ = self.env_right.reset(**kwargs)

        thread_left = threading.Thread(target=reset_env_left)
        thread_right = threading.Thread(target=reset_env_right)
        thread_left.start()
        thread_right.start()
        thread_left.join()
        thread_right.join()

        ob = self.combine_obs(ob_left, ob_right)
        return ob, {}

    def close(self):
        super().close()
