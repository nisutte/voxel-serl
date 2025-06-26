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
        if back[1] > 0.5:
            back[1] = 0.5
            self.move_to_pose(back)

    def move_to_pose(self, pose, velocity=0.001):
        self.env.update_currpos()
        old = self.env.curr_pos
        max_pos_diff = np.max(np.abs(old - pose)[:3])
        N = int(max_pos_diff / velocity)
        for i in range(N):
            alpha = (1. - np.cos(i / N * np.pi)) / 2.
            self.env.send_pos_command(alpha * pose + (1. - alpha) * old)
            time.sleep(0.02)
        self.env.send_pos_command(pose)

    def pickup(self) -> bool:
        self.env.update_currpos()
        if self.env.gripper_state[1] > 0.5:
            self.pickup_done.set()
            return True

        pickup_Q = [-0.7027, -0.8565, 1.1014, -1.8162, -1.5657, -0.7058]
        self.env.send_reset_command(np.asarray(pickup_Q))
        self.env.controller.reset_forces()
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
        """
        The goal of the env is always to give the parcel from right to left.
        Even if the env is inverted, the obs are from right to left.
        """
        self.goal_state_increment: int = 0
        self.inverted = False

    def combine_obs(self, ob_left, ob_right):
        if self.inverted:
            return super().combine_obs(ob_right, ob_left)
        else:
            return super().combine_obs(ob_left, ob_right)

    def step(self, action: np.ndarray) -> tuple:
        if self.inverted:
            action = np.concatenate((action[7:], action[:7]))
        return super().step(action)

    def reset(self, **kwargs):
        BTleft, BTright = SimpleBehaviorTree(self.env_left), SimpleBehaviorTree(self.env_right)
        self.env_left.update_currpos()
        self.env_right.update_currpos()

        already_picked_up = False
        if self.env_left.gripper_state[1] > 0.5 and self.env_right.gripper_state[1] < 0.5:
            # left gripper gripping, right gripper not gripping
            print("Env is inverted!")
            self.inverted = True
            already_picked_up = True
        elif self.env_right.gripper_state[1] > 0.5 and self.env_left.gripper_state[1] < 0.5:
            # right gripper gripping, left gripper not gripping
            self.inverted = False
            already_picked_up = True
        else:
            # both grippers not gripping or both gripping (also wrong)
            self.env_left._send_gripper_command(np.array(0))
            self.env_right._send_gripper_command(np.array(0))
            self.inverted = False

        def reset_env_left():
            global ob_left
            BTleft.retreat()
            if not already_picked_up:
                while not BTright.pickup_done.is_set():
                    time.sleep(0.1)

            self.env_left.controller.auto_release_gripper(not self.inverted)
            ob_left, _ = self.env_left.reset(**kwargs)
            self.env_left.controller.auto_release_gripper(True)

            if not self.inverted:
                self.env_left.controller.reset_forces()

        def reset_env_right():
            global ob_right
            BTright.retreat()
            time.sleep(0.5)
            if not already_picked_up:
                while not BTright.pickup():
                    time.sleep(0.5)

            self.env_right.controller.auto_release_gripper(self.inverted)
            ob_right, _ = self.env_right.reset(**kwargs)
            self.env_right.controller.auto_release_gripper(True)

            if self.inverted:
                self.env_right.controller.reset_forces()

        thread_left = threading.Thread(target=reset_env_left, daemon=True)
        thread_right = threading.Thread(target=reset_env_right, daemon=True)
        thread_left.start()
        thread_right.start()
        thread_left.join()
        thread_right.join()

        self.goal_state_increment = 0
        ob = self.combine_obs(ob_left, ob_right)
        return ob, {}

    def compute_reward(self, obs, action) -> float:
        state = obs["state"]

        step_cost = 0.1
        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.5 * np.sum(np.power(action - self.last_action, 2))
        self.last_action = action

        suction_reward = 0.3 * float(state["left/gripper_state"][1] > 0.5)
        suction_cost = 3. * float(state["left/gripper_state"][1] < -0.5)
        dropping_cost = 100 if self.dropped_parcel(obs) else 0

        cutoff_dist = 0.07
        pos_diff_left = state["left/tcp_pose"][:3] - self.env_left.curr_reset_pose[:3]
        pos_diff_right = state["right/tcp_pose"][:3] - self.env_right.curr_reset_pose[:3]
        position_cost_left = 10. * np.sum(
            np.where(np.abs(pos_diff_left) > cutoff_dist, np.abs(pos_diff_left - np.sign(pos_diff_left) * cutoff_dist),
                     0.0))
        position_cost_right = 10. * np.sum(
            np.where(np.abs(pos_diff_right) > cutoff_dist,
                     np.abs(pos_diff_right - np.sign(pos_diff_right) * cutoff_dist), 0.0))
        position_cost = position_cost_left + position_cost_right

        orientation_cost_left = 1. - sum(state["left/tcp_pose"][3:] * self.env_left.curr_reset_pose[3:]) ** 2
        orientation_cost_left = max(orientation_cost_left - 0.005, 0.) * 25.
        orientation_cost_right = 1. - sum(state["right/tcp_pose"][3:] * self.env_right.curr_reset_pose[3:]) ** 2
        orientation_cost_right = max(orientation_cost_right - 0.005, 0.) * 25.
        orientation_cost = orientation_cost_left + orientation_cost_right

        max_force_penalty = self.calculate_force_penalty(obs, max_force=10)

        retreat_reward = 10. * (-action[1] - action[7 + 1]) if self.goal_state_increment > 0 else 0.

        cost_info = dict(
            step_cost=step_cost,
            action_cost=action_cost,
            action_diff_cost=action_diff_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            dropping_cost=dropping_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            max_force_penalty=max_force_penalty,
            retreat_reward=retreat_reward,
            total_cost=-(-action_cost - action_diff_cost - step_cost + suction_reward - suction_cost - dropping_cost
                         - orientation_cost - position_cost - max_force_penalty + retreat_reward)
        )
        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])

        if self.reached_goal_state(obs, increment=False):
            self.last_action[:] = 0.
            return 100. - action_cost - action_diff_cost - orientation_cost - position_cost - max_force_penalty + retreat_reward
        else:
            return 0. - action_cost - action_diff_cost - step_cost + suction_reward - suction_cost - dropping_cost \
                - orientation_cost - position_cost - max_force_penalty + retreat_reward

    def dropped_parcel(self, obs) -> bool:
        state = obs["state"]
        # left gripper not gripping, right gripper not gripping
        # TODO check if good or if pressure is better
        return state['left/gripper_state'][1] < 0.5 and state["right/gripper_state"][1] < 0.5

    def calculate_force_penalty(self, obs, max_force=20.):
        # if gripper is gripping, ignore gravity
        state = obs["state"]
        penalty = 0.
        for both in ["left/", "right/"]:
            force = state[both + "tcp_force"]
            if state[both + "gripper_state"] > 0.5:
                force[2] = 0. if force[2] < 0. else force[2]
            penalty += np.linalg.norm(force)
        return min(0., penalty - 2 * max_force) * 0.1

    def reached_goal_state(self, obs, **kwargs) -> bool:
        state = obs["state"]
        # left gripper gripping, right gripper not gripping
        goal_state = state['left/gripper_state'][1] > 0.5 and state["right/gripper_state"][1] < 1.

        if not "increment" in kwargs or kwargs["increment"] == True:
            self.goal_state_increment = self.goal_state_increment + 1 if goal_state else 0
        return self.goal_state_increment > 4

    def _is_truncated(self, obs):
        collision = not self.collision_detector.is_collision_free()
        if collision:
            print(self.collision_detector.collision_msg)
        return self.dropped_parcel(obs) or collision

    def close(self):
        self.env_left._send_gripper_command(np.array(-1))
        self.env_right._send_gripper_command(np.array(-1))
        super().close()
