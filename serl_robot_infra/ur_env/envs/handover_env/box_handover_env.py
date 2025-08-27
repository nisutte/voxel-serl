import threading
import numpy as np
import time

from scipy.spatial.transform import Rotation as R

from ur_env.envs.dual_ur5_env import DualUR5Env
from ur_env.utils.transformations import pose_to_T


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
            if self.env.controller.is_truncated():
                return
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

            if self.env.curr_force[2] >= 5. and self.env.gripper_state[1] < 0.5:
                self.env.send_gripper_command(np.array(1))
                DOWN = -DOWN

            if self.env.curr_pos[2] < 0.01:
                DOWN = -DOWN

            new_pose = self.env.curr_pos + DOWN
            self.env.send_pos_command(new_pose)
            time.sleep(0.05)

        return False


def dropped_parcel(obs) -> bool:
    state = obs["state"]
    dropped = state['left/gripper_state'][1] < 0.5 and state["right/gripper_state"][1] < 0.5
    if dropped:
        print(f"parcel dropped!")
    return dropped


def calculate_force_penalty(obs, max_force=20.):
    # if gripper is gripping, ignore gravity
    state = obs["state"]
    penalty = 0.
    for both in ["left/", "right/"]:
        force = state[both + "tcp_force"]
        if state[both + "gripper_state"][1] > 0.5:
            force[2] = 0. if force[2] < 0. else force[2]
        penalty += np.linalg.norm(force)
    return max(0., penalty - 2 * max_force)


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
        # prevent parcel dropping
        receiving_env, box_env = (self.env_right, self.env_left) if self.inverted else (self.env_left, self.env_right)
        receiving_env.update_currpos()
        gripping_left = receiving_env.gripper_state[1] > 0.5
        would_drop = not gripping_left and action[-1] < -0.5
        if would_drop:
            print("receiving gripper is not gripping, but action is to drop parcel!")
            box_env.neutral_gripper_command = True

        # step
        if self.inverted:
            action = np.concatenate((action[7:], action[:7]))
        obs, reward, done, truncated, infos = super().step(action)

        if "dropping_cost" not in infos:
            infos["dropping_cost"] = 0
        if would_drop:
            reward -= 50
            infos["dropping_cost"] += 50
        return obs, reward, done, truncated, infos

    def reset(self, **kwargs):
        BTleft, BTright = SimpleBehaviorTree(self.env_left), SimpleBehaviorTree(self.env_right)
        self.env_left.update_currpos()
        self.env_right.update_currpos()

        if self.env_left.gripper_state[1] > 0.5 and self.env_right.gripper_state[1] > 0.5:
            # both grippers gripping, release the right one
            self.env_right.send_gripper_command(np.array(-1))

        thread_left = threading.Thread(target=BTleft.retreat, daemon=True)
        thread_right = threading.Thread(target=BTright.retreat, daemon=True)
        thread_left.start()
        thread_right.start()
        thread_left.join()
        thread_right.join()

        self.env_left.update_currpos()
        self.env_right.update_currpos()

        already_picked_up = False
        self.inverted = False
        if self.env_left.gripper_state[1] > 0.5 and self.env_right.gripper_state[1] < 0.5:
            # left gripper gripping, right gripper not gripping
            print("Env is inverted!")
            self.inverted = True
            already_picked_up = True
        elif self.env_right.gripper_state[1] > 0.5 and self.env_left.gripper_state[1] < 0.5:
            # right gripper gripping, left gripper not gripping
            already_picked_up = True
        elif self.env_left.gripper_state[1] > 0.5 and self.env_right.gripper_state[1] > 0.5:
            # both grippers gripping, release the left one
            self.env_left.send_gripper_command(np.array(-1))
            already_picked_up = True


        def reset_env_left():
            global ob_left
            self.env_left.controller.auto_release_gripper(not self.inverted)
            ob_left, _ = self.env_left.reset(**kwargs)
            self.env_left.controller.auto_release_gripper(True)

            if not self.inverted:
                self.env_left.controller.reset_forces()

        def reset_env_right():
            global ob_right
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
        action_diff_cost = 0.3 * np.sum(np.power(action - self.last_action, 2))
        self.last_action = action

        suction_reward = 0.3 * (float(state["left/gripper_state"][1] > 0.5) and action[6 + 7 * self.inverted] > 0.5)
        suction_cost = 2. * (float(state["left/gripper_state"][1] < -0.5) and action[6 + 7 * self.inverted] > -0.5)
        suction_cost += 2. * (float(state["right/gripper_state"][1] < -0.5) and action[6 + 7 * (not self.inverted)] > -0.5)

        cutoff_dist = np.array([0.05, 0.3, 0.05])  # lessen y direction (forward)
        pos_diff_left = state["left/tcp_pose"][:3] - self.env_left.curr_reset_pose[:3]
        pos_diff_right = state["right/tcp_pose"][:3] - self.env_right.curr_reset_pose[:3]
        position_cost_left = 10. * np.sum(
            np.where(np.abs(pos_diff_left) > cutoff_dist, np.abs(pos_diff_left - np.sign(pos_diff_left) * cutoff_dist),
                     0.0))
        position_cost_right = 20. * np.sum(
            np.where(np.abs(pos_diff_right) > cutoff_dist,
                     np.abs(pos_diff_right - np.sign(pos_diff_right) * cutoff_dist), 0.0))
        position_cost = position_cost_left + position_cost_right

        # todo do relative position cost

        w = np.array([0.2, 0.2, 0.05])  # x, y start after 22°, z after 45°
        def orientation_cost_fun(curr_quat, target_quat):
            rel_rot = R.from_quat(target_quat).inv() * R.from_quat(curr_quat)
            cost = sum(w * rel_rot.as_rotvec() ** 2)
            return max(cost - 0.03, 0.)

        orientation_cost_left = 20. * orientation_cost_fun(state["left/tcp_pose"][3:], self.env_left.curr_reset_pose[3:])
        orientation_cost_right = 20. * orientation_cost_fun(state["right/tcp_pose"][3:], self.env_right.curr_reset_pose[3:])
        orientation_cost = orientation_cost_left + orientation_cost_right

        allowed_rot_degrees = 30.
        T_l2r = np.linalg.inv(pose_to_T(state["left/tcp_pose"])) @ self.T_left2right @ pose_to_T(state["right/tcp_pose"])
        rel_rot_y = R.from_matrix(T_l2r[:3, :3]).as_euler("zyz")  # Y should be pi
        relative_orientation_cost = 2. * max(0., (1. - allowed_rot_degrees / 180.) * np.pi - float(rel_rot_y[1]))

        max_force_penalty = 0.4 * calculate_force_penalty(obs, max_force=10)
        retreat_reward = 0.5 * (-action[1] - action[7 + 1]) if self.goal_state_increment > 0 else 0.
        both_gripping = state["left/gripper_state"][1] > 0.5 and state["right/gripper_state"][1] > 0.5
        early_retreat_penalty = 1.0 * (action[1] + action[7 + 1]) if both_gripping else 0.
        both_gripping_huge_action_penalty = 0.5 * (np.sum(np.power(action[:6], 2)) + np.sum(np.power(action[7:13], 2))) if both_gripping else 0.

        cost_info = dict(
            step_cost=step_cost,
            action_cost=action_cost,
            action_diff_cost=action_diff_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            orientation_cost=orientation_cost,
            position_cost=position_cost,
            relative_orienation_cost=relative_orientation_cost,
            max_force_penalty=max_force_penalty,
            retreat_reward=retreat_reward,
            early_retreat_penalty=early_retreat_penalty,
            both_gripping_huge_action_penalty=both_gripping_huge_action_penalty,
            total_cost=-(-action_cost - action_diff_cost - step_cost + suction_reward - suction_cost
                 - orientation_cost - position_cost - max_force_penalty - relative_orientation_cost + retreat_reward - early_retreat_penalty - both_gripping_huge_action_penalty)
        )

        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])

        if self.reached_goal_state(obs, increment=False):
            self.last_action[:] = 0.
            return 500. - action_cost - action_diff_cost - orientation_cost - position_cost - max_force_penalty \
                - relative_orientation_cost + retreat_reward - early_retreat_penalty - both_gripping_huge_action_penalty
        else:
            return (0. - action_cost - action_diff_cost - step_cost + suction_reward - suction_cost - orientation_cost \
                    - position_cost - max_force_penalty - relative_orientation_cost + retreat_reward - early_retreat_penalty - \
                    both_gripping_huge_action_penalty)

    def reached_goal_state(self, obs, **kwargs) -> bool:
        state = obs["state"]
        # left gripper gripping, right gripper not gripping
        goal_state = state['left/gripper_state'][1] > 0.5 and state["right/gripper_state"][1] < 0.5

        if not "increment" in kwargs or kwargs["increment"] == True:
            self.goal_state_increment = self.goal_state_increment + 1 if goal_state else 0
        return self.goal_state_increment > 4

    def _is_truncated(self, obs):
        collision = not self.collision_detector.is_collision_free()
        if collision:
            print(self.collision_detector.collision_msg)
        dropped = dropped_parcel(obs)
        return collision or dropped

    def close(self):
        if not self.fake_env:
            self.env_left.send_gripper_command(np.array(-1))
            self.env_right.send_gripper_command(np.array(-1))
        super().close()


class UR5Handover90Degrees(UR5HandoverEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_reward(self, obs, action) -> float:
        state = obs["state"]

        step_cost = 0.1
        action_cost = 0.1 * np.sum(np.power(action, 2))
        action_diff_cost = 0.3 * np.sum(np.power(action - self.last_action, 2))
        self.last_action = action

        suction_reward = 0.3 * (float(state["left/gripper_state"][1] > 0.5) and action[6 + 7 * self.inverted] > 0.5)
        suction_cost = 2. * (float(state["left/gripper_state"][1] < -0.5) and action[6 + 7 * self.inverted] > -0.5)
        suction_cost += 2. * (float(state["right/gripper_state"][1] < -0.5) and action[6 + 7 * (not self.inverted)] > -0.5)

        relative_position_cost = 5 * max(0.0, -0.05 + np.linalg.norm(state["l2r/tcp_pose"][:3])) if not self.goal_state_increment else 0.

        allowed_rot_degrees = 20.
        T_l2r = pose_to_T(state["l2r/tcp_pose"])
        rel_rot_y = R.from_matrix(T_l2r[:3, :3]).as_euler("zyz")  # Y should be pi/2 for 90 degrees
        relative_orientation_cost = 2. * max(0., abs(abs(rel_rot_y[1]) - np.pi / 2. ) - allowed_rot_degrees)

        max_force_penalty = 0.4 * calculate_force_penalty(obs, max_force=10)
        retreat_reward = 0.5 * (-action[1] - action[7 + 1]) if self.goal_state_increment > 0 else 0.
        both_gripping = state["left/gripper_state"][1] > 0.5 and state["right/gripper_state"][1] > 0.5
        early_retreat_penalty = 1.0 * (action[1] + action[7 + 1]) if both_gripping else 0.

        cost_info = dict(
            step_cost=step_cost,
            action_cost=action_cost,
            action_diff_cost=action_diff_cost,
            suction_reward=suction_reward,
            suction_cost=suction_cost,
            relative_position_cost=relative_position_cost,
            relative_orientation_cost=relative_orientation_cost,
            max_force_penalty=max_force_penalty,
            retreat_reward=retreat_reward,
            early_retreat_penalty=early_retreat_penalty,
            total_cost=-(-action_cost - action_diff_cost - step_cost + suction_reward - suction_cost
                 - relative_position_cost - relative_orientation_cost - max_force_penalty + retreat_reward - early_retreat_penalty)
        )

        for key, info in cost_info.items():
            self.cost_infos[key] = info + (0. if key not in self.cost_infos else self.cost_infos[key])

        if self.reached_goal_state(obs, increment=False):
            self.last_action[:] = 0.
            return 1000. - action_cost - action_diff_cost - relative_position_cost - relative_orientation_cost- max_force_penalty \
                - relative_orientation_cost + retreat_reward - early_retreat_penalty
        else:
            return (0. - action_cost - action_diff_cost - step_cost + suction_reward - suction_cost - relative_position_cost \
                    - relative_orientation_cost - max_force_penalty + retreat_reward - early_retreat_penalty)