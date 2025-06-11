import gymnasium as gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os
import threading
from pynput import keyboard
from pprint import pprint

from ur_env.envs.camera_env.box_picking_camera_env import UR5Env
from ur_env.envs.dual_wrappers import DualToMrpWrapper, DualSpaceMouseIntervention, DualScaleObservationWrapper
from ur_env.envs.handover_env.box_handover_env import UR5HandoverEnv
from ur_env.envs.relative_env import DualRelativeFrame

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from serl_robot_infra.ur_env.envs.handover_env import UR5DualCameraConfigRight, UR5DualCameraConfigLeft

import ur_env

exit_program = threading.Event()


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()


if __name__ == "__main__":
    fake_env = False
    camera_mode = "none"

    left_env = UR5Env(
        fake_env = fake_env,
        config = UR5DualCameraConfigLeft,
        camera_mode=camera_mode,
        visualize_camera_mode=False,
    )

    right_env = UR5Env(
        fake_env = fake_env,
        config = UR5DualCameraConfigRight,
        camera_mode=camera_mode,
        visualize_camera_mode=False,
    )

    env = UR5HandoverEnv(
        env_left=left_env,
        env_right=right_env,
    )

    env = DualRelativeFrame(env)
    env = DualToMrpWrapper(env)
    env = DualScaleObservationWrapper(env)

    if not fake_env:
        env = DualSpaceMouseIntervention(env)

    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = 10
    total_count = 0
    pbar = tqdm(total=success_needed)

    info_dict = {'state': env.unwrapped.env_left.curr_pos, 'gripper_state': env.unwrapped.env_left.gripper_state,
                 'force': env.unwrapped.env_left.curr_force, 'reset_pose': env.unwrapped.env_left.curr_reset_pose}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"box_picking_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    try:
        running_reward = 0.
        while success_count < success_needed:
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before

            action = np.array([0., 0., 0., 0., 0., 0., 0.])     # for testing
            next_obs, rew, done, truncated, info = env.step(np.concatenate((action, action)))
            actions = info["intervene_action"]

            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            transitions.append(transition)
            # pprint(next_obs["state"])

            obs = next_obs
            running_reward += rew

            if done or truncated:
                success_count += int(rew > 0.99)
                total_count += 1
                print(
                    f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
                )
                pbar.update(int(rew > 0.99))
                obs, _ = env.reset()
                print("Reward total:", running_reward)
                running_reward = 0.

        with open(file_path, "wb") as f:
            pkl.dump(transitions, f)
            print(f"saved {success_needed} demos to {file_path}")

    except KeyboardInterrupt as e:
        print(f'\nProgram was interrupted, cleaning up...  ', e.__str__())

    finally:
        pbar.close()
        env.close()
        listener_1.stop()
        listener_2.stop()
